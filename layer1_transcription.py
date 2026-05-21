"""
=============================================================================
LAYER 1 — Real-Time Transcription (Deepgram Nova-3)
=============================================================================

This module handles live audio transcription using Deepgram's WebSocket
streaming API. It captures audio from the microphone (or accepts a pre-
recorded audio file), streams it to Deepgram's Nova-3 model, and returns
a clean, structured transcript with:
  - Speaker diarization (Speaker 0, Speaker 1, etc.)
  - Punctuation and smart formatting
  - Word-level timestamps

The output is a Python dict ready to be consumed by Layer 2.
=============================================================================
"""

import os
import json
import asyncio
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Deepgram SDK imports
# ---------------------------------------------------------------------------
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
    Microphone,
    PrerecordedOptions,
    FileSource,
)


# ===========================================================================
# OPTION A — Transcribe a pre-recorded audio file (most common for testing)
# ===========================================================================

async def transcribe_audio_file(file_path: str) -> dict:
    """
    Transcribe a local audio file using Deepgram's pre-recorded API.

    This is the recommended entry point for processing existing call
    recordings. It sends the entire file in one request and returns the
    full transcript with speaker labels.

    Args:
        file_path: Absolute or relative path to the audio file
                   (WAV, MP3, FLAC, OGG, etc.)

    Returns:
        dict with keys:
            - transcript (str): Full transcript text
            - speakers (list[dict]): Per-utterance breakdown with
              speaker, text, start, end, confidence
            - raw_response (dict): Unprocessed Deepgram API response
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPGRAM_API_KEY not found. Add it to your .env file."
        )

    try:
        # Initialise the Deepgram client
        deepgram = DeepgramClient(api_key)

        # Read the audio file into memory
        with open(file_path, "rb") as audio_file:
            buffer_data = audio_file.read()

        payload: FileSource = {"buffer": buffer_data}

        # Configure transcription options
        # ---------------------------------------------------------------
        # model        : nova-3 (latest, most accurate)
        # diarize      : label each speaker
        # punctuate    : add punctuation
        # smart_format : format numbers, dates, etc.
        # utterances   : group words into speaker turns
        # ---------------------------------------------------------------
        options = PrerecordedOptions(
            model="nova-3",
            language="en",
            diarize=True,
            punctuate=True,
            smart_format=True,
            utterances=True,
        )

        # Send to Deepgram and await the response
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        result = response.to_dict() if hasattr(response, "to_dict") else json.loads(response.to_json())

        # Parse the structured result
        return _parse_prerecorded_response(result)

    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Deepgram transcription failed: {e}") from e


def _parse_prerecorded_response(result: dict) -> dict:
    """
    Parse Deepgram's pre-recorded API response into our pipeline format.

    Extracts utterances (speaker turns) and builds a clean transcript
    with speaker labels and timestamps.
    """
    utterances = []
    full_transcript_parts = []

    # --- Extract utterances (speaker-labelled segments) ---
    raw_utterances = (
        result.get("results", {}).get("utterances", [])
    )

    for utt in raw_utterances:
        speaker_label = f"Speaker {utt.get('speaker', '?')}"
        text = utt.get("transcript", "").strip()
        start = utt.get("start", 0.0)
        end = utt.get("end", 0.0)
        confidence = utt.get("confidence", 0.0)

        utterances.append({
            "speaker": speaker_label,
            "text": text,
            "start": round(start, 2),
            "end": round(end, 2),
            "confidence": round(confidence, 4),
        })

        # Build a human-readable transcript line
        timestamp = _format_timestamp(start)
        full_transcript_parts.append(
            f"[{timestamp}] {speaker_label}: {text}"
        )

    full_transcript = "\n".join(full_transcript_parts)

    return {
        "transcript": full_transcript,
        "speakers": utterances,
        "raw_response": result,
    }


# ===========================================================================
# OPTION B — Live microphone transcription (real-time streaming)
# ===========================================================================

async def transcribe_live(duration_seconds: int = 30) -> dict:
    """
    Stream live audio from the default microphone to Deepgram and
    return the accumulated transcript when done.

    Args:
        duration_seconds: How long to record (default 30s).

    Returns:
        Same dict shape as transcribe_audio_file().
    """
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DEEPGRAM_API_KEY not found. Add it to your .env file."
        )

    # Storage for incoming results
    collected_utterances: list[dict] = []
    full_text_parts: list[str] = []

    try:
        # Initialise Deepgram client with keep-alive for streaming
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(api_key, config)

        # Create a live/streaming connection
        dg_connection = deepgram.listen.websocket.v("1")

        # -------------------------------------------------------------------
        # Event handlers
        # -------------------------------------------------------------------
        def on_message(self, result, **kwargs):
            """Fires every time Deepgram returns a transcript chunk."""
            sentence = result.channel.alternatives[0].transcript
            if not sentence:
                return

            # Extract speaker info if available
            words = result.channel.alternatives[0].words
            speaker = "Speaker ?"
            start_time = 0.0
            end_time = 0.0

            if words:
                speaker = f"Speaker {words[0].speaker}" if hasattr(words[0], "speaker") else "Speaker ?"
                start_time = words[0].start
                end_time = words[-1].end

            collected_utterances.append({
                "speaker": speaker,
                "text": sentence.strip(),
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "confidence": round(result.channel.alternatives[0].confidence, 4),
            })

            timestamp = _format_timestamp(start_time)
            full_text_parts.append(f"[{timestamp}] {speaker}: {sentence.strip()}")
            print(f"  🎙️  {speaker}: {sentence.strip()}")

        def on_error(self, error, **kwargs):
            print(f"  ❌  Deepgram streaming error: {error}")

        # Register event handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)

        # Configure live transcription options
        options = LiveOptions(
            model="nova-3",
            language="en",
            diarize=True,
            punctuate=True,
            smart_format=True,
            encoding="linear16",
            channels=1,
            sample_rate=16000,
            interim_results=False,    # Only final results
            endpointing=300,          # 300ms silence = end of utterance
        )

        # Start the connection
        if not dg_connection.start(options):
            raise RuntimeError("Failed to connect to Deepgram WebSocket.")

        # Open the microphone and stream audio
        microphone = Microphone(dg_connection.send)
        microphone.start()

        print(f"  🎤  Recording for {duration_seconds} seconds...")
        await asyncio.sleep(duration_seconds)

        # Cleanly shut down
        microphone.finish()
        dg_connection.finish()

        print(f"  ✅  Recording complete. {len(collected_utterances)} utterances captured.")

        return {
            "transcript": "\n".join(full_text_parts),
            "speakers": collected_utterances,
            "raw_response": {},
        }

    except Exception as e:
        raise RuntimeError(f"Live transcription failed: {e}") from e


# ===========================================================================
# Utility helpers
# ===========================================================================

def _format_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format for readable transcript output."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


# ===========================================================================
# Standalone test
# ===========================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If a file path is provided, transcribe that file
        audio_path = sys.argv[1]
        print(f"📂  Transcribing file: {audio_path}")
        result = asyncio.run(transcribe_audio_file(audio_path))
    else:
        # Otherwise, do a 15-second live recording
        print("🎤  Starting live transcription (15 seconds)...")
        result = asyncio.run(transcribe_live(duration_seconds=15))

    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)
    print(result["transcript"])
    print(f"\n📊  Total utterances: {len(result['speakers'])}")
