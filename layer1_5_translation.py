"""
=============================================================================
LAYER 1.5 — Hindi/Hinglish → English Translation (Gemini 2.5 Flash)
=============================================================================

This module sits between Layer 1 (Transcription) and Layer 2 (Breakdown).
It translates Hindi or Hinglish content in the transcript to clear English
while preserving:
  - All English words already present (no rephrasing)
  - Startup/business keywords (GTM, ICP, MVP, CAC, etc.)
  - Speaker labels and timestamps
  - The exact meaning and tone of each statement

The output schema matches the Layer 1 output schema exactly, so downstream
layers (2–5) require zero changes.
=============================================================================
"""

import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

import google.generativeai as genai


# ===========================================================================
# System Prompts
# ===========================================================================

TRANSLATOR_SYSTEM_PROMPT = """You are a professional Hindi/Hinglish-to-English translator specializing in startup mentor-founder call transcripts from Conquest, BITS Pilani, Asia's largest student-run startup accelerator.

YOUR TASK:
Translate the given transcript text from Hindi or Hinglish (mixed Hindi-English) into clear, natural English.

CRITICAL RULES:
1. PRESERVE all English words exactly as they appear in the original. Do not rephrase, simplify, or change any English word or phrase that is already in the text.
2. PRESERVE all startup, business, and technical keywords exactly: GTM, ICP, CAC, LTV, MVP, ARR, MRR, TAM, SAM, SOM, Series A, pitch deck, runway, burn rate, unit economics, PMF, product-market fit, scaling, outreach, onboarding, churn, retention, conversion, pipeline, funnel, B2B, B2C, SaaS, API, CRM, etc.
3. PRESERVE the exact meaning and tone of every statement. Do not add interpretation, do not remove nuance, do not editorialize.
4. PRESERVE all formatting exactly: timestamps in [MM:SS] format, speaker labels (Speaker 0, Speaker 1, names, etc.), and line breaks.
5. If a line is already entirely in English, return it UNCHANGED — character for character.
6. Translate ONLY the Hindi portions of each line to English. Keep the English portions untouched.
7. Return ONLY the translated text. No explanations, no commentary, no markdown fences, no preamble."""

BATCH_TRANSLATOR_SYSTEM_PROMPT = """You are a professional Hindi/Hinglish-to-English translator for startup mentor-founder call transcripts.

YOUR TASK:
You will receive a JSON array of text strings. Translate each string from Hindi/Hinglish to English.

CRITICAL RULES:
1. PRESERVE all English words exactly as they appear. Do not rephrase or simplify them.
2. PRESERVE all startup/business keywords exactly: GTM, ICP, CAC, LTV, MVP, ARR, pitch deck, runway, burn rate, unit economics, PMF, B2B, B2C, SaaS, etc.
3. PRESERVE the exact meaning and tone.
4. If a string is already entirely in English, return it UNCHANGED.
5. Return a valid JSON array of the same length with translated strings. Nothing else — no markdown fences, no explanations, no preamble."""


# ===========================================================================
# Main Entry Point
# ===========================================================================

def translate_transcript(transcript_data: dict) -> dict:
    """
    Translate a Layer 1 transcript output from Hindi/Hinglish to English.

    Takes the standard transcript_data dict (with keys: transcript, speakers,
    raw_response) and returns an identical dict with translated text.

    Args:
        transcript_data: Layer 1 output dict with keys:
            - transcript (str): Full formatted transcript text
            - speakers (list[dict]): Per-utterance breakdown
            - raw_response (dict): Unprocessed API response

    Returns:
        dict: Same schema as input, with Hindi/Hinglish text translated
              to English, plus a 'translation_metadata' key.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    genai.configure(api_key=api_key)

    transcript_text = transcript_data.get("transcript", "")
    speakers_list = transcript_data.get("speakers", [])

    if not transcript_text.strip():
        print("  ⚠️  Empty transcript — skipping translation.")
        return {
            **transcript_data,
            "translation_metadata": {
                "translated": False,
                "reason": "empty_transcript",
            }
        }

    print("  🌐  Translating transcript (Hindi/Hinglish → English)...")

    # --- Step 1: Translate the full transcript text block ---
    translated_transcript = _translate_text_block(transcript_text)

    # --- Step 2: Batch-translate speaker utterance texts ---
    translated_speakers = speakers_list  # default fallback
    if speakers_list:
        translated_speakers = _translate_speaker_texts(speakers_list)

    print(f"  ✅  Layer 1.5 translation complete. {len(speakers_list)} utterances processed.")

    return {
        "transcript": translated_transcript,
        "speakers": translated_speakers,
        "raw_response": transcript_data.get("raw_response", {}),
        "translation_metadata": {
            "translated": True,
            "source_language": "hi-en",
            "utterances_translated": len(speakers_list),
            "model_used": "gemini-2.5-flash",
        }
    }


# ===========================================================================
# Internal Helpers
# ===========================================================================

def _translate_text_block(text: str) -> str:
    """
    Translate a full transcript text block preserving all formatting
    (timestamps, speaker labels, line breaks).
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=TRANSLATOR_SYSTEM_PROMPT,
    )

    prompt = f"Translate this transcript to English:\n\n{text}"
    response = model.generate_content(prompt)
    result = response.text.strip()

    # Clean up any accidental markdown code fences
    if result.startswith("```"):
        result = result.split("\n", 1)[1]
    if result.endswith("```"):
        result = result.rsplit("```", 1)[0]

    return result.strip()


def _translate_speaker_texts(speakers_list: list[dict]) -> list[dict]:
    """
    Batch-translate all speaker utterance texts in a single API call.

    Sends the texts as a JSON array and parses the translated array back,
    preserving all non-text fields (speaker, start, end, confidence).
    """
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=BATCH_TRANSLATOR_SYSTEM_PROMPT,
    )

    # Extract just the text values
    texts = [s.get("text", "") for s in speakers_list]

    prompt = (
        f"Translate each text in this JSON array to English:\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}"
    )
    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Clean up markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        translated_texts = json.loads(raw)
        if not isinstance(translated_texts, list) or len(translated_texts) != len(speakers_list):
            print(
                f"  ⚠️  Translation array length mismatch "
                f"(expected {len(speakers_list)}, got {len(translated_texts) if isinstance(translated_texts, list) else 'non-list'}). "
                f"Falling back to original texts."
            )
            return speakers_list
    except json.JSONDecodeError:
        print("  ⚠️  Failed to parse batch translation response. Falling back to original texts.")
        return speakers_list

    # Rebuild speakers list with translated texts, preserving all other fields
    translated_speakers = []
    for original, new_text in zip(speakers_list, translated_texts):
        translated_speakers.append({
            **original,
            "text": new_text if isinstance(new_text, str) else original.get("text", ""),
        })

    return translated_speakers


# ===========================================================================
# Pretty-Print Helper
# ===========================================================================

def print_translation_result(transcript_data: dict) -> None:
    """Print translation metadata in a human-readable format."""
    meta = transcript_data.get("translation_metadata", {})
    if not meta.get("translated"):
        print(f"  ℹ️  Translation skipped: {meta.get('reason', 'unknown')}")
        return

    print(f"\n{'='*60}")
    print("  LAYER 1.5 — TRANSLATION RESULT")
    print(f"{'='*60}")
    print(f"  Source language : {meta.get('source_language', '?')}")
    print(f"  Utterances      : {meta.get('utterances_translated', 0)}")
    print(f"  Model           : {meta.get('model_used', '?')}")
    print(f"{'='*60}\n")


# ===========================================================================
# Standalone Test
# ===========================================================================

if __name__ == "__main__":
    test_data = {
        "transcript": (
            "[00:00] Speaker 0: Namaste, aaj hum GTM strategy ke baare mein baat karenge.\n"
            "[00:05] Speaker 1: Haan, humara ICP abhi clear nahi hai, hum cold outreach kar rahe hain.\n"
            "[00:12] Speaker 0: Aapka CAC kitna hai? Aur customer retention rate kya hai?\n"
            "[00:20] Speaker 1: CAC around 500 rupees hai, retention 60 percent hai.\n"
            "[00:28] Speaker 0: That's quite high. You need to bring CAC down below 200."
        ),
        "speakers": [
            {"speaker": "Speaker 0", "text": "Namaste, aaj hum GTM strategy ke baare mein baat karenge.", "start": 0.0, "end": 4.5, "confidence": 0.95},
            {"speaker": "Speaker 1", "text": "Haan, humara ICP abhi clear nahi hai, hum cold outreach kar rahe hain.", "start": 5.0, "end": 11.0, "confidence": 0.93},
            {"speaker": "Speaker 0", "text": "Aapka CAC kitna hai? Aur customer retention rate kya hai?", "start": 12.0, "end": 19.5, "confidence": 0.94},
            {"speaker": "Speaker 1", "text": "CAC around 500 rupees hai, retention 60 percent hai.", "start": 20.0, "end": 27.0, "confidence": 0.96},
            {"speaker": "Speaker 0", "text": "That's quite high. You need to bring CAC down below 200.", "start": 28.0, "end": 34.0, "confidence": 0.98},
        ],
        "raw_response": {},
    }

    result = translate_transcript(test_data)
    print_translation_result(result)

    print("TRANSLATED TRANSCRIPT:")
    print(result["transcript"])
    print(f"\nSPEAKER TEXTS:")
    for s in result["speakers"]:
        print(f"  {s['speaker']}: {s['text']}")
    print(f"\nMETADATA: {json.dumps(result['translation_metadata'], indent=2)}")
