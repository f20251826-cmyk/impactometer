"""
=============================================================================
MAIN PIPELINE ORCHESTRATOR
=============================================================================

This is the entry point for the AI Call Analysis Pipeline. It orchestrates
all four layers in sequence, saving results directly to Supabase and Google Docs
at each step:

  Layer 1 → Transcription (Deepgram Nova-3) -> Save Doc + Supabase Record
  Layer 2 → Talking Points Summarizer (Gemini 2.5 Flash) -> Save Doc + Supabase Update
  Layer 3 → Call Classification & Quality Score (Gemini 2.5 Flash) -> Save Doc + Supabase final update
  Layer 4 → Judge Agent (Placeholder)

All outputs between layers are passed as Python dicts/JSON so they
are easy to extend, inspect, and serialise.
=============================================================================
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables FIRST, before importing layer modules
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Import database and storage helpers
# ---------------------------------------------------------------------------
from db_helpers import (
    create_call_record,
    update_layer1_data,
    update_layer2_data,
    update_layer3_data,
    create_gdoc
)

# ---------------------------------------------------------------------------
# Import pipeline layers
# ---------------------------------------------------------------------------
from layer1_transcription import transcribe_audio_file, transcribe_live
from layer2_breakdown import breakdown_transcript, print_moments
from layer3_classification import classify_call, print_classification
from layer4_placeholder import judge_agent


# ===========================================================================
# Demo transcript (for testing without Deepgram API key)
# ===========================================================================

DEMO_TRANSCRIPT = {
    "transcript": (
        "[00:00] Speaker 0: Good morning, thanks for hopping on this call.\n"
        "[00:04] Speaker 1: Morning! Yeah, I wanted to chat about the onboarding flow.\n"
        "[00:10] Speaker 0: Sure, what's on your mind?\n"
        "[00:13] Speaker 1: So we've been getting a lot of drop-off at step three, where users have to verify their email.\n"
        "[00:22] Speaker 0: Hmm, what's the drop-off rate looking like?\n"
        "[00:25] Speaker 1: About 40 percent. It's really hurting our activation numbers.\n"
        "[00:32] Speaker 0: That's significant. Do we know why they're dropping off? Is the email not arriving?\n"
        "[00:38] Speaker 1: Partly. Our deliverability is around 85 percent, so some emails land in spam. But even users who get the email don't always click through.\n"
        "[00:50] Speaker 0: What if we added a magic link option? Skip the code entry entirely.\n"
        "[00:56] Speaker 1: I actually prototyped that last week. It reduced friction in our internal tests.\n"
        "[01:04] Speaker 0: Nice. Let's push that to staging this week. Can you own that?\n"
        "[01:10] Speaker 1: Absolutely. I'll have it on staging by Thursday.\n"
        "[01:15] Speaker 0: Perfect. And let's also look into the deliverability issue separately. Maybe we need to switch email providers.\n"
        "[01:24] Speaker 1: Agreed. I'll set up a comparison between SendGrid and Resend.\n"
        "[01:30] Speaker 0: Sounds good. Anything else?\n"
        "[01:33] Speaker 1: That's it for now. Thanks for the quick call.\n"
        "[01:36] Speaker 0: Thanks, talk soon. Bye!\n"
        "[01:38] Speaker 1: Bye!"
    ),
    "speakers": [
        {"speaker": "Speaker 0", "text": "Good morning, thanks for hopping on this call.", "start": 0.0, "end": 3.5, "confidence": 0.98},
        {"speaker": "Speaker 1", "text": "Morning! Yeah, I wanted to chat about the onboarding flow.", "start": 4.0, "end": 9.5, "confidence": 0.97},
        {"speaker": "Speaker 0", "text": "Sure, what's on your mind?", "start": 10.0, "end": 12.5, "confidence": 0.99},
        {"speaker": "Speaker 1", "text": "So we've been getting a lot of drop-off at step three, where users have to verify their email.", "start": 13.0, "end": 21.0, "confidence": 0.95},
        {"speaker": "Speaker 0", "text": "Hmm, what's the drop-off rate looking like?", "start": 22.0, "end": 24.5, "confidence": 0.96},
        {"speaker": "Speaker 1", "text": "About 40 percent. It's really hurting our activation numbers.", "start": 25.0, "end": 31.0, "confidence": 0.94},
        {"speaker": "Speaker 0", "text": "That's significant. Do we know why they're dropping off? Is the email not arriving?", "start": 32.0, "end": 37.5, "confidence": 0.97},
        {"speaker": "Speaker 1", "text": "Partly. Our deliverability is around 85 percent, so some emails land in spam. But even users who get the email don't always click through.", "start": 38.0, "end": 49.5, "confidence": 0.93},
        {"speaker": "Speaker 0", "text": "What if we added a magic link option? Skip the code entry entirely.", "start": 50.0, "end": 55.5, "confidence": 0.98},
        {"speaker": "Speaker 1", "text": "I actually prototyped that last week. It reduced friction in our internal tests.", "start": 56.0, "end": 63.0, "confidence": 0.96},
        {"speaker": "Speaker 0", "text": "Nice. Let's push that to staging this week. Can you own that?", "start": 64.0, "end": 69.5, "confidence": 0.97},
        {"speaker": "Speaker 1", "text": "Absolutely. I'll have it on staging by Thursday.", "start": 70.0, "end": 74.5, "confidence": 0.98},
        {"speaker": "Speaker 0", "text": "Perfect. And let's also look into the deliverability issue separately. Maybe we need to switch email providers.", "start": 75.0, "end": 83.5, "confidence": 0.95},
        {"speaker": "Speaker 1", "text": "Agreed. I'll set up a comparison between SendGrid and Resend.", "start": 84.0, "end": 89.5, "confidence": 0.97},
        {"speaker": "Speaker 0", "text": "Sounds good. Anything else?", "start": 90.0, "end": 92.5, "confidence": 0.99},
        {"speaker": "Speaker 1", "text": "That's it for now. Thanks for the quick call.", "start": 93.0, "end": 96.5, "confidence": 0.98},
        {"speaker": "Speaker 0", "text": "Thanks, talk soon. Bye!", "start": 97.0, "end": 99.0, "confidence": 0.99},
        {"speaker": "Speaker 1", "text": "Bye!", "start": 99.5, "end": 100.0, "confidence": 0.99},
    ],
    "raw_response": {},
}


# ===========================================================================
# Pipeline orchestrator
# ===========================================================================

async def run_pipeline(
    audio_path: str = None,
    live: bool = False,
    duration: int = 30,
    demo: bool = False,
    save_output: bool = True,
    user_context: dict = None,
) -> dict:
    """
    Run the full AI Call Analysis Pipeline end to end, syncing with Supabase and Google Docs.

    Args:
        audio_path: Path to an audio file
        live: If True, use live microphone transcription
        duration: Duration in seconds for live recording
        demo: If True, skip Layer 1 and use a built-in demo transcript
        save_output: If True, save the final JSON output to a local backup file
        user_context: Optional dictionary containing custom notes or summary formatting requirements

    Returns:
        dict containing the complete pipeline output with all layer results
    """
    pipeline_start = datetime.now()
    print("\n" + "=" * 60)
    print("  🚀  AI CALL ANALYSIS PIPELINE (SUPABASE & GDOCS READY)")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # DB INIT — Create initial Call Record in Supabase
    # -----------------------------------------------------------------------
    call_title = f"Call Analysis - {pipeline_start.strftime('%Y-%m-%d %H:%M')}"
    if audio_path:
        call_title = f"Call File: {os.path.basename(audio_path)}"
    elif live:
        call_title = f"Live Call - {pipeline_start.strftime('%Y-%m-%d %H:%M')}"
        
    print(f"\n  🗄️  Initialising session in Supabase...")
    call_id = create_call_record(call_title)

    # -----------------------------------------------------------------------
    # LAYER 1 — Transcription
    # -----------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("  LAYER 1 / 4 — Transcription (Deepgram Nova-3)")
    print(f"{'─'*60}")

    if demo:
        print("  📋  Using demo transcript (skipping Deepgram).")
        transcript_data = DEMO_TRANSCRIPT
    elif live:
        print(f"  🎤  Starting live transcription ({duration}s)...")
        transcript_data = await transcribe_live(duration_seconds=duration)
    elif audio_path:
        print(f"  📂  Transcribing file: {audio_path}")
        transcript_data = await transcribe_audio_file(audio_path)
    else:
        raise ValueError(
            "No input specified. Use --demo, --live, or provide an audio file path."
        )

    # Display the transcript
    print(f"\n  📝  Transcript ({len(transcript_data.get('speakers', []))} utterances):")
    print(f"  {'·'*50}")
    for line in transcript_data["transcript"].split("\n")[:10]:  # Show first 10 lines
        print(f"    {line}")
    total_lines = len(transcript_data["transcript"].split("\n"))
    if total_lines > 10:
        print(f"    ... ({total_lines - 10} more lines)")

    # 1B. Storage: Save raw transcript to Google Doc & push to Supabase
    print("\n  💾  Syncing Layer 1 to Google Docs & Supabase...")
    gdoc_l1_title = f"Transcript - {call_title}"
    gdoc_l1_content = (
        f"CALL TRANSCRIPT\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Call ID: {call_id}\n"
        f"{'='*40}\n\n"
        f"{transcript_data['transcript']}"
    )
    l1_doc_id = create_gdoc(gdoc_l1_title, gdoc_l1_content)
    update_layer1_data(
        call_id=call_id,
        doc_id=l1_doc_id,
        transcript=transcript_data["transcript"],
        speakers=transcript_data["speakers"]
    )

    # -----------------------------------------------------------------------
    # LAYER 2 — Talking Points Summarizer (UPDATED)
    # -----------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("  LAYER 2 / 4 — Talking Points Summarizer (Gemini Flash)")
    print(f"{'─'*60}")

    # Build the aggregated context payload
    layer2_payload = {
        "transcript": transcript_data["transcript"],
        "user_context": user_context or {
            "summary_length": "comprehensive (3-4 paragraphs)",
            "custom_notes": "None"
        }
    }

    try:
        breakdown_data = breakdown_transcript(layer2_payload)
        print_moments(breakdown_data)
    except Exception as e:
        print(f"  ❌  Layer 2 failed: {e}")
        print("  ⚠️  Continuing pipeline with empty breakdown...")
        breakdown_data = {"summary": "Summarizer failed.", "talking_points": [], "word_count": 0}

    # 2B. Storage: Save talking point summary to Google Doc & push to Supabase
    print("  💾  Syncing Layer 2 to Google Docs & Supabase...")
    gdoc_l2_title = f"Summary - {call_title}"
    gdoc_l2_bullets = "\n".join([f"• {p}" for p in breakdown_data.get('talking_points', [])])
    gdoc_l2_content = (
        f"TALKING POINTS SUMMARY\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Call ID: {call_id}\n"
        f"Sizing parameter: {layer2_payload['user_context'].get('summary_length')}\n"
        f"{'='*40}\n\n"
        f"### EXECUTIVE SUMMARY\n"
        f"{breakdown_data['summary']}\n\n"
        f"### KEY DISCUSSION POINTS\n"
        f"{gdoc_l2_bullets}"
    )
    l2_doc_id = create_gdoc(gdoc_l2_title, gdoc_l2_content)
    update_layer2_data(
        call_id=call_id,
        doc_id=l2_doc_id,
        summary=breakdown_data["summary"]
    )

    # -----------------------------------------------------------------------
    # LAYER 3 — Call Classification & Quality Score (UPDATED)
    # -----------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("  LAYER 3 / 4 — Call Classification & Quality (Gemini Flash)")
    print(f"{'─'*60}")

    try:
        classification_data = classify_call(transcript_data, breakdown_data)
        print_classification(classification_data)
    except Exception as e:
        print(f"  ❌  Layer 3 failed: {e}")
        print("  ⚠️  Continuing pipeline with default classification...")
        classification_data = {
            "call_type": "Unknown",
            "topic": "Unknown",
            "sentiment": "neutral",
            "summary": "Classification failed.",
            "participants_count": 0,
            "language_detected": "Unknown",
            "doc_quality_score": 0.0,
        }

    # -----------------------------------------------------------------------
    # LAYER 4 — Judge Agent (Placeholder)
    # -----------------------------------------------------------------------
    print(f"\n{'─'*60}")
    print("  LAYER 4 / 4 — Judge Agent (Placeholder)")
    print(f"{'─'*60}")

    judgement_data = judge_agent(transcript_data, breakdown_data, classification_data)

    # -----------------------------------------------------------------------
    # Finalize consolidated run & metadata
    # -----------------------------------------------------------------------
    pipeline_end = datetime.now()
    elapsed = (pipeline_end - pipeline_start).total_seconds()

    final_output = {
        "pipeline_metadata": {
            "timestamp": pipeline_start.isoformat(),
            "duration_seconds": round(elapsed, 2),
            "input_mode": "demo" if demo else ("live" if live else "file"),
            "input_source": audio_path if audio_path else ("microphone" if live else "demo"),
            "call_id": call_id,
        },
        "layer1_transcription": {
            "transcript": transcript_data.get("transcript", ""),
            "utterance_count": len(transcript_data.get("speakers", [])),
            "speakers": transcript_data.get("speakers", []),
            "google_doc_id": l1_doc_id,
        },
        "layer2_breakdown": {
            **breakdown_data,
            "google_doc_id": l2_doc_id,
        },
        "layer3_classification": {
            **classification_data,
            "google_doc_id": None, # Will be set below
        },
        "layer4_judgement": judgement_data,
    }

    # 3B. Storage: Save final evaluation to Google Doc & push to Supabase
    print("  💾  Syncing Layer 3 & final metadata to Google Docs & Supabase...")
    gdoc_l3_title = f"Evaluation - {call_title}"
    gdoc_l3_content = (
        f"CALL EVALUATION & METADATA\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Call ID: {call_id}\n"
        f"{'='*40}\n\n"
        f"Call Type: {classification_data.get('call_type')}\n"
        f"Main Topic: {classification_data.get('topic')}\n"
        f"Sentiment: {classification_data.get('sentiment')}\n"
        f"Language Detected: {classification_data.get('language_detected')}\n"
        f"Participant Count: {classification_data.get('participants_count')}\n"
        f"Document Quality Score: {classification_data.get('doc_quality_score')}/10.0\n\n"
        f"### Executive Summary:\n"
        f"{classification_data.get('summary')}"
    )
    l3_doc_id = create_gdoc(gdoc_l3_title, gdoc_l3_content)
    final_output["layer3_classification"]["google_doc_id"] = l3_doc_id

    # Complete Supabase sync
    update_layer3_data(
        call_id=call_id,
        doc_id=l3_doc_id,
        classification=classification_data,
        quality_score=classification_data.get("doc_quality_score", 0.0),
        metadata=final_output["pipeline_metadata"]
    )

    # -----------------------------------------------------------------------
    # Save output backup to local JSON file
    # -----------------------------------------------------------------------
    if save_output:
        os.makedirs("output", exist_ok=True)
        timestamp = pipeline_start.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("output", f"analysis_{timestamp}.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        print(f"\n  💾  Full backup output saved to: {output_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ✅  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  ⏱️   Total time     : {elapsed:.1f}s")
    print(f"  🗣️   Utterances     : {len(transcript_data.get('speakers', []))}")
    print(f"  📊  Doc Quality    : {classification_data.get('doc_quality_score', 0.0)}/10.0")
    print(f"  📞  Call type       : {classification_data.get('call_type', '?')}")
    print(f"  😊  Sentiment       : {classification_data.get('sentiment', '?')}")
    print(f"  ⚖️   Judge status   : {judgement_data.get('status', '?')}")
    print(f"{'='*60}\n")

    return final_output


# ===========================================================================
# CLI argument parser
# ===========================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Call Analysis Pipeline — Transcribe, analyse, and classify calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py recording.wav                          # Transcribe an audio file
  python main.py --live                                 # Live mic transcription (30s)
  python main.py --demo                                 # Run with built-in demo transcript
  python main.py --demo --summary-length "1 paragraph"  # Custom size summary run
        """,
    )

    # Positional argument: audio file path (optional)
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to an audio file to transcribe (WAV, MP3, FLAC, etc.)",
    )

    # Live transcription flag
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live microphone transcription instead of a file",
    )

    # Duration for live recording
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration in seconds for live recording (default: 30)",
    )

    # Demo mode
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with a built-in demo transcript (no Deepgram key needed)",
    )

    # Disable saving output
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the local backup output JSON file",
    )

    # Custom Summary sizing/depth configurations
    parser.add_argument(
        "--summary-length",
        type=str,
        default="comprehensive (3-4 paragraphs)",
        help="Target length/size parameter for the Layer 2 summary",
    )
    
    parser.add_argument(
        "--custom-notes",
        type=str,
        default="None provided.",
        help="Additional custom context or focus notes for the summary",
    )

    return parser.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()

    # Validate inputs
    if not args.demo and not args.live and not args.audio_file:
        print("❌  No input specified. Use --demo, --live, or provide an audio file.")
        print("    Run `python main.py --help` for usage info.")
        sys.exit(1)

    if args.audio_file and not os.path.exists(args.audio_file):
        print(f"❌  Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Assemble user context payload for Layer 2 dynamic sizing
    user_context = {
        "summary_length": args.summary_length,
        "custom_notes": args.custom_notes
    }

    # Run the pipeline
    try:
        result = asyncio.run(
            run_pipeline(
                audio_path=args.audio_file,
                live=args.live,
                duration=args.duration,
                demo=args.demo,
                save_output=not args.no_save,
                user_context=user_context,
            )
        )
    except KeyboardInterrupt:
        print("\n\n  ⛔  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌  Pipeline failed: {e}")
        sys.exit(1)
