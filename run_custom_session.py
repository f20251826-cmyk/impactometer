"""
=============================================================================
CUSTOM TRANSCRIPT RUNNER FOR IMPACTOMETER
=============================================================================

This script loads the raw transcript from `basel_session.txt`, formats it,
and runs the full Conquest 2026 Call Analysis Pipeline starting from Layer 2.
It saves the backup outputs to `output/` and records the quality score.
=============================================================================
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Reconfigure stdout/stderr to support Unicode (emojis) on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # For older Python versions that don't support reconfigure

# Ensure environment variables are loaded
load_dotenv()

# Import pipeline components
from layer1_5_translation import translate_transcript, print_translation_result
from layer2_breakdown import breakdown_transcript, print_moments
from layer3_classification import classify_call, print_classification
from layer4_intervention import analyze_interventions, print_intervention_analysis
from layer5_scoring import compute_solution_scores, print_solution_scores
from score_tracker import record_score
from db_helpers import create_call_record, update_layer1_data, update_layer2_data, update_layer3_data, create_gdoc

async def run_custom_pipeline():
    pipeline_start = datetime.now()
    print("\n" + "=" * 60)
    print("  🚀  IMPACTOMETER: CUSTOM SESSION RUN (CONQUEST 2026)")
    print("=" * 60)

    # 1. Read the custom transcript from file
    transcript_path = "basel_session.txt"
    if not os.path.exists(transcript_path):
        print(f"❌ Error: Custom transcript file not found at {transcript_path}")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        raw_transcript = f.read().strip()

    print(f"\n  📝 Loaded custom transcript ({len(raw_transcript.splitlines())} lines).")
    
    # 2. Build initial Supabase call record
    call_title = f"Conquest Mentor Session: Basel (Ram) & Gaurav Shah"
    print(f"\n  🗄️  Initialising session in Supabase...")
    call_id = create_call_record(call_title)

    # Compile mock Layer 1 stats for input
    # Gaurav Shah (mentor), Ram Ramarathnam (founder), VIHAAN ANIL GUPTA (host)
    layer1_metadata = {
        "speaker_count": 4,
        "duration": 2076,
        "confidence_scores": [0.99] * 50
    }
    
    transcript_data = {
        "transcript": raw_transcript,
        "speakers": [
            {"speaker": "Gaurav Shah", "text": "Mentor", "start": 0.0, "end": 0.0, "confidence": 0.99},
            {"speaker": "Ram Ramarathnam", "text": "Founder", "start": 0.0, "end": 0.0, "confidence": 0.99},
            {"speaker": "VIHAAN ANIL GUPTA", "text": "Host", "start": 0.0, "end": 0.0, "confidence": 0.99}
        ]
    }

    # Sync Layer 1 to Docs & Supabase
    print("\n  💾  Syncing Layer 1 to Google Docs & Supabase...")
    gdoc_l1_title = f"Transcript - {call_title}"
    gdoc_l1_content = (
        f"CALL TRANSCRIPT\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Call ID: {call_id}\n"
        f"{'='*40}\n\n"
        f"{raw_transcript}"
    )
    l1_doc_id = create_gdoc(gdoc_l1_title, gdoc_l1_content)
    update_layer1_data(
        call_id=call_id,
        doc_id=l1_doc_id,
        transcript=raw_transcript,
        speakers=transcript_data["speakers"]
    )

    # 2.5 LAYER 1.5 — Hindi/Hinglish → English Translation
    print(f"\n{'─'*60}")
    print("  LAYER 1.5 — Hindi/Hinglish → English Translation")
    print(f"{'─'*60}")
    try:
        transcript_data = translate_transcript(transcript_data)
        print_translation_result(transcript_data)
        # Update raw_transcript with translated version for downstream use
        raw_transcript = transcript_data["transcript"]
    except Exception as e:
        print(f"  ❌  Layer 1.5 translation failed: {e}")
        print("  ⚠️  Continuing pipeline with original transcript...")

    # 3. LAYER 2 — Summarizer Agent
    print(f"\n{'─'*60}")
    print("  LAYER 2 / 5 — Talking Points Summarizer (Gemini Flash)")
    print(f"{'─'*60}")

    user_context = {
        "summary_length": "standard",
        "focus_areas": ["GTM", "ICP", "pitch deck", "focus"],
        "custom_notes": "Mentor-founder session between Gaurav Shah (PE/VC Advisor) and Ram Ramarathnam (Founder of Basel - DC appliances/green energy)."
    }

    layer2_payload = {
        "transcript": raw_transcript,
        "user_context": user_context,
        "layer1_metadata": layer1_metadata
    }

    breakdown_data = breakdown_transcript(layer2_payload)
    print_moments(breakdown_data)

    # Sync Layer 2 to Docs & Supabase
    print("  💾  Syncing Layer 2 to Google Docs & Supabase...")
    gdoc_l2_title = f"Summary - {call_title}"
    
    bullets = []
    for param, statements in breakdown_data.get("discussion", {}).items():
        if statements:
            bullets.append(f"\n[{param}]")
            for st in statements:
                bullets.append(f"  • {st.get('speaker', '?')}: {st.get('statement', '')}")
    gdoc_l2_bullets = "\n".join(bullets)

    pitch_lines = []
    for param, val in breakdown_data.get("pitch_metrics", {}).items():
        if val:
            pitch_lines.append(f"  • {param}: {val}")
    gdoc_l2_pitch = "\n".join(pitch_lines) or "  (none extracted)"

    gdoc_l2_content = (
        f"TALKING POINTS SUMMARY (CONQUEST 2026)\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Call ID: {call_id}\n"
        f"Sizing parameter: standard\n"
        f"{'='*40}\n\n"
        f"### PITCH METRICS\n"
        f"{gdoc_l2_pitch}\n\n"
        f"### KEY DISCUSSION POINTS\n"
        f"{gdoc_l2_bullets}"
    )
    
    l2_doc_id = create_gdoc(gdoc_l2_title, gdoc_l2_content)
    update_layer2_data(
        call_id=call_id,
        doc_id=l2_doc_id,
        summary=json.dumps(breakdown_data.get("pitch_metrics", {}), ensure_ascii=False)
    )

    # 4. LAYER 3 — Call Classification & Quality Score
    print(f"\n{'─'*60}")
    print("  LAYER 3 / 5 — Call Classification & Quality (Gemini Flash)")
    print(f"{'─'*60}")

    classification_data = classify_call(transcript_data, breakdown_data, user_context=user_context)
    print_classification(classification_data)

    # 5. LAYER 4 — Mentor Intervention Analysis
    print(f"\n{'─'*60}")
    print("  LAYER 4 / 5 — Mentor Intervention Analysis (Gemini Flash)")
    print(f"{'─'*60}")

    try:
        intervention_data = analyze_interventions(
            transcript_data=transcript_data,
            breakdown_data=breakdown_data,
            classification_data=classification_data,
            layer1_metadata=layer1_metadata,
            user_context=user_context,
        )
        print_intervention_analysis(intervention_data)
    except Exception as e:
        print(f"  ❌  Layer 4 failed: {e}")
        from layer4_intervention import _default_layer4_output
        intervention_data = _default_layer4_output(f"Layer 4 failure: {e}")

    # 6. LAYER 5 — Numerical Solution Quality Scoring
    print(f"\n{'─'*60}")
    print("  LAYER 5 / 5 — Solution Quality Scoring (Deterministic)")
    print(f"{'─'*60}")

    scoring_data = compute_solution_scores(intervention_data, breakdown_data)
    print_solution_scores(scoring_data)

    # 7. Save consolidated run & record score
    pipeline_end = datetime.now()
    elapsed = (pipeline_end - pipeline_start).total_seconds()

    final_output = {
        "pipeline_metadata": {
            "timestamp": pipeline_start.isoformat(),
            "duration_seconds": round(elapsed, 2),
            "input_mode": "custom_file",
            "input_source": "basel_session.txt",
            "call_id": call_id,
        },
        "layer1_transcription": {
            "transcript": raw_transcript,
            "utterance_count": len(raw_transcript.splitlines()),
            "speakers": transcript_data["speakers"],
            "google_doc_id": l1_doc_id,
        },
        "layer2_breakdown": {
            **breakdown_data,
            "google_doc_id": l2_doc_id,
        },
        "layer3_classification": {
            **classification_data,
            "google_doc_id": None,
        },
        "layer4_intervention": intervention_data,
        "layer5_scoring": scoring_data,
    }

    # Sync Layer 3 & final metadata to Google Docs & Supabase
    print("  💾  Syncing Layer 3 & final metadata to Google Docs & Supabase...")
    gdoc_l3_title = f"Evaluation - {call_title}"
    eval_block = classification_data.get("layer3_evaluation", {})
    mentor = eval_block.get("mentor_engagement", {})
    founder = eval_block.get("founder_preparedness", {})
    gdoc_l3_content = (
        f"CALL EVALUATION & METADATA (CONQUEST 2026)\n"
        f"Generated: {datetime.now().isoformat()}\n"
        f"Call ID: {call_id}\n"
        f"{'='*40}\n\n"
        f"Mentor Engagement: {mentor.get('score', 0.0)}/10.0\n"
        f"  {mentor.get('observation', '')}\n\n"
        f"Founder Preparedness: {founder.get('score', 0.0)}/10.0\n"
        f"  {founder.get('observation', '')}\n\n"
        f"Overall Progress Metric: {eval_block.get('overall_progress_metric', 0.0)}/10.0\n\n"
        f"### Next Steps:\n"
        f"{json.dumps(eval_block.get('next_steps_for_founder', []), indent=2)}"
    )
    
    l3_doc_id = create_gdoc(gdoc_l3_title, gdoc_l3_content)
    final_output["layer3_classification"]["google_doc_id"] = l3_doc_id

    # Complete Supabase sync
    update_layer3_data(
        call_id=call_id,
        doc_id=l3_doc_id,
        classification=classification_data,
        quality_score=eval_block.get("overall_progress_metric", 0.0),
        metadata=final_output["pipeline_metadata"]
    )

    # Save output backup
    os.makedirs("output", exist_ok=True)
    timestamp = pipeline_start.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("output", f"analysis_basel_{timestamp}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"\n  💾  Full backup output saved to: {output_path}")

    # Record score
    try:
        entry = record_score(startup_id="basel", session_number=1, output_json_path=output_path)
        delta_sign = "+" if entry["delta_from_baseline"] >= 0 else ""
        print(f"\n  ✓  Score recorded")
        print(f"     Startup: basel")
        print(f"     Session: {entry['session']}")
        print(f"     Score: {entry['score']}")
        print(f"     Delta from baseline: {delta_sign}{entry['delta_from_baseline']}")
        print(f"     Maturity: {entry['maturity']}")
    except Exception as e:
        print(f"  ⚠️  Score tracking failed: {e}")

    # Final summary
    print(f"\n{'='*60}")
    print("  ✅  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  ⏱️   Total time     : {elapsed:.1f}s")
    print(f"  📊  Progress Metric: {eval_block.get('overall_progress_metric', 0.0)}/10.0")
    print(f"  🧑‍🏫 Mentor Score   : {mentor.get('score', 0.0)}/10.0")
    print(f"  🧑‍💼 Founder Score  : {founder.get('score', 0.0)}/10.0")
    l4_meta = intervention_data.get("layer4_metadata", {})
    l4_counts = l4_meta.get("intervention_type_counts", {})
    print(f"  🔬  Interventions  : {sum(l4_counts.values())} across {len(l4_meta.get('parameters_analysed', []))} parameter(s)")
    print(f"  📐  Solution Score : {scoring_data.get('overall_score', 0.0)}/4.0 ({scoring_data.get('overall_rating_label', '?')})")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    asyncio.run(run_custom_pipeline())
