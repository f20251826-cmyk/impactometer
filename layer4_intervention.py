"""
=============================================================================
LAYER 4 — Mentor Intervention Analysis (Gemini 2.5 Flash)
=============================================================================

Synthesises Layer 2 discussion and Layer 3 classification into a rigorous,
mentor-grounded problem and intervention map. Every output must be traceable
to something a mentor explicitly said in layer2_output.discussion.
=============================================================================
"""

import os
import sys
import json
import copy
from dotenv import load_dotenv

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

load_dotenv()

import google.generativeai as genai

# ===========================================================================
# Empty layer2_analysis scaffold (not yet produced by an upstream layer)
# ===========================================================================

def _empty_layer2_analysis() -> dict:
    null_cac = {
        "cac_strategy_explained": None,
        "mentor_response_to_cac": None,
        "mentor_flagged_cac_problems": None,
        "cac_trend_discussed": None,
        "mentor_flagged_cac_risks": None,
        "founder_agreed_to_change": None,
        "payback_period_discussed": None,
        "channel_concentration_risk_discussed": None,
    }
    return {
        "GTM": {
            "CAC": null_cac,
            "sales_motion": {
                "product_led_vs_sales_led_classified": None,
                "mentor_suggested_sales_motion_changes": None,
                "sales_cycle_repeatability_discussed": None,
                "revenue_per_headcount_discussed": None,
            },
            "retention_and_expansion": {
                "retention_expansion_overview": None,
                "cohort_retention_founder_view": None,
                "cohort_retention_mentor_comment": None,
                "nrr_founder_metrics": None,
                "nrr_mentor_comments": None,
                "nrr_building_advice": None,
            },
            "channel_and_moat": {
                "organic_distribution_discussed": None,
                "brand_defensibility_discussed": None,
            },
        },
        "market_structure_and_timing": None,
        "MVP": None,
        "unit_economics_and_capital_efficiency": None,
        "team_structure": None,
        "analysis_metadata": {
            "parameters_analysed": [],
            "uncovered_sub_parameters": [],
            "low_confidence_flags": [],
        },
    }


INTERVENTION_SYSTEM_PROMPT = """You are Layer 4 of the Conquest Call Analysis Pipeline. You synthesise the structured discussion and analysis produced by Layers 2 and 3 into a rigorous, mentor-grounded problem and intervention map. You do not invent, infer, or extrapolate. Every output you produce must be traceable to something a mentor explicitly said.

WHAT YOU READ
You receive the full accumulated context JSON with:
- layer1_output — raw transcript and diarization metadata
- layer2_output — cleaned post-pitch discussion classified under each parameter
- layer2_analysis — structured objective answers for active parameters (may be null)
- layer3_output — session classification, sentiment, founder receptiveness
- user_context — runtime parameters

You read from:
- layer2_output.discussion — cleaned, role-labeled post-pitch discussion per parameter
- layer2_output.pitch_metrics — founder-stated facts from the initial pitch
- layer2_analysis — structured answers for active parameters (GTM only when present)
- layer3_output.classification — session type, sentiment, founder receptiveness

You write exclusively into layer4_output. Do not modify any upstream key.

SOURCE QUOTE RULE
All source_statement and source_statements fields must quote mentor text exactly as it appears in layer2_output.discussion[parameter] entries where speaker is "mentor". synthesis_evidence must also quote from those same mentor discussion entries.

STEP 1 — Isolate Mentor Speech Per Parameter
For each active parameter in layer2_output.layer2_metadata.active_parameters, collect only mentor-attributed statements from layer2_output.discussion[parameter]. Ignore founder statements in this pass.

STEP 2 — Extract and Consolidate Problems
Read all mentor statements and identify every problem, gap, risk, or weakness the mentor surfaced — explicitly or through a pointed question.
Consolidation rule: If two or more mentor statements flag the same underlying problem, merge them into one problem entry.
For each consolidated problem write:
- problem_summary — single crisp sentence
- severity — critical | significant | moderate (use higher level when criteria conflict)
  - critical: mentor returned to it more than once, OR language indicates company cannot progress without resolving it, OR foundational assumption is questioned
  - significant: raised once with clear emphasis as material risk/blocker but not existential
  - moderate: noted in passing or as secondary concern
- source_statements — 1–3 mentor statements quoted exactly from layer2_output.discussion

STEP 3 — Classify and Evaluate Every Mentor Intervention
For every substantive suggestion, recommendation, redirect, warning, or flagged concern, produce one intervention entry.

intervention_type (exactly one):
- REBUTTAL — challenges founder strategy and demands a specific alternative or deliverable
- WARNING — alerts to a specific risk/failure mode without prescribing corrective action
- RECOMMENDATION — affirmatively proposes a specific course of action, focus, or strategic move
- FLAG_NO_SOLUTION — identifies a problem/gap and stops with no demand or recommended action

Tiebreakers:
- WARNING names a consequence; FLAG_NO_SOLUTION only notes absence
- REBUTTAL requires demanding/specifying an alternative

Alignment check against layer2_output.pitch_metrics[parameter] and layer2_analysis[parameter]:
- aligned_with_founder_strategy: reinforcing | contradicting | redirecting | gap
- alignment_note — one sentence citing specific founder fact/metric/strategy, or what founder failed to address for gap

For each intervention also write:
- intervention_summary — third-person single sentence capturing force and direction (not verbatim)
- source_statement — single mentor statement quoted exactly from layer2_output.discussion

STEP 4 — Parameter-Level Synthesis
Write parameter_synthesis (2–4 sentences) on mentor posture: aligned, corrective, or in disagreement.
Populate synthesis_evidence with 1–2 mentor statements from layer2_output.discussion quoted exactly.

STEP 5 — Metadata
Populate layer4_metadata:
- parameters_analysed — parameters with blocks produced
- intervention_type_counts — counts per type across all blocks
- low_confidence_flags — anything unresolved with confidence

RULES
- Mentor-only for problems and interventions
- No invention
- Consolidate, don't inflate
- Classification is semantic, not keyword-based
- No statement may appear twice within the same parameter block (problem OR intervention, not both)
- Parameter ownership inherited from Layer 2
- Empty blocks are valid when mentor only asked clarifying questions
- Inactive parameters are skipped entirely

OUTPUT
Return ONLY a JSON object for layer4_output with this exact structure:
{
  "parameter_analysis": {
    "<parameter_name>": {
      "problems_identified": [
        {
          "problem_summary": "",
          "severity": "critical | significant | moderate",
          "source_statements": ["", ""]
        }
      ],
      "interventions": [
        {
          "intervention_type": "REBUTTAL | WARNING | RECOMMENDATION | FLAG_NO_SOLUTION",
          "intervention_summary": "",
          "aligned_with_founder_strategy": "reinforcing | contradicting | redirecting | gap",
          "alignment_note": "",
          "source_statement": ""
        }
      ],
      "parameter_synthesis": "",
      "synthesis_evidence": ["", ""]
    }
  },
  "layer4_metadata": {
    "parameters_analysed": [],
    "intervention_type_counts": {
      "REBUTTAL": 0,
      "WARNING": 0,
      "RECOMMENDATION": 0,
      "FLAG_NO_SOLUTION": 0
    },
    "low_confidence_flags": []
  }
}
"""


# ===========================================================================
# Context assembly
# ===========================================================================

def _build_layer1_output(transcript_data: dict, layer1_metadata: dict) -> dict:
    speakers = transcript_data.get("speakers", [])
    if speakers and isinstance(speakers[0], dict) and "text" in speakers[0]:
        transcript_entries = speakers
    else:
        transcript_entries = [
            {
                "speaker": s.get("speaker", "UNKNOWN"),
                "text": s.get("text", ""),
                "start": s.get("start", 0.0),
                "end": s.get("end", 0.0),
                "confidence": s.get("confidence", 0.0),
            }
            for s in speakers
        ]

    confidences = layer1_metadata.get("confidence_scores", [])
    overall = (
        sum(confidences) / len(confidences) if confidences else 0.0
    )

    return {
        "transcript": transcript_entries,
        "layer1_metadata": {
            "speaker_count": layer1_metadata.get("speaker_count", len(speakers)),
            "duration_seconds": float(layer1_metadata.get("duration", 0.0)),
            "overall_confidence": round(overall, 4),
            "diarization_confidence": round(overall, 4),
        },
    }


def _strip_internal_keys(data: dict) -> dict:
    """Remove pipeline-internal keys before passing to Layer 4."""
    cleaned = copy.deepcopy(data)
    for key in ("google_doc_id", "layer3_evaluation"):
        cleaned.pop(key, None)
    return cleaned


def _build_layer3_output(classification_data: dict, breakdown_data: dict) -> dict:
    if classification_data.get("classification"):
        return {
            "classification": classification_data["classification"],
            "doc_quality_score": float(classification_data.get("doc_quality_score", 0.0)),
            "doc_quality_metadata": classification_data.get("doc_quality_metadata", {}),
            "layer3_metadata": classification_data.get("layer3_metadata", {}),
        }

    l3_eval = classification_data.get("layer3_evaluation", {})
    meta = breakdown_data.get("layer2_metadata", {})
    layer1_meta = breakdown_data.get("layer1_metadata", {})

    return {
        "classification": {
            "primary_topic": None,
            "secondary_topics": [],
            "sentiment": None,
            "sentiment_evidence": l3_eval.get("mentor_engagement", {}).get("observation"),
            "founder_receptiveness": None,
            "founder_receptiveness_evidence": l3_eval.get("founder_preparedness", {}).get("observation"),
            "session_type": "mixed",
            "participant_count": layer1_meta.get("speaker_count")
            or meta.get("speaker_count", 0),
        },
        "doc_quality_score": float(l3_eval.get("overall_progress_metric", 0.0)),
        "doc_quality_metadata": {},
        "layer3_metadata": {
            "topics_classified": len(meta.get("active_parameters", [])),
            "low_confidence_flags_inherited": len(meta.get("low_confidence_flags", [])),
        },
        "layer3_evaluation": l3_eval,
    }


def build_context_payload(
    transcript_data: dict,
    breakdown_data: dict,
    classification_data: dict,
    layer1_metadata: dict,
    user_context: dict = None,
    layer2_analysis: dict = None,
) -> dict:
    """Assemble the accumulated context JSON expected by Layer 4."""
    layer2_clean = _strip_internal_keys(breakdown_data)

    return {
        "layer1_output": _build_layer1_output(transcript_data, layer1_metadata),
        "layer2_output": layer2_clean,
        "layer2_analysis": layer2_analysis or _empty_layer2_analysis(),
        "layer3_output": _build_layer3_output(classification_data, breakdown_data),
        "user_context": user_context or {
            "summary_length": breakdown_data.get("summary_length_applied", "standard"),
            "focus_areas": breakdown_data.get("focus_areas_applied", []),
        },
    }


# ===========================================================================
# Main function
# ===========================================================================

def analyze_interventions(
    transcript_data: dict,
    breakdown_data: dict,
    classification_data: dict,
    layer1_metadata: dict,
    user_context: dict = None,
    layer2_analysis: dict = None,
) -> dict:
    """
    Run Layer 4 mentor intervention analysis.

    Returns:
        dict: layer4_output schema only.
    """
    active = breakdown_data.get("layer2_metadata", {}).get("active_parameters", [])
    id_conf = breakdown_data.get("layer2_metadata", {}).get("identification_confidence", "low")

    if not active or id_conf == "low":
        flags = breakdown_data.get("layer2_metadata", {}).get("low_confidence_flags", [])
        reason = flags[0] if flags else "No active parameters or low identification confidence."
        print(f"  ⚠️  Skipping Layer 4 analysis — {reason}")
        return _default_layer4_output(reason)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found. Add it to your .env file.")

    context = build_context_payload(
        transcript_data,
        breakdown_data,
        classification_data,
        layer1_metadata,
        user_context,
        layer2_analysis,
    )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=INTERVENTION_SYSTEM_PROMPT,
        )

        user_prompt = f"""Analyse mentor interventions according to your instructions.

YOUR INPUT:
{json.dumps(context, indent=2, ensure_ascii=False)}

Return ONLY the layer4_output JSON object."""

        print("  🔬  Analysing mentor interventions...")
        response = model.generate_content(user_prompt)

        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

        result = json.loads(raw_text)
        result = _validate_layer4_output(result)

        counts = result.get("layer4_metadata", {}).get("intervention_type_counts", {})
        total = sum(counts.values())
        params = result.get("layer4_metadata", {}).get("parameters_analysed", [])
        print(f"  ✅  Layer 4 complete — {len(params)} parameter(s), {total} intervention(s) mapped.")

        return result

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse Layer 4 response as JSON: {e}\nRaw: {raw_text[:500]}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Layer 4 intervention analysis failed: {e}") from e


def _default_layer4_output(reason: str) -> dict:
    return {
        "parameter_analysis": {},
        "layer4_metadata": {
            "parameters_analysed": [],
            "intervention_type_counts": {
                "REBUTTAL": 0,
                "WARNING": 0,
                "RECOMMENDATION": 0,
                "FLAG_NO_SOLUTION": 0,
            },
            "low_confidence_flags": [reason],
        },
    }


def _validate_layer4_output(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Layer 4 did not return a JSON object.")

    result.setdefault("parameter_analysis", {})
    meta = result.setdefault("layer4_metadata", {})
    meta.setdefault("parameters_analysed", list(result["parameter_analysis"].keys()))
    counts = meta.setdefault(
        "intervention_type_counts",
        {"REBUTTAL": 0, "WARNING": 0, "RECOMMENDATION": 0, "FLAG_NO_SOLUTION": 0},
    )
    meta.setdefault("low_confidence_flags", [])

    # Recompute intervention counts from actual data
    for itype in counts:
        counts[itype] = 0
    for block in result["parameter_analysis"].values():
        for intervention in block.get("interventions", []):
            itype = intervention.get("intervention_type", "")
            if itype in counts:
                counts[itype] += 1

    return result


# ===========================================================================
# Pretty-print helper
# ===========================================================================

def print_intervention_analysis(layer4_output: dict) -> None:
    """Print Layer 4 results in a readable format."""
    meta = layer4_output.get("layer4_metadata", {})
    params = layer4_output.get("parameter_analysis", {})

    print(f"\n{'='*60}")
    print("  CONQUEST 2026 — MENTOR INTERVENTION ANALYSIS (LAYER 4)")
    print(f"{'='*60}")

    if not params:
        print("  No parameter blocks produced.")
        for flag in meta.get("low_confidence_flags", []):
            print(f"  ⚠️  {flag}")
        print(f"{'='*60}\n")
        return

    counts = meta.get("intervention_type_counts", {})
    print(f"\n  Parameters analysed: {', '.join(meta.get('parameters_analysed', []))}")
    print(
        f"  Interventions: REBUTTAL={counts.get('REBUTTAL', 0)} | "
        f"WARNING={counts.get('WARNING', 0)} | "
        f"RECOMMENDATION={counts.get('RECOMMENDATION', 0)} | "
        f"FLAG_NO_SOLUTION={counts.get('FLAG_NO_SOLUTION', 0)}"
    )

    for param, block in params.items():
        problems = block.get("problems_identified", [])
        interventions = block.get("interventions", [])

        print(f"\n{'─'*60}")
        print(f"  [{param}]")
        print(f"{'─'*60}")

        if problems:
            print(f"\n  Problems ({len(problems)}):")
            for p in problems:
                print(f"    • [{p.get('severity', '?').upper()}] {p.get('problem_summary', '')}")

        if interventions:
            print(f"\n  Interventions ({len(interventions)}):")
            for i in interventions:
                print(
                    f"    • [{i.get('intervention_type', '?')}] "
                    f"{i.get('intervention_summary', '')}"
                )
                print(
                    f"      Alignment: {i.get('aligned_with_founder_strategy', '?')} — "
                    f"{i.get('alignment_note', '')}"
                )

        synthesis = block.get("parameter_synthesis", "")
        if synthesis:
            print(f"\n  Synthesis: {synthesis}")

    flags = meta.get("low_confidence_flags", [])
    if flags:
        print(f"\n{'─'*60}")
        print("  LOW CONFIDENCE FLAGS:")
        for flag in flags:
            print(f"  ⚠️  {flag}")

    print(f"\n{'='*60}\n")
