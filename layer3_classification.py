"""
=============================================================================
LAYER 3 — Call Classification & Quality Evaluator Agent (Gemini 2.5 Flash)
=============================================================================

This module takes the FULL TRANSCRIPT (from Layer 1) and the TALKING POINTS
SUMMARY (from Layer 2) as inputs, and uses Gemini 2.5 Flash to classify the call
and generate a numerical score (0.0 to 10.0) representing document quality.

Output: A structured JSON object with BITS Pilani Conquest 2026 specs.
=============================================================================
"""

import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

import google.generativeai as genai

EVALUATOR_SYSTEM_PROMPT = """You are a session evaluator for Conquest, BITS Pilani's startup accelerator program.

YOUR INPUT:
You will receive the full context JSON from Layer 2.

YOUR TASK:
Using the structured data from Layer 2, evaluate the session across four rubrics. Do not invent details; base your scores only on what was explicitly extracted.

Task 1 — Assess Mentor Engagement
Evaluate how effectively the mentor drove the session based on the nature of the extracted discussion.
- score: 0.0 to 10.0
- observation: One sentence citing specific advice given or questions asked.

Task 2 — Assess Founder Preparedness
Evaluate how well the founder understood their metrics, responded to feedback, and defended their assumptions.
- score: 0.0 to 10.0
- observation: One sentence citing their command of metrics from the pitch or their responses in the discussion.

Task 3 — Calculate Overall Progress Metric
Calculate a strict average of the two scores above.
- score: 0.0 to 10.0

Task 4 — Generate Next Steps / To-Dos
Identify 1-3 concrete next steps for the founder based on the discussion points. Only list actions explicitly discussed or strongly implied as necessary by the mentor's feedback.

OUTPUT — Context JSON:
You are the third layer in a pipeline. Inherit the Layer 2 JSON and append your evaluation under a new "layer3_evaluation" key. Do not modify or remove any data from Layer 2.

json
{
  ... (All Layer 2 fields unchanged),
  "layer3_evaluation": {
    "mentor_engagement": {
      "score": <float>,
      "observation": ""
    },
    "founder_preparedness": {
      "score": <float>,
      "observation": ""
    },
    "overall_progress_metric": <float>,
    "next_steps_for_founder": [
      ""
    ]
  }
}
"""

# ===========================================================================
# Main Function — Classification Agent
# ===========================================================================

def classify_call(transcript_data: dict, breakdown_data: dict, user_context: dict = None) -> dict:
    """
    Classify a call and evaluate document quality using the transcript and
    talking points summary.

    Args:
        transcript_data: Output dict from Layer 1 (must have "transcript" key)
        breakdown_data:  Output dict from Layer 2 (must have "talking_points" key)
        user_context:    User configuration parameters (optional)

    Returns:
        dict: Conquest 2026 Layer 3 Schema JSON.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    if not breakdown_data or breakdown_data.get("layer2_metadata", {}).get("identification_confidence") == "low":
        if not breakdown_data.get("discussion") and not breakdown_data.get("pitch_metrics"):
            print("  ⚠️  No substantive data received — returning default evaluation.")
            return _default_classification(breakdown_data)

    try:
        # Configure the Gemini API client
        genai.configure(api_key=api_key)

        # Initialise Gemini 2.5 Flash
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=EVALUATOR_SYSTEM_PROMPT,
        )

        # Build user prompt
        user_prompt = f"""Evaluate this session according to your instructions.

YOUR INPUT:
{json.dumps(breakdown_data, indent=2, ensure_ascii=False)}

Return the full merged JSON including your layer3_evaluation."""

        print("  🏷️  Evaluating session...")
        response = model.generate_content(user_prompt)

        # Parse JSON response
        raw_text = response.text.strip()

        # Clean up potential markdown code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

        classification = json.loads(raw_text)

        # Validate that all required structures exist
        if not isinstance(classification, dict):
            raise ValueError("Gemini did not return a JSON object.")

        if "layer3_evaluation" not in classification:
            classification["layer3_evaluation"] = {
                "mentor_engagement": {"score": 0.0, "observation": "Evaluation missing"},
                "founder_preparedness": {"score": 0.0, "observation": "Evaluation missing"},
                "overall_progress_metric": 0.0,
                "next_steps_for_founder": []
            }
        else:
            # Ensure float scores
            l3 = classification["layer3_evaluation"]
            if "mentor_engagement" in l3:
                l3["mentor_engagement"]["score"] = float(l3["mentor_engagement"].get("score", 0.0))
            if "founder_preparedness" in l3:
                l3["founder_preparedness"]["score"] = float(l3["founder_preparedness"].get("score", 0.0))
            l3["overall_progress_metric"] = float(l3.get("overall_progress_metric", 0.0))

        prog_metric = classification["layer3_evaluation"].get("overall_progress_metric", 0.0)
        print(f"  ✅  Call evaluated successfully. Overall Progress Metric: {prog_metric}/10.0")

        return classification

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse Gemini classification response as JSON: {e}\n"
            f"Raw response: {raw_text[:500]}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Classification agent failed: {e}") from e


# ===========================================================================
# Default / Fallback Helpers
# ===========================================================================

def _default_classification(breakdown_data: dict) -> dict:
    """Return a default evaluation when the input is empty."""
    # Start with a copy of layer 2 output if present
    res = dict(breakdown_data) if breakdown_data else {}
    res["layer3_evaluation"] = {
        "mentor_engagement": {
            "score": 0.0,
            "observation": "No data available."
        },
        "founder_preparedness": {
            "score": 0.0,
            "observation": "No data available."
        },
        "overall_progress_metric": 0.0,
        "next_steps_for_founder": []
    }
    return res


# ===========================================================================
# Pretty-Print Helper
# ===========================================================================

def print_classification(classification: dict) -> None:
    """Print the classification result in a readable format."""
    eval_data = classification.get("layer3_evaluation", {})

    print(f"\n{'='*60}")
    print("  CONQUEST 2026 - EVALUATION")
    print(f"{'='*60}")
    
    mentor = eval_data.get("mentor_engagement", {})
    print(f"  🧑‍🏫 Mentor Engagement Score: {mentor.get('score', 0.0)}/10.0")
    print(f"     Observation: {mentor.get('observation', '')}")
    
    founder = eval_data.get("founder_preparedness", {})
    print(f"  🧑‍💼 Founder Preparedness Score: {founder.get('score', 0.0)}/10.0")
    print(f"     Observation: {founder.get('observation', '')}")
    
    print(f"\n  📈 Overall Progress Metric: {eval_data.get('overall_progress_metric', 0.0)}/10.0")
    
    next_steps = eval_data.get("next_steps_for_founder", [])
    if next_steps:
        print(f"\n  🎯 Next Steps for Founder:")
        for step in next_steps:
            print(f"    • {step}")
            
    print(f"{'='*60}\n")


# ===========================================================================
# Standalone Test
# ===========================================================================

if __name__ == "__main__":
    # Test payloads
    sample_transcript = {
        "transcript": (
            "[00:00] Speaker 0: Hey, good morning! Thanks for joining the call. I'm your GTM mentor.\n"
            "[00:05] Speaker 1: Morning! Happy to be here. We are doing cold calling but conversion is very low.\n"
            "[00:12] Speaker 0: Let's figure out your ICP. What's the profile of your best customer?\n"
            "[00:22] Speaker 1: They are engineering leaders in India. I'll make sure to refine this ICP by Thursday.\n"
            "[00:48] Speaker 0: Excellent. Let's touch base again next week.\n"
            "[01:02] Speaker 1: Sounds perfect. Talk soon!"
        ),
        "speakers": [
            {"speaker": "Speaker 0"}, {"speaker": "Speaker 1"},
        ],
    }

    sample_breakdown = {
        "discussion": {
            "GTM": [
                {"speaker": "founder", "statement": "We are doing cold calling but conversion is very low."},
                {"speaker": "mentor", "statement": "Let's figure out your ICP. What's the profile of your best customer?"}
            ],
            "market_structure_and_timing": None,
            "MVP": None,
            "unit_economics_and_capital_efficiency": None,
            "team_structure": None
        },
        "layer2_metadata": {
            "speaker_identification": {"mentor": "Speaker 0", "founder": "Speaker 1"},
            "identification_confidence": "high",
            "active_parameters": ["GTM"]
        }
    }

    result = classify_call({}, sample_breakdown)
    print_classification(result)
