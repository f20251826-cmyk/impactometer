"""
=============================================================================
LAYER 2 — Transcript Talking Points Summarizer (Gemini 2.5 Flash)
=============================================================================

This module acts as the Talking Points Summarizer Agent. It parses the raw
transcript alongside an aggregated context payload (containing metadata from
Layer 1 and custom user instructions) and uses Gemini 2.5 Flash to synthesize
a detailed summary of all key talking points, dynamically sized according to
the user's context specifications.

Output: A structured JSON containing:
  - session_summary (str)
  - talking_points (list[dict])
  - action_items (list[dict])
  - low_confidence_flags (list[str])
  - layer2_metadata (dict)
=============================================================================
"""

import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

import google.generativeai as genai

# ===========================================================================
# System Prompt for the Summarizer Agent (Conquest 2026 Spec)
# ===========================================================================

SUMMARIZER_SYSTEM_PROMPT = """You are a call analysis agent processing a mentor-founder session transcript from Conquest, BITS Pilani, Asia's largest student-run startup accelerator. In these sessions, one speaker is an industry mentor and the other is an early-stage founder team from the Conquest cohort of 2026. Sessions focus on GTM strategy, ICP refinement, fundraising readiness, and execution accountability. The mentor asks probing questions, challenges assumptions, and offers suggestions or constructive criticism. The founder presents their startup and responds to feedback.

YOUR INPUT:
You will receive a JSON object with:
transcript: Full diarized transcript from Deepgram with speaker labels (SPEAKER_0, SPEAKER_1, etc.)
user_context: Runtime parameters including summary_length ("brief" | "standard" | "detailed") and focus_areas (list of topics to prioritize e.g. ["GTM", "ICP", "pricing"])
layer1_metadata: Diarization stats from Deepgram (confidence scores, speaker count, duration)

YOUR TASK:
Step 1 — Identify speakers Based on the nature of the conversation, determine which speaker label corresponds to the mentor (asks probing questions, challenges assumptions, gives advice) and which corresponds to the founder (presents their startup, responds to feedback). State your identification in layer2_metadata. If you cannot determine this confidently, flag it in low_confidence_flags.
Step 2 — Separate the pitch from the main discussion The call typically begins with the founder pitching their startup. Identify where the pitch ends and the back-and-forth discussion begins. Treat these two sections separately for all subsequent steps.
Step 3 — Detect active parameters Before extracting any discussion, scan the full post-pitch transcript and determine which of the five broader parameters were explicitly discussed. A parameter counts as active only if the mentor or founder directly addressed it — not if it was merely implied or tangentially mentioned. List active parameters in layer2_metadata under active_parameters. Only populate discussion entries for active parameters. For inactive parameters, set their discussion array to null and do not fabricate entries to fill them.
The five broader parameters are:
GTM
Market structure & timing
MVP
Unit economics & capital efficiency
Team structure
Step 4 — Extract pitch metrics From the initial pitch only, extract the metrics and facts the founder explicitly states that relate to each of the five parameters. Record these in pitch_metrics. If a parameter was not mentioned in the pitch, set its value to null. Do not carry any pitch content into the discussion sections.
Step 5 — Extract and classify post-pitch discussion For the main discussion (everything after the initial pitch):
Maintain the conversational flow and label each statement by role: mentor or founder.
Club all statements under the relevant active parameter.
Every statement must be assigned to exactly one parameter — the one it most directly addresses. If a statement touches two parameters, assign it to whichever parameter is its primary subject. Do not duplicate any statement across two parameters. When in doubt, ask: what is the speaker trying to solve or communicate with this statement? Assign it there.
Strip all filler, pleasantries, repetition, and fluff. Rewrite each statement in clean, direct language so that an agent reading it will not be confused about what was said or meant.
Do not invent, infer, or extrapolate beyond what speakers explicitly said.
Step 6 — Apply focus areas and summary length If focus_areas are specified in user_context, extract talking points for those topics first and in full before covering other topics. Do not skip non-focus active parameters — include them at "brief" depth regardless of the summary_length setting. Apply summary_length to control the depth of coverage for non-focus parameters.

OUTPUT — Context JSON:
You are the second layer in a pipeline of 4 layers. Your output must be a context JSON structured to coordinate between all layers. Use the following schema exactly:
json
{
  "layer2_metadata": {
    "speaker_identification": {
      "mentor": "SPEAKER_X",
      "founder": "SPEAKER_Y"
    },
    "identification_confidence": "high | medium | low",
    "active_parameters": [],
    "low_confidence_flags": []
  },
  "pitch_metrics": {
    "GTM": null,
    "market_structure_and_timing": null,
    "MVP": null,
    "unit_economics_and_capital_efficiency": null,
    "team_structure": null
  },
  "discussion": {
    "GTM": null,
    "market_structure_and_timing": null,
    "MVP": null,
    "unit_economics_and_capital_efficiency": null,
    "team_structure": null
  },
  "summary_length_applied": "brief | standard | detailed",
  "focus_areas_applied": []
}
Schema rules:
active_parameters: List only the parameters that were explicitly discussed in the post-pitch conversation. Inactive parameters must have their discussion value set to null.
pitch_metrics: Populate only from the initial pitch. Null if not mentioned.
discussion: Each active parameter holds an array of objects in the format { "speaker": "mentor | founder", "statement": "" }. Inactive parameters are null.
low_confidence_flags: List any ambiguous speaker attributions, unclear topic classifications, or statements that were difficult to assign to a single parameter.
Do not add any fields outside this schema. Do not include raw transcript text anywhere in the output.
"""

# ===========================================================================
# Main Function — Summarize Transcript
# ===========================================================================

def breakdown_transcript(context_payload: dict) -> dict:
    """
    Analyse a transcript and generate a custom-sized summary of talking points
    using Gemini 2.5 Flash, driven by the aggregated context payload.

    Args:
        context_payload: A dictionary containing:
            - transcript (str): The full formatted transcript text.
            - user_context (dict, optional): Custom instructions from the user,
              which may include "summary_length", "focus_areas", etc.
            - layer1_metadata (dict, optional): Deepgram diarization stats.

    Returns:
        dict: Conquest 2026 Layer 2 Schema JSON.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    # Ingest the transcript text and user context
    transcript_text = context_payload.get("transcript", "")
    user_ctx = context_payload.get("user_context", {}) or {}
    layer1_meta = context_payload.get("layer1_metadata", {}) or {}

    if not transcript_text.strip():
        print("  ⚠️  Empty transcript received — skipping summarization.")
        return {
            "layer2_metadata": {
                "speaker_identification": {"mentor": "?", "founder": "?"},
                "identification_confidence": "low",
                "active_parameters": [],
                "low_confidence_flags": ["Empty transcript"]
            },
            "pitch_metrics": {
                "GTM": None,
                "market_structure_and_timing": None,
                "MVP": None,
                "unit_economics_and_capital_efficiency": None,
                "team_structure": None
            },
            "discussion": {
                "GTM": None,
                "market_structure_and_timing": None,
                "MVP": None,
                "unit_economics_and_capital_efficiency": None,
                "team_structure": None
            },
            "summary_length_applied": summary_len,
            "focus_areas_applied": []
        }

    # Extract user configurations
    summary_len = user_ctx.get("summary_length", "standard")
    focus_areas = user_ctx.get("focus_areas", [])

    try:
        # Initialise the Gemini API client
        genai.configure(api_key=api_key)

        # Initialise Gemini 2.5 Flash model with system prompt
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SUMMARIZER_SYSTEM_PROMPT,
        )

        # Construct input payload mirroring specified JSON format
        input_data = {
            "transcript": transcript_text,
            "user_context": {
                "summary_length": summary_len,
                "focus_areas": focus_areas
            },
            "layer1_metadata": {
                "speaker_count": layer1_meta.get("speaker_count", 2),
                "duration": layer1_meta.get("duration", 0),
                "confidence_scores": layer1_meta.get("confidence_scores", [])
            }
        }

        user_prompt = f"""Analyze this session according to your system instructions.

YOUR INPUT:
{json.dumps(input_data, indent=2, ensure_ascii=False)}

Synthesize all parameters and return the JSON object exactly conforming to the schema."""

        print(f"  🔍  Summarising transcript talking points (Target Size: {summary_len})...")
        response = model.generate_content(user_prompt)

        # Parse the JSON response
        raw_text = response.text.strip()

        # Clean up potential markdown code fences
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
        if raw_text.endswith("```"):
            raw_text = raw_text.rsplit("```", 1)[0]
        raw_text = raw_text.strip()

        result = json.loads(raw_text)

        # Validate the structure
        if not isinstance(result, dict):
            raise ValueError("Gemini did not return a JSON object.")

        # Ensure keys exist
        if "layer2_metadata" not in result or not isinstance(result["layer2_metadata"], dict):
            result["layer2_metadata"] = {}
            
        l2_meta = result["layer2_metadata"]
        l2_meta.setdefault("speaker_identification", {"mentor": "SPEAKER_X", "founder": "SPEAKER_Y"})
        l2_meta.setdefault("identification_confidence", "medium")
        l2_meta.setdefault("active_parameters", [])
        l2_meta.setdefault("low_confidence_flags", [])

        if "pitch_metrics" not in result or not isinstance(result["pitch_metrics"], dict):
            result["pitch_metrics"] = {
                "GTM": None,
                "market_structure_and_timing": None,
                "MVP": None,
                "unit_economics_and_capital_efficiency": None,
                "team_structure": None
            }
            
        if "discussion" not in result or not isinstance(result["discussion"], dict):
            result["discussion"] = {
                "GTM": None,
                "market_structure_and_timing": None,
                "MVP": None,
                "unit_economics_and_capital_efficiency": None,
                "team_structure": None
            }
            
        result.setdefault("summary_length_applied", summary_len)
        result.setdefault("focus_areas_applied", [])

        print(f"  ✅  Layer 2 breakdown generated successfully.")

        return result

    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse Gemini summarizer response as JSON: {e}\n"
            f"Raw response: {raw_text[:500]}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Summarizer agent failed: {e}") from e


# ===========================================================================
# Pretty-Print Helper
# ===========================================================================

def print_moments(summary_result: dict) -> None:
    """Print the talking points summary in a human-readable format."""
    meta = summary_result.get("layer2_metadata", {})
    sp_id = meta.get("speaker_identification", {})
    flags = meta.get("low_confidence_flags", [])

    print(f"\n{'='*60}")
    print("  CONQUEST 2026 - TALKING POINTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\n  IDENTIFICATION (Confidence: {meta.get('identification_confidence', '?')})")
    print(f"  • Mentor : {sp_id.get('mentor', '?')}")
    print(f"  • Founder: {sp_id.get('founder', '?')}")
    
    pm = summary_result.get("pitch_metrics", {})
    print(f"\n{'-'*60}")
    print("  PITCH METRICS:")
    for key, val in pm.items():
        if val is not None:
            print(f"  • {key}: {val}")
            
    disc = summary_result.get("discussion", {})
    print(f"\n{'-'*60}")
    print("  KEY DISCUSSION POINTS:")
    for key, statements in disc.items():
        if statements:
            print(f"\n  [{key}]")
            for st in statements:
                print(f"    • {st.get('speaker', '?')}: {st.get('statement', '')}")
            
    if flags:
        print(f"\n{'-'*60}")
        print("  LOW CONFIDENCE FLAGS:")
        for flag in flags:
            print(f"  ⚠️ {flag}")
            
    print(f"\n{'='*60}\n")


# ===========================================================================
# Standalone Test
# ===========================================================================

if __name__ == "__main__":
    # Test payload mimicking Layer 1 output + user context
    test_payload = {
        "transcript": (
            "[00:00] Speaker 0: Hey, good morning! Thanks for joining the call. I'm a mentor here at Conquest.\n"
            "[00:05] Speaker 1: Morning! Happy to be here. I'm working on a SaaS for developers. Our GTM is cold outreach.\n"
            "[00:12] Speaker 0: Okay. I think cold outreach is hard. What is your ICP?\n"
            "[00:22] Speaker 1: We target CTOs in mid-sized firms. I'll make sure to write that down as a focus.\n"
            "[00:38] Speaker 0: Let's focus on setting up 5 outreach campaigns by Friday.\n"
            "[00:48] Speaker 1: Sounds perfect. I'll own setting up those campaigns.\n"
            "[00:55] Speaker 0: Great, talk soon.\n"
            "[01:02] Speaker 1: Bye!"
        ),
        "user_context": {
            "summary_length": "standard",
            "focus_areas": ["GTM", "ICP"]
        },
        "layer1_metadata": {
            "speaker_count": 2,
            "duration": 62,
            "confidence_scores": [0.98, 0.97, 0.95]
        }
    }

    result = breakdown_transcript(test_payload)
    print_moments(result)
