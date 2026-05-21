"""
=============================================================================
LAYER 3 — Call Classification & Quality Evaluator Agent (Gemini 2.5 Flash)
=============================================================================

This module takes the FULL TRANSCRIPT (from Layer 1) and the TALKING POINTS
SUMMARY (from Layer 2) as inputs, and uses Gemini 2.5 Flash to classify the call
and generate a numerical score (0.0 to 10.0) representing document quality.

Output: A structured JSON object with:
  - call_type          (str)  : e.g. "Sales call", "Support call", "Internal meeting"
  - topic              (str)  : Main topic discussed
  - sentiment          (str)  : "positive", "neutral", or "negative"
  - summary            (str)  : 2-3 sentence summary of the call
  - participants_count (int)  : Number of distinct speakers
  - language_detected  (str)  : Language of the conversation (e.g. "English")
  - doc_quality_score  (float): A rating from 0.0 to 10.0 of the document quality

The grading criteria for doc_quality_score can be dynamically customized
using parameters passed in the prompt.
=============================================================================
"""

import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

import google.generativeai as genai

# These are the call types the model knows about. Add/remove as needed.
CALL_TYPE_HINTS = [
    "Sales call",
    "Support / Help desk call",
    "Internal team meeting",
    "Client onboarding",
    "Interview",
    "Performance review",
    "Project standup / status update",
    "Negotiation",
    "Consultation",
    "Training / Walkthrough",
    "Escalation",
    "Follow-up",
    "Cold call / Outreach",
    "Other",
]

# Fields the model must return. Modify this list to add new fields.
REQUIRED_OUTPUT_FIELDS = [
    "call_type",
    "topic",
    "sentiment",
    "summary",
    "participants_count",
    "language_detected",
    "doc_quality_score",
]

CLASSIFICATION_SYSTEM_PROMPT = f"""You are a Call Classification & Quality Evaluator Agent.
Given a full call transcript and the talking points summary from Layer 2, you must classify the call and score the quality of the transcript document.

## YOUR TASK:

Analyse BOTH inputs to determine:

1. **call_type** — What kind of call this is. Pick the best match from:
   {json.dumps(CALL_TYPE_HINTS, indent=2)}
   If none of these fit, use "Other" and add a clarifier (e.g. "Other — Legal review").

2. **topic** — A concise phrase describing the main topic (e.g. "Deployment pipeline reliability", "Q3 sales targets").

3. **sentiment** — Overall sentiment of the call. Must be exactly one of:
   - "positive" — generally constructive, upbeat, or collaborative
   - "neutral" — balanced, informational, no strong emotion
   - "negative" — frustrated, conflicted, or adversarial

4. **summary** — A 2-3 sentence summary of what happened on the call. Be specific, mention key decisions or outcomes.

5. **participants_count** — How many distinct speakers are in the transcript (integer).

6. **language_detected** — The primary language spoken (e.g. "English", "Spanish", "Hindi").

7. **doc_quality_score** — A numerical rating from 0.0 to 10.0 assessing the "quality of the document".
   Assess the rating based on the following structural criteria:
   - Completeness: Does the transcript record complete thoughts rather than cut-offs?
   - Diarization Turn Structure: Are speaker turns organized logically and attributable?
   - Coherence: Is the conversation easy to comprehend?
   *Note: Detailed mathematical parameters for this score will be customized later. For now, grade the quality strictly between 0.0 (unusable) and 10.0 (pristine/perfect).*

## OUTPUT FORMAT:

Return a single valid JSON object with EXACTLY these keys: {json.dumps(REQUIRED_OUTPUT_FIELDS)}.
Do NOT wrap in markdown code fences. Do NOT add extra keys unless instructed.

Example:
{{
  "call_type": "Internal team meeting",
  "topic": "Deployment pipeline reliability",
  "sentiment": "neutral",
  "summary": "The team discussed recurring failures in the deployment pipeline. They identified the staging environment mismatch as the root cause and assigned an action item to set up a dedicated staging cluster.",
  "participants_count": 2,
  "language_detected": "English",
  "doc_quality_score": 8.5
}}
"""

# ===========================================================================
# Main Function — Classification Agent
# ===========================================================================

def classify_call(transcript_data: dict, breakdown_data: dict) -> dict:
    """
    Classify a call and evaluate document quality using the transcript and
    talking points summary.

    Args:
        transcript_data: Output dict from Layer 1 (must have "transcript" key)
        breakdown_data:  Output dict from Layer 2 (must have "summary" key)

    Returns:
        dict with keys: call_type, topic, sentiment, summary,
        participants_count, language_detected, doc_quality_score
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    transcript_text = transcript_data.get("transcript", "")
    summary_text = breakdown_data.get("summary", "")

    if not transcript_text.strip():
        print("  ⚠️  Empty transcript received — returning default classification.")
        return _default_classification()

    try:
        # Configure the Gemini API client
        genai.configure(api_key=api_key)

        # Initialise Gemini 2.5 Highlight with classification prompt
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=CLASSIFICATION_SYSTEM_PROMPT,
        )

        # Build user prompt
        user_prompt = f"""Classify and grade the quality of the following call.

--- FULL TRANSCRIPT ---
{transcript_text}
--- END TRANSCRIPT ---

--- TALKING POINTS SUMMARY (Layer 2) ---
{summary_text}
--- END SUMMARY ---

Return your classification and doc quality score as a JSON object."""

        print("  🏷️  Classifying call and evaluating document quality...")
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

        # Validate that all required fields exist
        if not isinstance(classification, dict):
            raise ValueError("Gemini did not return a JSON object.")

        for field in REQUIRED_OUTPUT_FIELDS:
            if field not in classification:
                print(f"  ⚠️  Missing field '{field}' — using default value.")
                classification[field] = _default_value(field)

        # Normalise sentiment
        if "sentiment" in classification:
            classification["sentiment"] = classification["sentiment"].lower()
            if classification["sentiment"] not in ("positive", "neutral", "negative"):
                classification["sentiment"] = "neutral"

        # Ensure participants_count is integer
        if "participants_count" in classification:
            try:
                classification["participants_count"] = int(classification["participants_count"])
            except (ValueError, TypeError):
                speakers = transcript_data.get("speakers", [])
                unique_speakers = set(s.get("speaker", "") for s in speakers)
                classification["participants_count"] = len(unique_speakers) or 2

        # Ensure doc_quality_score is float
        if "doc_quality_score" in classification:
            try:
                classification["doc_quality_score"] = float(classification["doc_quality_score"])
            except (ValueError, TypeError):
                classification["doc_quality_score"] = 7.0  # Fallback median grade

        print(f"  ✅  Call classified as: {classification.get('call_type', 'Unknown')}")
        print(f"      Sentiment: {classification.get('sentiment', '?')}  |  Doc Quality Score: {classification.get('doc_quality_score', '0.0')}/10.0")

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

def _default_classification() -> dict:
    """Return a default classification when the transcript is empty."""
    return {
        "call_type": "Unknown",
        "topic": "Unknown",
        "sentiment": "neutral",
        "summary": "No transcript available for classification.",
        "participants_count": 0,
        "language_detected": "Unknown",
        "doc_quality_score": 0.0,
    }


def _default_value(field: str):
    """Return a sensible default for a missing classification field."""
    defaults = {
        "call_type": "Unknown",
        "topic": "Unknown",
        "sentiment": "neutral",
        "summary": "Classification incomplete.",
        "participants_count": 0,
        "language_detected": "Unknown",
        "doc_quality_score": 5.0,
    }
    return defaults.get(field, "Unknown")


# ===========================================================================
# Pretty-Print Helper
# ===========================================================================

def print_classification(classification: dict) -> None:
    """Print the classification result in a readable format."""
    print(f"\n{'='*60}")
    print("  CALL CLASSIFICATION & EVALUATION")
    print(f"{'='*60}")
    print(f"  📞  Call Type    : {classification.get('call_type', '?')}")
    print(f"  📌  Topic        : {classification.get('topic', '?')}")
    print(f"  😊  Sentiment    : {classification.get('sentiment', '?')}")
    print(f"  👥  Participants : {classification.get('participants_count', '?')}")
    print(f"  🌐  Language     : {classification.get('language_detected', '?')}")
    print(f"  ⭐️  Doc Quality  : {classification.get('doc_quality_score', '?')}/10.0")
    print(f"  📝  Summary      : {classification.get('summary', '?')}")
    print(f"{'='*60}\n")


# ===========================================================================
# Standalone Test
# ===========================================================================

if __name__ == "__main__":
    # Test payloads
    sample_transcript = {
        "transcript": (
            "[00:00] Speaker 0: Hey, good morning! Thanks for joining the call.\n"
            "[00:05] Speaker 1: Morning! Happy to be here. Let's dive in.\n"
            "[00:12] Speaker 0: So the main issue is our deployment pipeline breaks every third release.\n"
            "[00:22] Speaker 1: That's concerning. Have you checked the CI config?\n"
            "[00:28] Speaker 0: Yes. The problem is in the staging environment — it doesn't mirror production.\n"
            "[00:38] Speaker 1: I think we should set up a dedicated staging cluster. I can take that as an action item.\n"
            "[00:48] Speaker 0: Perfect. Let's also add automated smoke tests after each deploy.\n"
            "[00:55] Speaker 1: Agreed. I'll draft a proposal by Friday.\n"
            "[01:02] Speaker 0: Great, thanks. Anything else?\n"
            "[01:05] Speaker 1: That's it from my side. Talk soon!\n"
            "[01:08] Speaker 0: Bye!"
        ),
        "speakers": [
            {"speaker": "Speaker 0"}, {"speaker": "Speaker 1"},
        ],
    }

    sample_breakdown = {
        "summary": (
            "The meeting centered around reliability issues in the team's deployment pipeline, "
            "specifically regarding staging environment mismatches with production. Speaker 1 "
            "committed to setting up a dedicated staging cluster, and both agreed to write a "
            "proposal to add automated smoke tests by Friday."
        ),
        "talking_points": [
            "Deployment pipeline breaks every third release.",
            "Staging environment does not match production config.",
            "Action item: Speaker 1 to configure staging cluster.",
            "Decision made: Set up automated smoke testing post-deploy."
        ]
    }

    result = classify_call(sample_transcript, sample_breakdown)
    print_classification(result)
