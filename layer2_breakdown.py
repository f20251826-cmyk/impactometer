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
  - summary (str)            : The full synthesized summary.
  - talking_points (list[str]): A list of key individual points/topics.
  - word_count (int)          : Estimated word count of the summary.
=============================================================================
"""

import os
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

import google.generativeai as genai

# ===========================================================================
# System Prompt for the Summarizer Agent
# ===========================================================================

SUMMARIZER_SYSTEM_PROMPT = """You are a Transcript Talking Points Summarizer Agent.
Your job is to digest a transcript and synthesize a highly detailed, professional summary of the discussion topics and talking points.

## RULES — Read carefully:

1. **Context Adherence:** You will receive both a transcript AND a 'user_context' containing custom configurations, notes, or specific size instructions.
2. **Dynamic Sizing:** Pay close attention to any length specifications (e.g. "detailed", "medium", "concise", or "X paragraphs") provided in the user context. Make your summary match the requested depth and length. If no length is specified, default to a robust, comprehensive 3-4 paragraph summary covering all core topics.
3. **Structured Breakdown:** Distil the call into structured talking points. Identify:
   - Core topics discussed
   - Decisions reached
   - Next steps / Action items
4. **Accuracy:** Keep all summaries factually accurate to the conversation. Do not invent details.
5. **JSON Formatting:** Return your response as a valid JSON object with EXACTLY these keys:
   - "summary": The full synthesized markdown-formatted summary text.
   - "talking_points": An array of strings, each summarizing a key point.
   - "word_count": An integer representing the estimated word count of the generated summary.

Do NOT wrap the response in markdown code fences.
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

    Returns:
        dict with keys:
            - summary (str): Synthesized markdown summary.
            - talking_points (list[str]): High-level bullet points.
            - word_count (int): Sized output word count.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found. Add it to your .env file."
        )

    # Ingest the transcript text and user context
    transcript_text = context_payload.get("transcript", "")
    user_ctx = context_payload.get("user_context", {}) or {}

    if not transcript_text.strip():
        print("  ⚠️  Empty transcript received — skipping summarization.")
        return {
            "summary": "No transcript available to summarize.",
            "talking_points": [],
            "word_count": 0
        }

    # Extract dynamic sizing instruction if present
    target_length = user_ctx.get("summary_length", "comprehensive (3-4 paragraphs)")
    custom_notes = user_ctx.get("custom_notes", "None provided.")

    try:
        # Initialise the Gemini API client
        genai.configure(api_key=api_key)

        # Initialise Gemini 2.5 Flash model with system prompt
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SUMMARIZER_SYSTEM_PROMPT,
        )

        # Build the user prompt instructing length and focusing context
        user_prompt = f"""Summarize the following call transcript.

--- CUSTOM USER CONTEXT CONFIGS ---
- Target Summary Size/Depth: {target_length}
- Focus / Additional Context: {custom_notes}
-----------------------------------

--- TRANSCRIPT ---
{transcript_text}
--- END TRANSCRIPT ---

Synthesize all talking points, align them with the sizing instruction, and return a JSON object."""

        print(f"  🔍  Summarising transcript talking points (Target Size: {target_length})...")
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
        if "summary" not in result:
            result["summary"] = "Summary extraction incomplete."
        if "talking_points" not in result or not isinstance(result["talking_points"], list):
            result["talking_points"] = []
        if "word_count" not in result:
            result["word_count"] = len(result["summary"].split())

        print(f"  ✅  Summary generated successfully ({result['word_count']} words, {len(result['talking_points'])} talking points).")

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
    summary = summary_result.get("summary", "")
    points = summary_result.get("talking_points", [])

    print(f"\n{'='*60}")
    print("  TALKING POINTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{summary}\n")
    
    if points:
        print(f"{'-'*60}")
        print("  KEY DISCUSSION POINTS:")
        for point in points:
            print(f"  • {point}")
            
    print(f"\n{'='*60}\n")


# ===========================================================================
# Standalone Test
# ===========================================================================

if __name__ == "__main__":
    # Test payload mimicking Layer 1 output + user context
    test_payload = {
        "transcript": (
            "[00:00] Speaker 0: Hey, good morning! Thanks for joining the call.\n"
            "[00:05] Speaker 1: Morning! Happy to be here. Let's dive in.\n"
            "[00:12] Speaker 0: So the main issue we're facing is that our deployment pipeline breaks every third release.\n"
            "[00:22] Speaker 1: That's concerning. Have you checked the CI config?\n"
            "[00:28] Speaker 0: Yes, we did. The problem is in the staging environment — it doesn't mirror production.\n"
            "[00:38] Speaker 1: I think we should set up a dedicated staging cluster. I can take that as an action item.\n"
            "[00:48] Speaker 0: Perfect. Let's also add automated smoke tests after each deploy.\n"
            "[00:55] Speaker 1: Agreed. I'll draft a proposal by Friday.\n"
            "[01:02] Speaker 0: Great, thanks. Anything else?\n"
            "[01:05] Speaker 1: That's it from my side. Talk soon!\n"
            "[01:08] Speaker 0: Bye!"
        ),
        "user_context": {
            "summary_length": "very concise (1 short paragraph with bullet points)",
            "custom_notes": "Focus heavily on the action items related to the staging environment."
        }
    }

    result = breakdown_transcript(test_payload)
    print_moments(result)
