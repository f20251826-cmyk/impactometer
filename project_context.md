# Impactometer Call Analysis Pipeline (Conquest 2026) — Developer Context

This document provides a complete overview of the project's architecture, data flow, pipeline layers, and system rules to help any incoming agent understand and work with the codebase.

---

## 1. Project Overview & Architecture
The project is an end-to-end multi-layer AI Call Analysis Pipeline designed to transcribe, summarize, evaluate, map, and score founder-mentor calls for Conquest 2026. The pipeline outputs are synced to **Supabase** and **Google Docs** at each step.

The pipeline consists of 5 sequential layers:

```
  [Audio / Mic / Demo]
           │
           ▼
┌──────────────────────────────┐
│  Layer 1: Transcription      │ ──► Save Transcript Doc & Supabase Call Record
│  (Deepgram Nova-3)           │
└──────────────────────────────┘
           │ (transcript text)
           ▼
┌──────────────────────────────┐
│  Layer 2: Summarizer         │ ──► Save Summary Doc & Update Supabase
│  (Gemini 2.5 Flash)          │
└──────────────────────────────┘
           │ (cleaned categorized discussion per parameter)
           ▼
┌──────────────────────────────┐
│  Layer 3: Session Evaluation │ ──► Save Evaluation Doc & Update Supabase
│  (Gemini 2.5 Flash)          │
└──────────────────────────────┘
           │ (quality score, sentiment, receptiveness)
           ▼
┌──────────────────────────────┐
│  Layer 4: Mentor Intervention│ ──► Build mentor problem & intervention map
│  (Gemini 2.5 Flash)          │
└──────────────────────────────┘
           │ (intervention types: REBUTTAL, WARNING, etc.)
           ▼
┌──────────────────────────────┐
│  Layer 5: deterministic score│ ──► Numerical rating (0-4) based on interventions
│  (Pure Python, no API)       │
└──────────────────────────────┘
```

---

## 2. Key Directories and Files

*   **[`main.py`](file:///c:/Users/Shashwat/Desktop/code/impact/main.py)**: The main orchestrator for live mic transcription, audio file transcription, and demo runs.
*   **[`run_custom_session.py`](file:///c:/Users/Shashwat/Desktop/code/impact/run_custom_session.py)**: Custom orchestrator designed to process transcripts from `basel_session.txt` directly starting from Layer 2.
*   **[`layer1_transcription.py`](file:///c:/Users/Shashwat/Desktop/code/impact/layer1_transcription.py)**: Manages Deepgram Nova-3 API for file and live mic speech-to-text.
*   **[`layer2_breakdown.py`](file:///c:/Users/Shashwat/Desktop/code/impact/layer2_breakdown.py)**: Cleans transcript and groups discussions and pitch metrics under five parameters: `GTM`, `market_structure_and_timing`, `MVP`, `unit_economics_and_capital_efficiency`, and `team_structure`.
*   **[`layer3_classification.py`](file:///c:/Users/Shashwat/Desktop/code/impact/layer3_classification.py)**: Analyzes call metadata, engagement levels, and calculates a `doc_quality_score` out of 10.
*   **[`layer4_intervention.py`](file:///c:/Users/Shashwat/Desktop/code/impact/layer4_intervention.py)**: Identifies mentor-surfaced problems and maps/classifies actions (e.g., `RECOMMENDATION`, `WARNING`). Uses Gemini.
*   **[`layer5_scoring.py`](file:///c:/Users/Shashwat/Desktop/code/impact/layer5_scoring.py)**: Pure Python implementation of deterministic rating scale rules (0 to 4) mapping solutions' richness.
*   **[`db_helpers.py`](file:///c:/Users/Shashwat/Desktop/code/impact/db_helpers.py)**: Interacts with Supabase (call records) and creates Google Docs (`create_gdoc`).
*   **[`score_tracker.py`](file:///c:/Users/Shashwat/Desktop/code/impact/score_tracker.py)**: Tracks historical scores of startup sessions in `data/scores.json`.

---

## 3. Data Schema & Payloads

### Layer 4 Context Payload (Input to Layer 4)
Passed from upstream layers:
*   `layer1_output`: raw transcript + diarization metadata
*   `layer2_output`: categorized discussion statements per active parameter
*   `layer2_analysis`: GTM sub-parameter structured data
*   `layer3_output`: sentiment, founder receptiveness, progress score
*   `user_context`: focus areas and custom notes

### Layer 4 Output Schema
```json
{
  "parameter_analysis": {
    "<parameter_name>": {
      "problems_identified": [
        {
          "problem_summary": "Crisp description.",
          "severity": "critical | significant | moderate",
          "source_statements": ["verbatim quote from mentor"]
        }
      ],
      "interventions": [
        {
          "intervention_type": "REBUTTAL | WARNING | RECOMMENDATION | FLAG_NO_SOLUTION",
          "intervention_summary": "Summary of guidance.",
          "aligned_with_founder_strategy": "reinforcing | contradicting | redirecting | gap",
          "alignment_note": "Comparison details.",
          "source_statement": "verbatim quote from mentor"
        }
      ],
      "parameter_synthesis": "Synthesis of mentor posture.",
      "synthesis_evidence": ["verbatim quote from mentor"]
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
```

### Layer 5 Scoring Schema (Deterministic 0-4 scale)
Scored using the following logic per parameter/sub-parameter:
*   `0`: No Solution (no interventions)
*   `1`: Vague solution (only `FLAG_NO_SOLUTION` interventions)
*   `2`: Vague solution (exactly one actionable type: `WARNING`, `RECOMMENDATION`, `REBUTTAL`)
*   `3`: Better solution (two or more actionable interventions, but all are of the *same* type)
*   `4`: Great solution (two or more actionable interventions spanning *different* types)

---

## 4. Operational Guidelines & Strict Rules

1.  **Strict Source Quote Rule (Layer 4)**: Every statement in `source_statements`, `source_statement`, and `synthesis_evidence` must quote the mentor text **verbatim** from `layer2_output.discussion`. No paraphrasing or hallucination is permitted.
2.  **Deterministic Rules (Layer 5)**: This layer must run in pure Python, without any LLM/AI calls, using keywords (`GTM_SUB_KEYWORDS`) to route GTM interventions to sub-parameters.
3.  **Third-Party Syncing**: Whenever running the pipeline, ensure the `.env` has valid credentials for Supabase and Google Docs. The helpers in `db_helpers.py` will handle auth automatically.
4.  **No direct CLI stdout pollution**: Console output is reconfigured to support emojis/Unicode even on Windows console environments.

---

## 5. Quick Commands & Verification

*   **Syntax Check**:
    ```bash
    python -m py_compile layer4_intervention.py layer5_scoring.py main.py run_custom_session.py
    ```
*   **Run Layer 5 Smoke Tests**:
    ```bash
    python layer5_scoring.py
    ```
*   **Run Custom Run (with custom transcript)**:
    ```bash
    python run_custom_session.py
    ```
*   **Run Demo Mode**:
    ```bash
    python main.py --demo
    ```
