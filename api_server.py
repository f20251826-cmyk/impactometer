"""
=============================================================================
IMPACTOMETER API SERVER
=============================================================================

A lightweight Flask server that lets the Conquest Impact Portal (the
dashboard at http://localhost:8000) drive the existing analysis pipeline
straight from the browser — no terminal required.

Endpoints
---------
POST /api/run-pipeline
    multipart/form-data with a "transcript" .txt file.
    Reads the file as plain text and runs it through the EXISTING
    Layer 2 (breakdown) -> Layer 3 (classification) pipeline, bypassing
    Layer 1 (transcription) since we already have the transcript as text.
    Returns the full pipeline JSON output.

GET /api/sessions
    Reads every JSON file in output/ and returns them as an array,
    sorted by timestamp (date) descending.

The analysis itself is untouched: we import and call the very same
`breakdown_transcript` and `classify_call` functions the CLI pipeline
uses. The only thing we skip here are the Supabase / Google Docs storage
side-syncs (they require extra credentials and are not part of the
analysis). The pipeline's behaviour and output schema are unchanged.

Run:  .venv/bin/python api_server.py   (serves on port 5001)
=============================================================================
"""

import os
import sys
import glob
import json
import traceback
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables (GEMINI_API_KEY etc.) before importing layers
load_dotenv()

from flask import Flask, request, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Import the EXISTING pipeline layers — analysis logic is reused verbatim.
# ---------------------------------------------------------------------------
from layer1_5_translation import translate_transcript
from layer2_breakdown import breakdown_transcript
from layer3_classification import classify_call

OUTPUT_DIR = "output"

app = Flask(__name__)
# Allow the dashboard (served from http://localhost:8000) to call this API.
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session_timestamp(session: dict) -> str:
    """Best-effort extraction of a sortable timestamp from a session dict."""
    meta = session.get("pipeline_metadata", {}) or {}
    return meta.get("timestamp", "") or ""


# ---------------------------------------------------------------------------
# POST /api/run-pipeline
# ---------------------------------------------------------------------------
@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline_endpoint():
    try:
        # --- Validate input -------------------------------------------------
        if "transcript" not in request.files:
            return jsonify({
                "error": "No transcript file provided. Send a multipart/form-data "
                         "request with a 'transcript' field containing a .txt file."
            }), 400

        upload = request.files["transcript"]
        if not upload.filename:
            return jsonify({"error": "Empty filename. Please choose a .txt transcript."}), 400

        raw_transcript = upload.read().decode("utf-8", errors="replace").strip()
        if not raw_transcript:
            return jsonify({"error": "The uploaded transcript file is empty."}), 400

        # Optional startup label (falls back to the filename stem)
        startup = (request.form.get("startup") or "").strip()
        if not startup:
            startup = os.path.splitext(os.path.basename(upload.filename))[0]

        pipeline_start = datetime.now()

        # --- LAYER 1 is bypassed: we already have text --------------------
        # Build a minimal Layer-1-style payload from the plain text.
        line_count = len(raw_transcript.splitlines())
        layer1_metadata = {
            "speaker_count": 2,
            "duration": 0,
            "confidence_scores": [],
        }
        transcript_data = {
            "transcript": raw_transcript,
            "speakers": [],
        }

        # --- LAYER 1.5 — Translation (if requested) ----------------------
        should_translate = (request.form.get("translate", "").lower() in ("true", "1", "yes"))
        if should_translate:
            transcript_data = translate_transcript(transcript_data)
            raw_transcript = transcript_data["transcript"]

        # --- LAYER 2 — Talking Points Summarizer (existing function) ------
        user_context = {
            "summary_length": request.form.get("summary_length", "standard"),
            "focus_areas": [],
            "custom_notes": f"Uploaded transcript for {startup} via the Impact Portal.",
        }
        layer2_payload = {
            "transcript": raw_transcript,
            "user_context": user_context,
            "layer1_metadata": layer1_metadata,
        }
        breakdown_data = breakdown_transcript(layer2_payload)

        # --- LAYER 3 — Classification & Evaluation (existing function) ----
        classification_data = classify_call(
            transcript_data, breakdown_data, user_context=user_context
        )

        # --- Assemble consolidated output (same schema as the CLI) --------
        pipeline_end = datetime.now()
        elapsed = (pipeline_end - pipeline_start).total_seconds()

        final_output = {
            "pipeline_metadata": {
                "timestamp": pipeline_start.isoformat(),
                "duration_seconds": round(elapsed, 2),
                "input_mode": "portal_upload",
                "input_source": upload.filename,
                "startup": startup,
                "call_id": None,
            },
            "layer1_transcription": {
                "transcript": raw_transcript,
                "utterance_count": line_count,
                "speakers": transcript_data["speakers"],
                "google_doc_id": None,
            },
            "layer2_breakdown": {
                **breakdown_data,
                "google_doc_id": None,
            },
            "layer3_classification": {
                **classification_data,
                "google_doc_id": None,
            },
        }

        # --- Save a backup into output/ so it shows up in /api/sessions ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = pipeline_start.strftime("%Y%m%d_%H%M%S")
        safe_stub = "".join(c if c.isalnum() else "_" for c in startup).strip("_").lower() or "session"
        out_path = os.path.join(OUTPUT_DIR, f"analysis_{safe_stub}_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)

        final_output["_output_file"] = os.path.basename(out_path)
        return jsonify(final_output), 200

    except Exception as e:  # noqa: BLE001 — surface any pipeline error as JSON
        traceback.print_exc()
        return jsonify({
            "error": f"Pipeline failed: {e}",
            "type": e.__class__.__name__,
        }), 500


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------
@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    try:
        sessions = []
        for path in glob.glob(os.path.join(OUTPUT_DIR, "*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_output_file"] = os.path.basename(path)
                sessions.append(data)
            except Exception as e:  # skip unreadable / malformed files
                print(f"  ⚠️  Skipping {path}: {e}")

        # Sort by timestamp (date) descending
        sessions.sort(key=_session_timestamp, reverse=True)
        return jsonify(sessions), 200

    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"Failed to read sessions: {e}"}), 500


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "impactometer-api"}), 200


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🚀  IMPACTOMETER API SERVER")
    print("=" * 60)
    print("  POST  /api/run-pipeline   (multipart 'transcript' .txt)")
    print("  GET   /api/sessions       (all output/*.json, date desc)")
    print("  GET   /api/health")
    print("  Listening on http://localhost:5001")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5001, debug=False)
