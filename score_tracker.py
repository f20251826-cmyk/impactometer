"""
=============================================================================
SCORE TRACKER
=============================================================================

Maintains data/scores.json — an append-only ledger of doc_quality_score
results across multiple pipeline runs, keyed by startup_id and session.

Each call to record_score():
  - Reads the layer3 doc_quality_score from a pipeline output JSON
  - Appends a session entry under the startup
  - Recomputes cohort averages across all startups
  - Returns the entry it just wrote (handy for printing)
=============================================================================
"""

import os
import json
from datetime import date
from typing import Optional


SCORES_DIR = "data"
SCORES_PATH = os.path.join(SCORES_DIR, "scores.json")


# ===========================================================================
# Maturity classification
# ===========================================================================

def _maturity_for_score(score: float) -> str:
    """Map a 0–10 doc quality score to a maturity stage label.

    Upper bound of each band belongs to the next stage (e.g. 4.0 → Hypothesis).
    """
    if score < 4:
        return "Pre-GTM"
    if score < 6.5:
        return "Hypothesis Stage"
    if score < 8.5:
        return "Validation Stage"
    return "Scaling Stage"


# ===========================================================================
# Disk I/O
# ===========================================================================

def _empty_store() -> dict:
    return {"startups": {}, "cohort_averages": []}


def _load_store() -> dict:
    if not os.path.exists(SCORES_PATH):
        return _empty_store()
    try:
        with open(SCORES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty_store()

    # Normalise structure if a partial/older file is present
    data.setdefault("startups", {})
    data.setdefault("cohort_averages", [])
    return data


def _save_store(store: dict) -> None:
    os.makedirs(SCORES_DIR, exist_ok=True)
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


# ===========================================================================
# Cohort averages
# ===========================================================================

def recalculate_cohort_averages(store: dict) -> list:
    """Recompute the cohort_averages array from scratch across all startups.

    For each session number that appears anywhere, average the score and the
    delta_from_baseline across every startup that has that session recorded.
    Result is sorted by session ascending and written back into the store.
    """
    by_session: dict[int, dict] = {}
    for startup in store.get("startups", {}).values():
        for entry in startup.get("sessions", []):
            session = entry.get("session")
            if session is None:
                continue
            bucket = by_session.setdefault(session, {"scores": [], "deltas": []})
            bucket["scores"].append(float(entry.get("score", 0.0)))
            bucket["deltas"].append(float(entry.get("delta_from_baseline", 0.0)))

    averages = []
    for session in sorted(by_session.keys()):
        scores = by_session[session]["scores"]
        deltas = by_session[session]["deltas"]
        averages.append({
            "session": session,
            "avg_score": round(sum(scores) / len(scores), 2),
            "avg_delta": round(sum(deltas) / len(deltas), 2),
        })

    store["cohort_averages"] = averages
    return averages


# ===========================================================================
# Public API
# ===========================================================================

def record_score(startup_id: str, session_number: int, output_json_path: str) -> dict:
    """Record one pipeline run's doc_quality_score into data/scores.json.

    Args:
        startup_id: Identifier for the startup (e.g. "startup_a").
        session_number: Integer session number for this run.
        output_json_path: Path to the pipeline's output JSON file.

    Returns:
        The session entry that was appended.
    """
    with open(output_json_path, "r", encoding="utf-8") as f:
        run_output = json.load(f)

    layer3 = run_output.get("layer3_classification", {}) or {}
    current_score = float(layer3.get("doc_quality_score", 0.0))

    store = _load_store()
    startups = store["startups"]

    if startup_id not in startups:
        startups[startup_id] = {"baseline_score": None, "sessions": []}

    startup = startups[startup_id]
    sessions = startup["sessions"]

    # Baseline: set on session 1, OR on the very first record for this startup
    # (defensive — if someone records session 3 first, we still have a baseline).
    if session_number == 1 or startup.get("baseline_score") is None:
        startup["baseline_score"] = current_score

    baseline = float(startup["baseline_score"])

    # Delta from previous session: 0.0 for session 1 or if no prior session exists
    if session_number == 1 or not sessions:
        delta_from_previous = 0.0
    else:
        prev_entry = max(sessions, key=lambda e: e.get("session", 0))
        delta_from_previous = current_score - float(prev_entry.get("score", 0.0))

    delta_from_baseline = current_score - baseline

    entry = {
        "session": int(session_number),
        "date": date.today().isoformat(),
        "score": round(current_score, 2),
        "delta_from_baseline": round(delta_from_baseline, 2),
        "delta_from_previous": round(delta_from_previous, 2),
        "maturity": _maturity_for_score(current_score),
        "run_file": os.path.basename(output_json_path),
    }

    sessions.append(entry)
    recalculate_cohort_averages(store)
    _save_store(store)

    return entry


def get_startup_history(startup_id: str) -> Optional[dict]:
    """Return all session data for a startup, or None if not found."""
    store = _load_store()
    return store.get("startups", {}).get(startup_id)


def get_all_startups() -> list:
    """Return list of all startup_ids currently tracked."""
    store = _load_store()
    return list(store.get("startups", {}).keys())
