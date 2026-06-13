"""
=============================================================================
LAYER 5 — Numerical Solution Quality Scoring (Pure Python)
=============================================================================

Produces deterministic numerical scores per parameter and sub-parameter based
on the mentor intervention map from Layer 4. No AI calls.

Rating scale (0–4):
  0 — No Solution: no mentor interventions for this scope
  1 — Vague (flagged only): only FLAG_NO_SOLUTION interventions
  2 — Vague (one actionable): exactly one WARNING / RECOMMENDATION / REBUTTAL
  3 — Better (two similar): two or more actionable interventions, all same type
  4 — Great (diverse kinds): two or more actionable types across interventions
=============================================================================
"""

from __future__ import annotations

from typing import Any

# ===========================================================================
# Rating scale
# ===========================================================================

ACTIONABLE_TYPES = frozenset({"WARNING", "RECOMMENDATION", "REBUTTAL"})
FLAG_TYPE = "FLAG_NO_SOLUTION"

RATING_LABELS = {
    0: "No Solution",
    1: "Vague solution (flagged only)",
    2: "Vague solution (one actionable)",
    3: "Better solution (two similar kinds)",
    4: "Great solution (two or three different kinds)",
}

GTM_SUB_PARAMETERS = (
    "CAC",
    "sales_motion",
    "retention_and_expansion",
    "channel_and_moat",
)

TOP_LEVEL_PARAMETERS = (
    "GTM",
    "market_structure_and_timing",
    "MVP",
    "unit_economics_and_capital_efficiency",
    "team_structure",
)

# Keywords used to route GTM interventions to sub-parameters
GTM_SUB_KEYWORDS: dict[str, tuple[str, ...]] = {
    "CAC": (
        "cac",
        "customer acquisition",
        "acquisition cost",
        "payback",
        "channel concentration",
        "top of funnel",
        "top-of-funnel",
        "conversion rate",
        "customer acquisition cost",
    ),
    "sales_motion": (
        "sales-led",
        "product-led",
        "sales cycle",
        "sales motion",
        "revenue per head",
        "enterprise sales",
        "repeatability",
        "channel strategy",
        "go-to-market",
        "gtm roadmap",
        "distributor",
        "anchor customer",
    ),
    "retention_and_expansion": (
        "retention",
        "nrr",
        "net revenue retention",
        "cohort",
        "expansion",
        "churn",
        "lifetime",
        "recurring revenue",
    ),
    "channel_and_moat": (
        "organic",
        "brand",
        "moat",
        "defensibility",
        "barrier",
        "partnership",
        "strategic partner",
        "competitive",
        "iot saas",
        "recurring revenue",
    ),
}


# ===========================================================================
# Core rating logic
# ===========================================================================

def rate_interventions(interventions: list[dict]) -> dict[str, Any]:
    """
    Apply the 0–4 rating rules to a list of Layer 4 intervention entries.

    Returns a dict with score, rating_label, and breakdown counts.
    """
    if not interventions:
        return _score_result(0, actionable_count=0, flag_count=0, intervention_types=[])

    actionable = [
        i for i in interventions
        if i.get("intervention_type") in ACTIONABLE_TYPES
    ]
    flags = [
        i for i in interventions
        if i.get("intervention_type") == FLAG_TYPE
    ]
    actionable_types = [i.get("intervention_type") for i in actionable]
    unique_actionable = set(actionable_types)

    if not actionable:
        return _score_result(
            1,
            actionable_count=0,
            flag_count=len(flags),
            intervention_types=[FLAG_TYPE] if flags else [],
        )

    if len(actionable) == 1:
        return _score_result(
            2,
            actionable_count=1,
            flag_count=len(flags),
            intervention_types=actionable_types,
        )

    if len(unique_actionable) == 1:
        return _score_result(
            3,
            actionable_count=len(actionable),
            flag_count=len(flags),
            intervention_types=actionable_types,
        )

    return _score_result(
        4,
        actionable_count=len(actionable),
        flag_count=len(flags),
        intervention_types=actionable_types,
    )


def _score_result(
    score: int,
    actionable_count: int,
    flag_count: int,
    intervention_types: list[str],
) -> dict[str, Any]:
    unique_types = sorted(set(intervention_types))
    return {
        "score": score,
        "rating_label": RATING_LABELS[score],
        "actionable_count": actionable_count,
        "flag_count": flag_count,
        "intervention_count": actionable_count + flag_count,
        "intervention_types": unique_types,
    }


def _intervention_text(intervention: dict) -> str:
    parts = [
        intervention.get("intervention_summary", ""),
        intervention.get("source_statement", ""),
        intervention.get("alignment_note", ""),
    ]
    return " ".join(p for p in parts if p).lower()


def _match_gtm_sub_parameter(intervention: dict) -> str | None:
    """Return the best-matching GTM sub-parameter for an intervention, if any."""
    text = _intervention_text(intervention)
    if not text:
        return None

    best_sub: str | None = None
    best_hits = 0
    for sub_param, keywords in GTM_SUB_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_hits = hits
            best_sub = sub_param
    return best_sub if best_hits > 0 else None


def _bucket_gtm_interventions(interventions: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {sub: [] for sub in GTM_SUB_PARAMETERS}
    unmatched: list[dict] = []

    for intervention in interventions:
        sub = _match_gtm_sub_parameter(intervention)
        if sub:
            buckets[sub].append(intervention)
        else:
            unmatched.append(intervention)

    # Broad GTM interventions with no keyword hit still count at parameter level;
    # attach unmatched to a general bucket used only when scoring sub-params
    # that received zero dedicated matches but GTM was active.
    buckets["_unmatched"] = unmatched
    return buckets


def _label_for_average(avg: float) -> str:
    """Map a fractional average to the nearest rating label."""
    rounded = round(avg)
    rounded = max(0, min(4, rounded))
    return RATING_LABELS[rounded]


# ===========================================================================
# Public API
# ===========================================================================

def compute_solution_scores(
    layer4_output: dict,
    breakdown_data: dict | None = None,
) -> dict[str, Any]:
    """
    Compute numerical solution-quality scores from Layer 4 intervention data.

    Args:
        layer4_output: Output from analyze_interventions() (layer4_output schema).
        breakdown_data: Optional Layer 2 output for active_parameters context.

    Returns:
        layer5_scoring schema dict.
    """
    breakdown_data = breakdown_data or {}
    active_parameters = (
        breakdown_data.get("layer2_metadata", {}).get("active_parameters")
        or layer4_output.get("layer4_metadata", {}).get("parameters_analysed")
        or []
    )

    parameter_analysis = layer4_output.get("parameter_analysis", {})
    sub_parameter_scores: dict[str, Any] = {}
    parameter_scores: dict[str, Any] = {}

    for param in TOP_LEVEL_PARAMETERS:
        if param not in active_parameters:
            continue

        block = parameter_analysis.get(param, {})
        interventions = block.get("interventions", [])
        param_result = rate_interventions(interventions)
        parameter_scores[param] = param_result

        if param == "GTM":
            buckets = _bucket_gtm_interventions(interventions)
            gtm_sub_scores: dict[str, Any] = {}

            for sub in GTM_SUB_PARAMETERS:
                sub_interventions = buckets[sub]
                gtm_sub_scores[sub] = rate_interventions(sub_interventions)

            sub_parameter_scores["GTM"] = gtm_sub_scores
        else:
            # Non-GTM parameters are scored as a single sub-parameter unit
            sub_parameter_scores[param] = param_result

    all_sub_scores: list[float] = []
    for param, scores in sub_parameter_scores.items():
        if param == "GTM" and isinstance(scores, dict):
            all_sub_scores.extend(s["score"] for s in scores.values())
        elif isinstance(scores, dict) and "score" in scores:
            all_sub_scores.append(float(scores["score"]))

    overall = round(sum(all_sub_scores) / len(all_sub_scores), 2) if all_sub_scores else 0.0

    return {
        "sub_parameter_scores": sub_parameter_scores,
        "parameter_scores": parameter_scores,
        "overall_score": overall,
        "overall_rating_label": _label_for_average(overall),
        "scoring_metadata": {
            "active_parameters": active_parameters,
            "sub_parameters_scored": len(all_sub_scores),
            "rating_scale": RATING_LABELS,
            "scoring_method": "deterministic_intervention_type_count",
        },
    }


def print_solution_scores(scoring_output: dict) -> None:
    """Print Layer 5 scores in a readable format."""
    print(f"\n{'='*60}")
    print("  CONQUEST 2026 — SOLUTION QUALITY SCORES (LAYER 5)")
    print(f"{'='*60}")

    overall = scoring_output.get("overall_score", 0.0)
    label = scoring_output.get("overall_rating_label", RATING_LABELS[0])
    print(f"\n  Overall Score: {overall}/4.0 — {label}")

    param_scores = scoring_output.get("parameter_scores", {})
    sub_scores = scoring_output.get("sub_parameter_scores", {})

    if not param_scores:
        print("\n  No active parameters to score.")
        print(f"{'='*60}\n")
        return

    print(f"\n{'─'*60}")
    print("  PARAMETER SCORES")
    print(f"{'─'*60}")
    for param, result in param_scores.items():
        print(
            f"  • {param}: {result['score']}/4 "
            f"({result['rating_label']}) "
            f"[{result['actionable_count']} actionable, {result['flag_count']} flagged]"
        )

    gtm_sub = sub_scores.get("GTM")
    if gtm_sub:
        print(f"\n{'─'*60}")
        print("  GTM SUB-PARAMETER SCORES")
        print(f"{'─'*60}")
        for sub, result in gtm_sub.items():
            types = ", ".join(result.get("intervention_types", [])) or "none"
            print(
                f"  • {sub}: {result['score']}/4 "
                f"({result['rating_label']}) — types: {types}"
            )

    print(f"\n{'='*60}\n")


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    examples = [
        ([], 0),
        ([{"intervention_type": "FLAG_NO_SOLUTION"}], 1),
        ([{"intervention_type": "RECOMMENDATION"}], 2),
        ([{"intervention_type": "RECOMMENDATION"}, {"intervention_type": "RECOMMENDATION"}], 3),
        ([{"intervention_type": "WARNING"}, {"intervention_type": "RECOMMENDATION"}], 4),
    ]
    for interventions, expected in examples:
        got = rate_interventions(interventions)["score"]
        status = "OK" if got == expected else "FAIL"
        print(f"  [{status}] expected={expected} got={got} — {interventions}")
