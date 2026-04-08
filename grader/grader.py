from __future__ import annotations

from typing import Any, Dict


def _clamp(v: float) -> float:
    """Clamp to strictly (0, 1) — never exactly 0.0 or 1.0."""
    return max(0.0001, min(0.9999, float(v)))


def _compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    tickets = results.get("tickets", [])
    if not tickets:
        return {
            "routing_accuracy":    0.0001,
            "response_quality":    0.0001,
            "escalation_accuracy": 0.0001,
            "sla_compliance":      0.0001,
            "resolution_rate":     0.0001,
        }

    n = len(tickets)

    routing_accuracy = _clamp(
        sum(1.0 for t in tickets if t["was_routed_correctly"]) / n
    )

    responded_scores = [t["keyword_score"] for t in tickets if t["was_responded_to"]]
    response_quality = _clamp(
        sum(responded_scores) / len(responded_scores) if responded_scores else 0.0
    )

    escalation_accuracy = _clamp(
        sum(
            1.0
            for t in tickets
            if (
                (t["requires_escalation"] and t["was_escalated_correctly"])
                or (not t["requires_escalation"] and not t["escalation_attempted"])
            )
        ) / n
    )

    sla_compliance = _clamp(
        sum(1.0 for t in tickets if t["resolved_before_sla"]) / n
    )

    resolution_rate = _clamp(
        sum(1.0 for t in tickets if t["properly_closed"]) / n
    )

    return {
        "routing_accuracy":    routing_accuracy,
        "response_quality":    response_quality,
        "escalation_accuracy": escalation_accuracy,
        "sla_compliance":      sla_compliance,
        "resolution_rate":     resolution_rate,
    }


def grade_task(results: Dict[str, Any]) -> float:
    """Grade a single task result. Returns strictly (0.0, 1.0)."""
    metrics = _compute_metrics(results)
    score = (
        0.35 * metrics["routing_accuracy"]
        + 0.30 * metrics["response_quality"]
        + 0.15 * metrics["escalation_accuracy"]
        + 0.10 * metrics["sla_compliance"]
        + 0.10 * metrics["resolution_rate"]
    )
    # Double clamp — metrics are already clamped but final score must also be strict
    score = _clamp(score)
    return round(score, 4)


def grade_all(summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """If summary is None, run inference and grade. Returns {scores, overall, passed}"""
    if summary is None:
        from inference import run_all_tasks
        summary = run_all_tasks()

    tasks = summary.get("tasks", {})
    scores: Dict[str, float] = {}
    metrics: Dict[str, Dict[str, float]] = {}
    for task_name in ("easy", "medium", "hard"):
        task_results = tasks.get(task_name, {})
        scores[task_name] = grade_task(task_results)
        metrics[task_name] = _compute_metrics(task_results)

    overall = _clamp(sum(scores.values()) / len(scores)) if scores else 0.0001
    return {
        "scores":  scores,
        "overall": round(overall, 4),
        "passed":  overall >= 0.65,
        "metrics": metrics,
    }