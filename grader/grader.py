from __future__ import annotations
from typing import Any, Dict

def _clamp(value: float) -> float:
    """Clamp a score to be strictly between 0 and 1 (exclusive)."""
    return max(0.0001, min(0.9999, value))

def _compute_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    tickets = results.get("tickets", [])
    if not tickets:
        return {
            "routing_accuracy": 0.0001,
            "response_quality": 0.0001,
            "escalation_accuracy": 0.0001,
            "sla_compliance": 0.0001,
            "resolution_rate": 0.0001,
        }

    routing_accuracy = sum(1.0 for ticket in tickets if ticket["was_routed_correctly"]) / len(tickets)

    responded_scores = [ticket["keyword_score"] for ticket in tickets if ticket["was_responded_to"]]
    response_quality = sum(responded_scores) / len(responded_scores) if responded_scores else 0.0001

    escalation_accuracy = (
        sum(
            1.0
            for ticket in tickets
            if (
                (ticket["requires_escalation"] and ticket["was_escalated_correctly"])
                or (not ticket["requires_escalation"] and not ticket["escalation_attempted"])
            )
        )
        / len(tickets)
    )

    sla_compliance = sum(1.0 for ticket in tickets if ticket["resolved_before_sla"]) / len(tickets)
    resolution_rate = sum(1.0 for ticket in tickets if ticket["properly_closed"]) / len(tickets)

    return {
        "routing_accuracy": _clamp(routing_accuracy),
        "response_quality": _clamp(response_quality),
        "escalation_accuracy": _clamp(escalation_accuracy),
        "sla_compliance": _clamp(sla_compliance),
        "resolution_rate": _clamp(resolution_rate),
    }

def grade_task(results: Dict[str, Any]) -> float:
    """Grade a single task result. Returns strictly between 0 and 1."""
    metrics = _compute_metrics(results)
    score = (
        0.35 * metrics["routing_accuracy"]
        + 0.30 * metrics["response_quality"]
        + 0.15 * metrics["escalation_accuracy"]
        + 0.10 * metrics["sla_compliance"]
        + 0.10 * metrics["resolution_rate"]
    )
    return round(_clamp(score), 4)

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

    overall = round(_clamp(sum(scores.values()) / len(scores)), 4) if scores else 0.0001

    return {
        "scores": scores,
        "overall": overall,
        "passed": overall >= 0.65,
        "metrics": metrics,
    }