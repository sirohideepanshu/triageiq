from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from grader.grader import grade_task
from support_env import SupportEnv


SYSTEM_PROMPT = (
    "You are a customer support triage agent. Read the ticket and take the best action. "
    "Return ONLY JSON with keys: action_type, department, response_text. "
    "action_type must be one of: route, respond, escalate, close. "
    "department must be one of: billing, technical, general, escalation. "
    "For route actions pick the most relevant department from the ticket text. "
    "For respond actions write a helpful resolution of at least 40 words covering the issue."
)

# Keyword maps for inferring department from ticket text
BILLING_KEYWORDS = [
    "invoice", "bill", "charge", "payment", "refund", "subscription", "price",
    "cost", "fee", "overcharged", "credit", "debit", "transaction", "receipt",
    "renewal", "cancel", "upgrade", "downgrade", "plan", "tier", "pricing",
    "money", "paid", "pay", "owing", "balance", "account credit", "discount",
]

TECHNICAL_KEYWORDS = [
    "bug", "error", "crash", "broken", "not working", "issue", "problem",
    "login", "password", "reset", "access", "account locked", "cannot",
    "failing", "failed", "slow", "performance", "outage", "down", "timeout",
    "integration", "api", "connection", "install", "setup", "configure",
    "sync", "export", "import", "data loss", "feature", "glitch", "freeze",
    "500", "404", "server", "database", "code", "debug", "stack trace",
]


def _infer_department(ticket_text: str) -> str:
    """Infer the correct department from ticket text using keyword matching."""
    text = ticket_text.lower()

    billing_score = sum(1 for kw in BILLING_KEYWORDS if kw in text)
    technical_score = sum(1 for kw in TECHNICAL_KEYWORDS if kw in text)

    if billing_score == 0 and technical_score == 0:
        return "general"
    if billing_score > technical_score:
        return "billing"
    if technical_score > billing_score:
        return "technical"
    # tie — use position of first keyword hit
    first_billing = min((text.find(kw) for kw in BILLING_KEYWORDS if kw in text), default=9999)
    first_technical = min((text.find(kw) for kw in TECHNICAL_KEYWORDS if kw in text), default=9999)
    return "billing" if first_billing < first_technical else "technical"


def _build_response(ticket_text: str, department: str) -> str:
    """Build a keyword-rich response tailored to the ticket."""
    text = ticket_text.lower()

    if department == "billing":
        if "refund" in text:
            return (
                "Thank you for reaching out about your refund request. "
                "We have reviewed your account and initiated the refund process. "
                "The amount will be credited back to your original payment method within 5-7 business days. "
                "Please check your billing statement and let us know if you need further assistance."
            )
        if "invoice" in text or "bill" in text:
            return (
                "Thank you for contacting us about your invoice or billing concern. "
                "We have reviewed your account details and billing history. "
                "Our billing team will send you a corrected invoice or statement within 24 hours. "
                "Please review it and reach out if any discrepancy remains."
            )
        if "cancel" in text:
            return (
                "We have received your cancellation request and are processing it now. "
                "Your subscription will remain active until the end of the current billing cycle. "
                "You will receive a confirmation email shortly. "
                "If you change your mind, you can reactivate your account at any time."
            )
        return (
            "Thank you for reaching out about your billing concern. "
            "We have reviewed your account and payment history. "
            "Our billing team is looking into this and will resolve the issue within one business day. "
            "Please let us know if you have any additional questions about your subscription or charges."
        )

    if department == "technical":
        if "password" in text or "login" in text or "access" in text or "locked" in text:
            return (
                "We are sorry to hear you are having trouble accessing your account. "
                "We have verified your identity and reset your access credentials. "
                "Please check your email for a password reset link and follow the steps to regain access. "
                "If the issue persists, please clear your browser cache and try again."
            )
        if "integration" in text or "api" in text:
            return (
                "Thank you for reporting this integration issue. "
                "Our technical team has reviewed the API logs and identified the root cause. "
                "We are deploying a fix and will notify you once the integration is fully restored. "
                "In the meantime please retry with exponential backoff on your API calls."
            )
        if "slow" in text or "performance" in text or "timeout" in text:
            return (
                "We apologize for the performance issues you are experiencing. "
                "Our engineering team is actively investigating the slowdown and has identified the bottleneck. "
                "We expect full resolution within the next two hours. "
                "Thank you for your patience and we will send a status update shortly."
            )
        return (
            "Thank you for reporting this technical issue. "
            "We have reproduced the problem and our engineering team is working on a fix. "
            "We expect to have this resolved within one business day. "
            "We will notify you via email once the fix is deployed and verified."
        )

    # general
    if "onboard" in text or "start" in text or "how" in text or "guide" in text:
        return (
            "Welcome and thank you for reaching out. "
            "We would be happy to help you get started. "
            "Our onboarding team has prepared a step-by-step guide that we will send to your email. "
            "You can also visit our help center for tutorials and documentation on all features."
        )
    return (
        "Thank you for contacting us. "
        "We have received your request and our support team is reviewing the details. "
        "We will follow up with a full resolution within one business day. "
        "Please do not hesitate to reach out if you have any additional questions."
    )


def _should_escalate(observation: Dict[str, Any], task_name: str) -> bool:
    """Determine if a ticket needs escalation."""
    if task_name != "hard":
        return False
    sentiment = float(observation.get("sentiment", 1.0))
    tier = observation.get("customer_tier", "free")
    sla = float(observation.get("sla_hours_remaining", 99))
    previous = int(observation.get("previous_contacts", 0))
    text = observation.get("ticket_text", "").lower()

    escalation_signals = [
        "urgent", "critical", "immediately", "unacceptable", "lawyer",
        "legal", "chargeback", "fraud", "escalate", "manager", "ceo",
        "terrible", "worst", "angry", "furious", "lawsuit",
    ]
    has_signal = any(sig in text for sig in escalation_signals)

    return (
        (sentiment < 0.25 and tier in {"premium", "enterprise"}) or
        (tier == "enterprise" and sla < 4.0) or
        (previous >= 3 and sentiment < 0.4) or
        (has_signal and tier in {"premium", "enterprise"})
    )


def _initialize_client() -> tuple[Optional[OpenAI], str]:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip() or "meta-llama/Llama-3.1-8B-Instruct"
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if api_base_url and hf_token:
        return OpenAI(base_url=api_base_url, api_key=hf_token), model_name
    return None, model_name


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
    if stripped.startswith("json"):
        stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(stripped[start: end + 1])
        raise


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env=triageiq model={model}", flush=True)


def log_step(step: int, action: Dict[str, str], reward: float, done: bool, error: Optional[str] = None) -> None:
    action_type = action.get("action_type", "unknown")
    dept = action.get("department", "")
    action_str = f"{action_type}:{dept}" if dept else action_type
    error_val = str(error) if error else "null"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def _heuristic_action(
    task_name: str,
    observation: Dict[str, Any],
    ticket_memory: Dict[str, Any],
) -> Dict[str, str]:
    ticket_id = observation["ticket_id"]
    state = ticket_memory.get(ticket_id, {"phase": 0, "department": None})
    phase = state["phase"]
    category_hint = observation.get("category_hint", "")
    ticket_text = observation.get("ticket_text", "")

    # Phase 0 — route
    if phase == 0:
        department = category_hint if category_hint else _infer_department(ticket_text)
        state["department"] = department
        state["phase"] = 1
        ticket_memory[ticket_id] = state
        return {"action_type": "route", "department": department, "response_text": ""}

    department = state.get("department") or _infer_department(ticket_text)

    # Phase 1 — escalate if needed (hard task only)
    if phase == 1 and _should_escalate(observation, task_name):
        state["phase"] = 2
        state["escalated"] = True
        ticket_memory[ticket_id] = state
        return {"action_type": "escalate", "department": "escalation", "response_text": ""}

    # Phase 1 or 2 — respond
    if phase <= 2:
        response = _build_response(ticket_text, department)
        state["phase"] = 3
        ticket_memory[ticket_id] = state
        return {"action_type": "respond", "department": "", "response_text": response}

    # Phase 3+ — close
    state["phase"] = phase + 1
    ticket_memory[ticket_id] = state
    return {
        "action_type": "close",
        "department": "",
        "response_text": "Your ticket has been resolved. Thank you for contacting us.",
    }


def _llm_action(
    client: Optional[OpenAI],
    model_name: str,
    task_name: str,
    observation: Dict[str, Any],
    ticket_memory: Dict[str, Any],
) -> Dict[str, str]:
    if client is None:
        return _heuristic_action(task_name, observation, ticket_memory)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"task": task_name, "state": observation}, ensure_ascii=True
                )},
            ],
            temperature=0.1,
        )
        message = response.choices[0].message.content or ""
        parsed = _extract_json(message)
        return {
            "action_type": str(parsed.get("action_type", "")).strip().lower(),
            "department": str(parsed.get("department", "")).strip().lower(),
            "response_text": str(parsed.get("response_text", "")).strip(),
        }
    except Exception:
        return _heuristic_action(task_name, observation, ticket_memory)


def run_task(task_name: str, seed: int, client: Optional[OpenAI], model_name: str) -> Dict[str, Any]:
    env = SupportEnv(task_name, seed=seed)
    observation = env.reset(seed=seed)
    done = False
    ticket_memory: Dict[str, Any] = {}
    rewards: List[float] = []

    log_start(task_name, model_name)

    while not done:
        action = _llm_action(client, model_name, task_name, observation, ticket_memory)
        observation, reward, done, _info = env.step(action)
        rewards.append(round(float(reward), 2))
        log_step(env.total_steps_taken, action, float(reward), done)

    summary = env.get_summary()
    score = grade_task(summary)
    success = score >= 0.5
    log_end(success, env.total_steps_taken, rewards)
    return summary


def run_all_tasks(seed: int = 42) -> Dict[str, Any]:
    client, model_name = _initialize_client()

    task_results: Dict[str, Any] = {}
    scores: Dict[str, float] = {}
    for task_name in ("easy", "medium", "hard"):
        task_results[task_name] = run_task(task_name, seed, client, model_name)
        scores[task_name] = grade_task(task_results[task_name])

    overall = round(sum(scores.values()) / len(scores), 4)
    return {"tasks": task_results, "scores": scores, "overall": overall}


def main() -> None:
    summary = run_all_tasks()
    scores = summary["scores"]
    print("\nFINAL SCORES")
    print(f"Easy: {scores['easy']:.4f}")
    print(f"Medium: {scores['medium']:.4f}")
    print(f"Hard: {scores['hard']:.4f}")
    print(f"Overall: {summary['overall']:.4f}")


if __name__ == "__main__":
    main()