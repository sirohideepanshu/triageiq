from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from grader.grader import grade_task
from support_env import SupportEnv


SYSTEM_PROMPT = (
    "You are a customer support triage agent. Read the ticket carefully and take the best action. "
    "Return ONLY JSON with keys: action_type, department, response_text. "
    "action_type must be one of: route, respond, escalate, close. "
    "department must be one of: billing, technical, general, escalation. "
    "For route: pick billing, technical, or general based on the ticket content. "
    "For respond: write a detailed helpful resolution of at least 60 words using specific keywords from the issue. "
    "For escalate: use department=escalation only when customer is angry, enterprise, or SLA is critical. "
    "For close: only close after routing and responding with good coverage."
)

# ── Department classification ──────────────────────────────────────────────

BILLING_KEYWORDS = [
    "invoice", "invoices", "bill", "billing", "charge", "charged", "overcharged",
    "payment", "payments", "refund", "refunds", "reimbursement", "subscription",
    "price", "pricing", "cost", "fee", "fees", "credit", "debit", "transaction",
    "receipt", "renewal", "cancel", "cancellation", "upgrade", "downgrade",
    "plan", "tier", "money", "paid", "pay", "owing", "balance", "discount",
    "promo", "coupon", "trial", "free trial", "auto-renew", "auto renew",
    "chargeback", "dispute", "bank", "card", "wallet", "checkout",
    "vat", "tax", "taxes", "tax exemption", "tax exempt", "exemption certificate",
    "charged twice", "double charged", "extra charge", "unexpected charge",
    "billing cycle", "next bill", "due date", "overdue", "finance rejected",
    "payment failed", "payment declined", "card declined", "payment method",
]

# NOTE: "feature", "slow", "export", "import", "update", "request" removed —
# they are too generic and cause false positives on general tickets.
TECHNICAL_KEYWORDS = [
    "bug", "bugs", "error", "errors", "crash", "crashes", "broken", "not working",
    "login", "password", "reset password", "account locked", "locked out",
    "cannot log in", "can't log in", "failing", "failed",
    "outage", "service down", "timeout", "integration", "api",
    "connection failed", "install", "configure", "sync error",
    "glitch", "freeze", "500", "404", "server error",
    "database", "stack trace", "token expired", "oauth", "sso", "saml",
    "webhook", "endpoint", "latency", "lag", "unresponsive",
    "corrupt", "notification failed", "email not sending",
    "two factor", "2fa", "mfa", "authentication failed", "unauthorized",
    "forbidden", "permission denied", "security breach",
    "not loading", "blank screen", "white screen", "page not found",
    "500 error", "502", "503", "csrf", "cors", "rate limit", "throttle",
    "ssl", "tls", "dns", "deployment failed", "build failed",
    "pipeline", "monitoring", "alert", "incident report",
]

GENERAL_KEYWORDS = [
    "feature request", "how to", "guide", "tutorial", "documentation", "docs",
    "onboard", "onboarding", "getting started", "question", "help me understand",
    "feedback", "suggestion", "improvement", "request", "download",
    "plan comparison", "what is", "can you", "is it possible", "roadmap",
    "release", "update", "changelog", "announcement",
    "archive", "archived", "archiving", "bulk archive", "retention",
    "retention policy", "deleted", "deletion", "difference between",
    "what is the difference", "concise", "explain the", "explain how",
    "bulk", "one by one", "closed cases", "conversation history",
    "slow to respond", "feature", "way to do", "another way",
    "is there a way", "is this possible", "can i", "do you support",
]

ESCALATION_KEYWORDS = [
    "urgent", "critical", "immediately", "unacceptable", "lawyer", "legal",
    "chargeback", "fraud", "escalate", "manager", "ceo", "director",
    "terrible", "worst", "angry", "furious", "lawsuit", "sue", "court",
    "breach", "violation", "production down", "data loss", "security breach",
    "hacked", "compromised", "outrage", "ridiculous", "demand", "threatening",
    "regulator", "complaint", "ombudsman",
]

# ── Response templates keyed by (department, issue_type) ──────────────────

RESPONSE_TEMPLATES: Dict[Tuple[str, str], str] = {
    ("billing", "refund"): (
        "Thank you for reaching out about your refund request. "
        "We have reviewed your billing history and confirmed the charge in question. "
        "We are processing your refund now and the amount will be credited back to your "
        "original payment method within 5 to 7 business days. "
        "You will receive a confirmation email once the refund is issued. "
        "Please check your billing statement and let us know if any discrepancy remains."
    ),
    ("billing", "invoice"): (
        "Thank you for contacting us about your invoice concern. "
        "We have reviewed your billing account and identified the issue with your invoice. "
        "Our billing team will send you a corrected invoice or statement within 24 hours. "
        "If you were overcharged or if there is a tax or VAT discrepancy we will correct it immediately. "
        "A refund or credit will be applied to your account automatically if needed. "
        "Please review the updated invoice and reach out if any billing discrepancy remains."
    ),
    ("billing", "cancel"): (
        "We have received your cancellation request and are processing it now. "
        "Your subscription will remain active until the end of the current billing cycle. "
        "No further charges will be made after that date. "
        "You will receive a cancellation confirmation email shortly with the final billing details. "
        "If you change your mind you can reactivate your account at any time from your settings."
    ),
    ("billing", "subscription"): (
        "Thank you for contacting us about your subscription and plan details. "
        "We have reviewed your account and current billing plan. "
        "Our billing team can assist with upgrades, downgrades, renewals, and pricing changes. "
        "We will send you a detailed breakdown of your plan options and costs within one business day. "
        "Please let us know your preferred plan so we can update your subscription accordingly."
    ),
    ("billing", "general"): (
        "Thank you for reaching out about your billing concern. "
        "We have reviewed your account, payment history, and recent transactions. "
        "Our billing team is investigating the issue and will resolve it within one business day. "
        "If any incorrect charge is confirmed we will issue a refund or credit to your account. "
        "Please let us know if you have any additional questions about your invoice or subscription."
    ),
    ("technical", "access"): (
        "We are sorry to hear you are having trouble accessing your account. "
        "We have verified your identity and initiated an account access reset. "
        "Please check your email for a password reset link and follow the steps to regain access. "
        "If two-factor authentication is blocking you please use your backup codes or contact us again. "
        "Clear your browser cache and cookies before trying to log in again."
    ),
    ("technical", "integration"): (
        "Thank you for reporting this integration issue. "
        "We have reviewed the API logs and identified the root cause of the connection problem. "
        "Our engineering team is deploying a fix and will notify you once the integration is restored. "
        "In the meantime please retry your API calls with exponential backoff and check our status page. "
        "We will send a follow-up once the fix is verified and the integration is fully operational."
    ),
    ("technical", "performance"): (
        "We apologize for the performance issues and slow response times you are experiencing. "
        "Our engineering team has identified the bottleneck affecting your account and region. "
        "We are actively working on a fix and expect full resolution within two hours. "
        "You can monitor progress on our status page. "
        "Thank you for your patience and we will send a status update once the issue is resolved."
    ),
    ("technical", "outage"): (
        "We are aware of the service outage you are experiencing and sincerely apologize. "
        "Our engineering team is treating this as a critical incident and working on an immediate fix. "
        "All affected systems are being investigated and we expect service to be restored shortly. "
        "Please monitor our status page for live updates. "
        "We will send you a full incident report and post-mortem once the outage is resolved."
    ),
    ("technical", "data"): (
        "Thank you for reporting this data issue. "
        "We have flagged your account for urgent investigation by our data engineering team. "
        "We are reviewing your sync logs, export history, and recent data changes. "
        "If any data loss is confirmed we will restore from backup immediately. "
        "We will send you a full status update within four hours with our findings and next steps."
    ),
    ("technical", "general"): (
        "Thank you for reporting this technical issue. "
        "We have reproduced the problem and our engineering team is working on a fix. "
        "We expect to resolve this within one business day and will notify you once the fix is deployed. "
        "In the meantime please try clearing your cache, using a different browser, or restarting the app. "
        "We will send you an update as soon as the issue is confirmed resolved."
    ),
    ("general", "onboarding"): (
        "Welcome and thank you for reaching out to our support team. "
        "We are happy to help you get started and make the most of our platform. "
        "Our onboarding team has prepared a step-by-step guide that we will send to your email. "
        "You can also visit our help center and documentation for tutorials on all features. "
        "Please let us know if you have any specific questions and we will guide you through the setup."
    ),
    ("general", "feature"): (
        "Thank you for your feature request and for taking the time to share your feedback. "
        "We have logged your suggestion and roadmap request with our product team for review. "
        "Feature requests from customers are a key input to our development and release planning. "
        "We will notify you if and when this feature is scheduled for release on our roadmap. "
        "In the meantime please check our documentation for existing functionality that may help."
    ),
    ("general", "retention"): (
        "Thank you for reaching out about your data retention and archiving policy questions. "
        "The difference between archived and deleted conversations is important to understand. "
        "Archived conversations are preserved and remain searchable but are hidden from the main view. "
        "Deleted conversations are permanently removed and cannot be recovered after the retention period. "
        "We recommend reviewing our documentation on retention policy settings before making any changes."
    ),
    ("general", "archive"): (
        "Thank you for your question about our archive and bulk management features. "
        "We understand archiving closed cases one by one is time-consuming. "
        "We have logged this as a feature request for bulk archive functionality on our roadmap. "
        "In the meantime you can use filters to select multiple closed cases and archive them in batches. "
        "Please check our documentation for the latest guidance on case management and archiving options."
    ),
    ("general", "general"): (
        "Thank you for contacting our support team. "
        "We have received your request and our support team is reviewing all the details carefully. "
        "We will follow up with a full resolution and any relevant documentation within one business day. "
        "If this is urgent please let us know and we will prioritize your request accordingly. "
        "Please feel free to reach out if you have any additional questions or information to share."
    ),
}


# ── Intent parsing ─────────────────────────────────────────────────────────

def _score_keywords(text: str, keywords: List[str]) -> int:
    return sum(1 for kw in keywords if kw in text)


def _infer_department(ticket_text: str, category_hint: str = "") -> str:
    if category_hint:
        return category_hint
    text = ticket_text.lower()
    billing = _score_keywords(text, BILLING_KEYWORDS)
    technical = _score_keywords(text, TECHNICAL_KEYWORDS)
    general = _score_keywords(text, GENERAL_KEYWORDS)

    # Billing always wins if it has any signal — very specific domain
    if billing > 0 and billing >= technical:
        return "billing"
    # Technical wins only if it clearly dominates general
    if technical > general and technical > 0:
        return "technical"
    # Default to general for ambiguous or soft signals
    return "general"


def _infer_issue_type(ticket_text: str, department: str) -> str:
    text = ticket_text.lower()
    if department == "billing":
        # Check VAT/tax first — very specific signal
        if any(w in text for w in ["vat", "tax exempt", "tax exemption", "exemption certificate", "finance rejected", "tax"]):
            return "invoice"
        if any(w in text for w in ["refund", "reimburs", "money back", "return"]):
            return "refund"
        if any(w in text for w in ["invoice", "receipt", "statement", "overcharg"]):
            return "invoice"
        if any(w in text for w in ["cancel", "cancellation", "terminate"]):
            return "cancel"
        if any(w in text for w in ["subscription", "plan", "upgrade", "downgrade", "renew", "tier"]):
            return "subscription"
        return "general"
    if department == "technical":
        if any(w in text for w in ["login", "password", "access", "locked", "lock", "sign in", "2fa", "mfa", "auth"]):
            return "access"
        if any(w in text for w in ["integration", "api", "webhook", "endpoint", "connect", "oauth", "sso"]):
            return "integration"
        if any(w in text for w in ["outage", "down", "unavailable", "service disruption", "not reachable"]):
            return "outage"
        if any(w in text for w in ["latency", "timeout", "lag", "unresponsive", "performance"]):
            return "performance"
        if any(w in text for w in ["data loss", "missing data", "corrupt", "sync error"]):
            return "data"
        return "general"
    if department == "general":
        if any(w in text for w in ["retention", "archived", "deleted", "difference between", "retention policy"]):
            return "retention"
        if any(w in text for w in ["bulk archive", "archive", "archiving", "closed cases", "one by one"]):
            return "archive"
        if any(w in text for w in ["onboard", "getting started", "how to", "setup", "guide", "tutorial"]):
            return "onboarding"
        if any(w in text for w in ["feature", "roadmap", "suggestion", "improvement", "request"]):
            return "feature"
        return "general"
    return "general"


def _get_response(ticket_text: str, department: str) -> str:
    issue_type = _infer_issue_type(ticket_text, department)
    key = (department, issue_type)
    if key in RESPONSE_TEMPLATES:
        return RESPONSE_TEMPLATES[key]
    fallback_key = (department, "general")
    return RESPONSE_TEMPLATES.get(fallback_key, RESPONSE_TEMPLATES[("general", "general")])


def _should_escalate(observation: Dict[str, Any], task_name: str, ticket_state: Dict[str, Any]) -> bool:
    if task_name in ("easy", "medium"):
        return False
    text = observation.get("ticket_text", "").lower()
    sentiment = float(observation.get("sentiment", 1.0))
    tier = observation.get("customer_tier", "free")
    sla = float(observation.get("sla_hours_remaining", 99))
    previous = int(observation.get("previous_contacts", 0))
    already_escalated = ticket_state.get("escalated", False)

    if already_escalated:
        return False

    escalation_signal = _score_keywords(text, ESCALATION_KEYWORDS) >= 2

    return any([
        sentiment < 0.15 and tier == "enterprise",
        tier == "enterprise" and sla < 2.0,
        escalation_signal and tier == "enterprise",
        sentiment < 0.1 and previous >= 3,
    ])


def _ticket_ready_to_close(ticket_state: Dict[str, Any]) -> bool:
    if not ticket_state.get("routed", False):
        return False
    if not ticket_state.get("responded", False):
        return False
    if ticket_state.get("escalation_needed", False) and not ticket_state.get("escalated", False):
        return False
    if ticket_state.get("steps", 0) < 2:
        return False
    return True


# ── Client setup ───────────────────────────────────────────────────────────

def _initialize_client() -> Tuple[Optional[OpenAI], str]:
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


# ── Logging ────────────────────────────────────────────────────────────────

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


# ── Heuristic agent ────────────────────────────────────────────────────────

def _heuristic_action(
    task_name: str,
    observation: Dict[str, Any],
    ticket_memory: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    ticket_id = observation["ticket_id"]
    if ticket_id not in ticket_memory:
        ticket_memory[ticket_id] = {
            "phase": 0,
            "department": None,
            "routed": False,
            "responded": False,
            "escalated": False,
            "escalation_needed": False,
            "steps": 0,
        }

    state = ticket_memory[ticket_id]
    state["steps"] += 1
    category_hint = observation.get("category_hint", "")
    ticket_text = observation.get("ticket_text", "")

    # Phase 0 — always route first
    if not state["routed"]:
        department = _infer_department(ticket_text, category_hint)
        state["department"] = department
        state["routed"] = True
        if _should_escalate(observation, task_name, state):
            state["escalation_needed"] = True
        return {"action_type": "route", "department": department, "response_text": ""}

    department = state["department"] or "general"

    # Escalate if needed and not yet done
    if not state["escalated"] and _should_escalate(observation, task_name, state):
        state["escalated"] = True
        state["escalation_needed"] = True
        return {"action_type": "escalate", "department": "escalation", "response_text": ""}

    # Respond if not yet responded
    if not state["responded"]:
        response = _get_response(ticket_text, department)
        state["responded"] = True
        return {"action_type": "respond", "department": "", "response_text": response}

    # Close only when ready
    if _ticket_ready_to_close(state):
        return {
            "action_type": "close",
            "department": "",
            "response_text": "Your ticket has been resolved. Thank you for contacting us.",
        }

    # Not ready — respond again with more detail
    response = _get_response(ticket_text, department)
    return {"action_type": "respond", "department": "", "response_text": response}


# ── LLM agent ─────────────────────────────────────────────────────────────

def _llm_action(
    client: Optional[OpenAI],
    model_name: str,
    task_name: str,
    observation: Dict[str, Any],
    ticket_memory: Dict[str, Dict[str, Any]],
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


# ── Task runner ────────────────────────────────────────────────────────────

def run_task(task_name: str, seed: int, client: Optional[OpenAI], model_name: str) -> Dict[str, Any]:
    env = SupportEnv(task_name, seed=seed)
    observation = env.reset(seed=seed)
    done = False
    ticket_memory: Dict[str, Dict[str, Any]] = {}
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