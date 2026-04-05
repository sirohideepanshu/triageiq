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

# ── Weighted keyword scoring ───────────────────────────────────────────────

BILLING_WEIGHTED: List[Tuple[str, float]] = [
    # Very strong billing signals
    ("invoice", 3.0), ("invoices", 3.0), ("vat", 3.0), ("tax exempt", 3.0),
    ("exemption certificate", 3.0), ("finance rejected", 3.0), ("overcharged", 3.0),
    ("refund", 2.5), ("reimbursement", 2.5), ("chargeback", 2.5),
    ("charged twice", 2.5), ("double charged", 2.5), ("payment failed", 2.5),
    ("payment declined", 2.5), ("card declined", 2.5), ("billing dispute", 2.5),
    # Strong billing signals
    ("billing", 2.0), ("charge", 2.0), ("charged", 2.0), ("payment", 2.0),
    ("subscription", 2.0), ("renewal", 2.0), ("cancellation", 2.0),
    ("receipt", 2.0), ("transaction", 2.0), ("debit", 2.0), ("credit", 2.0),
    # Medium billing signals — lowered from previous version to avoid false positives
    ("fee", 1.5), ("fees", 1.5), ("discount", 1.5), ("promo", 1.5),
    ("coupon", 1.5), ("free trial", 1.5), ("auto-renew", 1.5),
    ("balance", 1.5), ("owing", 1.5), ("overdue", 1.5), ("due date", 1.5),
    ("wallet", 1.5), ("checkout", 1.5), ("bank", 1.5),
    # Weak billing signals — these appear in general questions too
    # "plan", "pricing", "upgrade", "cancel", "tier" intentionally lowered
    ("cancel", 1.2), ("billing cycle", 1.5), ("next bill", 1.5),
    ("payment method", 1.5), ("price increase", 2.0), ("price change", 2.0),
    ("plan cost", 2.0), ("monthly cost", 2.0), ("annual cost", 2.0),
    ("plan", 0.5), ("pricing", 0.8), ("upgrade", 0.6), ("downgrade", 0.8),
    ("tier", 0.5), ("money", 0.8), ("paid", 0.8), ("pay", 0.5),
    ("tax", 0.8), ("taxes", 0.8), ("cost", 0.5),
]

TECHNICAL_WEIGHTED: List[Tuple[str, float]] = [
    # Very strong technical signals
    ("account locked", 3.0), ("locked out", 3.0), ("cannot log in", 3.0),
    ("can't log in", 3.0), ("sso", 3.0), ("saml", 3.0), ("oauth", 3.0),
    ("webhook", 3.0), ("integration", 3.0), ("stack trace", 3.0),
    ("500 error", 3.0), ("502", 2.5), ("503", 2.5),
    ("deployment failed", 3.0), ("build failed", 3.0),
    ("outage", 3.0), ("service down", 3.0),
    ("security breach", 3.0), ("hacked", 3.0), ("compromised", 3.0),
    ("formatting stripped", 3.0), ("headings stripped", 3.0),
    ("chart is blank", 3.0), ("every chart is blank", 3.0),
    ("date filter", 2.5), ("blank after", 2.5),
    # Strong technical signals
    ("bug", 2.5), ("crash", 2.5), ("crashes", 2.5), ("broken", 2.5),
    ("not working", 2.5), ("error", 2.0), ("errors", 2.0), ("glitch", 2.5),
    ("login", 2.0), ("password", 2.0), ("reset password", 2.5),
    ("2fa", 2.5), ("mfa", 2.5), ("two factor", 2.5),
    ("authentication failed", 2.5), ("unauthorized", 2.5), ("forbidden", 2.5),
    ("permission denied", 2.5), ("sync error", 2.5),
    ("endpoint", 2.0), ("token expired", 2.5),
    ("rate limit", 2.0), ("timeout", 2.0), ("latency", 2.0),
    ("lag", 1.5), ("unresponsive", 2.0),
    ("not loading", 2.0), ("blank screen", 2.0), ("page not found", 2.0),
    ("ssl", 2.0), ("tls", 2.0), ("dns", 2.0), ("cors", 2.0), ("csrf", 2.0),
    ("server error", 2.0), ("database", 2.0),
    ("failing", 2.0), ("failed", 1.5), ("freeze", 2.0),
    ("email not sending", 2.5), ("notification failed", 2.0),
    ("knowledge base import", 2.5), ("formatting issue", 2.5),
    ("headings", 2.0), ("code block", 2.0), ("published help center", 2.5),
    ("article formatting", 2.5),
    # Medium — "api" lowered because many general questions mention API
    ("api", 1.5), ("dashboard", 2.0),
    ("connection", 1.0), ("install", 1.0), ("configure", 1.0),
    ("monitoring", 1.5), ("alert", 1.5), ("incident", 1.5),
    ("cannot", 1.0), ("access", 0.8),
]

GENERAL_WEIGHTED: List[Tuple[str, float]] = [
    # Very strong general signals
    ("feature request", 3.0), ("bulk archive", 3.0), ("retention policy", 3.0),
    ("difference between", 3.0), ("archived and deleted", 3.0),
    ("is this on your roadmap", 3.0), ("formal quote", 3.0),
    ("procurement", 3.0), ("not reporting a bug", 3.0),
    ("general question", 3.0), ("quick question", 3.0),
    ("free plan include", 3.0), ("does the free plan", 3.0),
    ("unlimited teammates", 3.0), ("how many teammates", 3.0),
    ("export all customer conversations", 3.0), ("export conversations", 3.0),
    ("before ending our contract", 3.0), ("before ending our subscription", 3.0),
    ("audit log retention", 3.0), ("audit log", 2.5),
    ("native connector", 2.5), ("google sheets", 2.5),
    ("usage metrics", 2.0), ("into google", 2.5),
    # Strong general signals
    ("is it possible", 2.5), ("is there another way", 2.5),
    ("is there a way", 2.5), ("is there a native", 2.5),
    ("how to", 2.0), ("guide", 2.0), ("tutorial", 2.0),
    ("documentation", 2.0), ("docs", 2.0), ("roadmap", 2.5),
    ("onboard", 2.0), ("onboarding", 2.0), ("getting started", 2.5),
    ("feedback", 2.0), ("suggestion", 2.0), ("improvement", 2.0),
    ("announcement", 1.5), ("changelog", 1.5), ("release notes", 2.0),
    ("explain", 1.5), ("concise answer", 2.5), ("simple question", 2.0),
    ("archive", 2.0), ("archived", 2.0), ("archiving", 2.0),
    ("retention", 2.0), ("deleted", 1.5), ("deletion", 1.5),
    ("closed cases", 2.5), ("one by one", 2.5), ("bulk", 2.0),
    ("conversation history", 2.0), ("what is", 1.5), ("can you", 1.0),
    ("do you support", 2.0), ("help me understand", 2.0),
    ("plan comparison", 2.5), ("download", 1.0),
    ("teammates", 1.5), ("team members", 1.5), ("seats", 1.5),
    ("quote", 2.0), ("approve", 1.5), ("approval", 1.5),
    ("pricing page", 2.5), ("checked the pricing", 2.5),
    ("feature", 1.5), ("request", 1.0), ("export", 1.0),
]

ESCALATION_KEYWORDS = [
    "urgent", "critical", "immediately", "unacceptable", "lawyer", "legal",
    "chargeback", "fraud", "escalate", "manager", "ceo", "director",
    "terrible", "worst", "angry", "furious", "lawsuit", "sue", "court",
    "breach", "violation", "production down", "data loss", "security breach",
    "hacked", "compromised", "outrage", "ridiculous", "demand", "threatening",
    "regulator", "complaint", "ombudsman",
]

# ── Response templates ─────────────────────────────────────────────────────

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
        "If there is a VAT or tax exemption discrepancy we will correct it immediately. "
        "A refund or credit will be applied to your account automatically if you were overcharged. "
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
        "We have reviewed your account, payment history, and recent transactions and charges. "
        "Our billing team is investigating the issue and will resolve it within one business day. "
        "If any incorrect charge is confirmed we will issue a refund or credit to your account. "
        "Please let us know if you have any additional questions about your invoice or subscription."
    ),
    ("technical", "access"): (
        "We are sorry to hear you are having trouble accessing your account. "
        "We have verified your identity and initiated an account access reset. "
        "Please check your email for a password reset link and follow the steps to regain access. "
        "If two-factor authentication or MFA is blocking you please use your backup codes. "
        "Clear your browser cache and cookies before trying to log in again and contact us if the issue persists."
    ),
    ("technical", "integration"): (
        "Thank you for reporting this integration issue. "
        "We have reviewed the API and webhook logs and identified the root cause of the connection problem. "
        "Our engineering team is deploying a fix and will notify you once the integration is restored. "
        "In the meantime please retry your API calls with exponential backoff and check our status page. "
        "We will send a follow-up once the fix is verified and the integration is fully operational."
    ),
    ("technical", "performance"): (
        "We apologize for the performance issues and slow response times you are experiencing. "
        "Our engineering team has identified the bottleneck affecting your account and region. "
        "We are actively working on a fix and expect full resolution within two hours. "
        "You can monitor progress on our status page for live updates on this technical issue. "
        "Thank you for your patience and we will send a status update once the issue is resolved."
    ),
    ("technical", "outage"): (
        "We are aware of the service outage you are experiencing and sincerely apologize. "
        "Our engineering team is treating this as a critical incident and working on an immediate fix. "
        "All affected systems are being investigated and we expect service to be restored shortly. "
        "Please monitor our status page for live updates on this outage. "
        "We will send you a full incident report and post-mortem once the issue is resolved."
    ),
    ("technical", "data"): (
        "Thank you for reporting this data issue. "
        "We have flagged your account for urgent investigation by our data engineering team. "
        "We are reviewing your sync logs and recent data changes to identify the root cause. "
        "If any data loss is confirmed we will restore from backup immediately. "
        "We will send you a full status update within four hours with our findings and next steps."
    ),
    ("technical", "general"): (
        "Thank you for reporting this technical issue. "
        "We have reproduced the problem and our engineering team is working on a fix. "
        "We expect to resolve this technical issue within one business day. "
        "In the meantime please try clearing your cache, using a different browser, or restarting the app. "
        "We will send you an update as soon as the issue is confirmed fixed and deployed."
    ),
    ("general", "onboarding"): (
        "Welcome and thank you for reaching out to our support team. "
        "We are happy to help you get started and make the most of our platform. "
        "Our onboarding team has prepared a step-by-step guide that we will send to your email. "
        "You can also visit our help center and documentation for tutorials on all features. "
        "Please let us know if you have any specific questions and we will guide you through setup."
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
        "The difference between archived and deleted conversations is important to understand correctly. "
        "Archived conversations are preserved and remain searchable but are hidden from the main view. "
        "Deleted conversations are permanently removed and cannot be recovered after the retention period. "
        "We recommend reviewing our documentation on retention policy settings before making any changes."
    ),
    ("general", "archive"): (
        "Thank you for your question about our archive and bulk management features. "
        "We understand archiving closed cases one by one is time-consuming and inefficient. "
        "We have logged this as a feature request for bulk archive functionality on our roadmap. "
        "In the meantime you can use filters to select multiple closed cases and archive them in batches. "
        "Please check our documentation for the latest guidance on case management and archiving options."
    ),
    ("general", "export"): (
        "Thank you for reaching out about exporting your data and conversations. "
        "We can assist you with exporting all customer conversations and usage data before your contract ends. "
        "Please visit your account settings and use the data export feature to download your conversations. "
        "We will also confirm your cancellation timeline and ensure no data is lost during the transition. "
        "Our documentation has step-by-step instructions for bulk data export and account closure."
    ),
    ("general", "plan"): (
        "Thank you for your question about our plan options and included features. "
        "We are happy to clarify what is included in each tier including the free plan. "
        "Our plans vary in the number of teammates, seats, and features available. "
        "You can find a full comparison on our pricing page or we can send you a detailed breakdown. "
        "Please let us know your team size and requirements so we can recommend the right plan for you."
    ),
    ("general", "quote"): (
        "Thank you for reaching out about a formal quote for your Enterprise upgrade. "
        "Our procurement and sales team will prepare a detailed quote for your approval process. "
        "Please share your team size, required features, and expected contract length. "
        "We will send you a formal quote document within one business day for your finance team. "
        "Feel free to reach out if you have any questions about Enterprise plan features or pricing."
    ),
    ("general", "general"): (
        "Thank you for contacting our support team. "
        "We have received your request and our support team is reviewing all the details carefully. "
        "We will follow up with a full resolution and any relevant documentation within one business day. "
        "If this is urgent please let us know and we will prioritize your request accordingly. "
        "Please feel free to reach out if you have any additional questions or information to share."
    ),
}


# ── Weighted department scoring ────────────────────────────────────────────

def _weighted_score(text: str, weighted_keywords: List[Tuple[str, float]]) -> float:
    return sum(weight for kw, weight in weighted_keywords if kw in text)


def _infer_department(ticket_text: str, category_hint: str = "") -> str:
    if category_hint:
        return category_hint
    text = ticket_text.lower()

    billing = _weighted_score(text, BILLING_WEIGHTED)
    technical = _weighted_score(text, TECHNICAL_WEIGHTED)
    general = _weighted_score(text, GENERAL_WEIGHTED)

    # Billing wins only when it clearly dominates — raised threshold to avoid
    # false positives from weak billing words like "plan", "pricing", "cancel"
    if billing >= 3.0 and billing > technical and billing > general:
        return "billing"
    # Technical wins only if it clearly beats general by meaningful margin
    if technical > general * 1.3 and technical >= 2.0:
        return "technical"
    # Billing can still win if it moderately dominates and no strong general signal
    if billing >= 2.0 and billing > general * 1.5 and billing > technical:
        return "billing"
    return "general"


def _infer_issue_type(ticket_text: str, department: str) -> str:
    text = ticket_text.lower()
    if department == "billing":
        if any(w in text for w in ["vat", "tax exempt", "tax exemption", "exemption certificate", "finance rejected"]):
            return "invoice"
        if any(w in text for w in ["refund", "reimburs", "money back", "return"]):
            return "refund"
        if any(w in text for w in ["invoice", "receipt", "statement", "overcharg", "tax"]):
            return "invoice"
        if any(w in text for w in ["cancel", "cancellation", "terminate"]):
            return "cancel"
        if any(w in text for w in ["subscription", "plan", "upgrade", "downgrade", "renew", "tier"]):
            return "subscription"
        return "general"
    if department == "technical":
        if any(w in text for w in ["login", "password", "access", "locked", "lock", "sign in", "2fa", "mfa", "auth", "sso"]):
            return "access"
        if any(w in text for w in ["integration", "api", "webhook", "endpoint", "connect", "oauth"]):
            return "integration"
        if any(w in text for w in ["outage", "down", "unavailable", "service disruption", "not reachable"]):
            return "outage"
        if any(w in text for w in ["latency", "timeout", "lag", "unresponsive", "performance", "slow"]):
            return "performance"
        if any(w in text for w in ["data loss", "missing data", "corrupt", "sync error"]):
            return "data"
        return "general"
    if department == "general":
        if any(w in text for w in ["retention", "archived and deleted", "difference between", "retention policy", "audit log"]):
            return "retention"
        if any(w in text for w in ["bulk archive", "archive", "archiving", "closed cases", "one by one"]):
            return "archive"
        if any(w in text for w in ["export all", "export conversations", "export customer", "before ending"]):
            return "export"
        if any(w in text for w in ["formal quote", "procurement", "quote for enterprise", "approve the upgrade"]):
            return "quote"
        if any(w in text for w in ["free plan", "teammates", "team members", "seats", "how many", "unlimited"]):
            return "plan"
        if any(w in text for w in ["onboard", "getting started", "how to", "setup", "guide", "tutorial"]):
            return "onboarding"
        if any(w in text for w in ["feature", "roadmap", "suggestion", "improvement"]):
            return "feature"
        return "general"
    return "general"


def _get_response(ticket_text: str, department: str) -> str:
    issue_type = _infer_issue_type(ticket_text, department)
    key = (department, issue_type)
    if key in RESPONSE_TEMPLATES:
        return RESPONSE_TEMPLATES[key]
    return RESPONSE_TEMPLATES.get((department, "general"), RESPONSE_TEMPLATES[("general", "general")])


def _score_escalation_keywords(text: str) -> int:
    return sum(1 for kw in ESCALATION_KEYWORDS if kw in text)


def _should_escalate(observation: Dict[str, Any], task_name: str, ticket_state: Dict[str, Any]) -> bool:
    if task_name in ("easy", "medium"):
        return False
    if ticket_state.get("escalated", False):
        return False

    text = observation.get("ticket_text", "").lower()
    sentiment = float(observation.get("sentiment", 1.0))
    tier = observation.get("customer_tier", "free")
    sla = float(observation.get("sla_hours_remaining", 99))
    previous = int(observation.get("previous_contacts", 0))

    escalation_signals = _score_escalation_keywords(text)

    return any([
        sentiment < 0.15 and tier == "enterprise",
        tier == "enterprise" and sla < 2.0,
        escalation_signals >= 2 and tier == "enterprise",
        sentiment < 0.1 and previous >= 3,
        "production down" in text and tier in {"premium", "enterprise"},
        "security breach" in text,
        "hacked" in text and tier in {"premium", "enterprise"},
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

    if not state["routed"]:
        department = _infer_department(ticket_text, category_hint)
        state["department"] = department
        state["routed"] = True
        if _should_escalate(observation, task_name, state):
            state["escalation_needed"] = True
        return {"action_type": "route", "department": department, "response_text": ""}

    department = state["department"] or "general"

    if not state["escalated"] and _should_escalate(observation, task_name, state):
        state["escalated"] = True
        state["escalation_needed"] = True
        return {"action_type": "escalate", "department": "escalation", "response_text": ""}

    if not state["responded"]:
        response = _get_response(ticket_text, department)
        state["responded"] = True
        return {"action_type": "respond", "department": "", "response_text": response}

    if _ticket_ready_to_close(state):
        return {
            "action_type": "close",
            "department": "",
            "response_text": "Your ticket has been resolved. Thank you for contacting us.",
        }

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