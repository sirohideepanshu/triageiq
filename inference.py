from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from grader.grader import grade_task
from support_env import SupportEnv


SYSTEM_PROMPT = (
    "You are a customer support triage agent. Read the ticket and take the best action. "
    "Return ONLY JSON with keys: action_type, department, response_text"
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


def _heuristic_action(task_name: str, observation: Dict[str, Any], ticket_memory: Dict[str, int]) -> Dict[str, str]:
    ticket_id = observation["ticket_id"]
    phase = ticket_memory.get(ticket_id, 0)
    category_hint = observation.get("category_hint", "")
    if phase == 0:
        ticket_memory[ticket_id] = 1
        return {"action_type": "route", "department": category_hint or "general", "response_text": ""}
    if task_name == "hard" and phase == 1 and observation["sentiment"] < 0.2 and observation["customer_tier"] in {"premium", "enterprise"}:
        ticket_memory[ticket_id] = 2
        return {"action_type": "escalate", "department": "escalation", "response_text": ""}
    if phase <= 2:
        ticket_memory[ticket_id] = 3
        return {"action_type": "respond", "department": "", "response_text": "Thanks for the detailed report. We are reviewing your request and will provide a resolution update shortly after checking the account or service status."}
    ticket_memory[ticket_id] = phase + 1
    return {"action_type": "close", "department": "", "response_text": "Closing after sharing the next steps."}


def _llm_action(client: Optional[OpenAI], model_name: str, task_name: str, observation: Dict[str, Any], ticket_memory: Dict[str, int]) -> Dict[str, str]:
    if client is None:
        return _heuristic_action(task_name, observation, ticket_memory)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"task": task_name, "state": observation}, ensure_ascii=True)},
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
    ticket_memory: Dict[str, int] = {}
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