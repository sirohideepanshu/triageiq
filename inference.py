from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from grader.grader import grade_task
from support_env import SupportEnv


SYSTEM_PROMPT = (
    "You are a customer support triage agent. Read the ticket and take the best action. "
    "Return ONLY JSON with keys: action_type, department, response_text"
)


def _initialize_client() -> tuple[Optional[OpenAI], Dict[str, Any]]:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip() or "meta-llama/Llama-3.1-8B-Instruct"
    hf_token = os.getenv("HF_TOKEN", "").strip()

    if api_base_url and hf_token:
        client = OpenAI(base_url=api_base_url, api_key=hf_token)
        return client, {
            "api_base_url": api_base_url,
            "model_name": model_name,
            "hf_token_present": True,
        }
    return None, {
        "api_base_url": api_base_url,
        "model_name": model_name,
        "hf_token_present": bool(hf_token),
    }


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
            return json.loads(stripped[start : end + 1])
        raise


def _heuristic_action(task_name: str, observation: Dict[str, Any], ticket_memory: Dict[str, int]) -> Dict[str, str]:
    ticket_id = observation["ticket_id"]
    phase = ticket_memory.get(ticket_id, 0)
    category_hint = observation.get("category_hint", "")

    if phase == 0:
        ticket_memory[ticket_id] = 1
        department = category_hint or "general"
        return {"action_type": "route", "department": department, "response_text": ""}

    if task_name == "hard" and phase == 1 and observation["sentiment"] < 0.2 and observation["customer_tier"] in {"premium", "enterprise"}:
        ticket_memory[ticket_id] = 2
        return {"action_type": "escalate", "department": "escalation", "response_text": ""}

    if phase <= 2:
        ticket_memory[ticket_id] = 3
        response = (
            "Thanks for the detailed report. We are reviewing your request and will provide a "
            "resolution update shortly after checking the account or service status."
        )
        return {"action_type": "respond", "department": "", "response_text": response}

    ticket_memory[ticket_id] = phase + 1
    return {"action_type": "close", "department": "", "response_text": "Closing after sharing the next steps."}


def _llm_action(
    client: Optional[OpenAI],
    runtime: Dict[str, Any],
    task_name: str,
    observation: Dict[str, Any],
    ticket_memory: Dict[str, int],
) -> Dict[str, str]:
    if client is None:
        return _heuristic_action(task_name, observation, ticket_memory)

    try:
        response = client.chat.completions.create(
            model=runtime["model_name"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps({"task": task_name, "state": observation}, ensure_ascii=True),
                },
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


def run_task(task_name: str, seed: int, client: Optional[OpenAI], runtime: Dict[str, Any]) -> Dict[str, Any]:
    env = SupportEnv(task_name, seed=seed)
    observation = env.reset(seed=seed)
    done = False
    ticket_memory: Dict[str, int] = {}

    while not done:
        action = _llm_action(client, runtime, task_name, observation, ticket_memory)
        observation, reward, done, _info = env.step(action)
        print(
            f"[STEP] {json.dumps({'task': task_name, 'step': env.total_steps_taken, 'action_type': action.get('action_type', ''), 'department': action.get('department', ''), 'reward': reward, 'done': done}, ensure_ascii=True)}"
        )

    return env.get_summary()


def run_all_tasks(seed: int = 42) -> Dict[str, Any]:
    client, runtime = _initialize_client()
    print(
        f"[START] {json.dumps({'seed': seed, 'runtime': runtime, 'openai_client_initialized': client is not None}, ensure_ascii=True)}"
    )

    task_results: Dict[str, Any] = {}
    scores: Dict[str, float] = {}
    for task_name in ("easy", "medium", "hard"):
        task_results[task_name] = run_task(task_name, seed, client, runtime)
        scores[task_name] = grade_task(task_results[task_name])

    overall = round(sum(scores.values()) / len(scores), 4)
    print(f"[END] {json.dumps({'scores': scores, 'overall': overall}, ensure_ascii=True)}")
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
