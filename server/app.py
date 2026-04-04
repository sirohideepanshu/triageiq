from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from support_env import SupportEnv

app = FastAPI(title="TriageIQ", version="1.0.0")

_envs: Dict[str, SupportEnv] = {}


class ResetRequest(BaseModel):
    seed: Optional[int] = 42
    task: Optional[str] = None
    task_name: Optional[str] = None
    initial_conditions: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action_type: Optional[str] = None
    department: Optional[str] = ""
    response_text: Optional[str] = ""
    action: Optional[Dict[str, Any]] = None
    episode_id: Optional[str] = None


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>TriageIQ</title>
  <style>
    body {
      margin: 0;
      background: #07090f;
      color: #fff;
      font-family: Inter, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
    }
    .card {
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 20px;
      padding: 40px;
      max-width: 600px;
      width: 100%;
      text-align: center;
    }
    h1 { font-size: 2.5rem; margin: 0 0 8px; }
    p { color: #aaa; line-height: 1.6; }
    .links { display: flex; gap: 12px; justify-content: center; margin-top: 24px; flex-wrap: wrap; }
    a {
      padding: 10px 20px;
      border-radius: 8px;
      text-decoration: none;
      font-weight: 600;
      background: #4f8ef7;
      color: #fff;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>🎫 TriageIQ</h1>
    <p>Where every ticket finds its answer.</p>
    <p>An OpenEnv-compatible customer support triage environment where agents must route tickets, craft helpful responses, escalate high-risk customers, and close cases under SLA pressure.</p>
    <div class="links">
      <a href="/docs">API Docs</a>
      <a href="/healthz">Health Check</a>
      <a href="/schema">Schema</a>
      <a href="/tasks">Tasks</a>
      <a href="/metadata">Metadata</a>
    </div>
  </div>
</body>
</html>
""")


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "healthy", "service": "TriageIQ"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy", "service": "TriageIQ"}


@app.post("/reset")
def reset(request: ResetRequest = None) -> Dict[str, Any]:
    if request is None:
        request = ResetRequest()

    task = request.task_name or request.task or "easy"
    if task not in ("easy", "medium", "hard"):
        task = "easy"

    seed = request.seed if request.seed is not None else 42

    env = SupportEnv(task, seed=seed)
    obs = env.reset(seed=seed, initial_conditions=request.initial_conditions)

    episode_id = str(uuid4())
    _envs[episode_id] = env
    _envs["current"] = env

    result = dict(obs)
    result["episode_id"] = episode_id
    return result


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    episode_id = request.episode_id or "current"
    env = _envs.get(episode_id) or _envs.get("current")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    if request.action is not None:
        action = {
            "action_type": str(request.action.get("action_type", "")).strip().lower(),
            "department": str(request.action.get("department", "")).strip().lower(),
            "response_text": str(request.action.get("response_text", "")).strip(),
        }
    else:
        action = {
            "action_type": str(request.action_type or "").strip().lower(),
            "department": str(request.department or "").strip().lower(),
            "response_text": str(request.response_text or "").strip(),
        }

    try:
        obs, reward, done, info = env.step(action)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = dict(obs)
    result["reward"] = float(reward)
    result["done"] = bool(done)
    result["info"] = info
    result["observation"] = obs
    return result


@app.get("/state")
def state(episode_id: Optional[str] = None) -> Dict[str, Any]:
    env = _envs.get(episode_id or "current")
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["route", "respond", "escalate", "close"]
                },
                "department": {
                    "type": "string",
                    "enum": ["billing", "technical", "general", "escalation", ""]
                },
                "response_text": {"type": "string"}
            },
            "required": ["action_type"]
        },
        "observation": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "ticket_text": {"type": "string"},
                "customer_tier": {"type": "string"},
                "sentiment": {"type": "number"},
                "previous_contacts": {"type": "integer"},
                "sla_hours_remaining": {"type": "number"},
                "category_hint": {"type": "string"},
                "step": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "last_reward": {"type": "number"},
                "last_action": {"type": "object"}
            }
        },
        "state": {
            "type": "object",
            "description": "Current environment state, identical to observation"
        }
    }


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "name": "triageiq",
        "description": "Customer support ticket triage environment. Agent must route, respond, escalate, and close tickets across easy, medium, and hard tasks.",
        "tasks": ["easy", "medium", "hard"],
        "version": "1.0.0"
    }


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "1 ticket per episode, hint provided, objective is correct routing",
                "max_steps": 5,
                "sla_hours": 48
            },
            {
                "name": "medium",
                "description": "3 tickets per episode, no hint, agent must route, respond, and close",
                "max_steps": 30,
                "sla_hours": 24
            },
            {
                "name": "hard",
                "description": "5 tickets per episode, enterprise customers, escalation required, 8hr SLA",
                "max_steps": 75,
                "sla_hours": 8
            }
        ]
    }


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()