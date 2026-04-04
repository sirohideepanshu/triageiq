from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
from support_env import SupportEnv

app = FastAPI(title="TriageIQ", version="1.0.0")

_env: Optional[SupportEnv] = None
_task_name: str = "easy"


class ResetRequest(BaseModel):
    seed: Optional[int] = 42
    task: Optional[str] = "easy"
    initial_conditions: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action_type: str
    department: str = ""
    response_text: str = ""


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def root():
    return HTMLResponse("""
    <!DOCTYPE html><html><head><title>TriageIQ</title></head>
    <body style="background:#07090f;color:#fff;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
    <div style="text-align:center">
    <h1>🎫 TriageIQ</h1>
    <p>Where every ticket finds its answer.</p>
    <p><a href="/docs" style="color:#4f8ef7">API Docs</a> &nbsp;|&nbsp;
    <a href="/healthz" style="color:#4f8ef7">Health</a> &nbsp;|&nbsp;
    <a href="/schema" style="color:#4f8ef7">Schema</a></p>
    </div></body></html>
    """)


@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "TriageIQ"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "TriageIQ"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    global _env, _task_name
    if request is None:
        request = ResetRequest()
    task = request.task or "easy"
    if task not in ("easy", "medium", "hard"):
        task = "easy"
    _task_name = task
    seed = request.seed if request.seed is not None else 42
    _env = SupportEnv(task, seed=seed)
    obs = _env.reset(seed=seed, initial_conditions=request.initial_conditions)
    return obs


@app.post("/step")
def step(request: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    action = {
        "action_type": request.action_type,
        "department": request.department,
        "response_text": request.response_text,
    }
    obs, reward, done, info = _env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return _env.state()


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ["route", "respond", "escalate", "close"]},
                "department": {"type": "string", "enum": ["billing", "technical", "general", "escalation", ""]},
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
            "description": "Current environment state, same as observation"
        }
    }


@app.get("/metadata")
def metadata():
    return {
        "name": "triageiq",
        "description": "Customer support ticket triage environment. Agent must route, respond, escalate, and close tickets across easy, medium, and hard tasks.",
        "tasks": ["easy", "medium", "hard"],
        "version": "1.0.0"
    }


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {"name": "easy", "description": "1 ticket, hint provided, route only"},
            {"name": "medium", "description": "3 tickets, no hint, route+respond+close"},
            {"name": "hard", "description": "5 tickets, enterprise customers, escalation required"}
        ]
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()