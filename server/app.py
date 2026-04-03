from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from models import TriageAction, TriageObservation
from support_env import SupportEnv

try:
    from openenv.core.env_server.http_server import create_app as _openenv_create_app
except ImportError:  # pragma: no cover - local fallback when openenv is unavailable
    _openenv_create_app = None


def _fallback_create_app(env_cls, action_model, observation_model, env_name: str, max_concurrent_envs: int = 1):
    app = FastAPI(title=env_name)
    active_envs: Dict[str, SupportEnv] = {"default": env_cls("easy")}

    @app.post("/reset")
    def reset(payload: Dict[str, Any] | None = None):
        body = payload or {}
        task_name = body.get("task_name", "easy")
        seed = int(body.get("seed", 42))
        active_envs["default"] = env_cls(task_name, seed=seed)
        observation = active_envs["default"].reset(seed=seed, initial_conditions=body.get("initial_conditions"))
        return {"observation": observation, "reward": 0.0, "done": False, "info": {"task": task_name}}

    @app.get("/state")
    def state():
        env = active_envs["default"]
        return {"observation": env.state()}

    @app.post("/step")
    def step(action: Dict[str, Any]):
        env = active_envs["default"]
        observation, reward, done, info = env.step(action_model.model_validate(action).model_dump())
        return {"observation": observation, "reward": reward, "done": done, "info": info}

    return app


create_app = _openenv_create_app or _fallback_create_app

app = create_app(
    SupportEnv,
    TriageAction,
    TriageObservation,
    env_name="triageiq",
    max_concurrent_envs=1,
)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "service": "TriageIQ"}


@app.get("/", include_in_schema=False)
def root():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <title>TriageIQ</title>
          <style>
            :root {
              --bg: linear-gradient(135deg, #0f172a 0%, #123a5a 48%, #185f85 100%);
              --card: rgba(255, 255, 255, 0.12);
              --text: #f8fafc;
              --muted: #cbd5e1;
              --accent: #7dd3fc;
            }
            body {
              margin: 0;
              min-height: 100vh;
              font-family: "Trebuchet MS", "Segoe UI", sans-serif;
              background: var(--bg);
              color: var(--text);
              display: grid;
              place-items: center;
            }
            main {
              max-width: 760px;
              margin: 24px;
              padding: 32px;
              border-radius: 24px;
              background: var(--card);
              backdrop-filter: blur(10px);
              box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            }
            h1 {
              margin: 0 0 12px;
              font-size: clamp(2.3rem, 7vw, 4rem);
              letter-spacing: 0.04em;
            }
            p {
              color: var(--muted);
              line-height: 1.6;
              font-size: 1.05rem;
            }
            .pill {
              display: inline-block;
              margin-bottom: 18px;
              padding: 8px 14px;
              border-radius: 999px;
              background: rgba(125, 211, 252, 0.18);
              color: var(--accent);
              font-weight: 700;
              text-transform: uppercase;
              letter-spacing: 0.08em;
              font-size: 0.8rem;
            }
            code {
              padding: 2px 6px;
              border-radius: 6px;
              background: rgba(15, 23, 42, 0.4);
              color: #e2e8f0;
            }
          </style>
        </head>
        <body>
          <main>
            <div class="pill">Meta PyTorch OpenEnv x Scaler India AI Hackathon</div>
            <h1>TriageIQ</h1>
            <p>
              TriageIQ is an OpenEnv-compatible customer support triage environment where agents
              must route tickets, craft helpful responses, escalate high-risk customers, and close
              cases under SLA pressure.
            </p>
            <p>
              Start with <code>/healthz</code> for a quick liveness check, or connect through the
              OpenEnv HTTP interface to run the <code>easy</code>, <code>medium</code>, and
              <code>hard</code> tasks.
            </p>
          </main>
        </body>
        </html>
        """
    )
