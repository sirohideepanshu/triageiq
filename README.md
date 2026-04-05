---
title: TriageIQ
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
short_description: TriageIQ — Where every ticket finds its answer.
tags:
  - openenv
---

# TriageIQ

TriageIQ is a customer support ticket triage environment built for the Meta PyTorch OpenEnv x Scaler India AI Hackathon. It models a real workflow every software company deals with: reading incoming support messages, deciding who should handle them, responding with something useful, escalating high-risk situations, and closing tickets without missing SLA expectations.

Customer support triage is a strong benchmark for text-native agents because the work is sequential, high stakes, and grounded in language. Agents need to classify intent, reason about urgency, adapt to angry or high-value customers, and choose the next best action under limited steps. That makes the task practical, measurable, and well aligned with LLM-based decision making.

## What The Environment Does

The core environment lives in [support_env.py](./support_env.py). Each episode presents one or more deterministic customer tickets sampled from a seeded bank in [tickets.py](./tickets.py). The agent sees one ticket at a time and must decide among four actions:

- Route the ticket to the right department.
- Respond with a useful message that contains resolution cues.
- Escalate when the customer is at risk, high value, or close to SLA breach.
- Close the ticket once it has been handled well enough.

The project ships three task difficulties, lightweight server wiring in [server/app.py](./server/app.py), typed models in [models.py](./models.py), deterministic grading in [grader/grader.py](./grader/grader.py), and an offline-safe benchmark runner in [inference.py](./inference.py).

## Task Levels

### Easy

- 1 ticket per episode
- 5 max steps
- 48 SLA hours
- Only `free` and `premium` customers
- `category_hint` is always provided
- The main objective is correct routing

### Medium

- 3 tickets per episode
- 10 max steps per ticket, 30 total
- 24 SLA hours
- No category hint
- The agent should route, respond, and close each ticket

### Hard

- 5 tickets per episode
- 15 max steps per ticket, 75 total
- 8 SLA hours
- Includes `enterprise` customers
- Includes tickets that require escalation
- The agent must balance routing, response quality, escalation accuracy, and SLA pressure

## Observation Space

Each state returned by the environment matches `TriageObservation` in [models.py](./models.py):

- `ticket_id`: Unique ticket identifier such as `TKT-013`.
- `ticket_text`: The full customer message to reason over.
- `customer_tier`: One of `free`, `premium`, or `enterprise`.
- `sentiment`: A float from `0.0` to `1.0`, where lower values indicate angrier customers.
- `previous_contacts`: How many times the customer already contacted support.
- `sla_hours_remaining`: Remaining time before the ticket breaches its SLA budget.
- `category_hint`: A department hint on easy, and an empty string on medium and hard.
- `step`: The current episode step count.
- `max_steps`: The maximum total steps for the episode.
- `last_reward`: Reward from the previous action.
- `last_action`: The previous action payload as a dictionary.

## Action Space

The agent sends dictionaries matching `TriageAction` in [models.py](./models.py):

- `route`: Set `action_type="route"` and provide `department` as `billing`, `technical`, or `general`.
- `respond`: Set `action_type="respond"` and write a customer-facing `response_text`.
- `escalate`: Set `action_type="escalate"` and usually use `department="escalation"`.
- `close`: Set `action_type="close"` and optionally include a short closing message in `response_text`.

## Reward Function

The reward logic is implemented in [support_env.py](./support_env.py) and is fully deterministic.

### Routing

- `+0.8` for a correct first route
- `+0.4` for a correct late route
- `-0.6` for a wrong route
- `-0.3` for missing the route department

### Response Quality

- Keyword coverage is measured against each ticket's `resolution_keywords`
- Routed tickets earn up to `+0.9 * keyword_score`
- Unrouted tickets earn up to `+0.4 * keyword_score`
- Responses under 20 characters get `-0.2`
- Responses over 800 characters get `-0.1`

### Escalation

- `+0.85` for correctly escalating premium or enterprise risk tickets
- `+0.5` for correctly escalating free-tier risk tickets
- `-0.5` for unnecessary escalation
- A passive `-0.3` late-escalation penalty applies when a required escalation is still missing under 2 remaining SLA hours

### Closing

- `+1.0` for closing after correct routing and a good response
- `+0.5` for closing after correct routing but before a strong response
- `-0.4` for closing without correct routing
- `-0.8` for closing on the first step

### Global Modifiers

- Every step includes an SLA decay penalty based on remaining time
- Positive rewards are scaled by sentiment using `(0.7 + 0.3 * base_sentiment)`
- Final per-step rewards are clamped into `[-1.0, 1.0]`

## Grading

The deterministic grader in [grader/grader.py](./grader/grader.py) uses environment traces only. It computes:

- `routing_accuracy`
- `response_quality`
- `escalation_accuracy`
- `sla_compliance`
- `resolution_rate`

Final score:

```text
0.35 * routing_accuracy
+ 0.30 * response_quality
+ 0.15 * escalation_accuracy
+ 0.10 * sla_compliance
+ 0.10 * resolution_rate
```

## Baseline Scores

Running [inference.py](./inference.py) offline with the built-in heuristic fallback produces these deterministic baseline scores for seed `42`:

# * Easy: `0.7000`
* Medium: `0.9750`
* Hard: `0.8150`
* Overall: `0.8300`
# * Average across seeds 40–49: `0.7139`

If `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are provided, the script will query the configured model at every step and only fall back to heuristics if parsing or API calls fail.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn server_app:app --host 0.0.0.0 --port 8000
```

Run inference and grading:

```bash
python inference.py
```

## Docker

Build the image:

```bash
docker build -t triageiq .
```

Run the container:

```bash
docker run --rm -p 8000:8000 triageiq
```

The Docker configuration is defined in [Dockerfile](./Dockerfile), and the OpenEnv manifest is in [openenv.yaml](./openenv.yaml).

## Project Layout

- [support_env.py](./support_env.py): The single environment implementation
- [tickets.py](./tickets.py): Deterministic ticket bank
- [models.py](./models.py): Pydantic action and observation models
- [inference.py](./inference.py): LLM runner with offline heuristic fallback
- [grader/grader.py](./grader/grader.py): Deterministic scoring
- [tasks/easy.py](./tasks/easy.py): Easy task factory
- [tasks/medium.py](./tasks/medium.py): Medium task factory
- [tasks/hard.py](./tasks/hard.py): Hard task factory
- [server/app.py](./server/app.py): HTTP server app



## Team

**Team Tensor Titans** — built for the Meta PyTorch OpenEnv x Scaler India AI Hackathon 2026.

* **Deepanshu Sirohi**
* **Sahas Rastogi**
* **Yashraj Gulyani**

Built with a focus on realistic text-native workflows, deterministic grading, and easy local reproducibility.