from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class TriageObservation(BaseModel):
    ticket_id: str
    ticket_text: str
    customer_tier: str
    sentiment: float
    previous_contacts: int
    sla_hours_remaining: float
    category_hint: str
    step: int
    max_steps: int
    last_reward: float
    last_action: Dict[str, Any] = Field(default_factory=dict)


class TriageAction(BaseModel):
    action_type: str
    department: str = ""
    response_text: str = ""


class TriageReward(BaseModel):
    value: float
