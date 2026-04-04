from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from models import TriageAction, TriageObservation
from tickets import TicketBank


@dataclass(frozen=True)
class TaskConfig:
    ticket_count: int
    max_steps_per_ticket: int
    total_sla_hours: float
    provide_hint: bool


class SupportEnv:
    VALID_ACTIONS = {"route", "respond", "escalate", "close"}
    VALID_DEPARTMENTS = {"billing", "technical", "general", "escalation"}
    TASK_CONFIGS = {
        "easy": TaskConfig(ticket_count=1, max_steps_per_ticket=5, total_sla_hours=48.0, provide_hint=True),
        "medium": TaskConfig(ticket_count=3, max_steps_per_ticket=10, total_sla_hours=24.0, provide_hint=False),
        "hard": TaskConfig(ticket_count=5, max_steps_per_ticket=15, total_sla_hours=8.0, provide_hint=False),
    }

    def __init__(self, task_name: str = "easy", seed: int = 42) -> None:
        task_key = task_name.lower()
        if task_key not in self.TASK_CONFIGS:
            raise ValueError(f"Unsupported task_name '{task_name}'. Expected one of {sorted(self.TASK_CONFIGS)}")
        self.task_name = task_key
        self.seed = seed
        self.config = self.TASK_CONFIGS[self.task_name]
        self.ticket_bank = TicketBank()
        self.selected_tickets: List[Dict[str, Any]] = []
        self.ticket_states: List[Dict[str, Any]] = []
        self.current_ticket_index = 0
        self.total_steps_taken = 0
        self.total_max_steps = self.config.ticket_count * self.config.max_steps_per_ticket
        self.last_reward = 0.0
        self.last_action: Dict[str, Any] = {}
        self.total_reward = 0.0
        self.done = False
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None, initial_conditions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if seed is not None:
            self.seed = seed
        if initial_conditions and "seed" in initial_conditions:
            self.seed = int(initial_conditions["seed"])

        self.selected_tickets = self._select_tickets(self.ticket_bank.get_tickets(self.seed, self.task_name))
        self.ticket_states = [self._initialize_ticket_state(ticket, index) for index, ticket in enumerate(self.selected_tickets)]
        self.current_ticket_index = 0
        self.total_steps_taken = 0
        self.total_max_steps = self.config.ticket_count * self.config.max_steps_per_ticket
        self.last_reward = 0.0
        self.last_action = {}
        self.total_reward = 0.0
        self.done = False
        return self.state()

    def state(self) -> Dict[str, Any]:
        if self.done or not self.ticket_states:
            return self._terminal_state()

        ticket_state = self.ticket_states[self.current_ticket_index]
        return self._build_observation(ticket_state)

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            return self.state(), 0.0, True, {"message": "Episode already complete.", "summary": self.get_summary()}

        current = self.ticket_states[self.current_ticket_index]
        parsed_action, error = self._parse_action(action)
        reward = 0.0
        info: Dict[str, Any] = {"task": self.task_name, "ticket_id": current["ticket"]["ticket_id"]}

        reward += self._apply_passive_penalties(current)
        if error:
            reward -= 1.0
            info["error"] = error
        else:
            reward += self._apply_action(current, parsed_action)
            self.last_action = parsed_action
            current["last_valid_action"] = parsed_action

        self.total_steps_taken += 1
        current["steps_taken"] += 1
        self.last_reward = self._clamp_reward(reward)
        self.total_reward += self.last_reward
        current["action_history"].append({"action": parsed_action or action, "reward": self.last_reward})
        current["sla_hours_remaining"] = max(0.0, current["sla_hours_remaining"] - current["hours_per_step"])

        should_advance = self._should_advance_ticket(current, parsed_action)
        if should_advance:
            self._finalize_ticket(current)
            self._advance_ticket()

        observation = self.state()
        info["current_index"] = self.current_ticket_index
        if self.done:
            info["summary"] = self.get_summary()
        return observation, self.last_reward, self.done, info

    def get_summary(self) -> Dict[str, Any]:
        tickets: List[Dict[str, Any]] = []
        for state in self.ticket_states:
            ticket = state["ticket"]
            tickets.append(
                {
                    "ticket_id": ticket["ticket_id"],
                    "correct_department": ticket["correct_department"],
                    "requires_escalation": ticket["requires_escalation"],
                    "customer_tier": ticket["customer_tier"],
                    "base_sentiment": ticket["base_sentiment"],
                    "steps_taken": state["steps_taken"],
                    "total_sla_hours": state["total_sla_hours"],
                    "sla_hours_remaining": round(state["sla_hours_remaining"], 4),
                    "previous_contacts": state["previous_contacts"],
                    "was_routed_correctly": state["was_routed_correctly"],
                    "was_responded_to": state["was_responded_to"],
                    "keyword_score": round(state["keyword_score"], 4),
                    "escalation_attempted": state["escalation_attempted"],
                    "was_escalated_correctly": state["was_escalated_correctly"],
                    "was_closed": state["was_closed"],
                    "properly_closed": state["properly_closed"],
                    "resolved_before_sla": state["resolved_before_sla"],
                    "handled": state["handled"],
                }
            )

        return {
            "task": self.task_name,
            "seed": self.seed,
            "steps_taken": self.total_steps_taken,
            "max_steps": self.total_max_steps,
            "total_reward": round(self.total_reward, 4),
            "tickets": tickets,
        }

    def _select_tickets(self, shuffled_tickets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.task_name == "hard":
            chosen: List[Dict[str, Any]] = []

            def add_first(predicate) -> None:
                for ticket in shuffled_tickets:
                    if predicate(ticket) and ticket not in chosen:
                        chosen.append(ticket)
                        return

            add_first(lambda ticket: ticket["requires_escalation"])
            add_first(lambda ticket: ticket["customer_tier"] == "enterprise")
            for ticket in shuffled_tickets:
                if ticket not in chosen:
                    chosen.append(ticket)
                if len(chosen) == self.config.ticket_count:
                    break
            return chosen[: self.config.ticket_count]

        filtered = shuffled_tickets
        if self.task_name == "easy":
            filtered = [ticket for ticket in shuffled_tickets if ticket["customer_tier"] in {"free", "premium"}]
        elif self.task_name == "medium":
            filtered = [ticket for ticket in shuffled_tickets if not ticket["requires_escalation"]]
        return filtered[: self.config.ticket_count]

    def _initialize_ticket_state(self, ticket: Dict[str, Any], index: int) -> Dict[str, Any]:
        entropy = self.seed + (index + 1) * 97 + sum(ord(char) for char in ticket["ticket_id"])
        previous_contacts = entropy % 4
        if ticket["customer_tier"] == "premium":
            previous_contacts += 1
        elif ticket["customer_tier"] == "enterprise":
            previous_contacts += 2

        return {
            "ticket": ticket,
            "previous_contacts": previous_contacts,
            "sla_hours_remaining": self.config.total_sla_hours,
            "total_sla_hours": self.config.total_sla_hours,
            "hours_per_step": self.config.total_sla_hours / self.config.max_steps_per_ticket,
            "steps_taken": 0,
            "was_routed_correctly": False,
            "was_responded_to": False,
            "keyword_score": 0.0,
            "escalation_attempted": False,
            "was_escalated_correctly": False,
            "was_closed": False,
            "properly_closed": False,
            "resolved_before_sla": False,
            "handled": False,
            "action_history": [],
            "last_valid_action": {},
        }

    def _build_observation(self, ticket_state: Dict[str, Any]) -> Dict[str, Any]:
        ticket = ticket_state["ticket"]
        hint = ticket["correct_department"] if self.config.provide_hint else ""
        observation = TriageObservation(
            ticket_id=ticket["ticket_id"],
            ticket_text=ticket["ticket_text"],
            customer_tier=ticket["customer_tier"],
            sentiment=ticket["base_sentiment"],
            previous_contacts=ticket_state["previous_contacts"],
            sla_hours_remaining=round(ticket_state["sla_hours_remaining"], 2),
            category_hint=hint,
            step=self.total_steps_taken,
            max_steps=self.total_max_steps,
            last_reward=round(self.last_reward, 4),
            last_action=self.last_action,
        )
        return observation.model_dump()

    def _terminal_state(self) -> Dict[str, Any]:
        fallback_ticket = self.ticket_states[-1]["ticket"] if self.ticket_states else {
            "ticket_id": "DONE",
            "ticket_text": "Episode complete.",
            "customer_tier": "free",
            "base_sentiment": 1.0,
        }
        observation = TriageObservation(
            ticket_id=fallback_ticket["ticket_id"],
            ticket_text=fallback_ticket["ticket_text"],
            customer_tier=fallback_ticket["customer_tier"],
            sentiment=fallback_ticket["base_sentiment"],
            previous_contacts=0,
            sla_hours_remaining=0.0,
            category_hint="",
            step=self.total_steps_taken,
            max_steps=self.total_max_steps,
            last_reward=round(self.last_reward, 4),
            last_action=self.last_action,
        )
        return observation.model_dump()

    def _parse_action(self, action: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            parsed = TriageAction.model_validate(action).model_dump()
        except Exception as exc:  # pragma: no cover - defensive validation branch
            return None, f"Invalid action payload: {exc}"

        action_type = parsed["action_type"].strip().lower()
        department = parsed["department"].strip().lower()
        parsed["action_type"] = action_type
        parsed["department"] = department
        parsed["response_text"] = parsed["response_text"].strip()

        if action_type not in self.VALID_ACTIONS:
            return None, f"Invalid action_type '{action_type}'."
        if department and department not in self.VALID_DEPARTMENTS:
            return None, f"Invalid department '{department}'."
        return parsed, None

    def _apply_passive_penalties(self, ticket_state: Dict[str, Any]) -> float:
        reward = 0.0
        sla_ratio = ticket_state["sla_hours_remaining"] / ticket_state["total_sla_hours"]
        reward -= 0.05 * max(0.0, 1.0 - sla_ratio)
        if (
            ticket_state["ticket"]["requires_escalation"]
            and not ticket_state["was_escalated_correctly"]
            and ticket_state["sla_hours_remaining"] < 2.0
        ):
            reward -= 0.3
        return reward

    def _apply_action(self, ticket_state: Dict[str, Any], action: Dict[str, Any]) -> float:
        action_type = action["action_type"]
        if action_type == "route":
            reward = self._handle_route(ticket_state, action)
        elif action_type == "respond":
            reward = self._handle_respond(ticket_state, action)
        elif action_type == "escalate":
            reward = self._handle_escalate(ticket_state, action)
        else:
            reward = self._handle_close(ticket_state)
        return self._apply_sentiment_modifier(reward, ticket_state["ticket"]["base_sentiment"])

    def _handle_route(self, ticket_state: Dict[str, Any], action: Dict[str, Any]) -> float:
        department = action["department"]
        if not department:
            return -0.3
        if department == ticket_state["ticket"]["correct_department"]:
            first_action = ticket_state["steps_taken"] == 0
            ticket_state["was_routed_correctly"] = True
            if self.task_name == "easy":
                ticket_state["was_closed"] = True
                ticket_state["properly_closed"] = True
            return 0.8 if first_action else 0.4
        return -0.6

    def _handle_respond(self, ticket_state: Dict[str, Any], action: Dict[str, Any]) -> float:
        response_text = action["response_text"]
        lowered = response_text.lower()
        keywords = ticket_state["ticket"]["resolution_keywords"]
        hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
        keyword_score = min(1.0, hits / len(keywords))
        ticket_state["was_responded_to"] = bool(response_text)
        ticket_state["keyword_score"] = max(ticket_state["keyword_score"], keyword_score)

        reward = 0.9 * keyword_score if ticket_state["was_routed_correctly"] else 0.4 * keyword_score
        if len(response_text.strip()) < 20:
            reward -= 0.2
        if len(response_text) > 800:
            reward -= 0.1
        return reward

    def _handle_escalate(self, ticket_state: Dict[str, Any], action: Dict[str, Any]) -> float:
        ticket_state["escalation_attempted"] = True
        action_department = action["department"] or "escalation"
        if ticket_state["ticket"]["requires_escalation"]:
            ticket_state["was_escalated_correctly"] = action_department == "escalation"
            if ticket_state["ticket"]["customer_tier"] in {"premium", "enterprise"}:
                return 0.85 if ticket_state["was_escalated_correctly"] else -0.5
            return 0.5 if ticket_state["was_escalated_correctly"] else -0.5
        ticket_state["was_escalated_correctly"] = False
        return -0.5

    def _handle_close(self, ticket_state: Dict[str, Any]) -> float:
        ticket_state["was_closed"] = True
        if ticket_state["steps_taken"] == 0:
            return -0.8
        if not ticket_state["was_routed_correctly"]:
            return -0.4
        if ticket_state["keyword_score"] > 0.5:
            ticket_state["properly_closed"] = not ticket_state["ticket"]["requires_escalation"] or ticket_state["was_escalated_correctly"]
            return 1.0
        return 0.5

    def _apply_sentiment_modifier(self, reward: float, base_sentiment: float) -> float:
        if reward <= 0:
            return reward
        return reward * (0.7 + 0.3 * base_sentiment)

    def _should_advance_ticket(self, ticket_state: Dict[str, Any], action: Optional[Dict[str, Any]]) -> bool:
        if ticket_state["steps_taken"] >= self.config.max_steps_per_ticket:
            return True
        if self.task_name == "easy":
            return bool(action and action["action_type"] == "route" and ticket_state["was_routed_correctly"])
        return bool(action and action["action_type"] == "close")

    def _finalize_ticket(self, ticket_state: Dict[str, Any]) -> None:
        ticket = ticket_state["ticket"]
        if self.task_name == "easy" and ticket_state["was_routed_correctly"]:
            ticket_state["was_closed"] = True
            ticket_state["properly_closed"] = True

        if not ticket["requires_escalation"] and not ticket_state["escalation_attempted"]:
            ticket_state["was_escalated_correctly"] = True

        ticket_state["properly_closed"] = (
            ticket_state["properly_closed"]
            or (
                ticket_state["was_closed"]
                and ticket_state["was_routed_correctly"]
                and (self.task_name == "easy" or ticket_state["keyword_score"] > 0.5)
                and (not ticket["requires_escalation"] or ticket_state["was_escalated_correctly"])
            )
        )
        ticket_state["resolved_before_sla"] = ticket_state["properly_closed"] and ticket_state["sla_hours_remaining"] > 0.0
        ticket_state["handled"] = True

    def _advance_ticket(self) -> None:
        if self.total_steps_taken >= self.total_max_steps:
            self.done = True
            return
        self.current_ticket_index += 1
        if self.current_ticket_index >= len(self.ticket_states):
            self.done = True

    @staticmethod
    def _clamp_reward(reward: float) -> float:
        return round(max(-1.0, min(1.0, reward)), 4)

async def reset_async(self, seed=None, initial_conditions=None):
    return self.reset(seed=seed, initial_conditions=initial_conditions)

async def step_async(self, action):
    return self.step(action)

async def state_async(self):
    return self.state()

def close(self):
    pass