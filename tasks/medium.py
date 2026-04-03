from __future__ import annotations

from support_env import SupportEnv


def create_env(seed=None):
    return SupportEnv("medium", seed=seed or 42)
