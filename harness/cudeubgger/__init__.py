"""Cudeubgger agent framework.

Structured CUDA debugging agents used by the benchmark harness. The legacy
``harness.agent_loop`` module remains as a compatibility facade; new code should
import from this package.
"""

from .react import run_task_agent
from .blackboard import (
    run_task_blackboard_agent,
    run_task_multi_role_no_blackboard_agent,
)

__all__ = [
    "run_task_agent",
    "run_task_blackboard_agent",
    "run_task_multi_role_no_blackboard_agent",
]
