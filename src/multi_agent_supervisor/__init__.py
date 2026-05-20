"""Multi-agent supervisor with parallel specialist dispatch on LangGraph."""

from multi_agent_supervisor.state import SupervisorState
from multi_agent_supervisor.supervisor import build_supervisor

__version__ = "0.1.0"
__all__ = ["SupervisorState", "build_supervisor"]
