"""toolfence: provenance-based guard against indirect prompt injection in tool-using agents."""

from .agent import Trace, deny_all, run_agent
from .ledger import Ledger
from .policy import Decision, Guard, ToolCall, ToolSpec
from .tools import TOOL_SPECS, Environment

__all__ = ["Decision", "Environment", "Guard", "Ledger", "TOOL_SPECS", "ToolCall", "ToolSpec",
           "Trace", "deny_all", "run_agent"]
__version__ = "0.1.0"
