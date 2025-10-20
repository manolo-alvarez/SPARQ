"""
SGM Agent: A policy wrapper that sits between AgentGym environments and base ReAct agents,
adding retrieval-augmented priors, shallow simulations, and elastic-k memory selection.
"""

__version__ = "0.1.0"

from .base_agent_iface import (
    BaseAgent,
    Candidate,
    RetrievalModule,
    RetrievalResult,
    PriorModule,
    LookaheadModule,
    ElasticKModule,
    PolicyWrapper,
)
from .registry import register_sgm_agent, create_wrapped_agent

__all__ = [
    "BaseAgent",
    "Candidate",
    "RetrievalModule",
    "RetrievalResult",
    "PriorModule",
    "LookaheadModule",
    "ElasticKModule",
    "PolicyWrapper",
    "register_sgm_agent",
    "create_wrapped_agent",
]
