"""
Abstract interfaces for SPARQ Agent components.

This module defines the contracts between the policy wrapper and its modules,
enabling decoupling, testing, and extensibility across AgentGym environments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Candidate:
    """A proposed next action with optional policy logits and rationale."""
    
    action: str
    """The action text compatible with the environment."""
    
    logit: Optional[float] = None
    """Optional policy logit or log-probability."""
    
    rationale: Optional[str] = None
    """Optional ReAct-style reasoning trace."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional information for debugging or logging."""


@dataclass
class RetrievalResult:
    """Results from a kNN retrieval over trajectory memories."""
    
    neighbor_ids: List[str]
    """Identifiers of retrieved neighbors."""
    
    similarities: np.ndarray
    """Cosine similarities for each neighbor (length k)."""
    
    returns: Optional[np.ndarray] = None
    """Episodic returns or cumulative rewards (length k)."""
    
    success_logits: Optional[np.ndarray] = None
    """Success probability logits (length k)."""
    
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    """Per-neighbor metadata (e.g., env_id, stage, episode_id)."""
    
    query_embedding: Optional[np.ndarray] = None
    """The embedding vector used for this query."""


class BaseAgent(ABC):
    """
    Interface for any ReAct-style agent that can propose candidate actions.
    
    This remains the default fallback policy when memory or simulation is disabled.
    The wrapper delegates to this agent for action proposals and can fall back to
    its top candidate on error paths.
    """
    
    @abstractmethod
    def propose_actions(self, obs: Dict[str, Any], k: int = 5) -> List[Candidate]:
        """
        Generate up to k candidate next actions for the given observation.
        
        Args:
            obs: Environment observation dictionary (varies by environment).
            k: Number of candidates to generate.
            
        Returns:
            List of Candidate objects with action text, optional logits, and rationale.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset agent state for a new episode."""
        pass


class RetrievalModule(ABC):
    """
    Interface for trajectory memory retrieval with vector similarity search.
    
    Maintains a vector store over compressed trajectory summaries, partitioned
    by environment and stage, supporting fast cosine kNN queries.
    """
    
    @abstractmethod
    def query(
        self,
        context: Dict[str, Any],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve k nearest neighbors from memory.
        
        Args:
            context: Contains env_id, instruction text, observation tokens, and candidate action.
            k: Number of neighbors to retrieve.
            filters: Optional filters (e.g., stage tags, env_id).
            
        Returns:
            RetrievalResult with neighbor IDs, similarities, outcomes, and metadata.
        """
        pass
    
    @abstractmethod
    def add(
        self,
        trajectory_id: str,
        embedding: np.ndarray,
        return_value: float,
        success_logit: float,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add a trajectory summary to the memory store.
        
        Args:
            trajectory_id: Unique identifier for this trajectory.
            embedding: Summary embedding vector (normalized).
            return_value: Episodic return or cumulative reward.
            success_logit: Success probability logit.
            metadata: Additional information (env_id, stage, instruction_hash, etc.).
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the vector store to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load a persisted vector store from disk."""
        pass


class PriorModule(ABC):
    """
    Interface for computing similarity-weighted priors over candidate actions.
    
    Produces V₀(n) via temperature-weighted averaging over neighbor returns or
    success logits, with optional α annealing based on similarity confidence.
    """
    
    @abstractmethod
    def compute(
        self,
        candidate: Candidate,
        neighbors: RetrievalResult,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute the prior value V₀ for a candidate action.
        
        Args:
            candidate: The candidate action to score.
            neighbors: Retrieved neighbors with similarities and outcomes.
            params: Parameters including temperature τ, outcome type, calibration.
            
        Returns:
            Tuple of (V₀ value, confidence features dict).
            Confidence features may include mean similarity, entropy, concentration.
        """
        pass
    
    @abstractmethod
    def compute_alpha(
        self,
        confidence_features: Dict[str, Any],
        params: Dict[str, Any],
    ) -> float:
        """
        Compute the blend coefficient α via annealing.
        
        α = σ(γ(s̄ - s₀)) where s̄ is mean similarity, γ is temperature, s₀ is threshold.
        
        Args:
            confidence_features: Dictionary from compute() containing similarity stats.
            params: Parameters including gamma, s0 threshold.
            
        Returns:
            α ∈ [0, 1] for blending prior and lookahead values.
        """
        pass


class LookaheadModule(ABC):
    """
    Interface for shallow simulation scoring of candidate actions.
    
    Runs L short internal continuations of depth d for each candidate, scores
    via a deterministic rubric, and aggregates to produce V̂(n).
    """
    
    @abstractmethod
    def score(
        self,
        candidate: Candidate,
        obs: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score a candidate via shallow simulations.
        
        Args:
            candidate: The candidate action to simulate.
            obs: Current environment observation.
            params: Parameters including L (rollouts), d (depth), budget caps, rubric weights.
            
        Returns:
            Tuple of (V̂ value, diagnostics dict with rollout details).
        """
        pass


class ElasticKModule(ABC):
    """
    Interface for adaptive k selection based on uncertainty signals.
    
    Maps uncertainty (policy entropy, simulation variance, or similarity concentration)
    to retrieval breadth via k = clip(k_min + λ·u, k_min, k_max).
    """
    
    @abstractmethod
    def select_k(
        self,
        uncertainty: float,
        params: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select retrieval breadth k adaptively.
        
        Args:
            uncertainty: Scalar uncertainty signal (e.g., entropy, variance).
            params: Parameters including k_min, k_max, lambda scaling factor.
            
        Returns:
            Tuple of (selected k, diagnostics dict).
        """
        pass
    
    @abstractmethod
    def compute_uncertainty(
        self,
        candidates: List[Candidate],
        lookahead_scores: Optional[List[float]] = None,
        neighbor_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute uncertainty signal from available information.
        
        Can use policy entropy over candidate logits, variance of lookahead scores,
        or entropy of normalized neighbor weights.
        
        Args:
            candidates: List of proposed candidates with optional logits.
            lookahead_scores: Optional scores from shallow simulations.
            neighbor_weights: Optional normalized neighbor similarity weights.
            
        Returns:
            Scalar uncertainty ∈ [0, ∞).
        """
        pass


class PolicyWrapper(ABC):
    """
    Interface for the orchestrating policy wrapper.
    
    Integrates BaseAgent, RetrievalModule, PriorModule, LookaheadModule, and
    ElasticKModule to select the final action via score blending and argmax.
    """
    
    @abstractmethod
    def step(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Select the next action given an observation.
        
        Orchestrates: propose → retrieve → prior → lookahead → blend → select.
        Falls back to base agent's top candidate on any failure path.
        
        Args:
            obs: Environment observation dictionary.
            
        Returns:
            Tuple of (selected action, diagnostics dict for logging).
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset wrapper state for a new episode."""
        pass


# Extensibility hooks for future development

class SuccessorGuidedPriorModule(PriorModule):
    """
    Future extension: successor-guided prior using SR-style representations.
    
    Currently unimplemented; placeholder for upgrading to richer memory models.
    """
    
    def compute_successor_guided(
        self,
        candidate: Candidate,
        neighbors: RetrievalResult,
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute successor-guided prior (not yet implemented)."""
        raise NotImplementedError("Successor-guided prior not yet implemented")


class UCTSelector(ABC):
    """
    Future extension: UCT-style action selection with visit counts and exploration bonus.
    
    Currently a no-op; placeholder for converting the wrapper into a thin UCT planner.
    """
    
    @abstractmethod
    def select_with_exploration(
        self,
        candidates: List[Candidate],
        values: List[float],
        visit_counts: List[int],
        params: Dict[str, Any],
    ) -> int:
        """Select candidate index with UCT exploration bonus (not yet implemented)."""
        pass
