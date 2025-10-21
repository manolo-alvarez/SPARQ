"""
Policy wrapper integrating retrieval, prior, lookahead, and elastic-k modules.

Orchestrates: propose → retrieve → prior → lookahead → blend → select.
Falls back to base agent's top candidate on failure paths.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..base_agent_iface import (
    BaseAgent,
    Candidate,
    ElasticKModule,
    LookaheadModule,
    PolicyWrapper as PolicyWrapperInterface,
    PriorModule,
    RetrievalModule,
)


class SPARQPolicyWrapper(PolicyWrapperInterface):
    """
    Orchestrating policy wrapper for the SGM agent.
    
    Integrates BaseAgent, RetrievalModule, PriorModule, LookaheadModule, and
    ElasticKModule to select actions via score blending:
    
    Score(n) = α·V₀(n) + (1-α)·V̂(n)
    
    Falls back to base agent on error paths or when memory is empty.
    """
    
    def __init__(
        self,
        base_agent: BaseAgent,
        retrieval_module: RetrievalModule,
        prior_module: PriorModule,
        lookahead_module: LookaheadModule,
        elastic_k_module: ElasticKModule,
        embedding_fn: callable,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
    ):
        """
        Initialize the policy wrapper.
        
        Args:
            base_agent: Agent for proposing candidate actions.
            retrieval_module: Module for memory retrieval.
            prior_module: Module for computing similarity-weighted priors.
            lookahead_module: Module for shallow simulations.
            elastic_k_module: Module for adaptive k selection.
            embedding_fn: Function to embed (instruction, obs, action) into vector.
            config: Configuration dictionary with parameters.
            logger: Optional logging module for telemetry.
        """
        self.base_agent = base_agent
        self.retrieval_module = retrieval_module
        self.prior_module = prior_module
        self.lookahead_module = lookahead_module
        self.elastic_k_module = elastic_k_module
        self.embedding_fn = embedding_fn
        self.config = config
        self.logger = logger
        
        # Episode state
        self.episode_step = 0
        self.env_id = config.get("env_id", "default")
    
    def step(self, obs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Select the next action given an observation.
        
        Orchestrates: propose → retrieve → prior → lookahead → blend → select.
        
        Args:
            obs: Environment observation dictionary.
            
        Returns:
            Tuple of (selected action, diagnostics dict for logging).
        """
        step_start_time = time.time()
        self.episode_step += 1
        
        # Feature flags from config
        prior_enabled = not self.config.get("prior_off", False)
        lookahead_enabled = not self.config.get("lookahead_off", False)
        elastic_enabled = not self.config.get("elastic_off", False)
        
        diagnostics = {
            "episode_step": self.episode_step,
            "env_id": self.env_id,
            "prior_enabled": prior_enabled,
            "lookahead_enabled": lookahead_enabled,
            "elastic_enabled": elastic_enabled,
        }
        
        try:
            # Step 1: Propose candidates from base agent
            num_candidates = self.config.get("num_candidates", 5)
            candidates = self.base_agent.propose_actions(obs, k=num_candidates)
            
            if len(candidates) == 0:
                raise ValueError("Base agent returned no candidates")
            
            diagnostics["num_candidates"] = len(candidates)
            diagnostics["candidates"] = [
                {"action": c.action, "logit": c.logit, "rationale": c.rationale}
                for c in candidates
            ]
            
            # Step 2: Compute uncertainty and select k (if elastic-k enabled)
            if elastic_enabled:
                uncertainty = self.elastic_k_module.compute_uncertainty(candidates)
                k, elastic_diag = self.elastic_k_module.select_k(
                    uncertainty,
                    self.config.get("elastic_k_params", {}),
                )
            else:
                k = self.config.get("fixed_k", 32)
                elastic_diag = {"k_selected": k, "fixed": True}
            
            diagnostics["elastic_k"] = elastic_diag
            
            # Step 3: Score candidates
            candidate_scores = []
            
            for candidate in candidates:
                score_info = self._score_candidate(
                    candidate,
                    obs,
                    k,
                    prior_enabled,
                    lookahead_enabled,
                )
                candidate_scores.append(score_info)
            
            diagnostics["candidate_scores"] = candidate_scores
            
            # Step 4: Select best candidate
            best_idx = int(np.argmax([s["final_score"] for s in candidate_scores]))
            best_candidate = candidates[best_idx]
            selected_action = best_candidate.action
            
            diagnostics["selected_candidate_idx"] = best_idx
            diagnostics["selected_action"] = selected_action
            diagnostics["selection_method"] = "sparq_blend"
            
        except Exception as e:
            # Fallback to base agent's top candidate
            diagnostics["error"] = str(e)
            diagnostics["fallback"] = True
            
            try:
                fallback_candidates = self.base_agent.propose_actions(obs, k=1)
                if len(fallback_candidates) > 0:
                    selected_action = fallback_candidates[0].action
                    diagnostics["selected_action"] = selected_action
                    diagnostics["selection_method"] = "base_agent_fallback"
                else:
                    selected_action = "FALLBACK_ACTION"
                    diagnostics["selection_method"] = "hardcoded_fallback"
            except Exception as fallback_error:
                diagnostics["fallback_error"] = str(fallback_error)
                selected_action = "FALLBACK_ACTION"
                diagnostics["selection_method"] = "hardcoded_fallback"
        
        diagnostics["step_time"] = time.time() - step_start_time
        
        # Log to telemetry
        if self.logger:
            self.logger.log_decision(diagnostics)
        
        return selected_action, diagnostics
    
    def _score_candidate(
        self,
        candidate: Candidate,
        obs: Dict[str, Any],
        k: int,
        prior_enabled: bool,
        lookahead_enabled: bool,
    ) -> Dict[str, Any]:
        """
        Score a single candidate via retrieval, prior, and lookahead.
        
        Args:
            candidate: Candidate to score.
            obs: Current observation.
            k: Retrieval breadth.
            prior_enabled: Whether to use prior.
            lookahead_enabled: Whether to use lookahead.
            
        Returns:
            Dictionary with V₀, V̂, α, and final score.
        """
        score_info = {
            "candidate_action": candidate.action,
            "v0": 0.0,
            "v_hat": 0.0,
            "alpha": 0.0,
            "final_score": 0.0,
        }
        
        # Retrieve neighbors
        try:
            context = self._build_retrieval_context(candidate, obs)
            neighbors = self.retrieval_module.query(
                context,
                k,
                filters=self.config.get("retrieval_filters", {}),
            )
            score_info["neighbor_count"] = len(neighbors.neighbor_ids)
            score_info["neighbors"] = [
                {
                    "id": nid,
                    "similarity": float(sim),
                    "return": float(ret) if neighbors.returns is not None else None,
                }
                for nid, sim, ret in zip(
                    neighbors.neighbor_ids,
                    neighbors.similarities,
                    neighbors.returns if neighbors.returns is not None else [None] * len(neighbors.neighbor_ids),
                )
            ]
        except Exception as e:
            score_info["retrieval_error"] = str(e)
            neighbors = None
        
        # Compute prior V₀
        if prior_enabled and neighbors and len(neighbors.neighbor_ids) > 0:
            try:
                v0, conf_features = self.prior_module.compute(
                    candidate,
                    neighbors,
                    self.config.get("prior_params", {}),
                )
                score_info["v0"] = v0
                score_info["confidence_features"] = conf_features
                
                # Compute alpha
                alpha = self.prior_module.compute_alpha(
                    conf_features,
                    self.config.get("prior_params", {}),
                )
                score_info["alpha"] = alpha
            except Exception as e:
                score_info["prior_error"] = str(e)
                v0 = 0.0
                alpha = 0.0
        else:
            v0 = 0.0
            alpha = 0.0
        
        # Compute lookahead V̂
        if lookahead_enabled:
            try:
                v_hat, lookahead_diag = self.lookahead_module.score(
                    candidate,
                    obs,
                    self.config.get("lookahead_params", {}),
                )
                score_info["v_hat"] = v_hat
                score_info["lookahead_diagnostics"] = lookahead_diag
            except Exception as e:
                score_info["lookahead_error"] = str(e)
                v_hat = 0.0
        else:
            v_hat = 0.0
        
        # Blend scores: Score(n) = α·V₀ + (1-α)·V̂
        if prior_enabled and lookahead_enabled and neighbors and len(neighbors.neighbor_ids) > 0:
            final_score = alpha * v0 + (1 - alpha) * v_hat
        elif prior_enabled and neighbors and len(neighbors.neighbor_ids) > 0:
            final_score = v0
        elif lookahead_enabled:
            final_score = v_hat
        else:
            # Fallback to candidate logit if available
            final_score = candidate.logit if candidate.logit is not None else 0.0
        
        score_info["final_score"] = final_score
        
        return score_info
    
    def _build_retrieval_context(
        self,
        candidate: Candidate,
        obs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build retrieval context with embedding vector.
        
        Args:
            candidate: Candidate action.
            obs: Current observation.
            
        Returns:
            Context dictionary with env_id, embedding, and metadata.
        """
        # Extract instruction and observation text
        instruction = obs.get("instruction", "")
        obs_text = obs.get("observation", str(obs))
        
        # Combine for embedding
        combined_text = f"Instruction: {instruction}\nObservation: {obs_text}\nAction: {candidate.action}"
        
        # Generate embedding
        embedding = self.embedding_fn(combined_text)
        
        context = {
            "env_id": self.env_id,
            "embedding": embedding,
            "instruction": instruction,
            "observation": obs_text,
            "candidate_action": candidate.action,
        }
        
        return context
    
    def reset(self) -> None:
        """Reset wrapper state for a new episode."""
        self.base_agent.reset()
        self.episode_step = 0
        
        # Reset elastic-k hysteresis if applicable
        if hasattr(self.elastic_k_module, "reset"):
            self.elastic_k_module.reset()


class SimpleEmbeddingFunction:
    """
    Simple embedding function for testing (uses random projection or pre-trained).
    
    In production, replace with a proper sentence-transformer or similar.
    """
    
    def __init__(self, embedding_dim: int = 768, method: str = "random"):
        self.embedding_dim = embedding_dim
        self.method = method
        
        if method == "random":
            # Random projection matrix for testing
            self.projection = np.random.randn(1000, embedding_dim).astype(np.float32)
            self.projection = self.projection / np.linalg.norm(self.projection, axis=1, keepdims=True)
    
    def __call__(self, text: str) -> np.ndarray:
        """Embed text into vector."""
        if self.method == "random":
            # Simple hash + projection
            words = text.lower().split()[:1000]
            word_vec = np.zeros(1000, dtype=np.float32)
            for i, word in enumerate(words):
                word_vec[i] = hash(word) % 1000 / 1000.0
            embedding = word_vec @ self.projection
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding
        else:
            raise NotImplementedError(f"Embedding method {self.method} not implemented")
