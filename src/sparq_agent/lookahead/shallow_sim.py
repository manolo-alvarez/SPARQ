"""
Shallow simulation module for lookahead scoring of candidate actions.

Runs L short internal continuations of depth d for each candidate, scores via
a deterministic rubric, and aggregates to produce V̂(n).
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..base_agent_iface import Candidate, LookaheadModule


class ShallowSimulator(LookaheadModule):
    """
    Shallow simulation runner with deterministic scoring rubric and budget caps.
    
    Abstracts LLM continuation calls behind a strategy interface to support
    different simulation backends (model-based, learned dynamics, etc.).
    """
    
    def __init__(
        self,
        continuation_fn: Callable[[str, Dict[str, Any], int], List[str]],
        rubric: Optional['ScoringRubric'] = None,
    ):
        """
        Initialize the shallow simulator.
        
        Args:
            continuation_fn: Function that takes (action, obs, depth) and returns
                           a list of continuation strings (observations or trajectories).
            rubric: Scoring rubric instance. If None, uses DefaultRubric.
        """
        self.continuation_fn = continuation_fn
        self.rubric = rubric or DefaultRubric()
    
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
        num_rollouts = params.get("num_rollouts", 4)  # L
        depth = params.get("depth", 2)  # d
        max_time_seconds = params.get("max_time_seconds", 5.0)
        max_tokens = params.get("max_tokens", 2000)
        aggregation = params.get("aggregation", "mean")  # "mean" or "cvar"
        cvar_alpha = params.get("cvar_alpha", 0.25)
        
        start_time = time.time()
        rollout_scores = []
        rollout_details = []
        total_tokens = 0
        
        for rollout_idx in range(num_rollouts):
            # Check budget
            elapsed = time.time() - start_time
            if elapsed > max_time_seconds:
                break
            if total_tokens > max_tokens:
                break
            
            # Run continuation
            try:
                continuations = self.continuation_fn(
                    candidate.action,
                    obs,
                    depth,
                )
                
                # Score the rollout
                rollout_score, rubric_details = self.rubric.score(
                    continuations,
                    obs,
                    params,
                )
                
                rollout_scores.append(rollout_score)
                rollout_details.append({
                    "rollout_idx": rollout_idx,
                    "score": rollout_score,
                    "rubric_details": rubric_details,
                })
                
                # Estimate tokens (rough approximation)
                total_tokens += sum(len(c.split()) * 1.3 for c in continuations)
                
            except Exception as e:
                # Log and skip failed rollout
                rollout_details.append({
                    "rollout_idx": rollout_idx,
                    "error": str(e),
                })
        
        # Aggregate scores
        if len(rollout_scores) == 0:
            # No successful rollouts
            lookahead_value = 0.0
        elif aggregation == "mean":
            lookahead_value = float(np.mean(rollout_scores))
        elif aggregation == "cvar":
            # Conditional Value at Risk (lower α quantile)
            threshold = np.quantile(rollout_scores, cvar_alpha)
            cvar_scores = [s for s in rollout_scores if s <= threshold]
            lookahead_value = float(np.mean(cvar_scores)) if cvar_scores else float(np.min(rollout_scores))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        diagnostics = {
            "num_rollouts_completed": len(rollout_scores),
            "rollout_scores": rollout_scores,
            "rollout_variance": float(np.var(rollout_scores)) if len(rollout_scores) > 1 else 0.0,
            "elapsed_time": time.time() - start_time,
            "estimated_tokens": int(total_tokens),
            "rollout_details": rollout_details,
        }
        
        return lookahead_value, diagnostics


class ScoringRubric:
    """
    Abstract base class for scoring rubrics.
    
    A rubric takes a sequence of continuations (observations or trajectory snippets)
    and assigns a scalar score based on domain-specific heuristics.
    """
    
    def score(
        self,
        continuations: List[str],
        obs: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score a rollout based on continuations.
        
        Args:
            continuations: List of text continuations (length d).
            obs: Original observation for context.
            params: Rubric-specific parameters (e.g., weights, keywords).
            
        Returns:
            Tuple of (score, details dict).
        """
        raise NotImplementedError


class DefaultRubric(ScoringRubric):
    """
    Default scoring rubric using generic heuristics.
    
    Scores based on:
    - Presence of success/goal keywords
    - Absence of error/failure keywords
    - Trajectory length (longer may indicate progress)
    """
    
    def __init__(
        self,
        success_keywords: Optional[List[str]] = None,
        failure_keywords: Optional[List[str]] = None,
    ):
        self.success_keywords = success_keywords or [
            "success", "completed", "goal", "done", "solved", "correct"
        ]
        self.failure_keywords = failure_keywords or [
            "error", "failed", "impossible", "invalid", "incorrect", "timeout"
        ]
    
    def score(
        self,
        continuations: List[str],
        obs: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score based on keyword presence and trajectory length."""
        combined_text = " ".join(continuations).lower()
        
        # Count keyword matches
        success_count = sum(kw in combined_text for kw in self.success_keywords)
        failure_count = sum(kw in combined_text for kw in self.failure_keywords)
        
        # Weights
        success_weight = params.get("success_weight", 1.0)
        failure_weight = params.get("failure_weight", -1.0)
        length_weight = params.get("length_weight", 0.1)
        
        # Compute score
        score = (
            success_count * success_weight
            + failure_count * failure_weight
            + len(continuations) * length_weight
        )
        
        details = {
            "success_count": success_count,
            "failure_count": failure_count,
            "trajectory_length": len(continuations),
        }
        
        return float(score), details


class ToolPreconditionRubric(ScoringRubric):
    """
    Rubric for tool-use environments (e.g., ToolBench, WebArena).
    
    Scores based on tool precondition satisfaction and subgoal markers.
    """
    
    def __init__(self, precondition_checker: Optional[Callable] = None):
        self.precondition_checker = precondition_checker
    
    def score(
        self,
        continuations: List[str],
        obs: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score based on tool precondition satisfaction."""
        if self.precondition_checker is None:
            # Fallback to simple heuristic
            score = len(continuations)
            details = {"fallback": True}
        else:
            # Check preconditions for each step
            satisfied_count = 0
            for cont in continuations:
                if self.precondition_checker(cont, obs):
                    satisfied_count += 1
            
            score = float(satisfied_count)
            details = {"satisfied_preconditions": satisfied_count}
        
        return score, details


class SubgoalProgressRubric(ScoringRubric):
    """
    Rubric for environments with explicit subgoal hierarchies.
    
    Scores based on distance to subgoal completion or markers in the trajectory.
    """
    
    def __init__(self, subgoal_markers: Optional[List[str]] = None):
        self.subgoal_markers = subgoal_markers or []
    
    def score(
        self,
        continuations: List[str],
        obs: Dict[str, Any],
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score based on subgoal marker presence."""
        combined_text = " ".join(continuations).lower()
        
        marker_count = sum(marker.lower() in combined_text for marker in self.subgoal_markers)
        score = float(marker_count)
        
        details = {"subgoal_markers_found": marker_count}
        
        return score, details


def simple_continuation_fn(
    action: str,
    obs: Dict[str, Any],
    depth: int,
) -> List[str]:
    """
    Simple placeholder continuation function for testing.
    
    In production, this would call an LLM or learned dynamics model.
    
    Args:
        action: The action to simulate.
        obs: Current observation.
        depth: Number of steps to simulate.
        
    Returns:
        List of simulated observation strings.
    """
    # Placeholder: just return dummy continuations
    continuations = []
    for step in range(depth):
        continuations.append(f"[Simulated step {step+1} after action: {action}]")
    return continuations
