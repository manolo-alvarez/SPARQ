"""
Elastic-k controllers for adaptive retrieval breadth based on uncertainty.

Maps uncertainty signals (policy entropy, simulation variance, similarity concentration)
to retrieval breadth k via k = clip(k_min + λ·u, k_min, k_max).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..base_agent_iface import Candidate, ElasticKModule


class EntropyElasticK(ElasticKModule):
    """
    Elastic-k controller using policy entropy over candidate logits.
    
    High entropy → uncertain policy → increase k to gather more context.
    Low entropy → confident policy → decrease k to save compute.
    """
    
    def __init__(self, k_min: int = 8, k_max: int = 64, lambda_scale: float = 16.0):
        self.k_min = k_min
        self.k_max = k_max
        self.lambda_scale = lambda_scale
    
    def select_k(
        self,
        uncertainty: float,
        params: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select retrieval breadth k adaptively based on entropy.
        
        Args:
            uncertainty: Entropy of candidate distribution.
            params: Optional overrides for k_min, k_max, lambda.
            
        Returns:
            Tuple of (selected k, diagnostics dict).
        """
        k_min = params.get("k_min", self.k_min)
        k_max = params.get("k_max", self.k_max)
        lambda_scale = params.get("lambda_scale", self.lambda_scale)
        
        # k = k_min + λ·u
        k_raw = k_min + lambda_scale * uncertainty
        k = int(np.clip(k_raw, k_min, k_max))
        
        diagnostics = {
            "uncertainty": uncertainty,
            "k_raw": k_raw,
            "k_selected": k,
            "k_min": k_min,
            "k_max": k_max,
        }
        
        return k, diagnostics
    
    def compute_uncertainty(
        self,
        candidates: List[Candidate],
        lookahead_scores: Optional[List[float]] = None,
        neighbor_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute policy entropy from candidate logits.
        
        Args:
            candidates: List of candidates with optional logits.
            lookahead_scores: Ignored.
            neighbor_weights: Ignored.
            
        Returns:
            Entropy in nats (natural logarithm).
        """
        logits = [c.logit for c in candidates if c.logit is not None]
        
        if len(logits) == 0:
            # No logits available; return default uncertainty
            return 1.0
        
        # Convert logits to probabilities
        logits_array = np.array(logits)
        logits_array = logits_array - np.max(logits_array)  # Numerical stability
        probs = np.exp(logits_array) / np.sum(np.exp(logits_array))
        
        # Compute entropy: H = -Σ p log p
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        
        return float(entropy)


class VarianceElasticK(ElasticKModule):
    """
    Elastic-k controller using variance of lookahead scores.
    
    High variance → uncertain simulation outcomes → increase k.
    Low variance → consistent outcomes → decrease k.
    """
    
    def __init__(self, k_min: int = 8, k_max: int = 64, lambda_scale: float = 16.0):
        self.k_min = k_min
        self.k_max = k_max
        self.lambda_scale = lambda_scale
    
    def select_k(
        self,
        uncertainty: float,
        params: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Select k based on simulation variance."""
        k_min = params.get("k_min", self.k_min)
        k_max = params.get("k_max", self.k_max)
        lambda_scale = params.get("lambda_scale", self.lambda_scale)
        
        k_raw = k_min + lambda_scale * uncertainty
        k = int(np.clip(k_raw, k_min, k_max))
        
        diagnostics = {
            "uncertainty": uncertainty,
            "k_raw": k_raw,
            "k_selected": k,
        }
        
        return k, diagnostics
    
    def compute_uncertainty(
        self,
        candidates: List[Candidate],
        lookahead_scores: Optional[List[float]] = None,
        neighbor_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute normalized variance of lookahead scores.
        
        Args:
            candidates: Ignored.
            lookahead_scores: Scores from shallow simulations.
            neighbor_weights: Ignored.
            
        Returns:
            Normalized variance (standard deviation).
        """
        if lookahead_scores is None or len(lookahead_scores) < 2:
            return 1.0
        
        variance = float(np.var(lookahead_scores))
        std = float(np.sqrt(variance))
        
        # Normalize by mean to get coefficient of variation
        mean_abs = np.mean(np.abs(lookahead_scores))
        if mean_abs > 1e-8:
            normalized_std = std / mean_abs
        else:
            normalized_std = std
        
        return normalized_std


class SimilarityConcentrationElasticK(ElasticKModule):
    """
    Elastic-k controller using concentration of neighbor similarity weights.
    
    High concentration → few relevant neighbors → decrease k (focused retrieval).
    Low concentration → many similar neighbors → increase k (broad search).
    """
    
    def __init__(self, k_min: int = 8, k_max: int = 64, lambda_scale: float = 16.0):
        self.k_min = k_min
        self.k_max = k_max
        self.lambda_scale = lambda_scale
    
    def select_k(
        self,
        uncertainty: float,
        params: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Select k based on inverse concentration (dispersion)."""
        k_min = params.get("k_min", self.k_min)
        k_max = params.get("k_max", self.k_max)
        lambda_scale = params.get("lambda_scale", self.lambda_scale)
        
        k_raw = k_min + lambda_scale * uncertainty
        k = int(np.clip(k_raw, k_min, k_max))
        
        diagnostics = {
            "uncertainty": uncertainty,
            "k_raw": k_raw,
            "k_selected": k,
        }
        
        return k, diagnostics
    
    def compute_uncertainty(
        self,
        candidates: List[Candidate],
        lookahead_scores: Optional[List[float]] = None,
        neighbor_weights: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute dispersion (inverse concentration) of neighbor weights.
        
        Args:
            candidates: Ignored.
            lookahead_scores: Ignored.
            neighbor_weights: Normalized similarity weights.
            
        Returns:
            Dispersion ∈ [0, 1], where 1 is maximally dispersed.
        """
        if neighbor_weights is None or len(neighbor_weights) < 2:
            return 1.0
        
        # Compute entropy
        entropy = -np.sum(neighbor_weights * np.log(neighbor_weights + 1e-8))
        max_entropy = np.log(len(neighbor_weights))
        
        # Dispersion = entropy / max_entropy
        if max_entropy > 1e-8:
            dispersion = entropy / max_entropy
        else:
            dispersion = 0.0
        
        return float(dispersion)


class HysteresisElasticK(ElasticKModule):
    """
    Elastic-k controller with hysteresis to prevent oscillations.
    
    Wraps another ElasticKModule and smooths k changes over time.
    """
    
    def __init__(
        self,
        base_controller: ElasticKModule,
        smoothing: float = 0.5,
    ):
        """
        Initialize hysteresis controller.
        
        Args:
            base_controller: Underlying ElasticKModule.
            smoothing: Smoothing factor ∈ [0, 1]. Higher = more smoothing.
        """
        self.base_controller = base_controller
        self.smoothing = smoothing
        self.prev_k: Optional[int] = None
    
    def select_k(
        self,
        uncertainty: float,
        params: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Select k with hysteresis."""
        k_raw, diagnostics = self.base_controller.select_k(uncertainty, params)
        
        if self.prev_k is None:
            k_smooth = k_raw
        else:
            # Exponential smoothing: k = α·k_prev + (1-α)·k_raw
            k_smooth = int(
                self.smoothing * self.prev_k + (1 - self.smoothing) * k_raw
            )
        
        self.prev_k = k_smooth
        diagnostics["k_smooth"] = k_smooth
        
        return k_smooth, diagnostics
    
    def compute_uncertainty(
        self,
        candidates: List[Candidate],
        lookahead_scores: Optional[List[float]] = None,
        neighbor_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Delegate to base controller."""
        return self.base_controller.compute_uncertainty(
            candidates, lookahead_scores, neighbor_weights
        )
    
    def reset(self) -> None:
        """Reset hysteresis state for a new episode."""
        self.prev_k = None


class FixedKController(ElasticKModule):
    """
    Fixed-k controller for ablation studies (no adaptation).
    """
    
    def __init__(self, k_fixed: int = 32):
        self.k_fixed = k_fixed
    
    def select_k(
        self,
        uncertainty: float,
        params: Dict[str, Any],
    ) -> Tuple[int, Dict[str, Any]]:
        """Always return fixed k."""
        k = params.get("k_fixed", self.k_fixed)
        diagnostics = {"k_selected": k, "fixed": True}
        return k, diagnostics
    
    def compute_uncertainty(
        self,
        candidates: List[Candidate],
        lookahead_scores: Optional[List[float]] = None,
        neighbor_weights: Optional[np.ndarray] = None,
    ) -> float:
        """Return dummy uncertainty."""
        return 0.0
