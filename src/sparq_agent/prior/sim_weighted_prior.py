"""
Similarity-weighted prior computation with temperature scaling and alpha annealing.

Implements V₀(n) via weighted averaging over neighbor returns or success logits,
with confidence-based α adjustment for score blending.
"""

from typing import Any, Dict, Tuple
import numpy as np

from ..base_agent_iface import Candidate, PriorModule, RetrievalResult


class SimWeightedPrior(PriorModule):
    """
    Compute similarity-weighted priors using temperature-scaled neighbor outcomes.
    
    V₀(n) = Σᵢ wᵢ·rᵢ / Σᵢ wᵢ where wᵢ = exp(τ·cos(e(n), eᵢ))
    
    Supports both returns and success logits, with optional calibration to
    normalize scales across different environment types.
    """
    
    def __init__(
        self,
        outcome_type: str = "return",
        calibration_scale: float = 1.0,
        calibration_shift: float = 0.0,
    ):
        """
        Initialize the prior module.
        
        Args:
            outcome_type: "return" or "success_logit" to choose neighbor signal.
            calibration_scale: Multiplicative scaling for outcomes.
            calibration_shift: Additive shift for outcomes (applied after scale).
        """
        self.outcome_type = outcome_type
        self.calibration_scale = calibration_scale
        self.calibration_shift = calibration_shift
    
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
            params: Parameters including temperature τ.
            
        Returns:
            Tuple of (V₀ value, confidence features dict).
        """
        if len(neighbors.neighbor_ids) == 0:
            # No neighbors; return neutral prior
            return 0.0, {
                "mean_similarity": 0.0,
                "similarity_entropy": 0.0,
                "neighbor_count": 0,
            }
        
        # Extract parameters
        temperature = params.get("temperature", 10.0)
        
        # Compute weights: wᵢ = exp(τ · sᵢ)
        similarities = neighbors.similarities
        weights = np.exp(temperature * similarities)
        weights = weights / (np.sum(weights) + 1e-8)  # Normalize
        
        # Select outcomes
        if self.outcome_type == "return":
            outcomes = neighbors.returns
            if outcomes is None:
                raise ValueError("neighbors.returns is None but outcome_type='return'")
        elif self.outcome_type == "success_logit":
            outcomes = neighbors.success_logits
            if outcomes is None:
                raise ValueError("neighbors.success_logits is None but outcome_type='success_logit'")
        else:
            raise ValueError(f"Unknown outcome_type: {self.outcome_type}")
        
        # Calibrate outcomes
        outcomes = outcomes * self.calibration_scale + self.calibration_shift
        
        # Compute weighted average
        prior_value = np.sum(weights * outcomes)
        
        # Compute confidence features
        mean_similarity = np.mean(similarities)
        similarity_entropy = -np.sum(weights * np.log(weights + 1e-8))
        
        confidence_features = {
            "mean_similarity": float(mean_similarity),
            "similarity_entropy": float(similarity_entropy),
            "neighbor_count": len(neighbors.neighbor_ids),
            "weights": weights.tolist(),
        }
        
        return float(prior_value), confidence_features
    
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
        mean_similarity = confidence_features.get("mean_similarity", 0.0)
        gamma = params.get("gamma", 8.0)
        s0 = params.get("s0", 0.3)
        
        # Fixed alpha override
        fixed_alpha = params.get("fixed_alpha")
        if fixed_alpha is not None:
            return float(fixed_alpha)
        
        # Sigmoid annealing: α = 1 / (1 + exp(-γ(s̄ - s₀)))
        logit = gamma * (mean_similarity - s0)
        alpha = 1.0 / (1.0 + np.exp(-logit))
        
        return float(alpha)


class CalibrationUtility:
    """
    Utilities for calibrating returns and success logits across environments.
    
    Environments have different reward scales and success definitions; calibration
    helps ensure priors are on a comparable scale for blending.
    """
    
    @staticmethod
    def estimate_scale_shift(
        outcomes: np.ndarray,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Estimate calibration parameters to standardize outcomes.
        
        Args:
            outcomes: Array of observed outcomes (returns or success logits).
            target_mean: Desired mean after calibration.
            target_std: Desired standard deviation after calibration.
            
        Returns:
            Tuple of (scale, shift) parameters.
        """
        obs_mean = np.mean(outcomes)
        obs_std = np.std(outcomes) + 1e-8
        
        scale = target_std / obs_std
        shift = target_mean - scale * obs_mean
        
        return float(scale), float(shift)
    
    @staticmethod
    def apply_calibration(
        outcomes: np.ndarray,
        scale: float,
        shift: float,
    ) -> np.ndarray:
        """Apply calibration transformation."""
        return outcomes * scale + shift
    
    @staticmethod
    def fit_quantile_calibration(
        outcomes: np.ndarray,
        quantiles: np.ndarray = np.array([0.1, 0.5, 0.9]),
    ) -> Dict[str, Any]:
        """
        Fit a quantile-based calibration for robust scaling.
        
        Useful when outcome distributions have heavy tails or outliers.
        
        Args:
            outcomes: Array of observed outcomes.
            quantiles: Quantile points to match (default: [0.1, 0.5, 0.9]).
            
        Returns:
            Dictionary with quantile statistics.
        """
        quantile_values = np.quantile(outcomes, quantiles)
        
        return {
            "quantiles": quantiles.tolist(),
            "values": quantile_values.tolist(),
            "median": float(np.median(outcomes)),
            "iqr": float(np.percentile(outcomes, 75) - np.percentile(outcomes, 25)),
        }


def compute_concentration(weights: np.ndarray) -> float:
    """
    Compute concentration metric for similarity weights.
    
    High concentration → weights focused on few neighbors → high confidence.
    Low concentration → weights spread across many → low confidence.
    
    Uses inverse entropy normalized to [0, 1].
    
    Args:
        weights: Normalized weight vector (sums to 1).
        
    Returns:
        Concentration ∈ [0, 1], where 1 is maximally concentrated.
    """
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    max_entropy = np.log(len(weights))
    
    if max_entropy < 1e-8:
        return 1.0
    
    concentration = 1.0 - entropy / max_entropy
    return float(concentration)
