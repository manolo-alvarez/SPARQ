"""
Unit tests for elastic-k controllers.

Tests entropy, variance, and similarity-based uncertainty mappings.
"""

import numpy as np
import pytest

from sgm_agent.base_agent_iface import Candidate
from sgm_agent.elastic_k import (
    EntropyElasticK,
    VarianceElasticK,
    SimilarityConcentrationElasticK,
    HysteresisElasticK,
    FixedKController,
)


class TestEntropyElasticK:
    """Test cases for entropy-based elastic-k controller."""
    
    def test_uncertainty_high_entropy(self):
        """Test high entropy → high uncertainty."""
        controller = EntropyElasticK()
        
        # Uniform logits → high entropy
        candidates = [
            Candidate(action=f"a{i}", logit=0.0) for i in range(5)
        ]
        
        uncertainty = controller.compute_uncertainty(candidates)
        assert uncertainty > 1.0  # Should be near log(5) ≈ 1.6
    
    def test_uncertainty_low_entropy(self):
        """Test low entropy → low uncertainty."""
        controller = EntropyElasticK()
        
        # Peaked logits → low entropy
        candidates = [
            Candidate(action="a0", logit=10.0),
            Candidate(action="a1", logit=0.0),
            Candidate(action="a2", logit=0.0),
        ]
        
        uncertainty = controller.compute_uncertainty(candidates)
        assert uncertainty < 0.5
    
    def test_select_k(self):
        """Test k selection from uncertainty."""
        controller = EntropyElasticK(k_min=8, k_max=64, lambda_scale=16.0)
        
        # Low uncertainty → small k
        k_low, diag_low = controller.select_k(0.0, {})
        assert k_low == 8
        
        # High uncertainty → large k
        k_high, diag_high = controller.select_k(3.0, {})
        assert k_high > k_low
        assert k_high <= 64
    
    def test_k_clipping(self):
        """Test k is clipped to [k_min, k_max]."""
        controller = EntropyElasticK(k_min=10, k_max=50)
        
        # Very high uncertainty
        k, _ = controller.select_k(100.0, {})
        assert k == 50
        
        # Negative uncertainty
        k, _ = controller.select_k(-10.0, {})
        assert k == 10


class TestVarianceElasticK:
    """Test cases for variance-based elastic-k controller."""
    
    def test_uncertainty_high_variance(self):
        """Test high variance → high uncertainty."""
        controller = VarianceElasticK()
        
        lookahead_scores = [1.0, 5.0, 10.0, 2.0, 8.0]
        uncertainty = controller.compute_uncertainty(
            [], lookahead_scores=lookahead_scores
        )
        
        assert uncertainty > 0.5
    
    def test_uncertainty_low_variance(self):
        """Test low variance → low uncertainty."""
        controller = VarianceElasticK()
        
        lookahead_scores = [5.0, 5.1, 5.0, 4.9, 5.0]
        uncertainty = controller.compute_uncertainty(
            [], lookahead_scores=lookahead_scores
        )
        
        assert uncertainty < 0.1


class TestSimilarityConcentrationElasticK:
    """Test cases for similarity-based elastic-k controller."""
    
    def test_uncertainty_dispersed(self):
        """Test dispersed weights → high uncertainty."""
        controller = SimilarityConcentrationElasticK()
        
        # Uniform weights
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        uncertainty = controller.compute_uncertainty(
            [], neighbor_weights=weights
        )
        
        # High dispersion = high uncertainty
        assert uncertainty > 0.9
    
    def test_uncertainty_concentrated(self):
        """Test concentrated weights → low uncertainty."""
        controller = SimilarityConcentrationElasticK()
        
        # Concentrated weights
        weights = np.array([0.9, 0.05, 0.03, 0.02])
        uncertainty = controller.compute_uncertainty(
            [], neighbor_weights=weights
        )
        
        # Low dispersion = low uncertainty
        assert uncertainty < 0.5


class TestHysteresisElasticK:
    """Test cases for hysteresis wrapper."""
    
    def test_smoothing(self):
        """Test k smoothing over time."""
        base = EntropyElasticK(k_min=10, k_max=50, lambda_scale=10.0)
        controller = HysteresisElasticK(base, smoothing=0.5)
        
        # First call: no history
        k1, _ = controller.select_k(1.0, {})
        
        # Second call: should smooth
        k2, _ = controller.select_k(0.0, {})
        
        # k2 should be between k1 and new raw value
        assert k2 < k1  # Moving toward low uncertainty k
        assert k2 > 10  # But not fully there due to smoothing
    
    def test_reset(self):
        """Test reset clears history."""
        base = EntropyElasticK()
        controller = HysteresisElasticK(base, smoothing=0.8)
        
        # Build history
        controller.select_k(2.0, {})
        
        # Reset
        controller.reset()
        
        # Next selection should not use history
        assert controller.prev_k is None


class TestFixedKController:
    """Test cases for fixed-k controller."""
    
    def test_always_fixed(self):
        """Test k is always fixed."""
        controller = FixedKController(k_fixed=32)
        
        k1, _ = controller.select_k(0.0, {})
        k2, _ = controller.select_k(100.0, {})
        
        assert k1 == 32
        assert k2 == 32
    
    def test_override(self):
        """Test config override."""
        controller = FixedKController(k_fixed=32)
        
        k, _ = controller.select_k(0.0, {"k_fixed": 16})
        assert k == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
