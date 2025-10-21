"""
Unit tests for similarity-weighted prior computation.

Tests temperature weighting, alpha annealing, and calibration utilities.
"""

import numpy as np
import pytest

from sparq_agent.base_agent_iface import Candidate, RetrievalResult
from sparq_agent.prior import SimWeightedPrior, CalibrationUtility, compute_concentration


class TestSimWeightedPrior:
    """Test cases for similarity-weighted prior module."""
    
    def test_prior_basic(self):
        """Test basic prior computation."""
        prior_module = SimWeightedPrior(outcome_type="return")
        
        candidate = Candidate(action="test_action")
        neighbors = RetrievalResult(
            neighbor_ids=["n1", "n2", "n3"],
            similarities=np.array([0.9, 0.7, 0.5]),
            returns=np.array([10.0, 5.0, 2.0]),
        )
        
        params = {"temperature": 10.0}
        v0, conf = prior_module.compute(candidate, neighbors, params)
        
        # Higher similarity neighbors should have more weight
        assert v0 > 5.0  # Should be biased toward n1
        assert "mean_similarity" in conf
        assert conf["neighbor_count"] == 3
    
    def test_empty_neighbors(self):
        """Test prior with no neighbors."""
        prior_module = SimWeightedPrior()
        
        candidate = Candidate(action="test_action")
        neighbors = RetrievalResult(
            neighbor_ids=[],
            similarities=np.array([]),
            returns=np.array([]),
        )
        
        params = {"temperature": 10.0}
        v0, conf = prior_module.compute(candidate, neighbors, params)
        
        assert v0 == 0.0
        assert conf["neighbor_count"] == 0
    
    def test_temperature_effect(self):
        """Test that higher temperature increases weight concentration."""
        prior_module = SimWeightedPrior(outcome_type="return")
        
        candidate = Candidate(action="test_action")
        neighbors = RetrievalResult(
            neighbor_ids=["n1", "n2"],
            similarities=np.array([0.9, 0.5]),
            returns=np.array([10.0, 2.0]),
        )
        
        # Low temperature
        v0_low, _ = prior_module.compute(candidate, neighbors, {"temperature": 1.0})
        
        # High temperature
        v0_high, _ = prior_module.compute(candidate, neighbors, {"temperature": 20.0})
        
        # High temperature should concentrate more weight on n1
        assert v0_high > v0_low
    
    def test_alpha_annealing(self):
        """Test alpha annealing based on similarity confidence."""
        prior_module = SimWeightedPrior()
        
        # High mean similarity → high alpha
        conf_high = {"mean_similarity": 0.9}
        alpha_high = prior_module.compute_alpha(conf_high, {"gamma": 8.0, "s0": 0.3})
        
        # Low mean similarity → low alpha
        conf_low = {"mean_similarity": 0.2}
        alpha_low = prior_module.compute_alpha(conf_low, {"gamma": 8.0, "s0": 0.3})
        
        assert alpha_high > alpha_low
        assert 0.0 <= alpha_high <= 1.0
        assert 0.0 <= alpha_low <= 1.0
    
    def test_fixed_alpha(self):
        """Test fixed alpha override."""
        prior_module = SimWeightedPrior()
        
        conf = {"mean_similarity": 0.9}
        alpha = prior_module.compute_alpha(conf, {"fixed_alpha": 0.5})
        
        assert alpha == 0.5
    
    def test_calibration(self):
        """Test outcome calibration."""
        prior_module = SimWeightedPrior(
            outcome_type="return",
            calibration_scale=0.1,
            calibration_shift=5.0,
        )
        
        candidate = Candidate(action="test_action")
        neighbors = RetrievalResult(
            neighbor_ids=["n1"],
            similarities=np.array([1.0]),
            returns=np.array([100.0]),
        )
        
        params = {"temperature": 10.0}
        v0, _ = prior_module.compute(candidate, neighbors, params)
        
        # Should be 100 * 0.1 + 5.0 = 15.0
        assert abs(v0 - 15.0) < 0.01


class TestCalibrationUtility:
    """Test cases for calibration utilities."""
    
    def test_estimate_scale_shift(self):
        """Test scale/shift estimation."""
        outcomes = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        
        scale, shift = CalibrationUtility.estimate_scale_shift(
            outcomes, target_mean=0.0, target_std=1.0
        )
        
        # Apply calibration
        calibrated = CalibrationUtility.apply_calibration(outcomes, scale, shift)
        
        # Check standardization
        assert abs(np.mean(calibrated)) < 0.01
        assert abs(np.std(calibrated) - 1.0) < 0.01
    
    def test_quantile_calibration(self):
        """Test quantile-based calibration."""
        outcomes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier
        
        stats = CalibrationUtility.fit_quantile_calibration(outcomes)
        
        assert "median" in stats
        assert "iqr" in stats
        assert stats["median"] == pytest.approx(3.5)


class TestConcentration:
    """Test cases for concentration metric."""
    
    def test_uniform_weights(self):
        """Test concentration with uniform weights."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        concentration = compute_concentration(weights)
        
        # Uniform = low concentration
        assert concentration < 0.1
    
    def test_concentrated_weights(self):
        """Test concentration with concentrated weights."""
        weights = np.array([0.97, 0.01, 0.01, 0.01])
        concentration = compute_concentration(weights)
        
        # Concentrated = high concentration
        assert concentration > 0.85
    
    def test_single_weight(self):
        """Test concentration with single weight."""
        weights = np.array([1.0])
        concentration = compute_concentration(weights)
        
        assert concentration == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
