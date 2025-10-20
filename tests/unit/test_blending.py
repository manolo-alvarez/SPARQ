"""
Unit tests for score blending.

Tests the Score(n) = α·V₀(n) + (1-α)·V̂(n) formula.
"""

import pytest


class TestScoreBlending:
    """Test cases for score blending arithmetic."""
    
    def test_pure_prior(self):
        """Test α=1.0 → pure prior."""
        v0 = 10.0
        v_hat = 5.0
        alpha = 1.0
        
        score = alpha * v0 + (1 - alpha) * v_hat
        
        assert score == 10.0
    
    def test_pure_lookahead(self):
        """Test α=0.0 → pure lookahead."""
        v0 = 10.0
        v_hat = 5.0
        alpha = 0.0
        
        score = alpha * v0 + (1 - alpha) * v_hat
        
        assert score == 5.0
    
    def test_balanced_blend(self):
        """Test α=0.5 → balanced blend."""
        v0 = 10.0
        v_hat = 4.0
        alpha = 0.5
        
        score = alpha * v0 + (1 - alpha) * v_hat
        
        assert score == 7.0
    
    def test_prior_dominance(self):
        """Test α=0.8 → prior dominates."""
        v0 = 10.0
        v_hat = 2.0
        alpha = 0.8
        
        score = alpha * v0 + (1 - alpha) * v_hat
        
        # 0.8 * 10 + 0.2 * 2 = 8.4
        assert abs(score - 8.4) < 0.01
    
    def test_negative_values(self):
        """Test blending with negative values."""
        v0 = -5.0
        v_hat = 3.0
        alpha = 0.6
        
        score = alpha * v0 + (1 - alpha) * v_hat
        
        # 0.6 * (-5) + 0.4 * 3 = -3 + 1.2 = -1.8
        assert abs(score - (-1.8)) < 0.01


class TestArgmaxSelection:
    """Test cases for argmax candidate selection."""
    
    def test_simple_argmax(self):
        """Test selecting highest-scoring candidate."""
        scores = [1.0, 5.0, 3.0, 2.0]
        
        import numpy as np
        best_idx = int(np.argmax(scores))
        
        assert best_idx == 1
        assert scores[best_idx] == 5.0
    
    def test_tie_breaking(self):
        """Test tie-breaking (first occurrence)."""
        scores = [3.0, 5.0, 5.0, 2.0]
        
        import numpy as np
        best_idx = int(np.argmax(scores))
        
        # numpy.argmax returns first occurrence
        assert best_idx == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
