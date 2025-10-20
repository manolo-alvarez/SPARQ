"""
Unit tests for lookahead module.

Tests rubric scoring and shallow simulation aggregation.
"""

import pytest

from sgm_agent.base_agent_iface import Candidate
from sgm_agent.lookahead import (
    ShallowSimulator,
    DefaultRubric,
    simple_continuation_fn,
)


class TestDefaultRubric:
    """Test cases for default scoring rubric."""
    
    def test_success_keywords(self):
        """Test scoring with success keywords."""
        rubric = DefaultRubric()
        
        continuations = ["Task completed successfully", "Goal achieved"]
        obs = {}
        params = {"success_weight": 1.0, "failure_weight": -1.0, "length_weight": 0.0}
        
        score, details = rubric.score(continuations, obs, params)
        
        # Should detect "completed" and "goal"
        assert score > 0
        assert details["success_count"] > 0
    
    def test_failure_keywords(self):
        """Test scoring with failure keywords."""
        rubric = DefaultRubric()
        
        continuations = ["Error occurred", "Task failed"]
        obs = {}
        params = {"success_weight": 1.0, "failure_weight": -1.0, "length_weight": 0.0}
        
        score, details = rubric.score(continuations, obs, params)
        
        # Should detect "error" and "failed"
        assert score < 0
        assert details["failure_count"] > 0
    
    def test_length_weight(self):
        """Test trajectory length weighting."""
        rubric = DefaultRubric()
        
        short_cont = ["Step 1"]
        long_cont = ["Step 1", "Step 2", "Step 3", "Step 4"]
        obs = {}
        params = {"success_weight": 0.0, "failure_weight": 0.0, "length_weight": 1.0}
        
        score_short, _ = rubric.score(short_cont, obs, params)
        score_long, _ = rubric.score(long_cont, obs, params)
        
        assert score_long > score_short


class TestShallowSimulator:
    """Test cases for shallow simulator."""
    
    def test_basic_simulation(self):
        """Test basic simulation execution."""
        def mock_continuation(action, obs, depth):
            return [f"Step {i}" for i in range(depth)]
        
        simulator = ShallowSimulator(continuation_fn=mock_continuation)
        
        candidate = Candidate(action="test_action")
        obs = {}
        params = {
            "num_rollouts": 3,
            "depth": 2,
            "max_time_seconds": 10.0,
            "max_tokens": 10000,
        }
        
        score, diagnostics = simulator.score(candidate, obs, params)
        
        assert diagnostics["num_rollouts_completed"] == 3
        assert len(diagnostics["rollout_scores"]) == 3
    
    def test_budget_cap_time(self):
        """Test time budget cap."""
        import time
        
        def slow_continuation(action, obs, depth):
            time.sleep(0.5)
            return [f"Step {i}" for i in range(depth)]
        
        simulator = ShallowSimulator(continuation_fn=slow_continuation)
        
        candidate = Candidate(action="test_action")
        obs = {}
        params = {
            "num_rollouts": 10,
            "depth": 2,
            "max_time_seconds": 1.0,  # Should stop after ~2 rollouts
            "max_tokens": 10000,
        }
        
        score, diagnostics = simulator.score(candidate, obs, params)
        
        # Should not complete all 10 rollouts
        assert diagnostics["num_rollouts_completed"] < 10
    
    def test_aggregation_mean(self):
        """Test mean aggregation."""
        def mock_continuation(action, obs, depth):
            return ["success"]
        
        simulator = ShallowSimulator(continuation_fn=mock_continuation)
        
        candidate = Candidate(action="test_action")
        obs = {}
        params = {
            "num_rollouts": 4,
            "depth": 1,
            "max_time_seconds": 10.0,
            "max_tokens": 10000,
            "aggregation": "mean",
        }
        
        score, diagnostics = simulator.score(candidate, obs, params)
        
        # All rollouts should score the same, mean should equal individual score
        scores = diagnostics["rollout_scores"]
        assert abs(score - sum(scores) / len(scores)) < 0.01
    
    def test_error_handling(self):
        """Test simulator handles continuation errors gracefully."""
        def failing_continuation(action, obs, depth):
            raise ValueError("Mock failure")
        
        simulator = ShallowSimulator(continuation_fn=failing_continuation)
        
        candidate = Candidate(action="test_action")
        obs = {}
        params = {
            "num_rollouts": 3,
            "depth": 2,
            "max_time_seconds": 10.0,
            "max_tokens": 10000,
        }
        
        score, diagnostics = simulator.score(candidate, obs, params)
        
        # Should return 0 score when all rollouts fail
        assert diagnostics["num_rollouts_completed"] == 0
        assert score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
