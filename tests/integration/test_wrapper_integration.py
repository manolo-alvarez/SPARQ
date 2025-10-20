"""
Integration tests for SGM policy wrapper.

Tests full propose→retrieve→prior→lookahead→select pipeline with mock components.
"""

import numpy as np
import pytest
import tempfile
import os

from sgm_agent.base_agent_iface import BaseAgent, Candidate
from sgm_agent.elastic_k import FixedKController
from sgm_agent.lookahead import ShallowSimulator, DefaultRubric
from sgm_agent.prior import SimWeightedPrior
from sgm_agent.retrieval import InMemoryRetrievalStore
from sgm_agent.wrapper import SGMPolicyWrapper, SimpleEmbeddingFunction
from sgm_agent.logging import TelemetryLogger


class MockBaseAgent(BaseAgent):
    """Mock base agent for testing."""
    
    def __init__(self, candidates_to_return=None):
        self.candidates_to_return = candidates_to_return or [
            Candidate(action="action1", logit=2.0, rationale="Mock reason 1"),
            Candidate(action="action2", logit=1.5, rationale="Mock reason 2"),
            Candidate(action="action3", logit=1.0, rationale="Mock reason 3"),
        ]
        self.reset_called = False
    
    def propose_actions(self, obs, k=5):
        return self.candidates_to_return[:k]
    
    def reset(self):
        self.reset_called = True


def mock_continuation_fn(action, obs, depth):
    """Mock continuation function."""
    return [f"Step {i} after {action}" for i in range(depth)]


class TestPolicyWrapperIntegration:
    """Integration tests for the full wrapper pipeline."""
    
    def test_full_pipeline(self):
        """Test full propose→retrieve→prior→lookahead→select pipeline."""
        # Create mock components
        base_agent = MockBaseAgent()
        retrieval_module = InMemoryRetrievalStore(embedding_dim=64)
        prior_module = SimWeightedPrior()
        lookahead_module = ShallowSimulator(
            continuation_fn=mock_continuation_fn,
            rubric=DefaultRubric(),
        )
        elastic_k_module = FixedKController(k_fixed=3)
        embedding_fn = SimpleEmbeddingFunction(embedding_dim=64, method="random")
        
        # Add some synthetic memories
        for i in range(5):
            embedding = np.random.randn(64).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            retrieval_module.add(
                trajectory_id=f"traj_{i}",
                embedding=embedding,
                return_value=float(i * 2),
                success_logit=float(i * 0.5),
                metadata={"env_id": "test_env", "episode": i},
            )
        
        config = {
            "env_id": "test_env",
            "num_candidates": 3,
            "prior_params": {"temperature": 10.0, "gamma": 8.0, "s0": 0.3},
            "lookahead_params": {
                "num_rollouts": 2,
                "depth": 2,
                "max_time_seconds": 10.0,
                "max_tokens": 5000,
            },
        }
        
        wrapper = SGMPolicyWrapper(
            base_agent=base_agent,
            retrieval_module=retrieval_module,
            prior_module=prior_module,
            lookahead_module=lookahead_module,
            elastic_k_module=elastic_k_module,
            embedding_fn=embedding_fn,
            config=config,
            logger=None,
        )
        
        # Execute a step
        obs = {"instruction": "Test task", "observation": "Test state"}
        action, diagnostics = wrapper.step(obs)
        
        # Verify outputs
        assert action in ["action1", "action2", "action3"]
        assert "candidate_scores" in diagnostics
        assert len(diagnostics["candidate_scores"]) == 3
        assert "selected_action" in diagnostics
        assert diagnostics["selected_action"] == action
    
    def test_fallback_on_error(self):
        """Test wrapper falls back to base agent on errors."""
        base_agent = MockBaseAgent()
        
        # Create a failing retrieval module
        class FailingRetrieval:
            def query(self, *args, **kwargs):
                raise RuntimeError("Mock retrieval failure")
        
        retrieval_module = FailingRetrieval()
        prior_module = SimWeightedPrior()
        lookahead_module = ShallowSimulator(
            continuation_fn=mock_continuation_fn,
        )
        elastic_k_module = FixedKController()
        embedding_fn = SimpleEmbeddingFunction(embedding_dim=64)
        
        config = {"env_id": "test_env"}
        
        wrapper = SGMPolicyWrapper(
            base_agent=base_agent,
            retrieval_module=retrieval_module,
            prior_module=prior_module,
            lookahead_module=lookahead_module,
            elastic_k_module=elastic_k_module,
            embedding_fn=embedding_fn,
            config=config,
            logger=None,
        )
        
        obs = {"instruction": "Test", "observation": "State"}
        action, diagnostics = wrapper.step(obs)
        
        # Should fall back
        assert diagnostics.get("fallback", False)
        assert action == "action1"  # Base agent's top candidate
    
    def test_empty_memory(self):
        """Test wrapper handles empty memory gracefully."""
        base_agent = MockBaseAgent()
        retrieval_module = InMemoryRetrievalStore(embedding_dim=64)
        prior_module = SimWeightedPrior()
        lookahead_module = ShallowSimulator(
            continuation_fn=mock_continuation_fn,
        )
        elastic_k_module = FixedKController()
        embedding_fn = SimpleEmbeddingFunction(embedding_dim=64)
        
        config = {
            "env_id": "test_env",
            "prior_params": {"temperature": 10.0},
            "lookahead_params": {"num_rollouts": 2, "depth": 2},
        }
        
        wrapper = SGMPolicyWrapper(
            base_agent=base_agent,
            retrieval_module=retrieval_module,
            prior_module=prior_module,
            lookahead_module=lookahead_module,
            elastic_k_module=elastic_k_module,
            embedding_fn=embedding_fn,
            config=config,
            logger=None,
        )
        
        obs = {"instruction": "Test", "observation": "State"}
        action, diagnostics = wrapper.step(obs)
        
        # Should still select an action
        assert action in ["action1", "action2", "action3"]
        # Prior should be 0.0 due to empty memory
        for score_info in diagnostics["candidate_scores"]:
            assert score_info["v0"] == 0.0
    
    def test_with_logging(self):
        """Test wrapper with telemetry logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.jsonl")
            logger = TelemetryLogger(log_path, log_level="verbose")
            
            base_agent = MockBaseAgent()
            retrieval_module = InMemoryRetrievalStore(embedding_dim=64)
            prior_module = SimWeightedPrior()
            lookahead_module = ShallowSimulator(
                continuation_fn=mock_continuation_fn,
            )
            elastic_k_module = FixedKController()
            embedding_fn = SimpleEmbeddingFunction(embedding_dim=64)
            
            config = {"env_id": "test_env"}
            
            wrapper = SGMPolicyWrapper(
                base_agent=base_agent,
                retrieval_module=retrieval_module,
                prior_module=prior_module,
                lookahead_module=lookahead_module,
                elastic_k_module=elastic_k_module,
                embedding_fn=embedding_fn,
                config=config,
                logger=logger,
            )
            
            obs = {"instruction": "Test", "observation": "State"}
            wrapper.step(obs)
            
            logger.close()
            
            # Verify log was written
            assert os.path.exists(log_path)
            with open(log_path, "r") as f:
                lines = f.readlines()
                assert len(lines) > 0
    
    def test_reset(self):
        """Test wrapper reset."""
        base_agent = MockBaseAgent()
        retrieval_module = InMemoryRetrievalStore(embedding_dim=64)
        prior_module = SimWeightedPrior()
        lookahead_module = ShallowSimulator(
            continuation_fn=mock_continuation_fn,
        )
        elastic_k_module = FixedKController()
        embedding_fn = SimpleEmbeddingFunction(embedding_dim=64)
        
        config = {"env_id": "test_env"}
        
        wrapper = SGMPolicyWrapper(
            base_agent=base_agent,
            retrieval_module=retrieval_module,
            prior_module=prior_module,
            lookahead_module=lookahead_module,
            elastic_k_module=elastic_k_module,
            embedding_fn=embedding_fn,
            config=config,
            logger=None,
        )
        
        # Take a step
        obs = {"instruction": "Test", "observation": "State"}
        wrapper.step(obs)
        assert wrapper.episode_step == 1
        
        # Reset
        wrapper.reset()
        assert wrapper.episode_step == 0
        assert base_agent.reset_called


class TestBudgetGuards:
    """Tests for budget validation."""
    
    def test_token_budget_respected(self):
        """Test that token budget is respected in lookahead."""
        import time
        
        token_count = [0]
        
        def counting_continuation(action, obs, depth):
            # Simulate ~500 tokens per continuation
            token_count[0] += 500 * depth
            return ["Step"] * depth
        
        lookahead_module = ShallowSimulator(
            continuation_fn=counting_continuation,
        )
        
        candidate = Candidate(action="test")
        obs = {}
        params = {
            "num_rollouts": 10,
            "depth": 2,
            "max_time_seconds": 100.0,
            "max_tokens": 2000,  # Should stop after ~2 rollouts
        }
        
        score, diagnostics = lookahead_module.score(candidate, obs, params)
        
        # Should not exceed token budget significantly
        assert diagnostics["estimated_tokens"] <= 2500  # Small overshoot acceptable
    
    def test_time_budget_respected(self):
        """Test that time budget is respected in lookahead."""
        import time
        
        def slow_continuation(action, obs, depth):
            time.sleep(0.3)
            return ["Step"]
        
        lookahead_module = ShallowSimulator(
            continuation_fn=slow_continuation,
        )
        
        candidate = Candidate(action="test")
        obs = {}
        params = {
            "num_rollouts": 10,
            "depth": 2,
            "max_time_seconds": 1.0,  # Should stop after 3-4 rollouts
            "max_tokens": 100000,
        }
        
        start = time.time()
        score, diagnostics = lookahead_module.score(candidate, obs, params)
        elapsed = time.time() - start
        
        # Should respect time budget (with small tolerance)
        assert elapsed < 1.5
        assert diagnostics["num_rollouts_completed"] < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
