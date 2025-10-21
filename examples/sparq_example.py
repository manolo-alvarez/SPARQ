"""
Example usage of SPARQ Agent with a simple mock environment.

Demonstrates the full workflow: agent creation, episode execution, logging, and analysis.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import tempfile
import os

# Import SPARQ Agent components
from sparq_agent.base_agent_iface import BaseAgent, Candidate
from sparq_agent.registry import create_default_sparq_agent
from sparq_agent.logging import AggregatedMetrics


class SimpleReActAgent(BaseAgent):
    """
    A simple ReAct-style agent that proposes candidate actions.
    
    In production, this would be your actual AgentGym agent with LLM calls.
    """
    
    def __init__(self):
        self.step_count = 0
    
    def propose_actions(self, obs, k=5):
        """Generate k candidate actions based on observation."""
        # Mock candidates (in production, these come from LLM reasoning)
        candidates = [
            Candidate(
                action=f"search('{obs.get('query', 'item')}')",
                logit=2.0,
                rationale="Need to find relevant information"
            ),
            Candidate(
                action=f"click('{obs.get('target', 'button')}')",
                logit=1.5,
                rationale="This looks like the right element"
            ),
            Candidate(
                action="think('What should I do next?')",
                logit=1.0,
                rationale="Need more context"
            ),
            Candidate(
                action="scroll(down)",
                logit=0.8,
                rationale="Explore more options"
            ),
            Candidate(
                action="back()",
                logit=0.5,
                rationale="Might need to return"
            ),
        ]
        return candidates[:k]
    
    def reset(self):
        """Reset agent state for new episode."""
        self.step_count = 0


class MockEnvironment:
    """Mock environment for demonstration."""
    
    def __init__(self, max_steps=15):
        self.max_steps = max_steps
        self.step_count = 0
        self.goal = "find item and complete purchase"
    
    def reset(self):
        """Reset environment."""
        self.step_count = 0
        return {
            "instruction": self.goal,
            "observation": "You are on the homepage. There is a search bar and product categories.",
        }
    
    def step(self, action):
        """Execute action and return next observation."""
        self.step_count += 1
        
        # Mock reward based on action quality
        if "search" in action:
            reward = 1.0
            obs = "Search results: Found several matching items."
        elif "click" in action:
            reward = 2.0
            obs = "Clicked on item. Now viewing product page."
        elif "think" in action:
            reward = 0.0
            obs = "Still on the same page."
        else:
            reward = 0.5
            obs = "Action executed. Page updated."
        
        # Success condition
        done = self.step_count >= self.max_steps or (
            "click" in action and self.step_count >= 5
        )
        success = done and "click" in action
        
        return {
            "instruction": self.goal,
            "observation": obs,
        }, reward, done, {"success": success}


def run_example():
    """Run a complete example with SPARQ Agent."""
    print("="*60)
    print("SPARQ Agent Example")
    print("="*60)
    
    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "sgm_example.jsonl")
        
        # Create base ReAct agent
        base_agent = SimpleReActAgent()
        
        # Wrap with SGM
        print("\nCreating SPARQ-wrapped agent...")
        sparq_agent = create_default_sparq_agent(
            base_agent,
            env_id="mock_shop",
            log_path=log_path,
        )
        
        # Create environment
        env = MockEnvironment()
        
        # Run multiple episodes
        num_episodes = 3
        results = []
        
        for episode_id in range(num_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode_id + 1}/{num_episodes}")
            print(f"{'='*60}")
            
            sparq_agent.reset()
            obs = env.reset()
            
            episode_reward = 0.0
            episode_steps = 0
            
            done = False
            while not done:
                # Agent selects action
                action, diagnostics = sparq_agent.step(obs)
                
                print(f"\nStep {episode_steps + 1}:")
                print(f"  Observation: {obs['observation'][:60]}...")
                print(f"  Selected action: {action}")
                
                # Show some diagnostics
                if "candidate_scores" in diagnostics:
                    selected_idx = diagnostics.get("selected_candidate_idx", 0)
                    score_info = diagnostics["candidate_scores"][selected_idx]
                    print(f"  Prior V₀: {score_info.get('v0', 0.0):.3f}")
                    print(f"  Lookahead V̂: {score_info.get('v_hat', 0.0):.3f}")
                    print(f"  Alpha α: {score_info.get('alpha', 0.0):.3f}")
                    print(f"  Final score: {score_info.get('final_score', 0.0):.3f}")
                
                # Execute in environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
            
            success = info.get("success", False)
            results.append({
                "episode": episode_id,
                "success": success,
                "steps": episode_steps,
                "reward": episode_reward,
            })
            
            # Log episode summary
            if sparq_agent.logger:
                sparq_agent.logger.log_episode_summary(episode_id, success, episode_steps)
            
            print(f"\nEpisode {episode_id + 1} complete:")
            print(f"  Success: {success}")
            print(f"  Steps: {episode_steps}")
            print(f"  Total reward: {episode_reward:.1f}")
        
        # Close logger
        if sparq_agent.logger:
            sparq_agent.logger.close()
        
        # Analyze results
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_steps = sum(r["steps"] for r in results) / len(results)
        avg_reward = sum(r["reward"] for r in results) / len(results)
        
        print(f"Success rate: {success_rate:.1%}")
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Average reward: {avg_reward:.1f}")
        
        # Show telemetry stats
        print(f"\n{'='*60}")
        print("Telemetry Analysis")
        print(f"{'='*60}")
        
        if os.path.exists(log_path):
            records = AggregatedMetrics.load_log(log_path)
            decisions = [r for r in records if r.get("event_type") != "episode_summary"]
            
            if len(decisions) > 0:
                stats = AggregatedMetrics.compute_summary_stats(decisions)
                print(f"Total decisions: {stats['num_decisions']}")
                print(f"Mean k (retrieval breadth): {stats['mean_k']:.1f}")
                print(f"Mean α (blend coefficient): {stats['mean_alpha']:.3f}")
                print(f"Fallback rate: {stats['fallback_rate']:.1%}")
        
        print("\n" + "="*60)
        print("Example complete!")
        print("="*60)


if __name__ == "__main__":
    run_example()
