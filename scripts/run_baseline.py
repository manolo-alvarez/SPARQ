#!/usr/bin/env python3
"""
Run baseline experiments: Base ReAct, ReAct+prior, ReAct+lookahead, Full SGM.

This script orchestrates ablation studies across different configurations
under matched token budgets.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sgm_agent.registry import create_wrapped_agent
from sgm_agent.logging import AggregatedMetrics


def load_config(config_path: str, env_id: str, baseline: str) -> Dict[str, Any]:
    """
    Load configuration for a specific environment and baseline.
    
    Args:
        config_path: Path to sgm_default.yaml.
        env_id: Environment identifier.
        baseline: Baseline name (base_react, react_prior, etc.).
        
    Returns:
        Merged configuration dictionary.
    """
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    
    # Start with global defaults
    config = dict(full_config.get("global", {}))
    
    # Merge baseline config
    baseline_config = full_config.get("baselines", {}).get(baseline, {})
    config.update(baseline_config)
    
    # Merge environment-specific config
    env_config = full_config.get("environments", {}).get(env_id, {})
    for key, value in env_config.items():
        if isinstance(value, dict):
            config.setdefault(key, {}).update(value)
        else:
            config[key] = value
    
    # Merge other sections
    for section in ["candidates", "elastic_k", "prior", "lookahead", "retrieval"]:
        section_config = full_config.get(section, {})
        config.setdefault(f"{section}_params", {}).update(section_config)
    
    config["env_id"] = env_id
    
    return config


def run_experiment(
    env_name: str,
    baseline: str,
    num_episodes: int,
    config_path: str,
    output_dir: str,
    seed: int = 42,
):
    """
    Run a single experiment configuration.
    
    Args:
        env_name: Environment name (e.g., "webshop", "alfworld").
        baseline: Baseline configuration name.
        num_episodes: Number of episodes to run.
        config_path: Path to config file.
        output_dir: Directory to save results.
        seed: Random seed for reproducibility.
    """
    import numpy as np
    import random
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Running: {env_name} / {baseline}")
    print(f"Episodes: {num_episodes}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    # Load config
    config = load_config(config_path, env_name, baseline)
    
    # Set up logging
    log_path = os.path.join(output_dir, f"{env_name}_{baseline}_seed{seed}.jsonl")
    config["log_path"] = log_path
    config["log_level"] = "normal"
    
    # Create mock base agent (replace with actual AgentGym agent)
    from sgm_agent.base_agent_iface import BaseAgent, Candidate
    
    class DummyBaseAgent(BaseAgent):
        def propose_actions(self, obs, k=5):
            # Placeholder: return dummy candidates
            return [
                Candidate(action=f"action_{i}", logit=float(k-i))
                for i in range(k)
            ]
        
        def reset(self):
            pass
    
    base_agent = DummyBaseAgent()
    
    # Create wrapped agent
    wrapped_agent = create_wrapped_agent(base_agent, config)
    
    # Run episodes
    results = {
        "env_name": env_name,
        "baseline": baseline,
        "num_episodes": num_episodes,
        "seed": seed,
        "episodes": [],
    }
    
    for episode_id in range(num_episodes):
        print(f"Episode {episode_id + 1}/{num_episodes}")
        
        wrapped_agent.reset()
        
        # Mock episode (replace with actual AgentGym episode runner)
        episode_steps = 0
        success = False
        
        for step in range(20):  # Max 20 steps per episode
            obs = {
                "instruction": f"Task for episode {episode_id}",
                "observation": f"Step {step} state",
            }
            
            action, diagnostics = wrapped_agent.step(obs)
            episode_steps += 1
            
            # Mock success condition
            if step >= 10:
                success = True
                break
        
        # Log episode summary
        if wrapped_agent.logger:
            wrapped_agent.logger.log_episode_summary(episode_id, success, episode_steps)
        
        results["episodes"].append({
            "episode_id": episode_id,
            "success": success,
            "steps": episode_steps,
        })
    
    # Close logger
    if wrapped_agent.logger:
        wrapped_agent.logger.close()
    
    # Compute summary statistics
    successes = sum(1 for ep in results["episodes"] if ep["success"])
    avg_steps = sum(ep["steps"] for ep in results["episodes"]) / len(results["episodes"])
    
    results["summary"] = {
        "success_rate": successes / num_episodes,
        "avg_steps": avg_steps,
    }
    
    # Save results
    results_path = os.path.join(output_dir, f"{env_name}_{baseline}_seed{seed}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Success rate: {results['summary']['success_rate']:.2%}")
    print(f"Avg steps: {results['summary']['avg_steps']:.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run SGM agent baseline experiments")
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment name (webshop, alfworld, sciworld, etc.)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="sgm_full",
        choices=[
            "base_react",
            "react_prior",
            "react_lookahead",
            "react_prior_lookahead_fixed",
            "sgm_full",
        ],
        help="Baseline configuration to run",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sgm_default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    run_experiment(
        env_name=args.env,
        baseline=args.baseline,
        num_episodes=args.num_episodes,
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
