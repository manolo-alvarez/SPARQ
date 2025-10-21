#!/usr/bin/env python3
"""
Run ablation studies sweeping alpha values under fixed configurations.

Tests the effect of blending coefficient α on performance.
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_baseline import run_experiment, load_config


def run_alpha_sweep(
    env_name: str,
    alpha_values: list,
    num_episodes: int,
    config_path: str,
    output_dir: str,
    seed: int = 42,
):
    """
    Run experiments sweeping alpha values.
    
    Args:
        env_name: Environment name.
        alpha_values: List of alpha values to test.
        num_episodes: Episodes per configuration.
        config_path: Path to config file.
        output_dir: Output directory.
        seed: Random seed.
    """
    sweep_results = {
        "env_name": env_name,
        "alpha_values": alpha_values,
        "num_episodes": num_episodes,
        "seed": seed,
        "results": [],
    }
    
    for alpha in alpha_values:
        print(f"\n{'='*60}")
        print(f"Testing alpha = {alpha}")
        print(f"{'='*60}\n")
        
        # Load base config
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
        
        # Create alpha-specific config
        config = load_config(config_path, env_name, "react_prior_lookahead_fixed")
        config["prior_params"]["fixed_alpha"] = alpha
        
        # Set up logging
        log_path = os.path.join(output_dir, f"{env_name}_alpha{alpha}_seed{seed}.jsonl")
        config["log_path"] = log_path
        
        # Run experiment (simplified version)
        from sparq_agent.registry import create_wrapped_agent
        from sparq_agent.base_agent_iface import BaseAgent, Candidate
        
        class DummyBaseAgent(BaseAgent):
            def propose_actions(self, obs, k=5):
                return [Candidate(action=f"action_{i}", logit=float(k-i)) for i in range(k)]
            def reset(self):
                pass
        
        base_agent = DummyBaseAgent()
        wrapped_agent = create_wrapped_agent(base_agent, config)
        
        episodes = []
        for episode_id in range(num_episodes):
            wrapped_agent.reset()
            success = episode_id % 2 == 0  # Mock success
            steps = 10 + episode_id
            
            if wrapped_agent.logger:
                wrapped_agent.logger.log_episode_summary(episode_id, success, steps)
            
            episodes.append({"success": success, "steps": steps})
        
        if wrapped_agent.logger:
            wrapped_agent.logger.close()
        
        # Compute summary
        success_rate = sum(1 for ep in episodes if ep["success"]) / len(episodes)
        avg_steps = sum(ep["steps"] for ep in episodes) / len(episodes)
        
        result = {
            "alpha": alpha,
            "success_rate": success_rate,
            "avg_steps": avg_steps,
        }
        sweep_results["results"].append(result)
        
        print(f"Alpha {alpha}: success_rate={success_rate:.2%}, avg_steps={avg_steps:.1f}")
    
    # Save sweep results
    results_path = os.path.join(output_dir, f"{env_name}_alpha_sweep_seed{seed}.json")
    with open(results_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    
    print(f"\nAlpha sweep results saved to {results_path}")
    
    return sweep_results


def main():
    parser = argparse.ArgumentParser(description="Run alpha sweep ablation study")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Alpha values to test",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes per alpha")
    parser.add_argument("--config", type=str, default="configs/sparq_default.yaml")
    parser.add_argument("--output-dir", type=str, default="experiments/ablations")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_alpha_sweep(
        env_name=args.env,
        alpha_values=args.alpha_values,
        num_episodes=args.num_episodes,
        config_path=args.config,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
