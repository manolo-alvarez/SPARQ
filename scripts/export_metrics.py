#!/usr/bin/env python3
"""
Export performance metrics and generate plots from experiment results.

Produces CSV summaries and plots for performance vs. cost analysis.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparq_agent.logging import AggregatedMetrics, DashboardPlotter


def export_summary_csv(results_dir: str, output_csv: str):
    """
    Export CSV summary from all result JSON files in a directory.
    
    Args:
        results_dir: Directory containing *_results.json files.
        output_csv: Path to output CSV file.
    """
    import csv
    
    results_files = list(Path(results_dir).glob("*_results.json"))
    
    if len(results_files) == 0:
        print(f"No results found in {results_dir}")
        return
    
    summaries = []
    
    for results_file in results_files:
        with open(results_file, "r") as f:
            data = json.load(f)
        
        summary = {
            "env_name": data.get("env_name"),
            "baseline": data.get("baseline"),
            "seed": data.get("seed"),
            "num_episodes": data.get("num_episodes"),
            "success_rate": data.get("summary", {}).get("success_rate"),
            "avg_steps": data.get("summary", {}).get("avg_steps"),
        }
        
        # Try to load telemetry for additional metrics
        log_file = results_file.with_suffix(".jsonl")
        if log_file.exists():
            try:
                records = AggregatedMetrics.load_log(str(log_file))
                decisions = [r for r in records if r.get("event_type") != "episode_summary"]
                stats = AggregatedMetrics.compute_summary_stats(decisions)
                summary.update({
                    "mean_k": stats.get("mean_k"),
                    "mean_alpha": stats.get("mean_alpha"),
                    "fallback_rate": stats.get("fallback_rate"),
                })
            except Exception as e:
                print(f"Warning: Could not load telemetry for {results_file}: {e}")
        
        summaries.append(summary)
    
    if len(summaries) == 0:
        print("No summaries to export")
        return
    
    # Write CSV
    fieldnames = list(summaries[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    
    print(f"Exported {len(summaries)} summaries to {output_csv}")


def plot_performance_vs_cost(results_csv: str, output_plot: str):
    """
    Plot success rate vs. average steps (proxy for cost).
    
    Args:
        results_csv: Path to CSV with summary results.
        output_plot: Path to output plot image.
    """
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("matplotlib or pandas not available; skipping plot")
        return
    
    df = pd.read_csv(results_csv)
    
    plt.figure(figsize=(10, 6))
    
    # Group by baseline
    for baseline in df["baseline"].unique():
        subset = df[df["baseline"] == baseline]
        plt.scatter(
            subset["avg_steps"],
            subset["success_rate"],
            label=baseline,
            s=100,
            alpha=0.7,
        )
    
    plt.xlabel("Average Steps (Cost)")
    plt.ylabel("Success Rate")
    plt.title("Performance vs. Cost Across Baselines")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Performance vs. cost plot saved to {output_plot}")


def plot_alpha_sweep(sweep_json: str, output_plot: str):
    """
    Plot success rate vs. alpha from sweep results.
    
    Args:
        sweep_json: Path to alpha sweep JSON results.
        output_plot: Path to output plot image.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return
    
    with open(sweep_json, "r") as f:
        data = json.load(f)
    
    alphas = [r["alpha"] for r in data["results"]]
    success_rates = [r["success_rate"] for r in data["results"]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, success_rates, marker="o", linewidth=2, markersize=8)
    plt.xlabel("α (Prior Weight)")
    plt.ylabel("Success Rate")
    plt.title(f"Alpha Sweep: {data['env_name']}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Alpha sweep plot saved to {output_plot}")


def main():
    parser = argparse.ArgumentParser(description="Export metrics and generate plots")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory with result JSON files",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="experiments/summary.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plot-performance",
        action="store_true",
        help="Generate performance vs. cost plot",
    )
    parser.add_argument(
        "--plot-alpha-sweep",
        type=str,
        help="Path to alpha sweep JSON for plotting",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="experiments/plot.png",
        help="Output plot path",
    )
    
    args = parser.parse_args()
    
    # Export CSV
    export_summary_csv(args.results_dir, args.output_csv)
    
    # Generate plots
    if args.plot_performance:
        plot_performance_vs_cost(args.output_csv, args.output_plot)
    
    if args.plot_alpha_sweep:
        plot_alpha_sweep(args.plot_alpha_sweep, args.output_plot)


if __name__ == "__main__":
    main()
