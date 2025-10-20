"""
Telemetry and logging system for SGM agent decisions.

Provides JSONL logging of per-decision diagnostics, episode aggregation,
and utilities for reproducibility and ablation analysis.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class TelemetryLogger:
    """
    JSONL logger for SGM agent decisions.
    
    Logs per-decision diagnostics including:
    - Candidates, neighbors, priors, simulations, scores
    - Timing, token counts, uncertainty signals
    - Episode and environment metadata
    """
    
    def __init__(
        self,
        log_path: str,
        log_level: str = "normal",
        buffer_size: int = 100,
    ):
        """
        Initialize telemetry logger.
        
        Args:
            log_path: Path to JSONL log file.
            log_level: "minimal", "normal", or "verbose".
            buffer_size: Number of records to buffer before flushing to disk.
        """
        self.log_path = Path(log_path)
        self.log_level = log_level
        self.buffer_size = buffer_size
        
        # Create log directory
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open log file in append mode
        self.log_file = open(self.log_path, "a")
        
        # Buffer for batched writes
        self.buffer: List[Dict[str, Any]] = []
        
        # Episode-level aggregation
        self.episode_data: Dict[str, Any] = {}
        self.reset_episode()
    
    def log_decision(self, diagnostics: Dict[str, Any]) -> None:
        """
        Log a single decision.
        
        Args:
            diagnostics: Diagnostics dictionary from PolicyWrapper.step().
        """
        # Add timestamp and convert numpy types
        record = {
            "timestamp": time.time(),
            **self._serialize(diagnostics),
        }
        
        # Filter by log level
        if self.log_level == "minimal":
            record = self._filter_minimal(record)
        elif self.log_level == "normal":
            record = self._filter_normal(record)
        # verbose: keep everything
        
        # Add to buffer
        self.buffer.append(record)
        
        # Update episode aggregation
        self._update_episode_stats(diagnostics)
        
        # Flush if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self.flush()
    
    def log_episode_summary(self, episode_id: int, success: bool, steps: int) -> None:
        """
        Log episode-level summary.
        
        Args:
            episode_id: Episode identifier.
            success: Whether episode succeeded.
            steps: Number of steps taken.
        """
        summary = {
            "event_type": "episode_summary",
            "timestamp": time.time(),
            "episode_id": episode_id,
            "success": success,
            "steps": steps,
            **self.episode_data,
        }
        
        self.buffer.append(self._serialize(summary))
        self.flush()
        self.reset_episode()
    
    def reset_episode(self) -> None:
        """Reset episode-level aggregation."""
        self.episode_data = {
            "total_decisions": 0,
            "mean_k": 0.0,
            "mean_alpha": 0.0,
            "mean_v0": 0.0,
            "mean_v_hat": 0.0,
            "total_retrieval_time": 0.0,
            "total_lookahead_time": 0.0,
            "fallback_count": 0,
            "prior_changed_action_count": 0,
        }
    
    def _update_episode_stats(self, diagnostics: Dict[str, Any]) -> None:
        """Update running episode statistics."""
        n = self.episode_data["total_decisions"]
        self.episode_data["total_decisions"] += 1
        
        # Running averages
        if "elastic_k" in diagnostics:
            k = diagnostics["elastic_k"].get("k_selected", 0)
            self.episode_data["mean_k"] = (
                (self.episode_data["mean_k"] * n + k) / (n + 1)
            )
        
        if "candidate_scores" in diagnostics and len(diagnostics["candidate_scores"]) > 0:
            alpha = diagnostics["candidate_scores"][0].get("alpha", 0.0)
            v0 = diagnostics["candidate_scores"][0].get("v0", 0.0)
            v_hat = diagnostics["candidate_scores"][0].get("v_hat", 0.0)
            
            self.episode_data["mean_alpha"] = (
                (self.episode_data["mean_alpha"] * n + alpha) / (n + 1)
            )
            self.episode_data["mean_v0"] = (
                (self.episode_data["mean_v0"] * n + v0) / (n + 1)
            )
            self.episode_data["mean_v_hat"] = (
                (self.episode_data["mean_v_hat"] * n + v_hat) / (n + 1)
            )
        
        if diagnostics.get("fallback", False):
            self.episode_data["fallback_count"] += 1
    
    def flush(self) -> None:
        """Flush buffer to disk."""
        for record in self.buffer:
            self.log_file.write(json.dumps(record) + "\n")
        self.log_file.flush()
        self.buffer.clear()
    
    def close(self) -> None:
        """Close log file."""
        self.flush()
        self.log_file.close()
    
    def _serialize(self, obj: Any) -> Any:
        """Convert numpy types and non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _filter_minimal(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to minimal fields for small log size."""
        return {
            "timestamp": record.get("timestamp"),
            "episode_step": record.get("episode_step"),
            "selected_action": record.get("selected_action"),
            "k": record.get("elastic_k", {}).get("k_selected"),
            "alpha": record.get("candidate_scores", [{}])[0].get("alpha"),
            "fallback": record.get("fallback", False),
        }
    
    def _filter_normal(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to normal fields (exclude verbose details)."""
        filtered = dict(record)
        
        # Remove verbose fields
        if "candidate_scores" in filtered:
            for score in filtered["candidate_scores"]:
                if "neighbors" in score:
                    score["neighbor_count"] = len(score["neighbors"])
                    del score["neighbors"]
                if "lookahead_diagnostics" in score:
                    diag = score["lookahead_diagnostics"]
                    score["lookahead_summary"] = {
                        "num_rollouts": diag.get("num_rollouts_completed"),
                        "variance": diag.get("rollout_variance"),
                    }
                    del score["lookahead_diagnostics"]
        
        return filtered


class AggregatedMetrics:
    """
    Utilities for aggregating metrics across episodes and environments.
    
    Provides summary statistics for performance analysis and ablation studies.
    """
    
    @staticmethod
    def load_log(log_path: str) -> List[Dict[str, Any]]:
        """Load JSONL log file into list of records."""
        records = []
        with open(log_path, "r") as f:
            for line in f:
                records.append(json.loads(line))
        return records
    
    @staticmethod
    def aggregate_by_episode(records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        """
        Aggregate records by episode.
        
        Args:
            records: List of log records.
            
        Returns:
            Dictionary mapping episode_id to list of decision records.
        """
        episodes = defaultdict(list)
        
        for record in records:
            if record.get("event_type") == "episode_summary":
                continue
            episode_id = record.get("episode_id", 0)
            episodes[episode_id].append(record)
        
        return dict(episodes)
    
    @staticmethod
    def compute_summary_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute summary statistics across all records.
        
        Args:
            records: List of log records (decisions only).
            
        Returns:
            Dictionary of summary statistics.
        """
        if len(records) == 0:
            return {}
        
        k_values = [
            r.get("elastic_k", {}).get("k_selected")
            for r in records
            if "elastic_k" in r
        ]
        
        alpha_values = []
        v0_values = []
        v_hat_values = []
        
        for r in records:
            if "candidate_scores" in r and len(r["candidate_scores"]) > 0:
                alpha_values.append(r["candidate_scores"][0].get("alpha", 0.0))
                v0_values.append(r["candidate_scores"][0].get("v0", 0.0))
                v_hat_values.append(r["candidate_scores"][0].get("v_hat", 0.0))
        
        fallback_count = sum(1 for r in records if r.get("fallback", False))
        
        stats = {
            "num_decisions": len(records),
            "mean_k": float(np.mean(k_values)) if k_values else 0.0,
            "std_k": float(np.std(k_values)) if k_values else 0.0,
            "mean_alpha": float(np.mean(alpha_values)) if alpha_values else 0.0,
            "std_alpha": float(np.std(alpha_values)) if alpha_values else 0.0,
            "mean_v0": float(np.mean(v0_values)) if v0_values else 0.0,
            "mean_v_hat": float(np.mean(v_hat_values)) if v_hat_values else 0.0,
            "fallback_rate": fallback_count / len(records) if records else 0.0,
        }
        
        return stats
    
    @staticmethod
    def export_summary_csv(log_path: str, output_path: str) -> None:
        """
        Export episode summaries to CSV for plotting.
        
        Args:
            log_path: Path to JSONL log file.
            output_path: Path to output CSV file.
        """
        import csv
        
        records = AggregatedMetrics.load_log(log_path)
        summaries = [r for r in records if r.get("event_type") == "episode_summary"]
        
        if len(summaries) == 0:
            print("No episode summaries found in log")
            return
        
        # Extract fields
        fieldnames = list(summaries[0].keys())
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        
        print(f"Exported {len(summaries)} episode summaries to {output_path}")


class DashboardPlotter:
    """
    Lightweight plotting utilities for diagnostic dashboards.
    
    Generates plots of score decompositions, retrieval diagnostics, and
    performance over episodes.
    """
    
    @staticmethod
    def plot_score_decomposition(log_path: str, output_path: str) -> None:
        """
        Plot V₀, V̂, and final scores over time.
        
        Args:
            log_path: Path to JSONL log file.
            output_path: Path to output plot image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        
        records = AggregatedMetrics.load_log(log_path)
        decisions = [r for r in records if r.get("event_type") != "episode_summary"]
        
        v0_values = []
        v_hat_values = []
        final_scores = []
        
        for r in decisions:
            if "candidate_scores" in r and len(r["candidate_scores"]) > 0:
                selected_idx = r.get("selected_candidate_idx", 0)
                score = r["candidate_scores"][selected_idx]
                v0_values.append(score.get("v0", 0.0))
                v_hat_values.append(score.get("v_hat", 0.0))
                final_scores.append(score.get("final_score", 0.0))
        
        plt.figure(figsize=(10, 6))
        plt.plot(v0_values, label="V₀ (Prior)", alpha=0.7)
        plt.plot(v_hat_values, label="V̂ (Lookahead)", alpha=0.7)
        plt.plot(final_scores, label="Final Score", alpha=0.9)
        plt.xlabel("Decision Step")
        plt.ylabel("Score")
        plt.title("Score Decomposition Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"Score decomposition plot saved to {output_path}")
    
    @staticmethod
    def plot_k_over_time(log_path: str, output_path: str) -> None:
        """
        Plot k selection over time.
        
        Args:
            log_path: Path to JSONL log file.
            output_path: Path to output plot image.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        
        records = AggregatedMetrics.load_log(log_path)
        decisions = [r for r in records if r.get("event_type") != "episode_summary"]
        
        k_values = [
            r.get("elastic_k", {}).get("k_selected", 0)
            for r in decisions
            if "elastic_k" in r
        ]
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, marker="o", linestyle="-", markersize=3)
        plt.xlabel("Decision Step")
        plt.ylabel("k (Retrieval Breadth)")
        plt.title("Elastic-k Selection Over Time")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"k selection plot saved to {output_path}")
