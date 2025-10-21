# SPARQ Agent: Similarity Prior with Adaptive Retrieval and Quick lookahead Agent for AgentGym

A drop-in policy wrapper that enhances ReAct agents with retrieval-augmented priors, shallow simulations, and adaptive memory retrieval across AgentGym's 14 environments.

## Overview

SPARQ Agent implements a sophisticated decision-making system that:

- **Retrieval Module**: Maintains a vector store of trajectory summaries with fast cosine kNN queries
- **Prior Module**: Computes similarity-weighted priors V₀(n) via temperature-scaled averaging
- **Lookahead Module**: Runs shallow simulations to estimate V̂(n) with deterministic scoring rubrics
- **Elastic-k Module**: Adapts retrieval breadth based on uncertainty signals (entropy, variance, similarity concentration)
- **Policy Wrapper**: Orchestrates modules and blends scores: Score(n) = α·V₀(n) + (1-α)·V̂(n)

## Architecture

```
BaseAgent (ReAct) → PolicyWrapper → Environment
                         ↓
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
    Retrieval       Prior          Lookahead
     Module        Module           Module
         ↓              ↓              ↓
    ElasticK    Alpha Annealing   Score Blend
```

## Installation

```bash
# Install core dependencies
pip install numpy pyyaml

# Optional: FAISS for fast vector search
pip install faiss-cpu  # or faiss-gpu

# Optional: For visualization
pip install matplotlib pandas

# Development dependencies
pip install pytest pytest-cov
```

## Quick Start

```python
from sparq_agent import create_default_sparq_agent
from sparq_agent.base_agent_iface import BaseAgent, Candidate

# Your existing ReAct agent
class MyReActAgent(BaseAgent):
    def propose_actions(self, obs, k=5):
        # Return candidate actions
        return [Candidate(action=f"action_{i}", logit=score) for i, score in ...]
    
    def reset(self):
        pass

# Wrap with SPARQ
base_agent = MyReActAgent()
sparq_agent = create_default_sparq_agent(
    base_agent,
    env_id="webshop",
    log_path="logs/webshop.jsonl"
)

# Use as drop-in replacement
for episode in range(100):
    sparq_agent.reset()
    obs = env.reset()
    
    while not done:
        action, diagnostics = sparq_agent.step(obs)
        obs, reward, done, info = env.step(action)
```

## Configuration

See `configs/sparq_default.yaml` for per-environment defaults and ablation configurations.

Key parameters:
- `k_min`, `k_max`: Retrieval breadth bounds
- `temperature`: Similarity weighting temperature (τ)
- `gamma`, `s0`: Alpha annealing parameters
- `num_rollouts`, `depth`: Lookahead simulation parameters

## Running Experiments

### Baselines

```bash
# Base ReAct (no SPARQ)
python scripts/run_baseline.py --env webshop --baseline base_react --num-episodes 50

# ReAct + Prior (fixed k)
python scripts/run_baseline.py --env webshop --baseline react_prior --num-episodes 50

# Full SPARQ (elastic k)
python scripts/run_baseline.py --env webshop --baseline sparq_full --num-episodes 50
```

### Ablations

```bash
# Alpha sweep
python scripts/run_ablations.py --env alfworld --alpha-values 0.0 0.2 0.4 0.6 0.8 1.0

# Export results
python scripts/export_metrics.py --results-dir experiments/results --output-csv experiments/summary.csv --plot-performance
```

## Testing

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Coverage
pytest --cov=src/sparq_agent --cov-report=html
```

## Directory Structure

```
.
├── src/sparq_agent/          # Core implementation
│   ├── base_agent_iface.py # Abstract interfaces
│   ├── retrieval/          # Vector store & kNN
│   ├── prior/              # Similarity-weighted priors
│   ├── lookahead/          # Shallow simulations
│   ├── elastic_k/          # Adaptive k controllers
│   ├── wrapper/            # Policy orchestration
│   ├── logging/            # Telemetry & diagnostics
│   └── registry.py         # AgentGym integration
├── configs/                # Per-environment configs
├── tests/                  # Unit & integration tests
├── scripts/                # Experiment runners
└── experiments/            # Results & plots
```

## Key Formulas

**Similarity Weights**:
$$w_i = \exp(\tau \cdot \cos(e(n), e_i))$$

**Prior**:
$$V_0(n) = \frac{\sum_{i=1}^{k} w_i r_i}{\sum_{i=1}^{k} w_i}$$

**Alpha Annealing**:
$$\alpha = \sigma(\gamma(\bar{s} - s_0))$$

**Final Score**:
$$\text{Score}(n) = \alpha \cdot V_0(n) + (1-\alpha) \cdot \hat{V}(n)$$

**Elastic-k**:
$$k = \text{clip}(k_{\min} + \lambda \cdot u, k_{\min}, k_{\max})$$

## Supported Environments

- WebShop, ALFWorld, ScienceWorld, WebArena
- BabyAI, TextCraft, SearchQA, SQLGym
- LMRLGym (Maze, Wordle)
- Academia, Movie, Sheet, Todo, Weather

Each environment has tuned defaults in `configs/sparq_default.yaml`.

## Logging & Diagnostics

All decisions are logged in JSONL format with:
- Candidates, neighbors, priors, simulations, final scores
- Timing, token counts, uncertainty signals
- Episode-level aggregations

View diagnostics:
```python
from sparq_agent.logging import AggregatedMetrics, DashboardPlotter

# Load logs
records = AggregatedMetrics.load_log("logs/webshop.jsonl")
stats = AggregatedMetrics.compute_summary_stats(records)

# Plot score decomposition
DashboardPlotter.plot_score_decomposition("logs/webshop.jsonl", "plots/scores.png")
```

## Extensibility

Future hooks for:
- **Successor-guided priors**: Upgrade to SR-style representations
- **UCT selector**: Convert to thin UCT planner with exploration bonus
- **Custom rubrics**: Environment-specific scoring functions

## Performance Budgets

Typical per-decision costs:
- Retrieval: ~10ms (FAISS), ~50ms (in-memory)
- Prior computation: <1ms
- Lookahead (L=4, d=2): ~1-5s depending on continuation_fn
- Total: ~1-5s per decision

Tunable via `max_time_seconds` and `max_tokens` in lookahead params.

## Citation

Based on methods from:
- ReAct: Yao et al., 2022
- UCT: Kocsis & Szepesvári, 2006
- AgentGym: AgentGym Team, 2024

## License

MIT License - see LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest`
5. Submit a pull request

## Troubleshooting

**Q: FAISS import errors?**
A: Install with `pip install faiss-cpu`. Falls back to in-memory store if unavailable.

**Q: Slow lookahead?**
A: Reduce `num_rollouts` or `depth` in config, or implement faster continuation_fn.

**Q: Memory growing over episodes?**
A: Vector stores are persistent. Use `max_k` or periodic pruning for long runs.

**Q: Poor performance on new environment?**
A: Tune `temperature`, `gamma`, `s0` in config. Start with prior-only baseline to validate retrieval.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
