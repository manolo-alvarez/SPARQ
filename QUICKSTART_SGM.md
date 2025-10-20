# SGM Agent Quick Start Guide

## 1. Installation (2 minutes)

```bash
cd /Users/manoloalvarez/playground/SPARQ

# Install core dependencies
pip install -r requirements_sgm.txt

# Optional: Install FAISS for faster retrieval
pip install faiss-cpu

# Optional: Install visualization tools
pip install matplotlib pandas

# Install in development mode
pip install -e . -f setup_sgm.py
```

## 2. Run the Example (1 minute)

```bash
python examples/sgm_example.py
```

This demonstrates:
- Creating an SGM-wrapped agent
- Running episodes with retrieval, prior, and lookahead
- Logging decisions and analyzing results

## 3. Test the Implementation (2 minutes)

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration tests
pytest tests/integration/ -v

# Check coverage
pytest --cov=src/sgm_agent --cov-report=term-missing
```

## 4. Run Your First Experiment (5 minutes)

```bash
# Create output directories
mkdir -p experiments/results experiments/plots

# Run baseline experiment
python scripts/run_baseline.py \
    --env webshop \
    --baseline sgm_full \
    --num-episodes 20 \
    --output-dir experiments/results

# Compare with base ReAct
python scripts/run_baseline.py \
    --env webshop \
    --baseline base_react \
    --num-episodes 20 \
    --output-dir experiments/results

# Export metrics
python scripts/export_metrics.py \
    --results-dir experiments/results \
    --output-csv experiments/summary.csv \
    --plot-performance \
    --output-plot experiments/plots/performance.png
```

## 5. Integrate with Your AgentGym Environment

### Step 1: Wrap Your Agent

```python
from sgm_agent import create_wrapped_agent
from your_agentgym_code import YourReActAgent

# Your existing agent
base_agent = YourReActAgent()

# Load config
import yaml
with open("configs/sgm_default.yaml") as f:
    config = yaml.safe_load(f)

# Merge environment-specific settings
env_config = {
    **config["global"],
    **config["environments"]["webshop"],
    "env_id": "webshop",
    "log_path": "logs/webshop.jsonl",
}

# Create wrapped agent
sgm_agent = create_wrapped_agent(base_agent, env_config)
```

### Step 2: Run Episodes

```python
# Standard AgentGym episode loop
for episode in range(num_episodes):
    sgm_agent.reset()
    obs = env.reset()
    
    done = False
    while not done:
        # SGM handles retrieval, prior, lookahead internally
        action, diagnostics = sgm_agent.step(obs)
        
        # Execute in environment
        obs, reward, done, info = env.step(action)
    
    # Log episode summary
    if sgm_agent.logger:
        sgm_agent.logger.log_episode_summary(
            episode, 
            info["success"], 
            info["steps"]
        )
```

### Step 3: Analyze Results

```python
from sgm_agent.logging import AggregatedMetrics

# Load logs
records = AggregatedMetrics.load_log("logs/webshop.jsonl")
decisions = [r for r in records if r.get("event_type") != "episode_summary"]

# Compute stats
stats = AggregatedMetrics.compute_summary_stats(decisions)
print(f"Mean k: {stats['mean_k']:.1f}")
print(f"Mean α: {stats['mean_alpha']:.3f}")
print(f"Fallback rate: {stats['fallback_rate']:.1%}")
```

## 6. Customize for Your Environment

Edit `configs/sgm_default.yaml` to add your environment:

```yaml
environments:
  my_custom_env:
    prior:
      temperature: 12.0  # Adjust similarity weighting
      gamma: 6.0         # Adjust α annealing
    
    lookahead:
      num_rollouts: 6    # More rollouts for complex tasks
      depth: 3           # Deeper simulations
      rubric_type: custom  # Your custom rubric
    
    elastic_k:
      k_min: 10
      k_max: 60
      lambda_scale: 14.0
```

## 7. Common Customizations

### Custom Continuation Function

```python
from sgm_agent.lookahead import ShallowSimulator

def my_continuation_fn(action, obs, depth):
    """Use your LLM or dynamics model."""
    continuations = []
    for step in range(depth):
        # Call your model
        next_obs = my_model.predict(action, obs)
        continuations.append(next_obs)
        obs = next_obs
    return continuations

# Pass to wrapper
lookahead_module = ShallowSimulator(
    continuation_fn=my_continuation_fn,
    rubric=your_rubric
)
```

### Custom Scoring Rubric

```python
from sgm_agent.lookahead import ScoringRubric

class MyRubric(ScoringRubric):
    def score(self, continuations, obs, params):
        # Your domain-specific scoring
        score = 0.0
        for cont in continuations:
            if self.check_subgoal_satisfied(cont):
                score += 1.0
        
        return score, {"details": "..."}
```

### Custom Embedding Function

```python
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def __call__(self, text):
        return self.model.encode(text, normalize_embeddings=True)

# Use in wrapper
embedding_fn = SentenceTransformerEmbedding()
```

## 8. Troubleshooting

**Slow performance?**
- Reduce `num_rollouts` or `depth` in lookahead config
- Use FAISS instead of in-memory retrieval
- Implement faster continuation_fn

**Poor results?**
- Check retrieval is working: verify `neighbor_count > 0` in logs
- Tune `temperature` for your similarity distribution
- Adjust `gamma` and `s0` for α annealing
- Try baseline ablations to isolate issues

**Memory issues?**
- Reduce `k_max` to limit retrieval size
- Periodically prune vector store
- Use FAISS with quantization

## 9. Next Steps

1. **Populate memory**: Run base agent to collect trajectories, then add to retrieval store
2. **Tune hyperparameters**: Use alpha sweeps and elastic-k ablations
3. **Profile performance**: Check timing breakdowns in diagnostics
4. **Scale to multi-environment**: Use per-environment configs

## 10. Getting Help

- Check `README_SGM.md` for detailed documentation
- Review test files for usage examples
- See `examples/sgm_example.py` for complete workflow
- Open GitHub issues for bugs or questions

---

**You're ready to go!** Start with the example, then integrate with your AgentGym setup.
