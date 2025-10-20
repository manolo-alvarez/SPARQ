"""
Registry hooks to plug SGM agent into AgentGym runners.

Provides factory functions to create wrapped agents without modifying core AgentGym code.
"""

from typing import Any, Callable, Dict, Optional

from .base_agent_iface import BaseAgent
from .elastic_k import EntropyElasticK, FixedKController, HysteresisElasticK
from .lookahead import ShallowSimulator, DefaultRubric, simple_continuation_fn
from .prior import SimWeightedPrior
from .retrieval import FAISSRetrievalStore, InMemoryRetrievalStore
from .wrapper import SGMPolicyWrapper, SimpleEmbeddingFunction
from .logging import TelemetryLogger


def create_wrapped_agent(
    base_agent: BaseAgent,
    config: Dict[str, Any],
    continuation_fn: Optional[Callable] = None,
    embedding_fn: Optional[Callable] = None,
) -> SGMPolicyWrapper:
    """
    Create an SGM-wrapped agent from a base ReAct agent.
    
    Args:
        base_agent: Base agent for proposing candidate actions.
        config: Configuration dictionary with parameters.
        continuation_fn: Optional continuation function for shallow simulations.
        embedding_fn: Optional embedding function for retrieval.
        
    Returns:
        SGMPolicyWrapper instance ready to use.
    """
    # Set up retrieval module
    use_faiss = config.get("use_faiss", False)
    embedding_dim = config.get("embedding_dim", 768)
    
    if use_faiss:
        retrieval_module = FAISSRetrievalStore(
            embedding_dim=embedding_dim,
            use_gpu=config.get("use_gpu", False),
        )
    else:
        retrieval_module = InMemoryRetrievalStore(embedding_dim=embedding_dim)
    
    # Load existing memory if path provided
    memory_path = config.get("memory_path")
    if memory_path:
        try:
            retrieval_module.load(memory_path)
        except FileNotFoundError:
            pass  # No existing memory; start fresh
    
    # Set up prior module
    prior_module = SimWeightedPrior(
        outcome_type=config.get("outcome_type", "return"),
        calibration_scale=config.get("calibration_scale", 1.0),
        calibration_shift=config.get("calibration_shift", 0.0),
    )
    
    # Set up lookahead module
    if continuation_fn is None:
        continuation_fn = simple_continuation_fn
    
    rubric_type = config.get("rubric_type", "default")
    if rubric_type == "default":
        rubric = DefaultRubric()
    else:
        rubric = DefaultRubric()  # Extend for other rubric types
    
    lookahead_module = ShallowSimulator(
        continuation_fn=continuation_fn,
        rubric=rubric,
    )
    
    # Set up elastic-k module
    elastic_mode = config.get("elastic_mode", "entropy")
    k_min = config.get("k_min", 8)
    k_max = config.get("k_max", 64)
    lambda_scale = config.get("lambda_scale", 16.0)
    
    if config.get("elastic_off", False):
        elastic_k_module = FixedKController(k_fixed=config.get("fixed_k", 32))
    elif elastic_mode == "entropy":
        elastic_k_module = EntropyElasticK(k_min, k_max, lambda_scale)
    elif elastic_mode == "variance":
        from .elastic_k import VarianceElasticK
        elastic_k_module = VarianceElasticK(k_min, k_max, lambda_scale)
    elif elastic_mode == "similarity":
        from .elastic_k import SimilarityConcentrationElasticK
        elastic_k_module = SimilarityConcentrationElasticK(k_min, k_max, lambda_scale)
    else:
        raise ValueError(f"Unknown elastic_mode: {elastic_mode}")
    
    # Add hysteresis if requested
    if config.get("use_hysteresis", False):
        elastic_k_module = HysteresisElasticK(
            elastic_k_module,
            smoothing=config.get("hysteresis_smoothing", 0.5),
        )
    
    # Set up embedding function
    if embedding_fn is None:
        embedding_fn = SimpleEmbeddingFunction(
            embedding_dim=embedding_dim,
            method=config.get("embedding_method", "random"),
        )
    
    # Set up logging
    logger = None
    log_path = config.get("log_path")
    if log_path:
        logger = TelemetryLogger(
            log_path=log_path,
            log_level=config.get("log_level", "normal"),
            buffer_size=config.get("log_buffer_size", 100),
        )
    
    # Create wrapper
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
    
    return wrapper


def register_sgm_agent(
    agent_registry: Dict[str, Callable],
    agent_name: str = "sgm_react",
    config_path: Optional[str] = None,
) -> None:
    """
    Register SGM agent factory in an AgentGym-style registry.
    
    Args:
        agent_registry: AgentGym agent registry dictionary.
        agent_name: Name to register the agent under.
        config_path: Optional path to config file for defaults.
    """
    import yaml
    
    # Load default config if provided
    default_config = {}
    if config_path:
        with open(config_path, "r") as f:
            default_config = yaml.safe_load(f)
    
    def sgm_agent_factory(base_agent: BaseAgent, env_config: Dict[str, Any]) -> SGMPolicyWrapper:
        """Factory function for creating SGM-wrapped agents."""
        # Merge configs: defaults < env_config
        merged_config = {**default_config, **env_config}
        return create_wrapped_agent(base_agent, merged_config)
    
    # Register in AgentGym registry
    agent_registry[agent_name] = sgm_agent_factory


# Convenience function for standalone usage
def create_default_sgm_agent(
    base_agent: BaseAgent,
    env_id: str = "default",
    log_path: Optional[str] = None,
) -> SGMPolicyWrapper:
    """
    Create an SGM agent with sensible defaults.
    
    Args:
        base_agent: Base agent for proposing candidate actions.
        env_id: Environment identifier.
        log_path: Optional path to log file.
        
    Returns:
        SGMPolicyWrapper instance.
    """
    config = {
        "env_id": env_id,
        "num_candidates": 5,
        "embedding_dim": 768,
        "use_faiss": False,
        "outcome_type": "return",
        "elastic_mode": "entropy",
        "k_min": 8,
        "k_max": 64,
        "lambda_scale": 16.0,
        "prior_params": {
            "temperature": 10.0,
            "gamma": 8.0,
            "s0": 0.3,
        },
        "lookahead_params": {
            "num_rollouts": 4,
            "depth": 2,
            "max_time_seconds": 5.0,
            "max_tokens": 2000,
            "aggregation": "mean",
        },
        "log_path": log_path,
        "log_level": "normal",
    }
    
    return create_wrapped_agent(base_agent, config)
