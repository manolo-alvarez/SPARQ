"""
FAISS-backed vector store for trajectory memory retrieval.

Provides per-environment partitioning, fast cosine kNN queries, and persistence
for reusing memories across evaluation runs.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    import warnings
    warnings.warn("FAISS not available; falling back to in-memory retrieval")

from ..base_agent_iface import RetrievalModule, RetrievalResult


class FAISSRetrievalStore(RetrievalModule):
    """
    FAISS-backed vector store with per-environment collections and stage tags.
    
    Maintains separate indices per environment to avoid cross-contamination and
    supports fast cosine similarity search via normalized vectors.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        use_gpu: bool = False,
        metric: str = "cosine",
    ):
        """
        Initialize the retrieval store.
        
        Args:
            embedding_dim: Dimensionality of embedding vectors.
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu).
            metric: Distance metric ("cosine" or "l2").
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.metric = metric
        
        # Per-environment indices and metadata
        self.indices: Dict[str, Any] = {}
        self.metadata_store: Dict[str, List[Dict[str, Any]]] = {}
        self.embeddings_store: Dict[str, List[np.ndarray]] = {}
        self.trajectory_ids: Dict[str, List[str]] = {}
        
        if not FAISS_AVAILABLE:
            self._use_fallback = True
        else:
            self._use_fallback = False
    
    def _get_or_create_index(self, env_id: str) -> Any:
        """Get or create a FAISS index for the given environment."""
        if env_id not in self.indices:
            if self._use_fallback:
                # Simple in-memory fallback
                self.indices[env_id] = None
                self.embeddings_store[env_id] = []
                self.metadata_store[env_id] = []
                self.trajectory_ids[env_id] = []
            else:
                # FAISS index with inner product (cosine with normalized vectors)
                if self.metric == "cosine":
                    index = faiss.IndexFlatIP(self.embedding_dim)
                else:
                    index = faiss.IndexFlatL2(self.embedding_dim)
                
                if self.use_gpu and faiss.get_num_gpus() > 0:
                    index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, index
                    )
                
                self.indices[env_id] = index
                self.metadata_store[env_id] = []
                self.trajectory_ids[env_id] = []
        
        return self.indices[env_id]
    
    def add(
        self,
        trajectory_id: str,
        embedding: np.ndarray,
        return_value: float,
        success_logit: float,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Add a trajectory summary to the memory store.
        
        Args:
            trajectory_id: Unique identifier for this trajectory.
            embedding: Summary embedding vector (will be normalized for cosine).
            return_value: Episodic return or cumulative reward.
            success_logit: Success probability logit.
            metadata: Additional information (env_id, stage, instruction_hash, etc.).
        """
        env_id = metadata.get("env_id", "default")
        index = self._get_or_create_index(env_id)
        
        # Normalize embedding for cosine similarity
        if self.metric == "cosine":
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Store metadata
        full_metadata = {
            "trajectory_id": trajectory_id,
            "return": return_value,
            "success_logit": success_logit,
            **metadata,
        }
        
        if self._use_fallback:
            self.embeddings_store[env_id].append(embedding)
            self.metadata_store[env_id].append(full_metadata)
            self.trajectory_ids[env_id].append(trajectory_id)
        else:
            # Add to FAISS index
            embedding_2d = embedding.reshape(1, -1).astype(np.float32)
            index.add(embedding_2d)
            self.metadata_store[env_id].append(full_metadata)
            self.trajectory_ids[env_id].append(trajectory_id)
    
    def query(
        self,
        context: Dict[str, Any],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Retrieve k nearest neighbors from memory.
        
        Args:
            context: Contains env_id, embedding vector for the query.
            k: Number of neighbors to retrieve.
            filters: Optional filters (e.g., stage tags).
            
        Returns:
            RetrievalResult with neighbor IDs, similarities, outcomes, and metadata.
        """
        env_id = context.get("env_id", "default")
        query_embedding = context.get("embedding")
        
        if query_embedding is None:
            raise ValueError("Context must contain 'embedding' key")
        
        # Get index for this environment
        if env_id not in self.indices:
            # No memories for this environment yet
            return RetrievalResult(
                neighbor_ids=[],
                similarities=np.array([]),
                returns=np.array([]),
                success_logits=np.array([]),
                metadata=[],
                query_embedding=query_embedding,
            )
        
        index = self.indices[env_id]
        all_metadata = self.metadata_store[env_id]
        
        # Normalize query embedding
        if self.metric == "cosine":
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Apply filters to metadata
        valid_indices = list(range(len(all_metadata)))
        if filters:
            stage = filters.get("stage")
            if stage is not None:
                valid_indices = [
                    i for i in valid_indices
                    if all_metadata[i].get("stage") == stage
                ]
        
        if len(valid_indices) == 0:
            # No matching memories
            return RetrievalResult(
                neighbor_ids=[],
                similarities=np.array([]),
                returns=np.array([]),
                success_logits=np.array([]),
                metadata=[],
                query_embedding=query_embedding,
            )
        
        # Clamp k to available neighbors
        k = min(k, len(valid_indices))
        
        if self._use_fallback:
            # In-memory cosine similarity
            embeddings = np.array([self.embeddings_store[env_id][i] for i in valid_indices])
            similarities = embeddings @ query_embedding
            top_k_indices = np.argsort(-similarities)[:k]
            top_similarities = similarities[top_k_indices]
            top_metadata_indices = [valid_indices[i] for i in top_k_indices]
        else:
            # FAISS search
            if len(valid_indices) < len(all_metadata):
                # Need to filter; use fallback for simplicity
                # (In production, use FAISS IDSelector for filtering)
                warnings.warn("Filtering with FAISS not yet optimized; using brute force")
                # For now, rebuild a temporary index (not efficient)
                temp_index = faiss.IndexFlatIP(self.embedding_dim) if self.metric == "cosine" else faiss.IndexFlatL2(self.embedding_dim)
                temp_embeddings = []
                for i in valid_indices:
                    # Re-extract embeddings (not stored separately in FAISS mode)
                    # This is a limitation; in production, maintain separate embedding store
                    pass
                # Fallback to simple approach
                raise NotImplementedError("Filtered FAISS queries require embedding store")
            else:
                # No filtering; use FAISS directly
                query_2d = query_embedding.reshape(1, -1).astype(np.float32)
                similarities, indices = index.search(query_2d, k)
                top_similarities = similarities[0]
                top_metadata_indices = indices[0].tolist()
        
        # Extract results
        neighbor_ids = [self.trajectory_ids[env_id][i] for i in top_metadata_indices]
        neighbor_metadata = [all_metadata[i] for i in top_metadata_indices]
        returns = np.array([m["return"] for m in neighbor_metadata])
        success_logits = np.array([m["success_logit"] for m in neighbor_metadata])
        
        return RetrievalResult(
            neighbor_ids=neighbor_ids,
            similarities=top_similarities,
            returns=returns,
            success_logits=success_logits,
            metadata=neighbor_metadata,
            query_embedding=query_embedding,
        )
    
    def save(self, path: str) -> None:
        """Persist the vector store to disk."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata and trajectory IDs
        with open(save_dir / "metadata.pkl", "wb") as f:
            pickle.dump({
                "metadata_store": self.metadata_store,
                "trajectory_ids": self.trajectory_ids,
                "embedding_dim": self.embedding_dim,
                "metric": self.metric,
            }, f)
        
        # Save FAISS indices or embeddings
        if self._use_fallback:
            with open(save_dir / "embeddings.pkl", "wb") as f:
                pickle.dump(self.embeddings_store, f)
        else:
            for env_id, index in self.indices.items():
                index_path = save_dir / f"index_{env_id}.faiss"
                if self.use_gpu:
                    # Move to CPU before saving
                    index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(index, str(index_path))
    
    def load(self, path: str) -> None:
        """Load a persisted vector store from disk."""
        load_dir = Path(path)
        
        # Load metadata
        with open(load_dir / "metadata.pkl", "rb") as f:
            data = pickle.load(f)
            self.metadata_store = data["metadata_store"]
            self.trajectory_ids = data["trajectory_ids"]
            self.embedding_dim = data["embedding_dim"]
            self.metric = data["metric"]
        
        # Load FAISS indices or embeddings
        if self._use_fallback:
            with open(load_dir / "embeddings.pkl", "rb") as f:
                self.embeddings_store = pickle.load(f)
            # Rebuild in-memory structure
            for env_id in self.embeddings_store:
                self.indices[env_id] = None
        else:
            for env_id in self.metadata_store:
                index_path = load_dir / f"index_{env_id}.faiss"
                if index_path.exists():
                    index = faiss.read_index(str(index_path))
                    if self.use_gpu and faiss.get_num_gpus() > 0:
                        index = faiss.index_cpu_to_gpu(
                            faiss.StandardGpuResources(), 0, index
                        )
                    self.indices[env_id] = index


class InMemoryRetrievalStore(RetrievalModule):
    """
    Simple in-memory retrieval store for testing without FAISS dependency.
    
    Uses brute-force cosine similarity; suitable for small memory stores.
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.memories: Dict[str, List[Dict[str, Any]]] = {}
    
    def add(
        self,
        trajectory_id: str,
        embedding: np.ndarray,
        return_value: float,
        success_logit: float,
        metadata: Dict[str, Any],
    ) -> None:
        env_id = metadata.get("env_id", "default")
        
        if env_id not in self.memories:
            self.memories[env_id] = []
        
        # Normalize for cosine
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        self.memories[env_id].append({
            "trajectory_id": trajectory_id,
            "embedding": embedding,
            "return": return_value,
            "success_logit": success_logit,
            "metadata": metadata,
        })
    
    def query(
        self,
        context: Dict[str, Any],
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        env_id = context.get("env_id", "default")
        query_embedding = context.get("embedding")
        
        if query_embedding is None:
            raise ValueError("Context must contain 'embedding' key")
        
        if env_id not in self.memories or len(self.memories[env_id]) == 0:
            return RetrievalResult(
                neighbor_ids=[],
                similarities=np.array([]),
                returns=np.array([]),
                success_logits=np.array([]),
                metadata=[],
                query_embedding=query_embedding,
            )
        
        # Normalize query
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Filter memories
        valid_memories = self.memories[env_id]
        if filters:
            stage = filters.get("stage")
            if stage is not None:
                valid_memories = [
                    m for m in valid_memories
                    if m["metadata"].get("stage") == stage
                ]
        
        if len(valid_memories) == 0:
            return RetrievalResult(
                neighbor_ids=[],
                similarities=np.array([]),
                returns=np.array([]),
                success_logits=np.array([]),
                metadata=[],
                query_embedding=query_embedding,
            )
        
        # Compute cosine similarities
        embeddings = np.array([m["embedding"] for m in valid_memories])
        similarities = embeddings @ query_embedding
        
        # Get top k
        k = min(k, len(similarities))
        top_k_indices = np.argsort(-similarities)[:k]
        
        neighbor_ids = [valid_memories[i]["trajectory_id"] for i in top_k_indices]
        top_similarities = similarities[top_k_indices]
        returns = np.array([valid_memories[i]["return"] for i in top_k_indices])
        success_logits = np.array([valid_memories[i]["success_logit"] for i in top_k_indices])
        metadata = [valid_memories[i]["metadata"] for i in top_k_indices]
        
        return RetrievalResult(
            neighbor_ids=neighbor_ids,
            similarities=top_similarities,
            returns=returns,
            success_logits=success_logits,
            metadata=metadata,
            query_embedding=query_embedding,
        )
    
    def save(self, path: str) -> None:
        """Save to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.memories, f)
    
    def load(self, path: str) -> None:
        """Load from pickle file."""
        with open(path, "rb") as f:
            self.memories = pickle.load(f)
