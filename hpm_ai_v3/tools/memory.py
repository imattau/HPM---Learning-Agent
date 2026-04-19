"""
memory_tools.py - Persistent memory tools for HPM agents.
Uses FAISS for vector search and JSON for key-value storage.
"""

import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Union
from collections import deque
import warnings

from .registry import ToolRegistry

# ----------------------------------------------------------------------
# Vector Memory (FAISS)
# ----------------------------------------------------------------------
_vector_indexes = {}
_vector_metadata = {}

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("faiss not installed. Vector memory tools will use fallback (numpy).")


def _get_vector_index(name: str, dimension: int = 384):
    """Get or create a FAISS index for a named collection."""
    if name not in _vector_indexes:
        if FAISS_AVAILABLE:
            index = faiss.IndexFlatL2(dimension)
            _vector_indexes[name] = index
            _vector_metadata[name] = []
        else:
            # Fallback to numpy arrays
            _vector_indexes[name] = {"vectors": [], "metadata": []}
            _vector_metadata[name] = []
    return _vector_indexes[name], _vector_metadata.get(name, [])


def vector_store(embedding: Union[np.ndarray, torch.Tensor, List[float]], 
                 metadata: Dict[str, Any] = None,
                 collection: str = "default") -> Dict[str, Any]:
    """
    Store an embedding vector with associated metadata.
    """
    if metadata is None:
        metadata = {}
    
    # Convert to numpy
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().detach().numpy()
    elif isinstance(embedding, list):
        embedding = np.array(embedding, dtype=np.float32)
    else:
        embedding = np.asarray(embedding, dtype=np.float32)
    
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    dimension = embedding.shape[1]
    index, meta_list = _get_vector_index(collection, dimension)
    
    idx = len(meta_list)
    
    if FAISS_AVAILABLE:
        index.add(embedding)
    else:
        index["vectors"].append(embedding[0])
        index["metadata"].append(metadata)
    
    metadata["_id"] = idx
    meta_list.append(metadata)
    
    return {"id": idx, "status": "stored", "collection": collection}


def vector_search(query: Union[np.ndarray, torch.Tensor, List[float]],
                  collection: str = "default",
                  top_k: int = 5) -> Dict[str, Any]:
    """
    Search for similar vectors in a collection.
    """
    if collection not in _vector_indexes:
        return {"results": [], "collection": collection, "status": "empty"}
    
    # Convert query
    if isinstance(query, torch.Tensor):
        query = query.cpu().detach().numpy()
    elif isinstance(query, list):
        query = np.array(query, dtype=np.float32)
    else:
        query = np.asarray(query, dtype=np.float32)
    
    if query.ndim == 1:
        query = query.reshape(1, -1)
    
    index, meta_list = _vector_indexes[collection], _vector_metadata[collection]
    
    if FAISS_AVAILABLE:
        distances, indices = index.search(query, min(top_k, len(meta_list)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(meta_list):
                results.append({
                    "id": int(idx),
                    "distance": float(dist),
                    "metadata": meta_list[idx].copy()
                })
    else:
        # Fallback: linear search
        vectors = np.array(index["vectors"])
        if len(vectors) == 0:
            return {"results": [], "collection": collection, "status": "empty"}
        # Compute L2 distances
        dists = np.linalg.norm(vectors - query[0], axis=1)
        top_indices = np.argsort(dists)[:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "id": int(idx),
                "distance": float(dists[idx]),
                "metadata": index["metadata"][idx].copy()
            })
    
    return {"results": results, "collection": collection, "status": "success"}


def clear_vector_collection(collection: str = "default") -> Dict[str, str]:
    """Clear a vector collection."""
    if collection in _vector_indexes:
        if FAISS_AVAILABLE:
            _vector_indexes[collection].reset()
        else:
            _vector_indexes[collection] = {"vectors": [], "metadata": []}
        _vector_metadata[collection] = []
    return {"status": "cleared", "collection": collection}


# ----------------------------------------------------------------------
# Key-Value Store (JSON file)
# ----------------------------------------------------------------------
_kv_store: Dict[str, Dict[str, Any]] = {}
_kv_persist_path: Optional[str] = None


def set_kv_persistence(path: str):
    """Set a file path for persistent key-value storage."""
    global _kv_persist_path
    _kv_persist_path = path
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                global _kv_store
                _kv_store = json.load(f)
        except:
            pass


def save_kv_store():
    """Persist key-value store to disk if path is set."""
    if _kv_persist_path:
        with open(_kv_persist_path, 'w') as f:
            json.dump(_kv_store, f)


def key_value_store(key: str, value: Any, namespace: str = "default") -> Dict[str, str]:
    """
    Store a value under a key in a namespace.
    Value must be JSON-serializable.
    """
    if namespace not in _kv_store:
        _kv_store[namespace] = {}
    _kv_store[namespace][key] = value
    save_kv_store()
    return {"status": "stored", "namespace": namespace, "key": key}


def key_value_retrieve(key: str, namespace: str = "default") -> Dict[str, Any]:
    """Retrieve a value by key from a namespace."""
    if namespace not in _kv_store:
        return {"value": None, "found": False}
    value = _kv_store[namespace].get(key)
    return {"value": value, "found": value is not None, "key": key, "namespace": namespace}


def key_value_delete(key: str, namespace: str = "default") -> Dict[str, Any]:
    """Delete a key from a namespace."""
    if namespace in _kv_store and key in _kv_store[namespace]:
        del _kv_store[namespace][key]
        save_kv_store()
        return {"status": "deleted", "key": key}
    return {"status": "not_found", "key": key}


def key_value_list(namespace: str = "default") -> Dict[str, Any]:
    """List all keys in a namespace."""
    if namespace in _kv_store:
        return {"keys": list(_kv_store[namespace].keys()), "namespace": namespace}
    return {"keys": [], "namespace": namespace}


# ----------------------------------------------------------------------
# Episodic Buffer (in-memory, per-agent)
# ----------------------------------------------------------------------
_episodic_buffers: Dict[str, deque] = {}


def episodic_append(event: Dict[str, Any], buffer_name: str = "default", maxlen: int = 1000) -> Dict[str, Any]:
    """Append an event to an episodic buffer."""
    if buffer_name not in _episodic_buffers:
        _episodic_buffers[buffer_name] = deque(maxlen=maxlen)
    _episodic_buffers[buffer_name].append(event)
    return {"status": "appended", "buffer": buffer_name, "length": len(_episodic_buffers[buffer_name])}


def episodic_get_recent(n: int = 10, buffer_name: str = "default") -> Dict[str, Any]:
    """Get the most recent n events from the buffer."""
    if buffer_name not in _episodic_buffers:
        return {"events": [], "buffer": buffer_name}
    buf = _episodic_buffers[buffer_name]
    events = list(buf)[-n:]
    return {"events": events, "buffer": buffer_name, "count": len(events)}


def episodic_clear(buffer_name: str = "default") -> Dict[str, str]:
    """Clear an episodic buffer."""
    if buffer_name in _episodic_buffers:
        _episodic_buffers[buffer_name].clear()
    return {"status": "cleared", "buffer": buffer_name}


# ----------------------------------------------------------------------
# Register Tools
# ----------------------------------------------------------------------
def register_memory_tools(persistence_path: Optional[str] = None):
    """Register all memory tools with ToolRegistry."""
    if persistence_path:
        set_kv_persistence(persistence_path)
    
    ToolRegistry.register(
        name="vector_store",
        tool_fn=vector_store,
        input_keys=["embedding", "metadata", "collection"],
        output_key="store_result",
        cost=0.02,
        description="Store an embedding vector with metadata in a collection."
    )
    
    ToolRegistry.register(
        name="vector_search",
        tool_fn=vector_search,
        input_keys=["query", "collection", "top_k"],
        output_key="search_results",
        cost=0.03,
        description="Search for similar vectors in a collection."
    )
    
    ToolRegistry.register(
        name="key_value_store",
        tool_fn=key_value_store,
        input_keys=["key", "value", "namespace"],
        output_key="kv_result",
        cost=0.01,
        description="Store a JSON-serializable value under a key."
    )
    
    ToolRegistry.register(
        name="key_value_retrieve",
        tool_fn=key_value_retrieve,
        input_keys=["key", "namespace"],
        output_key="retrieved",
        cost=0.01,
        description="Retrieve a value by key."
    )
    
    ToolRegistry.register(
        name="key_value_list",
        tool_fn=key_value_list,
        input_keys=["namespace"],
        output_key="key_list",
        cost=0.005,
        description="List all keys in a namespace."
    )
    
    ToolRegistry.register(
        name="episodic_append",
        tool_fn=episodic_append,
        input_keys=["event", "buffer_name"],
        output_key="episodic_result",
        cost=0.005,
        description="Append an event to an episodic memory buffer."
    )
    
    ToolRegistry.register(
        name="episodic_recent",
        tool_fn=episodic_get_recent,
        input_keys=["n", "buffer_name"],
        output_key="recent_events",
        cost=0.005,
        description="Retrieve recent events from episodic buffer."
    )
    
    print(f"[MemoryTools] Registered {len(ToolRegistry.list_tools())} total tools.")


# Auto-register on import (without persistence by default)
register_memory_tools()
