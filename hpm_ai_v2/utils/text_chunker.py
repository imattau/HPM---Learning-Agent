"""Utility for chunking sentences into overlapping passages."""
from typing import List

def chunk_passages(sentences: List[str], window_size: int = 5, overlap: int = 2) -> List[str]:
    """Split sentences into overlapping windows."""
    passages = []
    if not sentences:
        return passages
    
    step = window_size - overlap
    if step <= 0:
        step = 1 # Avoid infinite loop
        
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + window_size]
        if chunk:
            passages.append(" ".join(chunk))
        
        # If we reached the end, stop
        if i + window_size >= len(sentences):
            break
            
    return passages
