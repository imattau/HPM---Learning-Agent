"""
online_buffer.py - Manages failure-triggered online fetching from Wikipedia.
Implements the curiosity-driven learning buffer for HPM LanguageModelPatterns.
"""

import urllib.request
import urllib.parse
import json
import re
from typing import List, Dict, Any, Optional


class OnlineLearningBuffer:
    """
    Budget-capped buffer for online knowledge acquisition.
    Triggered by failure in linguistic tasks.
    """
    def __init__(self, 
                 max_buffer_chars: int = 10000, 
                 max_fetches_per_phase: int = 5,
                 failure_threshold: int = 3):
        self.max_buffer_chars = max_buffer_chars
        self.max_fetches_per_phase = max_fetches_per_phase
        self.failure_threshold = failure_threshold
        
        self.buffer = ""
        self.fetch_counts = {} # phase_id -> count
        self.consecutive_failures = {} # domain_key -> count
        
        # Simple stop words for keyword extraction
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "when", 
            "at", "from", "by", "for", "with", "about", "against", "between", 
            "into", "through", "during", "before", "after", "above", "below", 
            "to", "of", "in", "on", "is", "are", "was", "were", "be", "been", 
            "being", "have", "has", "had", "do", "does", "did", "from", "what",
            "which", "who", "whom", "this", "that", "these", "those", "i", 
            "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
            "its", "our", "their", "extract", "numeric", "values", "from", "text"
        }

    def extract_keywords(self, text: str) -> List[str]:
        """Derive 2-3 significant keywords for Wikipedia lookup."""
        # Remove non-alphanumeric
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = clean.split()
        
        # Filter stop words and short words
        candidates = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        # Count frequencies
        counts = {}
        for w in candidates:
            counts[w] = counts.get(w, 0) + 1
            
        # Sort by frequency and length
        sorted_candidates = sorted(counts.keys(), key=lambda x: (counts[x], len(x)), reverse=True)
        return sorted_candidates[:5]

    def fetch_wikipedia(self, keywords: List[str]) -> Optional[str]:
        """Fetch summary from Wikipedia API based on keywords."""
        if not keywords: return None
        
        title = "_".join(keywords).capitalize()
        # Fallback to just the first keyword if joined fails
        titles_to_try = [title, keywords[0].capitalize()]
        
        for t in titles_to_try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(t)}"
            try:
                headers = {'User-Agent': 'HPM-Learning-Agent/1.0 (matt@example.com)'}
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    extract = data.get('extract')
                    if extract and len(extract) > 100:
                        print(f"  [OnlineBuffer] Fetched Wikipedia: {t}")
                        return extract
            except Exception as e:
                # print(f"  [OnlineBuffer] Fetch error for {t}: {e}")
                continue
        return None

    def add_to_buffer(self, text: str):
        """Append text to buffer, rotating out oldest when full."""
        if not text: return
        self.buffer += "\n" + text
        if len(self.buffer) > self.max_buffer_chars:
            # Rotate: Keep the last max_buffer_chars
            self.buffer = self.buffer[-self.max_buffer_chars:]

    def can_fetch(self, phase_id: str, domain_key: str) -> bool:
        """Check if fetch should be triggered based on budget and failures."""
        phase_count = self.fetch_counts.get(phase_id, 0)
        failures = self.consecutive_failures.get(domain_key, 0)
        
        if phase_count >= self.max_fetches_per_phase:
            return False
            
        if failures >= self.failure_threshold:
            return True
            
        return False

    def record_failure(self, domain_key: str):
        """Increment consecutive failure counter."""
        self.consecutive_failures[domain_key] = self.consecutive_failures.get(domain_key, 0) + 1

    def record_success(self, domain_key: str):
        """Reset consecutive failure counter."""
        self.consecutive_failures[domain_key] = 0

    def mark_fetched(self, phase_id: str):
        """Increment phase fetch counter."""
        self.fetch_counts[phase_id] = self.fetch_counts.get(phase_id, 0) + 1
