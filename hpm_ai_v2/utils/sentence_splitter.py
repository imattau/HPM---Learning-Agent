"""Lightweight sentence splitter with abbreviation handling."""
import re
from typing import List

class SentenceSplitter:
    def __init__(self):
        # Abbreviations that do NOT end a sentence
        self.abbrev = {
            "mr", "mrs", "ms", "dr", "prof", "gen", "col", "maj", "capt",
            "vs", "etc", "al", "inc", "ltd", "co",
            "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
        }

    def split(self, text: str) -> List[str]:
        # Replace newlines with spaces
        text = text.replace('\n', ' ')
        
        # Use regex to find potential boundaries
        # Look for . ! ? followed by whitespace and an uppercase letter
        # We'll split on the whitespace
        
        # First, protect abbreviations by temporarily replacing their periods
        protected_text = text
        for a in self.abbrev:
            # Use a lambda to preserve case of the matched abbreviation
            pattern = r'\b(' + re.escape(a) + r')\.'
            protected_text = re.sub(pattern, lambda m: m.group(1).replace('.', '|') + '|', protected_text, flags=re.IGNORECASE)
            
        # Also protect initials like "A."
        protected_text = re.sub(r'\b([A-Z])\.', r'\1|', protected_text)
        
        # Now split on [.!?] followed by whitespace
        # We don't need lookahead for capital letter if we protected abbreviations correctly
        # but it's safer to keep it.
        splits = re.split(r'([.!?])\s+(?=[A-Z])', protected_text)
        
        sentences = []
        for i in range(0, len(splits), 2):
            s = splits[i]
            if i + 1 < len(splits):
                s += splits[i+1]
            
            # Restore periods
            s = s.replace('|', '.')
            sentences.append(s.strip())
            
        return [s for s in sentences if s]
        
    def _fallback_split(self, text: str) -> List[str]:
        # Very basic fallback
        return [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
