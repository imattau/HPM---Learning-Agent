import json
import re
from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import List, Any, Union, Optional, Sequence, Dict
import matplotlib.pyplot as plt

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr as _parse_expr
except Exception:  # pragma: no cover - optional dependency
    sp = None
    _parse_expr = None

class InputAdapter:
    """Base class for converting raw input to HPM observation tokens."""
    @property
    def obs_dim(self) -> int:
        return 2 # Default

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        raise NotImplementedError

class VisionAdapter(InputAdapter):
    """Convert images to discrete tokens via quantization."""
    def __init__(self, image_size=(64, 64), num_bins=8):
        self.image_size = image_size
        self.num_bins = num_bins

    @property
    def obs_dim(self) -> int:
        return self.num_bins

    def to_observations(self, image_path_or_array, max_length=100) -> List[int]:
        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array).convert('L')
        else:
            img = Image.fromarray(image_path_or_array).convert('L')
            
        img = img.resize(self.image_size)
        pixels = np.array(img).flatten()
        # Quantize pixels into bins
        tokens = np.digitize(pixels, bins=np.linspace(0, 255, self.num_bins)) - 1
        # Map to valid range [0, num_bins-1]
        tokens = np.clip(tokens, 0, self.num_bins - 1)
        return tokens[:max_length].tolist()

class TextAdapter(InputAdapter):
    """Convert natural language to character or word tokens."""
    def __init__(self, vocab: Optional[dict] = None):
        self.vocab = vocab if vocab else {chr(i): i - 32 for i in range(32, 127)}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def obs_dim(self) -> int:
        return 256 # character space

    def to_observations(self, text: str, max_length: int = 100) -> List[int]:
        # Simple character-level tokenization
        tokens = [self.vocab.get(ch, 0) % 256 for ch in text[:max_length]]
        return tokens


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    sentence_type: str
    start: int
    end: int
    confidence: float


@dataclass(frozen=True)
class ParagraphSpan:
    text: str
    sentences: List[SentenceSpan]
    start: int
    end: int
    confidence: float


class SentenceAdapter(InputAdapter):
    """Sentence-level abstraction built on top of the char/text substrate."""

    SENTENCE_TYPES = (
        "declarative",
        "question",
        "request",
        "clarification",
        "closing",
        "exclamation",
        "fragment",
    )
    TYPE_TO_ID = {name: idx for idx, name in enumerate(SENTENCE_TYPES)}
    ID_TO_TYPE = {idx: name for name, idx in TYPE_TO_ID.items()}
    _SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
    _PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
    _WHITESPACE_RE = re.compile(r"\s+")
    PARA_START_TOKEN = len(SENTENCE_TYPES) + 16
    PARA_END_TOKEN = len(SENTENCE_TYPES) + 17

    def __init__(self, obs_dim: int = 32, use_spacy: bool = False, spacy_model: str = "en_core_web_sm"):
        self._obs_dim = max(8, int(obs_dim))
        self._text_adapter = TextAdapter()
        self._use_spacy = bool(use_spacy)
        self._spacy_model = str(spacy_model)
        self._nlp = self._load_spacy() if self._use_spacy else None

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_text(self, raw_input: Any) -> str:
        return self._canonicalize(str(raw_input or ""))

    def from_text(self, text: str) -> str:
        return self._canonicalize(text)

    def segment(self, text: str) -> List[SentenceSpan]:
        paragraphs = self.segment_paragraphs(text)
        if not paragraphs:
            return []
        spans: List[SentenceSpan] = []
        for paragraph in paragraphs:
            spans.extend(paragraph.sentences)
        return spans

    def segment_paragraphs(self, text: str) -> List[ParagraphSpan]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return []
        paragraph_texts = [piece.strip() for piece in self._PARAGRAPH_SPLIT_RE.split(raw_text) if piece.strip()]
        if not paragraph_texts:
            paragraph_texts = [raw_text]
        paragraphs: List[ParagraphSpan] = []
        cursor = 0
        for paragraph_text in paragraph_texts:
            start = raw_text.find(paragraph_text, cursor)
            if start < 0:
                start = cursor
            end = start + len(paragraph_text)
            sentences = self._segment_sentences(self._canonicalize(paragraph_text), base_offset=start)
            confidence = float(sum(sentence.confidence for sentence in sentences) / max(1, len(sentences))) if sentences else 0.0
            paragraphs.append(
                ParagraphSpan(
                    text=paragraph_text,
                    sentences=sentences,
                    start=start,
                    end=end,
                    confidence=confidence,
                )
            )
            cursor = end
        return paragraphs

    def _segment_sentences(self, text: str, base_offset: int = 0) -> List[SentenceSpan]:
        if self._nlp is not None:
            spans = self._segment_sentences_spacy(text, base_offset=base_offset)
            if spans:
                return spans
        spans: List[SentenceSpan] = []
        cursor = 0
        for chunk in self._split(text):
            start = text.find(chunk, cursor)
            if start < 0:
                start = cursor
            end = start + len(chunk)
            sentence_type = self.classify_sentence(chunk)
            confidence = self._sentence_confidence(chunk, sentence_type)
            spans.append(
                SentenceSpan(
                    text=chunk,
                    sentence_type=sentence_type,
                    start=base_offset + start,
                    end=base_offset + end,
                    confidence=confidence,
                )
            )
            cursor = end
        return spans

    def _segment_sentences_spacy(self, text: str, base_offset: int = 0) -> List[SentenceSpan]:
        if self._nlp is None:
            return []
        try:
            doc = self._nlp(text)
        except Exception:
            return []
        spans: List[SentenceSpan] = []
        cursor = 0
        for sent in getattr(doc, "sents", []):
            chunk = str(sent).strip()
            if not chunk:
                continue
            start = text.find(chunk, cursor)
            if start < 0:
                start = cursor
            end = start + len(chunk)
            sentence_type = self.classify_sentence(chunk)
            confidence = self._sentence_confidence(chunk, sentence_type)
            try:
                sent_tokens = [tok for tok in sent if not tok.is_space]
                if sent_tokens:
                    if any(tok.text == "?" for tok in sent_tokens):
                        sentence_type = "question"
                    elif any(tok.text == "!" for tok in sent_tokens):
                        sentence_type = "exclamation"
                    if any(tok.dep_ in {"aux", "cop"} for tok in sent_tokens if hasattr(tok, "dep_")):
                        confidence = min(1.0, confidence + 0.05)
            except Exception:
                pass
            spans.append(
                SentenceSpan(
                    text=chunk,
                    sentence_type=sentence_type,
                    start=base_offset + start,
                    end=base_offset + end,
                    confidence=confidence,
                )
            )
            cursor = end
        return spans

    def classify_sentence(self, sentence: str) -> str:
        text = self._canonicalize(sentence).lower()
        if not text:
            return "fragment"
        if text.endswith("?"):
            return "question"
        if text.endswith("!"):
            return "exclamation"
        if any(text.startswith(prefix) for prefix in ("tell me", "give me", "show me", "help me", "please", "explain")):
            return "request"
        if any(phrase in text for phrase in ("what do you mean", "can you clarify", "which part", "clarify that", "say more")):
            return "clarification"
        if any(text.startswith(prefix) for prefix in ("bye", "goodbye", "thanks", "thank you", "see you")):
            return "closing"
        if len(text.split()) < 2:
            return "fragment"
        return "declarative"

    def _sentence_confidence(self, sentence: str, sentence_type: str) -> float:
        words = self._canonicalize(sentence).split()
        if not words:
            return 0.0
        base = 0.25
        if sentence_type in {"question", "request", "clarification", "closing"}:
            base += 0.25
        if sentence.endswith((".", "?", "!", ":")):
            base += 0.20
        if len(words) >= 4:
            base += 0.15
        if len(words) >= 8:
            base += 0.10
        return float(max(0.0, min(1.0, base)))

    def to_observations(
        self,
        raw_input: Any,
        max_length: int = 100,
        include_paragraph_markers: bool = False,
    ) -> List[int]:
        text = self.to_text(raw_input)
        paragraphs = self.segment_paragraphs(text)
        if not paragraphs:
            return []

        tokens: List[int] = []
        for paragraph in paragraphs:
            if include_paragraph_markers:
                tokens.append(self.PARA_START_TOKEN)
            for span in paragraph.sentences:
                if len(tokens) >= max_length:
                    break
                type_id = self.TYPE_TO_ID.get(span.sentence_type, self.TYPE_TO_ID["fragment"])
                length_bucket = min(7, max(0, len(span.text.split()) // 4))
                confidence_bucket = min(7, max(0, int(round(span.confidence * 7))))
                tokens.extend([
                    type_id,
                    len(self.SENTENCE_TYPES) + length_bucket,
                    len(self.SENTENCE_TYPES) + 8 + confidence_bucket,
                ])
                if len(tokens) >= max_length:
                    break
            if include_paragraph_markers:
                tokens.append(self.PARA_END_TOKEN)
        return tokens[:max_length]

    def from_observations(self, tokens: Sequence[int]) -> str:
        if not tokens:
            return ""
        spans: List[str] = []
        for idx in range(0, len(tokens), 3):
            type_id = int(tokens[idx]) % len(self.SENTENCE_TYPES)
            length_bucket = int(tokens[idx + 1]) if idx + 1 < len(tokens) else 0
            confidence_bucket = int(tokens[idx + 2]) if idx + 2 < len(tokens) else 0
            sentence_type = self.ID_TO_TYPE.get(type_id, "fragment")
            length_label = f"len{max(0, length_bucket - len(self.SENTENCE_TYPES))}"
            confidence_label = f"conf{max(0, confidence_bucket - len(self.SENTENCE_TYPES) - 8)}"
            spans.append(f"<{sentence_type}:{length_label}:{confidence_label}>")
        return " ".join(spans)

    def paragraph_markers(self, text: str) -> List[int]:
        tokens: List[int] = []
        for paragraph in self.segment_paragraphs(text):
            tokens.extend([self.PARA_START_TOKEN, self.PARA_END_TOKEN])
        return tokens

    def act(self, token: int, context: Any = None):
        if isinstance(context, dict) and "text" in context:
            return self.to_text(context["text"])
        if token == self.PARA_START_TOKEN:
            return "<PARA_START>"
        if token == self.PARA_END_TOKEN:
            return "<PARA_END>"
        return self.ID_TO_TYPE.get(int(token) % len(self.SENTENCE_TYPES), "fragment")

    def split(self, text: str) -> List[str]:
        """Split text into individual sentences."""
        return self._split(text)

    def _split(self, text: str) -> List[str]:
        pieces = [piece.strip() for piece in self._SPLIT_RE.split(text) if piece.strip()]
        return pieces or [text.strip()]

    def _canonicalize(self, text: str) -> str:
        text = str(text or "").strip()
        text = self._WHITESPACE_RE.sub(" ", text)
        text = re.sub(r"\s+([.!?,;:])", r"\1", text)
        text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
        return text.strip()

    def _load_spacy(self):
        if spacy is None:  # pragma: no cover - optional dependency
            return None
        try:
            if hasattr(spacy, "util") and spacy.util.is_package(self._spacy_model):
                return spacy.load(self._spacy_model, disable=["ner"])
        except Exception:
            pass
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None


class WordAdapter(InputAdapter):
    """Word-level surface adapter with a bounded dynamic vocabulary."""

    TOKEN_RE = re.compile(r"\n|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", re.UNICODE)
    FUNCTION_WORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "so", "because", "while", "when", "where",
        "of", "to", "in", "on", "at", "for", "from", "by", "with", "without", "into", "over", "under",
        "is", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "should", "may", "might", "must",
    }
    PRONOUNS = {
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "their", "our", "its", "this", "that", "these", "those",
    }
    QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "which", "whom", "whose"}
    COMMON_VERBS = {
        "say", "says", "said", "see", "saw", "seen", "make", "made", "go", "went", "gone",
        "know", "knew", "known", "think", "thought", "take", "took", "taken", "write", "wrote", "written",
        "chase", "chased", "sit", "sat", "move", "moved", "run", "ran", "help", "helped", "notice", "noticed",
        "admire", "admired", "like", "liked", "watch", "watched", "tell", "told", "read", "read", "give", "gave",
        "given", "become", "became", "find", "found", "ask", "asked", "need", "needed", "want", "wanted",
        "have", "had", "do", "did", "done", "be", "is", "are", "was", "were", "been",
    }
    COMMON_ADJECTIVES = {
        "big", "small", "large", "little", "quick", "slow", "fast", "red", "blue", "green", "young",
        "old", "new", "good", "bad", "happy", "sad", "warm", "cold", "bright", "dark", "round",
        "clear", "simple", "complex", "useful", "helpful", "true", "false",
    }
    DEFAULT_CANONICAL_ALIASES = {
        "feline": "cat",
        "kitty": "cat",
        "canine": "dog",
        "puppy": "dog",
        "automobile": "car",
        "vehicle": "car",
        "kid": "child",
        "youth": "child",
        "purchase": "buy",
        "buying": "buy",
        "begin": "start",
        "commence": "start",
        "finish": "end",
        "finishs": "end",
        "assist": "help",
        "assistive": "help",
        "reply": "answer",
        "respond": "answer",
        "quickly": "quick",
        "rapid": "fast",
        "large": "big",
        "little": "small",
        "glad": "happy",
        "sadness": "sad",
    }
    VOCAB_CONTRACT = {
        "stable_tokenization": True,
        "bounded_vocab": True,
        "canonicalization": "alias + light stemming",
        "reserved_tokens": ("<UNK>", "<BOS>", "<EOS>", "<NL>", "<PARA>"),
        "semantic_ordering": "frequency + lexical bucket",
        "bundle_keys": ("max_vocab_size", "lowercase", "word_vocab", "canonical_aliases"),
    }

    def __init__(
        self,
        max_vocab_size: int = 5000,
        lowercase: bool = True,
        vocab: Optional[Dict[str, int]] = None,
        canonical_aliases: Optional[Dict[str, str]] = None,
    ):
        self._max_vocab_size = max(8, int(max_vocab_size))
        self.lowercase = bool(lowercase)
        self._canonical_aliases = {
            str(alias).strip().lower() if self.lowercase else str(alias).strip(): str(canonical).strip().lower() if self.lowercase else str(canonical).strip()
            for alias, canonical in dict(canonical_aliases or {}).items()
            if str(alias).strip() and str(canonical).strip()
        }
        self.UNK_TOKEN = "<UNK>"
        self.BOS_TOKEN = "<BOS>"
        self.EOS_TOKEN = "<EOS>"
        self.NEWLINE_TOKEN = "<NL>"
        self.PARA_TOKEN = "<PARA>"
        base_vocab: Dict[str, int] = {
            self.UNK_TOKEN: 0,
            self.BOS_TOKEN: 1,
            self.EOS_TOKEN: 2,
            self.NEWLINE_TOKEN: 3,
            self.PARA_TOKEN: 4,
        }
        if vocab:
            for tok, idx in vocab.items():
                tok = str(tok)
                idx = int(idx)
                if 0 <= idx < self._max_vocab_size:
                    base_vocab[tok] = idx
        self._word_to_id = dict(sorted(base_vocab.items(), key=lambda item: item[1]))
        self._id_to_word: Dict[int, str] = {idx: tok for tok, idx in self._word_to_id.items()}

    @classmethod
    def vocab_contract(cls) -> Dict[str, Any]:
        return {
            "stable_tokenization": bool(cls.VOCAB_CONTRACT["stable_tokenization"]),
            "bounded_vocab": bool(cls.VOCAB_CONTRACT["bounded_vocab"]),
            "canonicalization": str(cls.VOCAB_CONTRACT["canonicalization"]),
            "reserved_tokens": list(cls.VOCAB_CONTRACT["reserved_tokens"]),
            "semantic_ordering": str(cls.VOCAB_CONTRACT["semantic_ordering"]),
            "bundle_keys": list(cls.VOCAB_CONTRACT["bundle_keys"]),
        }

    @classmethod
    def vocab_checklist(cls) -> List[str]:
        return [
            "Tokenization is stable across runs.",
            "Vocabulary size is bounded.",
            "Canonical aliases are persisted with the bundle.",
            "Reserved tokens are always present.",
            "Vocabulary ordering is frequency- and bucket-aware.",
        ]

    @property
    def obs_dim(self) -> int:
        return self._max_vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self._word_to_id)

    def tokenize(self, text: str) -> List[str]:
        raw = str(text or "")
        tokens: List[str] = []
        for tok in self.TOKEN_RE.findall(raw):
            if not tok or tok.isspace():
                continue
            if tok == "\n":
                tokens.append(self.NEWLINE_TOKEN)
                continue
            tokens.append(self._canonical_token(tok))
        return tokens

    def _canonical_token(self, token: str) -> str:
        tok = str(token or "").strip()
        if not tok:
            return self.UNK_TOKEN
        if tok == "\n":
            return self.NEWLINE_TOKEN
        if self.lowercase and tok not in {self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.NEWLINE_TOKEN, self.PARA_TOKEN}:
            tok = tok.lower()
        tok = self._canonical_aliases.get(tok, tok)
        if tok.endswith("'s") and len(tok) > 3:
            tok = tok[:-2]
        if tok.endswith("s") and len(tok) > 3 and not tok.endswith("ss") and tok not in self.COMMON_VERBS:
            singular = tok[:-1]
            tok = self._canonical_aliases.get(singular, singular)
        return tok

    def encode_token(self, token: str) -> int:
        tok = self._canonical_token(token)
        if not tok:
            return self._word_to_id[self.UNK_TOKEN]
        if tok in self._word_to_id:
            return self._word_to_id[tok]
        if len(self._word_to_id) >= self._max_vocab_size:
            return self._word_to_id[self.UNK_TOKEN]
        idx = len(self._word_to_id)
        self._word_to_id[tok] = idx
        self._id_to_word[idx] = tok
        return idx

    def decode_token(self, token_id: int) -> str:
        return self._id_to_word.get(int(token_id), self.UNK_TOKEN)

    def encode(self, token_id: int) -> int:
        return int(token_id) if 0 <= int(token_id) < self._max_vocab_size else self._word_to_id[self.UNK_TOKEN]

    def encode_char(self, ch: str) -> int:
        return self.encode_token(ch)

    def decode_class(self, class_id: int) -> str:
        return self.decode_token(class_id)

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        tokens = [self.encode_token(tok) for tok in self.tokenize(str(raw_input or ""))]
        if not tokens:
            tokens = [self._word_to_id[self.UNK_TOKEN]]
        return tokens[:max_length]

    def from_observations(self, tokens: Sequence[int]) -> str:
        pieces: List[str] = []
        for token_id in tokens:
            tok = self.decode_token(int(token_id))
            if tok == self.NEWLINE_TOKEN:
                pieces.append("\n")
                continue
            if tok == self.PARA_TOKEN:
                pieces.append("\n\n")
                continue
            if not pieces:
                pieces.append(tok)
                continue
            if tok in {".", ",", ";", ":", "!", "?", ")", "]", "}"}:
                pieces[-1] = pieces[-1].rstrip() + tok
            elif pieces[-1] in {"(", "[", "{"}:
                pieces.append(tok)
            else:
                pieces.append(" " + tok)
        return "".join(pieces).strip()

    @classmethod
    def semantic_bucket(cls, token: str) -> int:
        tok = str(token or "").strip().lower()
        if not tok or tok in {"<unk>", "<bos>", "<eos>", "<nl>", "<para>"}:
            return 0
        if tok in cls.PRONOUNS or tok in cls.QUESTION_WORDS:
            return 1
        if tok in cls.FUNCTION_WORDS:
            return 2
        if tok.isdigit():
            return 3
        if tok in cls.COMMON_VERBS or tok.endswith("ed") or tok.endswith("ing"):
            return 4
        if tok in cls.COMMON_ADJECTIVES or tok.endswith("ous") or tok.endswith("ful") or tok.endswith("able"):
            return 5
        if tok.isalpha():
            return 6
        return 7

    @classmethod
    def build_semantic_vocab(
        cls,
        chunks: Sequence[str],
        *,
        max_vocab_size: int = 5000,
        lowercase: bool = True,
        min_freq: int = 2,
        canonical_aliases: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        adapter = cls(max_vocab_size=max_vocab_size, lowercase=lowercase, canonical_aliases=canonical_aliases)
        counts: Dict[str, int] = {}
        for chunk in chunks:
            for tok in adapter.tokenize(chunk):
                if tok in {adapter.UNK_TOKEN, adapter.BOS_TOKEN, adapter.EOS_TOKEN, adapter.NEWLINE_TOKEN, adapter.PARA_TOKEN}:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        ordered_tokens = sorted(
            (
                token
                for token, count in counts.items()
                if count >= min_freq
            ),
            key=lambda token: (cls.semantic_bucket(token), -counts[token], token),
        )
        vocab: Dict[str, int] = {
            adapter.UNK_TOKEN: 0,
            adapter.BOS_TOKEN: 1,
            adapter.EOS_TOKEN: 2,
            adapter.NEWLINE_TOKEN: 3,
            adapter.PARA_TOKEN: 4,
        }
        next_idx = len(vocab)
        for token in ordered_tokens:
            if token in vocab:
                continue
            if next_idx >= max_vocab_size:
                break
            vocab[token] = next_idx
            next_idx += 1
        return vocab

    def _sentence_confidence(self, sentence: str, sentence_type: str) -> float:
        words = sentence.split()
        if not words:
            return 0.0
        base = 0.25
        if sentence_type in {"question", "request", "clarification", "closing"}:
            base += 0.25
        if sentence.endswith((".", "?", "!", ":")):
            base += 0.20
        if len(words) >= 4:
            base += 0.15
        if len(words) >= 8:
            base += 0.10
        return float(max(0.0, min(1.0, base)))


class StructuredTextAdapter(InputAdapter):
    """Canonical JSON adapter for record-like text structures."""

    def __init__(self, obs_dim: int = 256):
        self._obs_dim = max(2, int(obs_dim))
        self._text_adapter = TextAdapter()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_text(self, raw_input: Any) -> str:
        structured = self._normalize(raw_input)
        if isinstance(structured, (dict, list)):
            return json.dumps(structured, sort_keys=True, separators=(",", ":"))
        return json.dumps({"value": structured}, sort_keys=True, separators=(",", ":"))

    def from_text(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            return text

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        canonical = self.to_text(raw_input)
        return self._text_adapter.to_observations(canonical, max_length=max_length)

    def from_observations(self, tokens: Sequence[int]) -> Any:
        text = "".join(self._text_adapter.reverse_vocab.get(int(tok) % 256, "?") for tok in tokens)
        return self.from_text(text)

    def act(self, token: int, context: Any = None):
        if isinstance(context, dict):
            return self.to_text(context)
        return self._text_adapter.reverse_vocab.get(int(token) % 256, chr(int(token) % 95 + 32))

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): self._normalize(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
        if isinstance(value, (list, tuple)):
            return [self._normalize(v) for v in value]
        if isinstance(value, np.ndarray):
            return [self._normalize(v) for v in value.tolist()]
        if isinstance(value, (np.bool_, bool, np.integer, int, np.floating, float, str)) or value is None:
            return value
        return repr(value)


class MathTextAdapter(InputAdapter):
    """Canonical adapter for mixed natural-language text with inline math."""

    MATH_SPAN_RE = re.compile(
        r"(?<!\w)(?:[A-Za-z_]\w*|\d+(?:\.\d+)?)"
        r"(?:\s*(?:[+\-*/^=<>±×÷]|<=|>=|!=|≈|∈|∉)\s*(?:[A-Za-z_]\w*|\d+(?:\.\d+)?))*"
        r"(?:\s*(?:\(|\)|\^)\s*(?:[A-Za-z_]\w*|\d+(?:\.\d+)?))*"
    )
    MATH_SYMBOL_RE = re.compile(r"[=+\-*/^<>±×÷∑∫√≈≠≤≥∈∉∞πθλμσΔαβγ]")
    WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, obs_dim: int = 256):
        self._obs_dim = max(2, int(obs_dim))
        self._text_adapter = TextAdapter()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_text(self, raw_input: Any) -> str:
        if isinstance(raw_input, dict) and "text" in raw_input:
            text = str(raw_input["text"])
        else:
            text = str(raw_input)
        return self._canonicalize(text)

    def from_text(self, text: str) -> str:
        return self._canonicalize(text)

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        canonical = self.to_text(raw_input)
        return self._text_adapter.to_observations(canonical, max_length=max_length)

    def from_observations(self, tokens: Sequence[int]) -> str:
        return "".join(self._text_adapter.reverse_vocab.get(int(tok) % 256, "?") for tok in tokens)

    def extract_math_spans(self, text: str) -> List[str]:
        canonical = self._canonicalize(text)
        return [match.group(0).strip() for match in self.MATH_SPAN_RE.finditer(canonical) if self.MATH_SYMBOL_RE.search(match.group(0))]

    def act(self, token: int, context: Any = None):
        if isinstance(context, dict) and "text" in context:
            return self.to_text(context)
        return self._text_adapter.reverse_vocab.get(int(token) % 256, chr(int(token) % 95 + 32))

    def _canonicalize(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\u2212", "-").replace("\u00d7", "×").replace("\u00f7", "÷")
        text = self.WHITESPACE_RE.sub(" ", text.strip())
        text = re.sub(r"\s*([=+\-*/^<>±×÷])\s*", r" \1 ", text)
        text = re.sub(r"\s*([(){}\[\],:])\s*", r"\1", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class SympyMathAdapter(InputAdapter):
    """Symbolic math adapter backed by SymPy.

    This is the math equivalent of a dict/grammar validator:
    - parseability
    - simplification
    - equivalence
    - solve/evaluate feedback
    """

    def __init__(self, obs_dim: int = 256):
        self._obs_dim = max(2, int(obs_dim))
        self._text_adapter = MathTextAdapter(obs_dim=obs_dim)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_text(self, raw_input: Any) -> str:
        return self._text_adapter.to_text(raw_input)

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        canonical = self.to_text(raw_input)
        return self._text_adapter.to_observations(canonical, max_length=max_length)

    def from_observations(self, tokens: Sequence[int]) -> str:
        return self._text_adapter.from_observations(tokens)

    def parse(self, text: str):
        if sp is None or _parse_expr is None:
            return None
        canonical = self._text_adapter.to_text(text)
        try:
            lhs, rhs = self._split_equation(canonical)
            if rhs is not None:
                return sp.Eq(self._parse_side(lhs), self._parse_side(rhs))
            return self._parse_side(canonical)
        except Exception:
            return None

    def simplify(self, text: str) -> Optional[str]:
        expr = self.parse(text)
        if expr is None:
            return None
        try:
            if isinstance(expr, sp.Equality):
                lhs = sp.simplify(expr.lhs)
                rhs = sp.simplify(expr.rhs)
                return str(sp.Eq(lhs, rhs))
            return str(sp.simplify(expr))
        except Exception:
            return None

    def equivalent(self, left: str, right: str) -> bool:
        if sp is None or _parse_expr is None:
            return False
        left_expr = self.parse(left)
        right_expr = self.parse(right)
        if left_expr is None or right_expr is None:
            return False
        try:
            if isinstance(left_expr, sp.Equality) and isinstance(right_expr, sp.Equality):
                return sp.simplify((left_expr.lhs - left_expr.rhs) - (right_expr.lhs - right_expr.rhs)) == 0
            if isinstance(left_expr, sp.Equality):
                return sp.simplify(left_expr.lhs - left_expr.rhs - right_expr) == 0
            if isinstance(right_expr, sp.Equality):
                return sp.simplify(left_expr - (right_expr.lhs - right_expr.rhs)) == 0
            return sp.simplify(left_expr - right_expr) == 0
        except Exception:
            return False

    def evaluate(self, text: str, substitutions: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        if sp is None or _parse_expr is None:
            return None
        expr = self.parse(text)
        if expr is None:
            return None
        subs = self._sympy_substitutions(substitutions)
        try:
            if isinstance(expr, sp.Equality):
                lhs = sp.simplify(expr.lhs.subs(subs))
                rhs = sp.simplify(expr.rhs.subs(subs))
                return sp.simplify(lhs - rhs)
            return sp.simplify(expr.subs(subs))
        except Exception:
            return None

    def solve(self, equation_text: str, symbol: Optional[str] = None) -> List[str]:
        if sp is None or _parse_expr is None:
            return []
        expr = self.parse(equation_text)
        if expr is None:
            return []
        try:
            if isinstance(expr, sp.Equality):
                lhs = expr.lhs
                rhs = expr.rhs
                target = sp.Symbol(symbol) if symbol else self._first_symbol(lhs, rhs)
                if target is None:
                    return []
                sol = sp.solve(sp.Eq(lhs, rhs), target)
            else:
                target = sp.Symbol(symbol) if symbol else self._first_symbol(expr)
                if target is None:
                    return []
                sol = sp.solve(expr, target)
            if not isinstance(sol, (list, tuple)):
                sol = [sol]
            return [str(item) for item in sol]
        except Exception:
            return []

    def feedback(self, text: str, target_text: Optional[str] = None, substitutions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        parsed = self.parse(text)
        parseable = parsed is not None
        simplified = self.simplify(text) if parseable else None
        equivalent = self.equivalent(text, target_text) if target_text else False
        evaluated = self.evaluate(text, substitutions=substitutions) if substitutions else None
        return {
            "parseable": parseable,
            "equivalent": equivalent,
            "simplified": simplified,
            "evaluated": str(evaluated) if evaluated is not None else None,
            "symbolic_score": float(parseable) + (0.5 if equivalent else 0.0),
        }

    def act(self, token: int, context: Any = None):
        return self._text_adapter.act(token, context=context)

    def _parse_side(self, text: str):
        canonical = text.replace("^", "**")
        return _parse_expr(canonical, evaluate=True)

    def _split_equation(self, text: str) -> tuple[str, Optional[str]]:
        if "=" not in text or "==" in text:
            return text, None
        left, right = text.split("=", 1)
        return left.strip(), right.strip()

    def _first_symbol(self, *exprs):
        for expr in exprs:
            free = list(getattr(expr, "free_symbols", []))
            if free:
                return free[0]
        return None

    def _sympy_substitutions(self, substitutions: Optional[Dict[str, Any]]) -> Dict[Any, Any]:
        subs: Dict[Any, Any] = {}
        if not substitutions:
            return subs
        for key, value in substitutions.items():
            try:
                sym = sp.Symbol(str(key)) if sp is not None else str(key)
                subs[sym] = value
            except Exception:
                continue
        return subs


class CodeDSLAdapter(InputAdapter):
    """Canonical adapter for a tiny stack-machine DSL.

    The DSL is intentionally small and exact:
        PUSH <int>
        ADD | SUB | MUL | DIV | DUP | SWAP | POP
        RETURN
    """

    INSTRUCTIONS = {"PUSH", "ADD", "SUB", "MUL", "DIV", "DUP", "SWAP", "POP", "RETURN"}

    def __init__(self, obs_dim: int = 256):
        self._obs_dim = max(2, int(obs_dim))
        self._text_adapter = TextAdapter()

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_text(self, raw_input: Any) -> str:
        if isinstance(raw_input, str):
            program = raw_input
        elif isinstance(raw_input, dict) and "program" in raw_input:
            program = str(raw_input["program"])
        else:
            program = str(raw_input)
        parsed = self.parse_program(program)
        if not parsed:
            return self._canonicalize_fallback(program)
        return "\n".join(self._format_instruction(op, arg) for op, arg in parsed)

    def from_text(self, text: str) -> List[tuple[str, Optional[int]]]:
        return self.parse_program(text)

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        canonical = self.to_text(raw_input)
        return self._text_adapter.to_observations(canonical, max_length=max_length)

    def from_observations(self, tokens: Sequence[int]) -> str:
        text = "".join(self._text_adapter.reverse_vocab.get(int(tok) % 256, "?") for tok in tokens)
        return self.to_text(text)

    def parse_program(self, program: str) -> List[tuple[str, Optional[int]]]:
        instructions: List[tuple[str, Optional[int]]] = []
        for raw_line in program.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if not parts:
                continue
            op = parts[0].upper()
            if op not in self.INSTRUCTIONS:
                continue
            arg: Optional[int] = None
            if op == "PUSH":
                if len(parts) < 2:
                    continue
                try:
                    arg = int(parts[1])
                except Exception:
                    continue
            instructions.append((op, arg))
        return instructions

    def execute(self, program: str) -> Optional[int]:
        stack: List[int] = []
        parsed = self.parse_program(program)
        if not parsed:
            return None
        for op, arg in parsed:
            if op == "PUSH":
                stack.append(int(arg or 0))
            elif op == "ADD" and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
            elif op == "SUB" and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
            elif op == "MUL" and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b)
            elif op == "DIV" and len(stack) >= 2:
                b = stack.pop()
                a = stack.pop()
                stack.append(0 if b == 0 else int(a / b))
            elif op == "DUP" and stack:
                stack.append(stack[-1])
            elif op == "SWAP" and len(stack) >= 2:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            elif op == "POP" and stack:
                stack.pop()
            elif op == "RETURN":
                break
        return stack[-1] if stack else None

    def act(self, token: int, context: Any = None):
        return self._text_adapter.reverse_vocab.get(int(token) % 256, chr(int(token) % 95 + 32))

    def _format_instruction(self, op: str, arg: Optional[int]) -> str:
        if op == "PUSH":
            return f"PUSH {int(arg or 0)}"
        return op

    def _canonicalize_fallback(self, program: str) -> str:
        lines = []
        for line in program.splitlines():
            line = line.strip()
            if not line:
                continue
            lines.append(line.upper())
        return "\n".join(lines)

class ContinuousSignalAdapter(InputAdapter):
    """Convert time series data to discrete tokens via binning."""
    def __init__(self, num_bins=10, low=-1.0, high=1.0):
        self.num_bins = num_bins
        self.bins = np.linspace(low, high, num_bins + 1)

    @property
    def obs_dim(self) -> int:
        return self.num_bins

    def to_observations(self, signal_array, max_length=100) -> List[int]:
        inds = np.digitize(signal_array, self.bins) - 1
        # Clip to [0, num_bins-1]
        inds = np.clip(inds, 0, self.num_bins - 1)
        return inds[:max_length].tolist()

class DiscreteInputAdapter(InputAdapter):
    """Simple pass-through for already discretized integer tokens."""
    def __init__(self, obs_dim=2):
        self._obs_dim = obs_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        if isinstance(raw_input, int):
            return [raw_input % self._obs_dim]
        elif isinstance(raw_input, (list, np.ndarray)):
            return [int(x) % self._obs_dim for x in raw_input][:max_length]
        else:
            return [0]

class OutputAdapter:
    """Base class for converting internal pattern output to external action."""
    def act(self, token: int, context: Any) -> Any:
        raise NotImplementedError

class MotorAdapter(OutputAdapter):
    """Interpret tokens as basic directions (0:Left, 1:Right, 2:Up, 3:Down)."""
    def act(self, token: int, context: Any = None):
        mapping = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}
        dx, dy = mapping.get(token, (0, 0))
        return dx, dy

class ConsoleOutputAdapter(OutputAdapter):
    """Simple adapter that prints the prediction to the console."""
    def act(self, token: int, context: Any = None):
        print(f"[HPM Action] Predicted next observation: {token}")
        return token

class VisualisationAdapter(OutputAdapter):
    """Render a pattern's transition matrices into an image file."""
    def act(self, pattern, path="pattern_viz.png"):
        if pattern.complexity < 2:
            print("Visualisation: Pattern is too simple (flat).")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Plot top and lower level matrices
        axes[0, 0].imshow(pattern.A3, cmap='Blues')
        axes[0, 0].set_title('A3 (top-level transition)')
        
        axes[0, 1].imshow(pattern.A32, cmap='Blues')
        axes[0, 1].set_title('A32 (z3 -> z2)')
        
        axes[1, 0].imshow(pattern.A21, cmap='Blues')
        axes[1, 0].set_title('A21 (z2 -> z1)')
        
        axes[1, 1].imshow(pattern.B, cmap='Greens')
        axes[1, 1].set_title('B (Emission)')
        
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

class CharClassAdapter:
    """
    Maps 95 printable ASCII character IDs (ord(ch)-32 for ch in range(32,127))
    to 5 coarse character classes, reducing obs_dim from 95 to 5.

    Classes:
        0 = letter      (A-Z: IDs 33-58, a-z: IDs 65-90)
        1 = digit       (0-9: IDs 16-25)
        2 = space       (ID 0)
        3 = punctuation (all other printable ASCII)
        4 = newline     (ord('\n')-32 = -22, passed as sentinel)
    """

    CLASS_NAMES = ['letter', 'digit', 'space', 'punctuation', 'newline']
    NEWLINE_ID = -22  # ord('\n') - 32

    def __init__(self):
        # Build lookup table for IDs 0-94
        self._table = {}
        for char_id in range(95):
            ch = chr(char_id + 32)
            if ch == ' ':
                self._table[char_id] = 2
            elif ch.isdigit():
                self._table[char_id] = 1
            elif ch.isalpha():
                self._table[char_id] = 0
            else:
                self._table[char_id] = 3

    @property
    def obs_dim(self) -> int:
        return 5

    def encode(self, char_id: int) -> int:
        """Map a character ID (ord(ch)-32) to a class ID 0-4."""
        if char_id == self.NEWLINE_ID:
            return 4
        return self._table.get(char_id, 3)  # default to punctuation

    def decode_class(self, class_id: int) -> str:
        """Return the name of the character class."""
        if class_id < 0 or class_id >= len(self.CLASS_NAMES):
            raise ValueError(f"class_id {class_id} out of range [0, 4]")
        return self.CLASS_NAMES[class_id]

    def encode_char(self, ch: str) -> int:
        """Map a raw character to a class ID 0-4."""
        if ch == '\n':
            return 4
        char_id = ord(ch) - 32
        return self.encode(char_id)


class AsciiCharAdapter:
    """
    Surface adapter that preserves printable ASCII distinctions.

    This expands the observation space from 5 coarse character classes to 95
    printable ASCII slots plus the newline sentinel, while still exposing the
    coarse bucket labels used by the decoders as a fallback prior.
    """

    CLASS_NAMES = ['letter', 'digit', 'space', 'punctuation', 'newline']
    NEWLINE_ID = 94

    def __init__(self):
        self._table = {}
        for char_id in range(95):
            ch = chr(char_id + 32)
            if ch == ' ':
                self._table[char_id] = char_id
            else:
                self._table[char_id] = char_id

    @property
    def obs_dim(self) -> int:
        return 95

    def encode(self, char_id: int) -> int:
        if char_id == self.NEWLINE_ID:
            return self.NEWLINE_ID
        return self._table.get(int(char_id), self.NEWLINE_ID)

    def encode_char(self, ch: str) -> int:
        if ch == '\n':
            return self.NEWLINE_ID
        char_id = ord(ch) - 32
        if 0 <= char_id < 95:
            return self.encode(char_id)
        return self.NEWLINE_ID

    def decode_class(self, class_id: int) -> str:
        """Return a coarse bucket name for compatibility with old decoders."""
        if class_id == self.NEWLINE_ID:
            return 'newline'
        ch = chr(int(class_id) + 32)
        if ch == ' ':
            return 'space'
        if ch.isdigit():
            return 'digit'
        if ch.isalpha():
            return 'letter'
        return 'punctuation'

    def bucket_for_token(self, token_id: int) -> str:
        return self.decode_class(int(token_id))


class EnvironmentStateAdapter(InputAdapter):
    """Encode structured environment state into discrete observation tokens."""

    def __init__(self, obs_dim: int = 8, keys: Optional[Sequence[str]] = None):
        self._obs_dim = max(2, int(obs_dim))
        self.keys = tuple(keys or ("state", "obs", "observation", "value", "family", "phase", "reward"))

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        tokens: List[int] = []
        self._append_tokens(tokens, raw_input)
        if not tokens:
            tokens = [0]
        return tokens[:max_length]

    def _append_tokens(self, tokens: List[int], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (bool, np.bool_)):
            tokens.append(int(value) % self._obs_dim)
            return
        if isinstance(value, (int, np.integer)):
            tokens.append(int(value) % self._obs_dim)
            return
        if isinstance(value, (float, np.floating)):
            tokens.append(int(round(float(value))) % self._obs_dim)
            return
        if isinstance(value, str):
            tokens.append(self._string_code(value))
            return
        if isinstance(value, dict):
            ordered = [key for key in self.keys if key in value]
            ordered.extend(key for key in value.keys() if key not in ordered)
            for key in ordered:
                self._append_tokens(tokens, value.get(key))
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                self._append_tokens(tokens, item)
            return
        tokens.append(self._string_code(repr(value)))

    def _string_code(self, text: str) -> int:
        text = text.strip()
        if not text:
            return 0
        if text.isdigit():
            return int(text) % self._obs_dim
        return sum(ord(ch) for ch in text) % self._obs_dim


class ToolActionAdapter(InputAdapter, OutputAdapter):
    """Round-trip adapter for tool-family names and tool action tokens."""

    def __init__(self, actions: Optional[Sequence[str]] = None):
        self.actions = list(actions or ("inspect", "shift", "flip", "commit"))
        self._obs_dim = max(2, len(self.actions))
        self._name_to_id = {name: idx for idx, name in enumerate(self.actions)}

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        tokens: List[int] = []
        if isinstance(raw_input, dict):
            for key in ("action", "tool", "name", "sequence"):
                if key in raw_input:
                    self._append(tokens, raw_input[key])
        else:
            self._append(tokens, raw_input)
        if not tokens:
            tokens = [0]
        return tokens[:max_length]

    def act(self, token: int, context: Any = None):
        if not self.actions:
            return int(token)
        return self.actions[int(token) % len(self.actions)]

    def _append(self, tokens: List[int], value: Any) -> None:
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                self._append(tokens, item)
            return
        if isinstance(value, (int, np.integer)):
            tokens.append(int(value) % self._obs_dim)
            return
        if isinstance(value, str):
            if value in self._name_to_id:
                tokens.append(self._name_to_id[value])
            else:
                tokens.append(sum(ord(ch) for ch in value) % self._obs_dim)
            return
        if value is not None:
            tokens.append(sum(ord(ch) for ch in repr(value)) % self._obs_dim)


class CurriculumAdapter(InputAdapter, OutputAdapter):
    """Adapter for task-family and curriculum metadata."""

    def __init__(self, families: Optional[Sequence[str]] = None, obs_dim: int = 16):
        self.families = list(families or [])
        self._obs_dim = max(2, int(obs_dim))
        self._family_to_id = {name: idx for idx, name in enumerate(self.families)}

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        tokens: List[int] = []
        if isinstance(raw_input, dict):
            for key in ("family", "phase", "stage", "task_family", "difficulty"):
                if key in raw_input:
                    self._append(tokens, raw_input[key])
        else:
            self._append(tokens, raw_input)
        if not tokens:
            tokens = [0]
        return tokens[:max_length]

    def act(self, token: int, context: Any = None):
        if self.families:
            return self.families[int(token) % len(self.families)]
        return f"family_{int(token) % self._obs_dim}"

    def _append(self, tokens: List[int], value: Any) -> None:
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                self._append(tokens, item)
            return
        if isinstance(value, (int, np.integer)):
            tokens.append(int(value) % self._obs_dim)
            return
        if isinstance(value, str):
            if value in self._family_to_id:
                tokens.append(self._family_to_id[value])
            else:
                tokens.append(sum(ord(ch) for ch in value) % self._obs_dim)
            return
        if isinstance(value, (float, np.floating)):
            tokens.append(int(round(float(value))) % self._obs_dim)
            return
        if value is not None:
            tokens.append(sum(ord(ch) for ch in repr(value)) % self._obs_dim)


class EpisodeBundleAdapter(InputAdapter, OutputAdapter):
    """Encode bundle descriptors and reload metadata into discrete tokens."""

    def __init__(self, obs_dim: int = 32):
        self._obs_dim = max(2, int(obs_dim))

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def to_observations(self, raw_input: Any, max_length: int = 100) -> List[int]:
        tokens: List[int] = []
        self._append(tokens, raw_input)
        if not tokens:
            tokens = [0]
        return tokens[:max_length]

    def act(self, token: int, context: Any = None):
        return self.decode_token(token)

    def encode_bundle(self, descriptor: Dict[str, Any]) -> List[int]:
        return self.to_observations(descriptor)

    def decode_bundle(self, tokens: Sequence[int]) -> Dict[str, Any]:
        tokens = [int(t) % self._obs_dim for t in tokens]
        return {
            "kind": self.decode_token(tokens[0]) if tokens else "bundle",
            "phase": self.decode_token(tokens[1]) if len(tokens) > 1 else "unknown",
            "obs_dim": int(tokens[2]) if len(tokens) > 2 else self._obs_dim,
            "count": int(tokens[3]) if len(tokens) > 3 else len(tokens),
        }

    def decode_token(self, token: int) -> str:
        token = int(token) % self._obs_dim
        if token == 0:
            return "bundle"
        if token == 1:
            return "save"
        if token == 2:
            return "load"
        if token == 3:
            return "train"
        if token == 4:
            return "validation"
        return f"bundle_{token}"

    def _append(self, tokens: List[int], value: Any) -> None:
        if isinstance(value, dict):
            for key in ("kind", "phase", "level", "state", "mode"):
                if key in value:
                    self._append(tokens, value[key])
            for key in sorted(value.keys()):
                if key not in {"kind", "phase", "level", "state", "mode"}:
                    self._append(tokens, value[key])
            return
        if isinstance(value, (list, tuple, np.ndarray)):
            for item in value:
                self._append(tokens, item)
            return
        if isinstance(value, (int, np.integer)):
            tokens.append(int(value) % self._obs_dim)
            return
        if isinstance(value, (float, np.floating)):
            tokens.append(int(round(float(value))) % self._obs_dim)
            return
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"bundle", "save", "load", "train", "validation"}:
                tokens.append({"bundle": 0, "save": 1, "load": 2, "train": 3, "validation": 4}[lowered])
            else:
                tokens.append(sum(ord(ch) for ch in value) % self._obs_dim)
            return
        if value is not None:
            tokens.append(sum(ord(ch) for ch in repr(value)) % self._obs_dim)
