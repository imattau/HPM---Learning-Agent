import json
import numpy as np
from PIL import Image
from typing import List, Any, Union, Optional, Sequence, Dict
import matplotlib.pyplot as plt

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
