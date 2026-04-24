import numpy as np
from PIL import Image
from typing import List, Any, Union, Optional
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
