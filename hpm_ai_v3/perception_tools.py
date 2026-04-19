"""
perception_tools.py - Vision and language tools for HPM agents.
Uses torchvision, transformers, and PIL. Falls back gracefully if libraries missing.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union
from PIL import Image
import warnings
import os

from .tool_registry import ToolRegistry

# ----------------------------------------------------------------------
# Helper: Get device (respect HPM's global device if set)
# ----------------------------------------------------------------------
def _get_device() -> torch.device:
    try:
        from .pattern import HPMPattern
        if HPMPattern._global_device is not None:
            return HPMPattern._global_device
    except:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# Vision Tools
# ----------------------------------------------------------------------

# Cache models to avoid reloading
_vision_models = {}


def _load_resnet18():
    if "resnet18" not in _vision_models:
        try:
            from torchvision import models, transforms
            # Use Weights enum as pretrained=True is deprecated
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            model.eval()
            # Remove final classification layer for feature extraction
            feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
            feature_extractor.to(_get_device())
            
            preprocess = weights.transforms()
            _vision_models["resnet18"] = (feature_extractor, preprocess)
        except ImportError:
            warnings.warn("torchvision not installed. Vision tools unavailable.")
            _vision_models["resnet18"] = (None, None)
    return _vision_models["resnet18"]


def extract_resnet_features(image: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> torch.Tensor:
    """
    Extract 512-dim features from an image using ResNet18.
    """
    model, preprocess = _load_resnet18()
    if model is None:
        raise RuntimeError("ResNet18 not available. Install torchvision.")
    
    # Load image if path/URL
    if isinstance(image, str):
        if image.startswith("http"):
            import requests
            from io import BytesIO
            response = requests.get(image, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
    elif isinstance(image, torch.Tensor):
        from torchvision.transforms import ToPILImage
        if image.dim() == 4: # Batch
            image = image[0]
        if image.dim() == 3:
            # Check if (C, H, W) or (H, W, C)
            if image.shape[0] in [1, 3]: # likely (C, H, W)
                img = ToPILImage()(image.cpu())
            else:
                img = Image.fromarray((image.cpu().numpy() * 255).astype('uint8')).convert("RGB")
        else:
            raise ValueError(f"Tensor must be 3D (C,H,W), got {image.shape}")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[0] in [1, 3]: # (C, H, W)
                image = image.transpose(1, 2, 0)
            if image.max() <= 1.0:
                image = (image * 255).astype('uint8')
            img = Image.fromarray(image.astype('uint8')).convert("RGB")
        else:
            img = Image.fromarray(image.astype('uint8')).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    img_tensor = preprocess(img).unsqueeze(0).to(_get_device())
    with torch.no_grad():
        features = model(img_tensor).squeeze()
    return features.cpu()


def classify_image_resnet(image: Union[str, np.ndarray, torch.Tensor, Image.Image]) -> Dict[str, Any]:
    """
    Classify image using ResNet50 (1000 ImageNet classes).
    """
    if "resnet50_classifier" not in _vision_models:
        try:
            from torchvision import models, transforms
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
            model.eval()
            model.to(_get_device())
            
            preprocess = weights.transforms()
            classes = weights.meta["categories"]
            _vision_models["resnet50_classifier"] = (model, preprocess, classes)
        except ImportError:
            _vision_models["resnet50_classifier"] = (None, None, [])
    
    model, preprocess, classes = _vision_models["resnet50_classifier"]
    if model is None:
        raise RuntimeError("ResNet50 classifier not available.")
    
    # Load image
    if isinstance(image, str):
        if image.startswith("http"):
            import requests
            from io import BytesIO
            response = requests.get(image, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image.astype('uint8')).convert("RGB")
    elif isinstance(image, torch.Tensor):
        from torchvision.transforms import ToPILImage
        img = ToPILImage()(image.cpu())
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    
    img_tensor = preprocess(img).unsqueeze(0).to(_get_device())
    with torch.no_grad():
        logits = model(img_tensor).squeeze()
        probs = torch.softmax(logits, dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)
    
    results = []
    for i in range(5):
        idx = top5_idx[i].item()
        results.append({
            "class_id": idx,
            "class_name": classes[idx] if idx < len(classes) else f"class_{idx}",
            "probability": top5_prob[i].item()
        })
    return {"predictions": results}


# ----------------------------------------------------------------------
# Text Tools
# ----------------------------------------------------------------------

_text_models = {}


def _load_sentence_transformer():
    if "sentence_transformer" not in _text_models:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.to(_get_device())
            _text_models["sentence_transformer"] = model
        except ImportError:
            warnings.warn("sentence-transformers not installed. Text embedding unavailable.")
            _text_models["sentence_transformer"] = None
    return _text_models["sentence_transformer"]


def embed_text(text: Union[str, List[str]]) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Convert text to 384-dim embedding using all-MiniLM-L6-v2.
    """
    model = _load_sentence_transformer()
    if model is None:
        raise RuntimeError("Sentence transformer not available.")
    
    is_single = isinstance(text, str)
    texts = [text] if is_single else text
    embeddings = model.encode(texts, convert_to_tensor=True, device=_get_device())
    if is_single:
        return embeddings[0].cpu()
    return [emb.cpu() for emb in embeddings]


def tokenize_text(text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """
    Tokenize text using BERT tokenizer.
    """
    if "bert_tokenizer" not in _text_models:
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            _text_models["bert_tokenizer"] = tokenizer
        except ImportError:
            warnings.warn("transformers not installed. Tokenization unavailable.")
            _text_models["bert_tokenizer"] = None
    tokenizer = _text_models["bert_tokenizer"]
    if tokenizer is None:
        raise RuntimeError("BERT tokenizer not available.")
    
    encoded = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    return {k: v.squeeze(0) for k, v in encoded.items()}


def sentiment_analysis(text: str) -> Dict[str, float]:
    """
    Simple sentiment analysis using TextBlob (fallback).
    """
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        return {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity
        }
    except ImportError:
        # Fallback: use a basic rule
        positive_words = ['good', 'great', 'excellent', 'love', 'happy']
        negative_words = ['bad', 'terrible', 'hate', 'sad', 'awful']
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        total = pos + neg + 1e-6
        polarity = (pos - neg) / total
        subjectivity = total / (len(text_lower.split()) + 1e-6)
        return {"polarity": polarity, "subjectivity": subjectivity}


# ----------------------------------------------------------------------
# Register Tools
# ----------------------------------------------------------------------
def register_perception_tools():
    """Register all perception tools with ToolRegistry."""
    
    # Vision
    ToolRegistry.register(
        name="extract_features",
        tool_fn=extract_resnet_features,
        input_keys=["image"],
        output_key="features",
        cost=0.05,
        description="Extract 512-dim ResNet18 features from an image."
    )
    
    ToolRegistry.register(
        name="classify_image",
        tool_fn=classify_image_resnet,
        input_keys=["image"],
        output_key="classification",
        cost=0.08,
        description="Classify image into 1000 ImageNet categories."
    )
    
    # Text
    ToolRegistry.register(
        name="embed_text",
        tool_fn=embed_text,
        input_keys=["text"],
        output_key="embedding",
        cost=0.02,
        description="Convert text to 384-dim sentence embedding."
    )
    
    ToolRegistry.register(
        name="tokenize_text",
        tool_fn=tokenize_text,
        input_keys=["text"],
        output_key="tokens",
        cost=0.01,
        description="Tokenize text using BERT tokenizer."
    )
    
    ToolRegistry.register(
        name="sentiment",
        tool_fn=sentiment_analysis,
        input_keys=["text"],
        output_key="sentiment",
        cost=0.01,
        description="Analyze sentiment polarity and subjectivity."
    )
    
    print(f"[PerceptionTools] Registered {len(ToolRegistry.list_tools())} tools.")


# Auto-register on import
register_perception_tools()
