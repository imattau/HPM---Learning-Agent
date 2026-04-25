"""
example_vision_tool.py - Demonstrate HPM agent using vision tool.
"""

import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from hpm_ai_v3.tool_registry import ToolRegistry
from hpm_ai_v3.augmented_agent import AugmentedHPMAgent
from hpm_ai_v3.classification_pattern import ClassificationPattern
from data.permuted_mnist import get_permuted_mnist


# Register vision tool
def vision_encoder(image):
    """Extract features from image using pre-trained ResNet."""
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    
    # Ensure image is in right format for ResNet (3-channel, 224x224)
    if image.dim() == 2:
        image = image.unsqueeze(0)
    
    # Expand to 3 channels and resize
    # image is (1, H, W) -> (3, H, W)
    img_rgb = image.repeat(3, 1, 1).unsqueeze(0)
    img_resized = torch.nn.functional.interpolate(img_rgb, size=(224, 224), mode='bilinear', align_corners=False)
    
    model = models.resnet18(pretrained=True)
    model.eval()
    # Remove final classification layer
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    
    with torch.no_grad():
        features = model(img_resized)
    return features.squeeze()  # 512-dim vector

ToolRegistry.register(
    name="vision_resnet18",
    tool_fn=vision_encoder,
    input_keys=["image"],
    output_key="features",
    cost=0.05,
    description="ResNet18 feature extractor"
)


def run_augmented_agent():
    print("Initializing Augmented HPM Agent...")
    # Create agent with vision tool
    # The base patterns will take the 512-dim features as input
    agent = AugmentedHPMAgent(
        tool_names=["vision_resnet18"],
        base_patterns=[ClassificationPattern(input_dim=512, num_classes=10) for _ in range(3)],
        context_feature_dim=16
    )
    
    # Get MNIST data
    print("Loading MNIST data...")
    train_loader = get_permuted_mnist(task_id=0, batch_size=1, train=True)
    
    print("Starting training loop...")
    for i, (image, label) in enumerate(train_loader):
        # Raw input: 28x28 image tensor
        raw_input = {"image": image.squeeze(0)}  # Remove batch dim
        
        # Agent step
        context = agent.step(raw_input, target=label)
        
        if i % 10 == 0:
            selected_tool = agent.tool_usage_history[-1] if agent.tool_usage_history else 'None'
            print(f"Step {i}: Selected tool = {selected_tool}")
            
        if i >= 50:  # Short run for validation
            break
    
    # Show tool usage statistics
    print("\nTool usage distribution:")
    tool_counts = {}
    for tool in agent.tool_usage_history:
        tool_counts[tool] = tool_counts.get(tool, 0) + 1
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count} times")


if __name__ == "__main__":
    run_augmented_agent()
