import sys, os
sys.path.append(os.path.abspath("hpm_ai_v3"))

import torch
import numpy as np
from PIL import Image
from hpm_ai_v3.tools.registry import ToolRegistry
from hpm_ai_v3.tools.perception import register_perception_tools

def main():
    print("Testing Perception Tools...")
    # Tools are registered on import of perception_tools, but let's be explicit
    # register_perception_tools() 
    
    tools = ToolRegistry.list_tools()
    print(f"Registered tools: {tools}")
    
    assert "extract_features" in tools
    assert "classify_image" in tools
    assert "embed_text" in tools
    
    # Test text embedding (fast)
    embed_tool = ToolRegistry.create_pattern("embed_text")
    out = embed_tool.sample({"text": "Hello HPM!"})
    print(f"Text embedding shape: {out['embedding'].shape}")
    
    # Test sentiment
    sentiment_tool = ToolRegistry.create_pattern("sentiment")
    out = sentiment_tool.sample({"text": "This is amazing!"})
    print(f"Sentiment: {out['sentiment']}")
    
    print("Perception tools test passed.")

if __name__ == "__main__":
    main()
