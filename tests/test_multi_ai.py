#!/usr/bin/env python3
"""
Test Multi-AI Integration
"""
import asyncio
import logging
from core.updated_multi_ai_integration import multi_ai

logging.basicConfig(level=logging.INFO)

async def test_multi_ai():
    """Test all available AI models"""
    print("🧪 Testing Multi-AI Integration\n")
    
    # Get available models
    models = multi_ai.get_available_models()
    print(f"Available models: {models}\n")
    
    # Test prompt
    test_prompt = "What is the meaning of life? (Please answer in one sentence)"
    
    # Test each model
    for model in models:
        print(f"Testing {model}...")
        try:
            response = await multi_ai.query(test_prompt, model=model)
            print(f"✅ {model}: {response}\n")
        except Exception as e:
            print(f"❌ {model} error: {e}\n")
    
    # Test fallback
    print("Testing fallback mechanism...")
    response = await multi_ai.query(test_prompt, model="nonexistent-model")
    print(f"Fallback response: {response}\n")

if __name__ == "__main__":
    asyncio.run(test_multi_ai())
