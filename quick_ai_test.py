#!/usr/bin/env python3
"""Quick test of enhanced JARVIS"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.working_multi_ai import multi_ai

async def quick_test():
    print("🧪 Quick JARVIS Multi-AI Test")
    print("=" * 40)
    
    # Initialize
    print("\n1️⃣ Initializing AI services...")
    success = await multi_ai.initialize()
    
    if not success:
        print("❌ No AI services available")
        return
        
    models = list(multi_ai.available_models.keys())
    print(f"✅ Connected to: {models}")
    
    # Test a query
    print("\n2️⃣ Testing AI response...")
    response = await multi_ai.query(
        "Hello! I am JARVIS. Please respond with a brief greeting.",
        preferred_model=models[0] if models else None
    )
    
    if response.get("success"):
        print(f"✅ Response from {response.get('model')}:")
        print(f"   '{response['response']}'")
    else:
        print(f"❌ Error: {response.get('error')}")
    
    # Test all models if multiple available
    if len(models) > 1:
        print("\n3️⃣ Testing all models...")
        results = await multi_ai.query_all("What is 2+2?")
        
        for model, result in results.items():
            if result.get("success"):
                print(f"✅ {model}: {result['response'][:50]}...")
            else:
                print(f"❌ {model}: Failed")
    
    print("\n✨ Test complete! Your JARVIS can now use AI!")

if __name__ == "__main__":
    asyncio.run(quick_test())
