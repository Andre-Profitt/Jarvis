#!/usr/bin/env python3
"""
Comprehensive AI Integration Test for JARVIS
"""
import os
import asyncio
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def test_all_ais():
    """Test all AI integrations comprehensively"""
    print("🚀 JARVIS Multi-AI Integration Test\n")
    print("="*50 + "\n")
    
    results = {}
    
    # Test OpenAI
    print("1️⃣ Testing OpenAI GPT-4...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Test with GPT-3.5 first
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are JARVIS, an AI assistant."},
                     {"role": "user", "content": "Introduce yourself in one sentence."}],
            max_tokens=50
        )
        gpt_response = response.choices[0].message.content
        print(f"✅ GPT-3.5 Response: {gpt_response}")
        results["openai"] = True
        
        # Try GPT-4 if available
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Say 'GPT-4 active' if you're GPT-4"}],
                max_tokens=10
            )
            print(f"✅ GPT-4 Available: {response.choices[0].message.content}")
        except:
            print("⚠️  GPT-4 not available (check subscription)")
            
    except Exception as e:
        print(f"❌ OpenAI Error: {str(e)[:150]}...")
        results["openai"] = False
    
    # Test Gemini with correct model
    print("\n2️⃣ Testing Google Gemini...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        # List available models first
        models = genai.list_models()
        available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
        print(f"Available Gemini models: {available_models[:3]}...")
        
        # Use the correct model
        model = genai.GenerativeModel('gemini-1.5-flash')  # Changed from gemini-pro
        response = model.generate_content("You are JARVIS. Introduce yourself in one sentence.")
        print(f"✅ Gemini Response: {response.text}")
        results["gemini"] = True
        
    except Exception as e:
        print(f"❌ Gemini Error: {str(e)[:150]}...")
        results["gemini"] = False
    
    # Test ElevenLabs
    print("\n3️⃣ Testing ElevenLabs Voice Synthesis...")
    try:
        from elevenlabs import ElevenLabs
        client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
        # Get voices
        voices = client.voices.get_all()
        print(f"✅ Found {len(voices.voices)} voices")
        
        # Show first 3 voices
        for i, voice in enumerate(voices.voices[:3]):
            print(f"   • {voice.name} ({voice.voice_id})")
        
        results["elevenlabs"] = True
        
    except Exception as e:
        print(f"❌ ElevenLabs Error: {str(e)[:150]}...")
        results["elevenlabs"] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 INTEGRATION SUMMARY")
    print("="*50)
    
    working = sum(results.values())
    total = len(results)
    
    for service, status in results.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {service.title()}: {'Working' if status else 'Failed'}")
    
    print(f"\n🎯 Overall: {working}/{total} services operational")
    
    if working == total:
        print("🎉 All AI services are working perfectly!")
        print("\n🚀 JARVIS is ready for full multi-AI operation!")
    else:
        print("\n⚠️  Some services need attention")
    
    print("\n💡 Next Steps:")
    print("1. Restart Claude Desktop to activate MCP")
    print("2. Run: python3 test_multi_ai.py")
    print("3. Launch full JARVIS: python3 LAUNCH-JARVIS-REAL.py")
    
    return results

async def test_multi_ai_integration():
    """Test the actual multi-AI integration module"""
    print("\n\n🧪 Testing Multi-AI Integration Module...")
    print("="*50 + "\n")
    
    try:
        # Import after fixing the module
        from core.updated_multi_ai_integration import multi_ai
        
        # Test each model
        models = multi_ai.get_available_models()
        print(f"Available models in config: {models}\n")
        
        test_prompt = "You are JARVIS. Say hello in exactly 5 words."
        
        for model in models:
            if model != "claude-desktop":  # Skip Claude Desktop for now
                print(f"Testing {model}...")
                try:
                    response = await multi_ai.query(test_prompt, model=model)
                    print(f"✅ {model}: {response}\n")
                except Exception as e:
                    print(f"❌ {model}: {str(e)[:100]}\n")
        
        # Test fallback
        print("Testing fallback mechanism...")
        response = await multi_ai.query(test_prompt, model="nonexistent")
        print(f"Fallback result: {response}\n")
        
    except Exception as e:
        print(f"❌ Integration module error: {e}")

async def main():
    """Run all tests"""
    # Test individual APIs
    results = await test_all_ais()
    
    # If at least 2 services work, test integration
    if sum(results.values()) >= 2:
        await test_multi_ai_integration()

if __name__ == "__main__":
    asyncio.run(main())
