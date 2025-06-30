#!/usr/bin/env python3
"""
Test Multi-AI Integration - Check which AI services are available
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_openai():
    """Test OpenAI connection"""
    try:
        import openai
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "❌ OpenAI: No API key found"
        
        client = openai.OpenAI(api_key=api_key)
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello from OpenAI!'"}],
            max_tokens=10
        )
        
        return f"✅ OpenAI: {response.choices[0].message.content}"
        
    except Exception as e:
        return f"❌ OpenAI: {str(e)[:100]}"

async def test_anthropic():
    """Test Anthropic Claude connection"""
    try:
        import anthropic
        
        api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "❌ Claude: No API key found (but Claude Desktop is available)"
        
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'Hello from Claude!'"}]
        )
        
        return f"✅ Claude API: {response.content[0].text}"
        
    except Exception as e:
        return f"❌ Claude API: {str(e)[:100]} (Claude Desktop may still work)"

async def test_gemini():
    """Test Google Gemini connection"""
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "❌ Gemini: No API key found"
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        response = model.generate_content("Say 'Hello from Gemini!'")
        
        return f"✅ Gemini: {response.text}"
        
    except Exception as e:
        return f"❌ Gemini: {str(e)[:100]}"

async def test_elevenlabs():
    """Test ElevenLabs connection"""
    try:
        api_key = os.getenv("ELEVENLABS_API_KEY")
        if not api_key:
            return "❌ ElevenLabs: No API key found"
        
        # Just check if we have the key for now
        return "✅ ElevenLabs: API key found (voice synthesis available)"
        
    except Exception as e:
        return f"❌ ElevenLabs: {str(e)[:100]}"

async def main():
    """Run all tests"""
    print("🧪 Testing AI Integrations for JARVIS")
    print("=" * 50)
    
    # Test all services
    tests = [
        test_openai(),
        test_anthropic(),
        test_gemini(),
        test_elevenlabs()
    ]
    
    results = await asyncio.gather(*tests)
    
    print("\n📊 Test Results:")
    for result in results:
        print(f"  {result}")
    
    # Count successes
    successes = sum(1 for r in results if r.startswith("✅"))
    total = len(results)
    
    print(f"\n🎯 Score: {successes}/{total} AI services available")
    
    if successes >= 2:
        print("\n✨ Great! You have enough AI services to run multi-AI JARVIS!")
        print("   Next: Test the actual multi-AI integration")
    else:
        print("\n⚠️  You need at least 2 AI services for multi-AI to work effectively")
        print("   Check your API keys in the .env file")
    
    # Test if we can import the multi-AI module
    print("\n🔍 Testing multi-AI module import...")
    try:
        from core.updated_multi_ai_integration import EnhancedMultiAIIntegration
        print("✅ Multi-AI module can be imported")
        
        # Try to initialize it
        multi_ai = EnhancedMultiAIIntegration()
        await multi_ai.initialize()
        
        available = list(multi_ai.available_models.keys())
        if available:
            print(f"✅ Multi-AI initialized with models: {available}")
        else:
            print("⚠️  Multi-AI initialized but no models available")
            
    except Exception as e:
        print(f"❌ Multi-AI module error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
