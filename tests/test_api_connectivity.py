#!/usr/bin/env python3
"""
Test actual AI API connectivity
"""
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_apis():
    """Test actual API connections"""
    print("🧪 Testing Actual AI API Connections\n")
    
    # Test OpenAI
    print("1️⃣ Testing OpenAI GPT-4...")
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using 3.5 for testing
            messages=[{"role": "user", "content": "Say 'Hello from GPT!' in 5 words or less"}],
            max_tokens=20
        )
        print(f"✅ GPT Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ OpenAI Error: {str(e)[:100]}...")
    
    # Test Gemini
    print("\n2️⃣ Testing Google Gemini...")
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Say 'Hello from Gemini!' in 5 words or less")
        print(f"✅ Gemini Response: {response.text}")
    except Exception as e:
        print(f"❌ Gemini Error: {str(e)[:100]}...")
    
    # Test ElevenLabs
    print("\n3️⃣ Testing ElevenLabs Voice...")
    try:
        from elevenlabs import client as elevenlabs_client
        client = elevenlabs_client.ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        
        # Just test that we can connect
        voices = client.voices.get_all()
        print(f"✅ ElevenLabs: Found {len(voices.voices)} voices available")
    except Exception as e:
        print(f"❌ ElevenLabs Error: {str(e)[:100]}...")
    
    print("\n📊 Summary:")
    print("- If you see ✅ responses above, the APIs are working!")
    print("- If you see ❌ errors, check your API keys in .env")
    print("- Claude Desktop works through MCP (restart Claude to see)")

if __name__ == "__main__":
    asyncio.run(test_apis())
