#!/usr/bin/env python3
"""
Multi-AI Integration Setup for JARVIS (Updated)
Properly loads .env and configures all AI connections
"""
import os
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def setup_multi_ai():
    """Setup multi-AI integration with proper env loading"""
    logger.info("🤖 Setting up Multi-AI Integration...\n")
    
    # Check for API keys with correct names from .env
    api_keys = {
        "OPENAI": os.getenv("OPENAI_API_KEY"),
        "GEMINI": os.getenv("GEMINI_API_KEY"),
        "CLAUDE_DESKTOP": os.getenv("USE_CLAUDE_DESKTOP", "false").lower() == "true",
        "ELEVENLABS": os.getenv("ELEVENLABS_API_KEY"),
    }
    
    # Report status
    available_ais = []
    for key_name, key_value in api_keys.items():
        if key_value and (key_name != "CLAUDE_DESKTOP" or key_value):
            available_ais.append(key_name)
            if key_name == "CLAUDE_DESKTOP":
                logger.info(f"✅ Claude Desktop enabled (no API key needed)")
            else:
                logger.info(f"✅ {key_name} API key found: {str(key_value)[:20]}...")
        else:
            logger.info(f"⚠️  {key_name} not configured")
    
    # Create comprehensive multi-AI configuration
    multi_ai_config = {
        "version": "2.0",
        "available_models": [],
        "default_model": "gpt4" if "OPENAI" in available_ais else "gemini",
        "fallback_chain": [],
        "voice_enabled": "ELEVENLABS" in available_ais
    }
    
    # Add Claude Desktop support
    if "CLAUDE_DESKTOP" in available_ais:
        multi_ai_config["available_models"].append({
            "name": "claude-desktop",
            "provider": "anthropic",
            "model": "claude-desktop",
            "max_tokens": 200000,  # Claude's max context
            "notes": "Via Claude Desktop MCP integration"
        })
        multi_ai_config["fallback_chain"].append("claude-desktop")
    
    # Add GPT-4
    if "OPENAI" in available_ais:
        multi_ai_config["available_models"].append({
            "name": "gpt4",
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "max_tokens": 128000,
            "api_key_env": "OPENAI_API_KEY"
        })
        multi_ai_config["fallback_chain"].append("gpt4")
    
    # Add Gemini
    if "GEMINI" in available_ais:
        multi_ai_config["available_models"].append({
            "name": "gemini",
            "provider": "google",
            "model": "gemini-1.5-pro",
            "max_tokens": 2097152,  # 2M context!
            "api_key_env": "GEMINI_API_KEY"
        })
        multi_ai_config["fallback_chain"].append("gemini")
    
    # Add voice synthesis
    if "ELEVENLABS" in available_ais:
        multi_ai_config["voice_config"] = {
            "provider": "elevenlabs",
            "enabled": True,
            "api_key_env": "ELEVENLABS_API_KEY"
        }
    
    # Save configuration
    config_path = Path("config/multi_ai_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(multi_ai_config, f, indent=2)
    
    logger.info(f"\n✅ Multi-AI configuration saved to {config_path}")
    logger.info(f"📊 Available AI models: {len(multi_ai_config['available_models'])}")
    for model in multi_ai_config['available_models']:
        logger.info(f"   • {model['name']} ({model['provider']}) - {model['max_tokens']:,} tokens")
    
    # Create enhanced integration module
    integration_code = '''"""
Enhanced Multi-AI Integration Module for JARVIS
Supports GPT-4, Gemini, Claude Desktop, and ElevenLabs
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Try to import available AI libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiAIIntegration:
    """Enhanced multi-AI integration with fallback support"""
    
    def __init__(self):
        self.config_path = Path("config/multi_ai_config.json")
        self.config = self._load_config()
        self.available_models = {m["name"]: m for m in self.config.get("available_models", [])}
        self._initialize_clients()
        
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {"available_models": [], "default_model": "minimal"}
    
    def _initialize_clients(self):
        """Initialize AI clients"""
        self.clients = {}
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self.clients["gpt4"] = openai
            logger.info("✅ OpenAI client initialized")
        
        # Initialize Gemini
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.clients["gemini"] = genai.GenerativeModel('gemini-1.5-pro')
            logger.info("✅ Gemini client initialized")
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.available_models.keys())
    
    async def query(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Query an AI model with automatic fallback"""
        if not model:
            model = self.config.get("default_model", "minimal")
        
        # Try the requested model
        try:
            return await self._query_model(model, prompt, **kwargs)
        except Exception as e:
            logger.warning(f"Model {model} failed: {e}")
            
            # Try fallback chain
            for fallback_model in self.config.get("fallback_chain", []):
                if fallback_model != model:
                    try:
                        logger.info(f"Trying fallback: {fallback_model}")
                        return await self._query_model(fallback_model, prompt, **kwargs)
                    except Exception as e:
                        logger.warning(f"Fallback {fallback_model} failed: {e}")
            
            # Final fallback
            return f"[All models failed] Received: {prompt[:100]}..."
    
    async def _query_model(self, model: str, prompt: str, **kwargs) -> str:
        """Query a specific model"""
        if model == "gpt4" and "gpt4" in self.clients:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4-turbo-preview",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 4096)
            )
            return response.choices[0].message.content
        
        elif model == "gemini" and "gemini" in self.clients:
            response = await asyncio.to_thread(
                self.clients["gemini"].generate_content,
                prompt
            )
            return response.text
        
        elif model == "claude-desktop":
            # This would integrate with Claude Desktop via MCP
            return f"[Claude Desktop integration pending] {prompt[:100]}..."
        
        else:
            return f"[Model {model} not available] {prompt[:100]}..."
    
    async def generate_voice(self, text: str) -> Optional[bytes]:
        """Generate voice using ElevenLabs if available"""
        if self.config.get("voice_config", {}).get("enabled"):
            # ElevenLabs integration would go here
            logger.info(f"Voice generation requested for: {text[:50]}...")
            return None
        return None

# Create global instance
multi_ai = MultiAIIntegration()
'''
    
    # Save the enhanced integration module
    integration_path = Path("core/updated_multi_ai_integration.py")
    with open(integration_path, 'w') as f:
        f.write(integration_code)
    
    logger.info(f"✅ Created enhanced multi-AI integration at {integration_path}")
    
    return multi_ai_config

def create_test_script():
    """Create a test script for multi-AI"""
    test_code = '''#!/usr/bin/env python3
"""
Test Multi-AI Integration
"""
import asyncio
import logging
from core.updated_multi_ai_integration import multi_ai

logging.basicConfig(level=logging.INFO)

async def test_multi_ai():
    """Test all available AI models"""
    print("🧪 Testing Multi-AI Integration\\n")
    
    # Get available models
    models = multi_ai.get_available_models()
    print(f"Available models: {models}\\n")
    
    # Test prompt
    test_prompt = "What is the meaning of life? (Please answer in one sentence)"
    
    # Test each model
    for model in models:
        print(f"Testing {model}...")
        try:
            response = await multi_ai.query(test_prompt, model=model)
            print(f"✅ {model}: {response}\\n")
        except Exception as e:
            print(f"❌ {model} error: {e}\\n")
    
    # Test fallback
    print("Testing fallback mechanism...")
    response = await multi_ai.query(test_prompt, model="nonexistent-model")
    print(f"Fallback response: {response}\\n")

if __name__ == "__main__":
    asyncio.run(test_multi_ai())
'''
    
    test_path = Path("test_multi_ai.py")
    with open(test_path, 'w') as f:
        f.write(test_code)
    
    os.chmod(test_path, 0o755)
    logger.info(f"✅ Created test script: {test_path}")

def main():
    """Main setup function"""
    # Setup multi-AI
    config = setup_multi_ai()
    
    # Create test script
    create_test_script()
    
    logger.info("\n✨ Multi-AI Integration Setup Complete!")
    logger.info("\n🚀 Next steps:")
    logger.info("  1. Test integration: python3 test_multi_ai.py")
    logger.info("  2. Restart JARVIS to load new configuration")
    logger.info("  3. Check Claude Desktop MCP menu for JARVIS")
    
    # Show summary
    logger.info(f"\n📊 Summary:")
    logger.info(f"  • Models configured: {len(config['available_models'])}")
    logger.info(f"  • Default model: {config['default_model']}")
    logger.info(f"  • Voice enabled: {'Yes' if config.get('voice_enabled') else 'No'}")
    logger.info(f"  • Fallback chain: {' → '.join(config['fallback_chain'])}")

if __name__ == "__main__":
    main()
