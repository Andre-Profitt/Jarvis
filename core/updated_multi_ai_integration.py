"""
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
    from openai import OpenAI
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
        
        # Initialize OpenAI (new v1.0+ syntax)
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            self.clients["gpt4"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("✅ OpenAI client initialized")
        
        # Initialize Gemini
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.clients["gemini"] = genai.GenerativeModel('gemini-1.5-flash')
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
            # Use new OpenAI v1.0+ syntax
            response = await asyncio.to_thread(
                self.clients["gpt4"].chat.completions.create,
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
            try:
                from elevenlabs import ElevenLabs
                client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
                
                # Get first available voice
                voices = client.voices.get_all()
                if voices.voices:
                    voice_id = voices.voices[0].voice_id
                    
                    # Generate audio
                    audio = client.generate(
                        text=text[:1000],  # Limit text length
                        voice=voice_id,
                        model="eleven_monolingual_v1"
                    )
                    
                    logger.info(f"✅ Voice generated for: {text[:50]}...")
                    return audio
            except Exception as e:
                logger.error(f"Voice generation failed: {e}")
        return None

# Create global instance
multi_ai = MultiAIIntegration()
