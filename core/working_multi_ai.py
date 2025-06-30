#!/usr/bin/env python3
"""
Working Multi-AI Integration for JARVIS
Simplified version that actually works with available APIs
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Load environment
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkingMultiAI:
    """Multi-AI integration that actually works"""
    
    def __init__(self):
        self.available_models = {}
        self.clients = {}
        
    async def initialize(self):
        """Initialize available AI services"""
        logger.info("Initializing Multi-AI System...")
        
        # Initialize OpenAI
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.clients["openai"] = openai.OpenAI(api_key=api_key)
                # Test connection
                test = self.clients["openai"].chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                self.available_models["openai"] = True
                logger.info("✅ OpenAI initialized")
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
            
        # Initialize Gemini
        try:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                genai.configure(api_key=api_key)
                # Use the correct model name
                self.clients["gemini"] = genai.GenerativeModel('gemini-1.5-flash')
                # Test connection
                test = self.clients["gemini"].generate_content("test")
                self.available_models["gemini"] = True
                logger.info("✅ Gemini initialized")
        except Exception as e:
            logger.warning(f"Gemini not available: {e}")
            
        # Initialize Claude (if API key exists)
        try:
            import anthropic
            api_key = os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                self.clients["claude"] = anthropic.Anthropic(api_key=api_key)
                self.available_models["claude"] = True
                logger.info("✅ Claude API initialized")
        except Exception as e:
            logger.warning(f"Claude API not available: {e}")
            
        # Note about Claude Desktop
        if os.getenv("USE_CLAUDE_DESKTOP") == "true":
            logger.info("ℹ️  Claude Desktop is configured (MCP integration)")
            
        logger.info(f"Available models: {list(self.available_models.keys())}")
        return len(self.available_models) > 0
        
    async def query(self, prompt: str, preferred_model: Optional[str] = None) -> Dict[str, Any]:
        """Query an available AI model"""
        
        # Select model
        if preferred_model and preferred_model in self.available_models:
            model = preferred_model
        else:
            # Use first available model
            if not self.available_models:
                return {"error": "No AI models available", "response": "I'm running in offline mode."}
            model = list(self.available_models.keys())[0]
            
        try:
            if model == "openai":
                response = self.clients["openai"].chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return {
                    "response": response.choices[0].message.content,
                    "model": "gpt-4-turbo",
                    "success": True
                }
                
            elif model == "gemini":
                response = self.clients["gemini"].generate_content(prompt)
                return {
                    "response": response.text,
                    "model": "gemini-1.5-flash", 
                    "success": True
                }
                
            elif model == "claude":
                response = self.clients["claude"].messages.create(
                    model="claude-3-opus-20240229",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return {
                    "response": response.content[0].text,
                    "model": "claude-3-opus",
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error querying {model}: {e}")
            # Try another model if available
            other_models = [m for m in self.available_models if m != model]
            if other_models:
                logger.info(f"Trying fallback model: {other_models[0]}")
                return await self.query(prompt, preferred_model=other_models[0])
                
            return {"error": str(e), "model": model, "success": False}
            
    async def query_all(self, prompt: str) -> Dict[str, Any]:
        """Query all available models in parallel"""
        
        tasks = []
        for model in self.available_models:
            task = self.query(prompt, preferred_model=model)
            tasks.append((model, task))
            
        results = {}
        for model, task in tasks:
            try:
                result = await task
                results[model] = result
            except Exception as e:
                results[model] = {"error": str(e)}
                
        return results

# Create a singleton instance
multi_ai = WorkingMultiAI()

# For compatibility with existing code
EnhancedMultiAIIntegration = WorkingMultiAI
