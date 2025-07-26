"""
AI Router - Intelligent model selection and response generation
Routes to GPT-4, Claude, Gemini, or local models based on task
"""
import asyncio
from typing import Dict, Any, Optional, List
from enum import Enum
import time

import openai
import anthropic
import google.generativeai as genai
from transformers import pipeline

from ..logger import setup_logger

logger = setup_logger(__name__)


class ModelType(Enum):
    """Available AI models"""
    GPT4 = "gpt-4"
    CLAUDE = "claude-3"
    GEMINI = "gemini-pro"
    LOCAL = "local"
    

class AIRouter:
    """Routes requests to appropriate AI model"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.default_model = ModelType.GPT4
        
        # Model selection rules
        self.task_model_map = {
            "code": ModelType.CLAUDE,
            "creative": ModelType.GPT4,
            "analysis": ModelType.CLAUDE,
            "general": ModelType.GPT4,
            "quick": ModelType.LOCAL,
            "vision": ModelType.GEMINI,
        }
        
    async def initialize(self):
        """Initialize AI models"""
        logger.info("Initializing AI router...")
        
        # Initialize models based on available API keys
        tasks = []
        
        if api_key := self.config.get("ai.openai_api_key"):
            tasks.append(self._init_openai(api_key))
            
        if api_key := self.config.get("ai.anthropic_api_key"):
            tasks.append(self._init_anthropic(api_key))
            
        if api_key := self.config.get("ai.google_api_key"):
            tasks.append(self._init_gemini(api_key))
            
        # Always initialize local model as fallback
        tasks.append(self._init_local_model())
        
        await asyncio.gather(*tasks)
        
        logger.info(f"Initialized {len(self.models)} AI models")
        
    async def _init_openai(self, api_key: str):
        """Initialize OpenAI GPT-4"""
        openai.api_key = api_key
        self.models[ModelType.GPT4] = {
            "client": openai,
            "model": "gpt-4-turbo-preview"
        }
        logger.info("✓ OpenAI GPT-4 initialized")
        
    async def _init_anthropic(self, api_key: str):
        """Initialize Anthropic Claude"""
        self.models[ModelType.CLAUDE] = {
            "client": anthropic.Anthropic(api_key=api_key),
            "model": "claude-3-opus-20240229"
        }
        logger.info("✓ Anthropic Claude initialized")
        
    async def _init_gemini(self, api_key: str):
        """Initialize Google Gemini"""
        genai.configure(api_key=api_key)
        self.models[ModelType.GEMINI] = {
            "client": genai.GenerativeModel('gemini-pro'),
            "model": "gemini-pro"
        }
        logger.info("✓ Google Gemini initialized")
        
    async def _init_local_model(self):
        """Initialize local model for fast responses"""
        try:
            self.models[ModelType.LOCAL] = {
                "client": pipeline(
                    "text-generation",
                    model="microsoft/DialoGPT-medium",
                    device=-1  # CPU
                ),
                "model": "DialoGPT"
            }
            logger.info("✓ Local model initialized")
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            
    async def generate_response(
        self,
        text: str,
        context: Dict[str, Any],
        intent: str,
        stream: bool = True
    ) -> str:
        """Generate response using appropriate model"""
        
        # Select model based on intent
        model_type = self._select_model(intent, text)
        
        if model_type not in self.models:
            # Fallback to available model
            model_type = next(iter(self.models.keys()))
            
        logger.info(f"Using {model_type.value} for intent: {intent}")
        
        # Generate response
        start_time = time.time()
        
        try:
            if model_type == ModelType.GPT4:
                response = await self._generate_gpt4(text, context, stream)
            elif model_type == ModelType.CLAUDE:
                response = await self._generate_claude(text, context, stream)
            elif model_type == ModelType.GEMINI:
                response = await self._generate_gemini(text, context, stream)
            else:
                response = await self._generate_local(text, context)
                
            elapsed = time.time() - start_time
            logger.info(f"Generated response in {elapsed:.2f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error generating a response. Please try again."
            
    def _select_model(self, intent: str, text: str) -> ModelType:
        """Select best model for the task"""
        # Check for specific keywords
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["code", "program", "function", "debug"]):
            return ModelType.CLAUDE
        elif any(word in text_lower for word in ["image", "picture", "analyze", "vision"]):
            return ModelType.GEMINI
        elif len(text.split()) < 10:  # Short queries
            return ModelType.LOCAL
            
        # Use intent mapping
        for task_type, model in self.task_model_map.items():
            if task_type in intent:
                return model
                
        return self.default_model
        
    async def _generate_gpt4(self, text: str, context: Dict[str, Any], stream: bool) -> str:
        """Generate response using GPT-4"""
        messages = self._build_messages(text, context)
        
        response = await openai.ChatCompletion.acreate(
            model=self.models[ModelType.GPT4]["model"],
            messages=messages,
            temperature=0.7,
            stream=stream
        )
        
        if stream:
            return await self._process_stream(response)
        else:
            return response.choices[0].message.content
            
    async def _generate_claude(self, text: str, context: Dict[str, Any], stream: bool) -> str:
        """Generate response using Claude"""
        client = self.models[ModelType.CLAUDE]["client"]
        
        response = await client.messages.create(
            model=self.models[ModelType.CLAUDE]["model"],
            messages=[{"role": "user", "content": text}],
            system=self._build_system_prompt(context),
            max_tokens=1000,
            stream=stream
        )
        
        if stream:
            return await self._process_claude_stream(response)
        else:
            return response.content[0].text
            
    async def _generate_gemini(self, text: str, context: Dict[str, Any], stream: bool) -> str:
        """Generate response using Gemini"""
        model = self.models[ModelType.GEMINI]["client"]
        
        prompt = self._build_prompt(text, context)
        response = await model.generate_content_async(prompt, stream=stream)
        
        if stream:
            return await self._process_gemini_stream(response)
        else:
            return response.text
            
    async def _generate_local(self, text: str, context: Dict[str, Any]) -> str:
        """Generate response using local model"""
        model = self.models[ModelType.LOCAL]["client"]
        
        # Simple context injection
        prompt = f"Context: {context.get('summary', '')}\nUser: {text}\nAssistant:"
        
        response = model(prompt, max_length=200, temperature=0.7)[0]
        return response['generated_text'].split("Assistant:")[-1].strip()
        
    def _build_messages(self, text: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build message history for chat models"""
        messages = [
            {"role": "system", "content": self._build_system_prompt(context)}
        ]
        
        # Add conversation history
        if history := context.get("history", []):
            for exchange in history[-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
                
        messages.append({"role": "user", "content": text})
        return messages
        
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with context"""
        prompt = """You are JARVIS, an advanced AI assistant. You are helpful, knowledgeable, and conversational.
        
Key traits:
- Concise but friendly responses
- Proactive in offering help
- Remember context from the conversation
- Technical expertise when needed
"""
        
        if user_prefs := context.get("user_preferences"):
            prompt += f"\nUser preferences: {user_prefs}"
            
        if current_context := context.get("current_context"):
            prompt += f"\nCurrent context: {current_context}"
            
        return prompt
        
    def _build_prompt(self, text: str, context: Dict[str, Any]) -> str:
        """Build prompt for non-chat models"""
        prompt = self._build_system_prompt(context)
        prompt += f"\n\nUser: {text}\nJARVIS:"
        return prompt
        
    async def _process_stream(self, stream) -> str:
        """Process streaming response"""
        response_text = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text
        
    async def _process_claude_stream(self, stream) -> str:
        """Process Claude streaming response"""
        response_text = ""
        async for chunk in stream:
            if chunk.type == "content_block_delta":
                response_text += chunk.delta.text
        return response_text
        
    async def _process_gemini_stream(self, stream) -> str:
        """Process Gemini streaming response"""
        response_text = ""
        async for chunk in stream:
            response_text += chunk.text
        return response_text
        
    async def shutdown(self):
        """Cleanup AI router"""
        logger.info("Shutting down AI router...")
        # Cleanup if needed