"""
Main Assistant orchestrator
Coordinates all components for seamless interaction
"""
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from .nlp.pipeline import NLPPipeline
from .voice.manager import VoiceManager
from .memory.context import ContextManager
from .ai.router import AIRouter
from .plugins.manager import PluginManager
from .logger import setup_logger

logger = setup_logger(__name__)


class Assistant:
    """Main JARVIS assistant orchestrator"""
    
    def __init__(self, config):
        self.config = config
        self.nlp = NLPPipeline(config)
        self.voice = VoiceManager(config)
        self.memory = ContextManager(config)
        self.ai = AIRouter(config)
        self.plugins = PluginManager(config)
        
        self.active = False
        self.conversation_id = None
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing assistant components...")
        
        # Initialize in parallel for speed
        await asyncio.gather(
            self.nlp.initialize(),
            self.voice.initialize(),
            self.memory.initialize(),
            self.ai.initialize(),
            self.plugins.initialize()
        )
        
        # Start new conversation
        self.conversation_id = self.memory.create_conversation()
        logger.info(f"Started conversation: {self.conversation_id}")
        
    async def run(self):
        """Main assistant loop"""
        self.active = True
        
        # Start voice listening
        voice_task = asyncio.create_task(self._voice_loop())
        
        try:
            while self.active:
                await asyncio.sleep(0.1)
        finally:
            voice_task.cancel()
            
    async def _voice_loop(self):
        """Handle voice interactions"""
        async for audio_input in self.voice.listen():
            if audio_input.is_wake_word:
                await self._handle_wake_word()
            elif audio_input.transcription:
                await self._process_input(audio_input.transcription)
                
    async def _handle_wake_word(self):
        """Handle wake word detection"""
        logger.info("Wake word detected!")
        await self.voice.play_acknowledgment()
        
    async def _process_input(self, text: str):
        """Process user input through the full pipeline"""
        logger.info(f"Processing: {text}")
        
        try:
            # 1. NLP Analysis
            nlp_result = await self.nlp.process(text)
            
            # 2. Update context
            context = await self.memory.update_context(
                text=text,
                intent=nlp_result.intent,
                entities=nlp_result.entities
            )
            
            # 3. Check for plugin command
            if nlp_result.intent.startswith("plugin:"):
                plugin_name = nlp_result.intent.split(":", 1)[1]
                response = await self.plugins.execute(
                    plugin_name, 
                    nlp_result.entities
                )
            else:
                # 4. Get AI response
                response = await self.ai.generate_response(
                    text=text,
                    context=context,
                    intent=nlp_result.intent
                )
            
            # 5. Store in memory
            await self.memory.add_exchange(text, response)
            
            # 6. Speak response
            await self.voice.speak(response)
            
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            await self.voice.speak("I encountered an error. Please try again.")
            
    async def process_text(self, text: str) -> str:
        """Process text input (for API/UI access)"""
        await self._process_input(text)
        return self.memory.last_response
        
    async def shutdown(self):
        """Gracefully shutdown assistant"""
        logger.info("Shutting down assistant...")
        self.active = False
        
        # Save conversation
        await self.memory.save_conversation(self.conversation_id)
        
        # Shutdown components
        await asyncio.gather(
            self.voice.shutdown(),
            self.memory.shutdown(),
            self.ai.shutdown(),
            self.plugins.shutdown()
        )