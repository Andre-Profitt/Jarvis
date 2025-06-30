#!/usr/bin/env python3
"""
Minimal JARVIS - A working starting point
"""

import asyncio
import os
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalJARVIS:
    """Minimal working JARVIS implementation"""
    
    def __init__(self):
        self.name = "JARVIS"
        self.active = False
        self.ai_backend = None
        
    async def initialize(self):
        """Initialize JARVIS systems"""
        logger.info(f"Initializing {self.name}...")
        
        # Try to load AI integration
        try:
            from core.updated_multi_ai_integration import EnhancedMultiAIIntegration
            self.ai_backend = EnhancedMultiAIIntegration()
            logger.info("AI backend loaded successfully")
        except Exception as e:
            logger.warning(f"AI backend not available: {e}")
            logger.info("Running in offline mode")
        
        self.active = True
        logger.info(f"{self.name} is now active!")
        
    async def chat(self, message: str) -> str:
        """Process a chat message"""
        if not self.active:
            return "JARVIS is not active. Please initialize first."
            
        # If we have AI backend, use it
        if self.ai_backend:
            try:
                response = await self.ai_backend.generate_response(message)
                return response
            except Exception as e:
                logger.error(f"AI error: {e}")
        
        # Fallback responses
        responses = {
            "hello": "Hello! I'm JARVIS, your AI assistant. How can I help you today?",
            "help": "I can help you with various tasks. Try asking me questions!",
            "status": "All systems operational. Running in minimal mode.",
        }
        
        message_lower = message.lower()
        for key, response in responses.items():
            if key in message_lower:
                return response
                
        return "I understand you said: " + message + ". I'm still learning!"
        
    async def shutdown(self):
        """Shutdown JARVIS"""
        logger.info("Shutting down JARVIS...")
        self.active = False

async def main():
    """Main entry point"""
    jarvis = MinimalJARVIS()
    await jarvis.initialize()
    
    print("\n" + "="*50)
    print("JARVIS Minimal Edition - Type 'exit' to quit")
    print("="*50 + "\n")
    
    while jarvis.active:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
                
            response = await jarvis.chat(user_input)
            print(f"JARVIS: {response}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            
    await jarvis.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
