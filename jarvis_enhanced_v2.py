#!/usr/bin/env python3
"""
Enhanced JARVIS with Multi-AI Integration
This version actually connects to real AI services!
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our working multi-AI
from core.working_multi_ai import multi_ai

class EnhancedJARVIS:
    """JARVIS with real AI capabilities"""
    
    def __init__(self):
        self.name = "JARVIS"
        self.active = False
        self.conversation_history = []
        self.start_time = None
        
    async def initialize(self):
        """Initialize JARVIS with AI services"""
        print(f"""
╔══════════════════════════════════════════╗
║      🚀 JARVIS ENHANCED EDITION 🚀       ║
║         Powered by Real AI Models         ║
╚══════════════════════════════════════════╝
""")
        
        logger.info("Initializing JARVIS Enhanced...")
        self.start_time = datetime.now()
        
        # Initialize AI services
        print("🔧 Connecting to AI services...")
        success = await multi_ai.initialize()
        
        if success:
            models = list(multi_ai.available_models.keys())
            print(f"\n✅ Connected to {len(models)} AI service(s): {models}")
            self.active = True
            
            # Announce readiness
            if models:
                response = await multi_ai.query(
                    "You are JARVIS, an advanced AI assistant. Say hello and mention you're now online with enhanced capabilities.",
                    preferred_model=models[0]
                )
                if response.get("success"):
                    print(f"\nJARVIS: {response['response']}")
        else:
            print("\n⚠️  No AI services available - running in limited mode")
            self.active = True
            
        print("\nType 'help' for available commands or 'exit' to quit\n")
        
    async def chat(self, message: str) -> str:
        """Process a chat message with AI"""
        
        # Handle special commands
        if message.lower() == "help":
            return self._get_help()
        elif message.lower() == "status":
            return self._get_status()
        elif message.lower() == "models":
            return self._get_models_info()
        elif message.lower().startswith("use "):
            return await self._switch_model(message[4:].strip())
        elif message.lower() == "compare":
            return await self._compare_models()
            
        # Regular chat with AI
        if multi_ai.available_models:
            # Add context about being JARVIS
            enhanced_prompt = f"""You are JARVIS, an advanced AI assistant created to help users with various tasks. 
You are helpful, knowledgeable, and professional.

User: {message}

JARVIS:"""
            
            response = await multi_ai.query(enhanced_prompt)
            
            if response.get("success"):
                # Store in history
                self.conversation_history.append({
                    "user": message,
                    "jarvis": response["response"],
                    "model": response.get("model", "unknown"),
                    "timestamp": datetime.now()
                })
                
                return response["response"]
            else:
                return f"I encountered an error: {response.get('error', 'Unknown error')}"
        else:
            # Offline mode responses
            return "I'm currently in offline mode. My AI services are not connected."
            
    def _get_help(self) -> str:
        """Get help information"""
        return """
🤖 JARVIS Enhanced Commands:
─────────────────────────────
  help     - Show this help message
  status   - Check system status
  models   - Show available AI models
  use [model] - Switch to a specific model
  compare  - Compare responses from all models
  exit     - Shutdown JARVIS
  
Just type normally to chat with me!
"""
        
    def _get_status(self) -> str:
        """Get system status"""
        uptime = datetime.now() - self.start_time if self.start_time else "Unknown"
        models = list(multi_ai.available_models.keys())
        
        return f"""
📊 JARVIS System Status:
─────────────────────────
  Status: {'Online' if self.active else 'Offline'}
  Uptime: {str(uptime).split('.')[0]}
  Available Models: {len(models)}
  Models: {', '.join(models) if models else 'None'}
  Chat History: {len(self.conversation_history)} messages
"""
        
    def _get_models_info(self) -> str:
        """Get information about available models"""
        if not multi_ai.available_models:
            return "No AI models are currently available."
            
        info = "🧠 Available AI Models:\n─────────────────────────\n"
        
        model_info = {
            "openai": "GPT-4 Turbo - Advanced reasoning and code generation",
            "gemini": "Gemini 1.5 Flash - Fast responses with large context",
            "claude": "Claude 3 Opus - Deep analysis and creative tasks"
        }
        
        for model in multi_ai.available_models:
            info += f"  • {model}: {model_info.get(model, 'AI model')}\n"
            
        return info
        
    async def _switch_model(self, model: str) -> str:
        """Switch to a specific model"""
        if model in multi_ai.available_models:
            # Test the model
            response = await multi_ai.query("Confirm you are online.", preferred_model=model)
            if response.get("success"):
                return f"✅ Switched to {model} model"
            else:
                return f"❌ Failed to switch to {model}: {response.get('error')}"
        else:
            return f"❌ Model '{model}' is not available. Available: {list(multi_ai.available_models.keys())}"
            
    async def _compare_models(self) -> str:
        """Compare responses from all available models"""
        if len(multi_ai.available_models) < 2:
            return "Need at least 2 models for comparison. Currently available: " + str(list(multi_ai.available_models.keys()))
            
        test_prompt = "What makes you unique as an AI model? (Answer in 2 sentences)"
        
        print("🔄 Querying all models...")
        results = await multi_ai.query_all(test_prompt)
        
        comparison = "🔍 Model Comparison:\n" + "─" * 50 + "\n"
        
        for model, result in results.items():
            if result.get("success"):
                comparison += f"\n{model.upper()}:\n{result['response']}\n"
            else:
                comparison += f"\n{model.upper()}:\n❌ Error: {result.get('error')}\n"
                
        return comparison
        
    async def shutdown(self):
        """Shutdown JARVIS"""
        logger.info("Shutting down JARVIS Enhanced...")
        self.active = False
        
        # Save conversation history if needed
        if self.conversation_history:
            logger.info(f"Processed {len(self.conversation_history)} messages this session")

async def main():
    """Main entry point for Enhanced JARVIS"""
    jarvis = EnhancedJARVIS()
    await jarvis.initialize()
    
    while jarvis.active:
        try:
            # Get user input
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye', 'shutdown']:
                print("\nJARVIS: Goodbye! It was a pleasure assisting you.")
                break
                
            # Process with JARVIS
            response = await jarvis.chat(user_input)
            print(f"\nJARVIS: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nJARVIS: Shutdown signal received.")
            break
        except Exception as e:
            print(f"\nJARVIS: I encountered an error: {e}\n")
            logger.error(f"Chat error: {e}")
            
    await jarvis.shutdown()

if __name__ == "__main__":
    # Run the enhanced JARVIS
    print("Starting JARVIS Enhanced Edition...")
    asyncio.run(main())
