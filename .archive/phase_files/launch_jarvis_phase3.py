#!/usr/bin/env python3
"""
JARVIS Phase 3 Launcher
======================
Launches JARVIS with full Phase 3 intelligent processing capabilities.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Phase 1 components
from core.unified_input_pipeline import UnifiedInputPipeline
from core.fluid_state_management import FluidStateManager
from core.jarvis_enhanced_core import JARVISEnhancedCore

# Import Phase 3 components
from core.memory_enhanced_processing import enhance_jarvis_with_phase3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3Launcher:
    """Launcher for JARVIS with Phase 3 enhancements"""
    
    def __init__(self):
        self.jarvis = None
        self.phase3 = None
        self.running = False
        
    async def initialize(self):
        """Initialize JARVIS with Phase 3"""
        logger.info("🚀 Initializing JARVIS Phase 3...")
        
        try:
            # Create JARVIS Enhanced Core
            logger.info("📦 Loading JARVIS Enhanced Core...")
            self.jarvis = JARVISEnhancedCore()
            await self.jarvis.initialize()
            
            # Enhance with Phase 3
            logger.info("🧠 Enhancing with Phase 3 capabilities...")
            self.phase3 = await enhance_jarvis_with_phase3(self.jarvis)
            
            logger.info("✅ JARVIS Phase 3 initialized successfully!")
            
            # Show initial status
            await self.show_status()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize: {e}")
            raise
    
    async def show_status(self):
        """Show current system status"""
        try:
            # Get intelligence insights
            insights = await self.jarvis.get_intelligence()
            
            print("\n" + "="*60)
            print("🤖 JARVIS PHASE 3 STATUS")
            print("="*60)
            print(f"⚡ Intelligence Score: {insights.get('intelligence_score', 0)}/100")
            print(f"🧠 Total Memories: {insights['memory_insights']['total_memories']}")
            print(f"📚 Learned Patterns: {insights['predictive_insights']['learned_patterns']}")
            print(f"🔮 Active Predictions: {insights['predictive_insights']['active_predictions']}")
            print(f"💬 Active Conversations: {insights['context_insights']['active_threads']}")
            print(f"🎯 Current Activities: {insights['context_insights']['active_activities']}")
            print("="*60 + "\n")
            
        except Exception as e:
            logger.error(f"Error getting status: {e}")
    
    async def run_interactive(self):
        """Run interactive mode"""
        self.running = True
        
        print("\n🎮 JARVIS Phase 3 Interactive Mode")
        print("="*50)
        print("Commands:")
        print("  'quit' or 'exit' - Exit JARVIS")
        print("  'status' - Show system status")
        print("  'predictions' - Show active predictions")
        print("  'recall <query>' - Search memories")
        print("  'context' - Show current context")
        print("  'help' - Show this help message")
        print("="*50)
        print("\nStart chatting with JARVIS!\n")
        
        while self.running:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    print("\nGoodbye! 👋")
                    break
                    
                elif user_input.lower() == 'status':
                    await self.show_status()
                    continue
                    
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  'quit' or 'exit' - Exit JARVIS")
                    print("  'status' - Show system status")
                    print("  'predictions' - Show active predictions")
                    print("  'recall <query>' - Search memories")
                    print("  'context' - Show current context")
                    print("  'help' - Show this help message\n")
                    continue
                    
                elif user_input.lower() == 'predictions':
                    predictions = await self.jarvis.get_predictions()
                    if predictions:
                        print("\n🔮 Active Predictions:")
                        for i, pred in enumerate(predictions[:5], 1):
                            print(f"{i}. {pred['content']} ({pred['confidence']*100:.0f}% confidence)")
                        print()
                    else:
                        print("\nNo active predictions.\n")
                    continue
                    
                elif user_input.lower().startswith('recall '):
                    query = user_input[7:]
                    memories = await self.jarvis.recall(query, max_results=3)
                    if memories:
                        print(f"\n🧠 Memories related to '{query}':")
                        for i, memory in enumerate(memories, 1):
                            content = memory.chunks[0].content
                            if isinstance(content, dict):
                                content = content.get('content', str(content))
                            print(f"{i}. {str(content)[:100]}...")
                        print()
                    else:
                        print(f"\nNo memories found for '{query}'.\n")
                    continue
                    
                elif user_input.lower() == 'context':
                    context = await self.jarvis.get_context_state()
                    print("\n📋 Current Context:")
                    print(f"Active Threads: {context['active_conversation_threads']}")
                    print(f"Active Activities: {context['active_activities']}")
                    print(f"Context Switches Today: {context['context_switches_today']}")
                    if context['top_topics']:
                        print("Top Topics:")
                        for topic, count in context['top_topics'][:3]:
                            print(f"  - {topic}: {count}")
                    print()
                    continue
                
                # Process normal input
                start_time = datetime.now()
                
                # Process with memory
                result = await self.jarvis.process_with_memory(user_input)
                
                # Get response (using the original process method)
                response = await self.jarvis.process(user_input)
                
                # Calculate processing time
                process_time = (datetime.now() - start_time).total_seconds()
                
                # Display response
                if isinstance(response, dict):
                    print(f"\nJARVIS: {response.get('response', response)}")
                else:
                    print(f"\nJARVIS: {response}")
                
                # Show context info
                info_parts = []
                
                if result.context.get("conversation_thread"):
                    thread = result.context["conversation_thread"]
                    info_parts.append(f"Thread: {thread.topic[:20]}")
                
                if result.context.get("current_activity"):
                    activity = result.context["current_activity"]
                    info_parts.append(f"Activity: {activity.activity_type}")
                
                if result.memory_utilized:
                    info_parts.append("Memory: ✓")
                
                if result.predictions:
                    info_parts.append(f"Predictions: {len(result.predictions)}")
                
                info_parts.append(f"Time: {process_time:.2f}s")
                
                print(f"[{' | '.join(info_parts)}]\n")
                
                # Check for high-confidence predictions
                if result.predictions:
                    high_conf = [p for p in result.predictions 
                               if p.get('confidence', 0) > 0.7]
                    if high_conf:
                        print(f"💡 Prediction: {high_conf[0]['content']} "
                              f"({high_conf[0]['confidence']*100:.0f}% likely)\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                print(f"\n❌ Error: {e}\n")
    
    async def run_batch(self, commands: list):
        """Run batch commands"""
        logger.info(f"Running {len(commands)} commands in batch mode")
        
        for i, command in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] Processing: {command}")
            
            try:
                result = await self.jarvis.process_with_memory(command)
                response = await self.jarvis.process(command)
                
                if isinstance(response, dict):
                    print(f"Response: {response.get('response', response)}")
                else:
                    print(f"Response: {response}")
                
                if result.predictions:
                    print(f"Generated {len(result.predictions)} predictions")
                
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(f"Error: {e}")
            
            # Small delay between commands
            await asyncio.sleep(0.5)
        
        # Show final status
        await self.show_status()
    
    async def shutdown(self):
        """Shutdown JARVIS"""
        logger.info("Shutting down JARVIS Phase 3...")
        
        if self.phase3:
            await self.phase3.shutdown()
        
        if self.jarvis:
            # Jarvis shutdown if it has one
            pass
        
        logger.info("Shutdown complete.")


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JARVIS Phase 3 Launcher")
    parser.add_argument("--batch", nargs="+", help="Run batch commands")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--test", action="store_true", help="Run tests")
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = Phase3Launcher()
    
    try:
        # Initialize
        await launcher.initialize()
        
        if args.test:
            # Run tests
            from test_jarvis_phase3 import Phase3Tester
            tester = Phase3Tester()
            tester.jarvis_core = launcher.jarvis
            tester.phase3_integration = launcher.phase3
            await tester.run_all_tests()
            
        elif args.status:
            # Just show status
            pass
            
        elif args.batch:
            # Run batch commands
            await launcher.run_batch(args.batch)
            
        else:
            # Run interactive mode
            await launcher.run_interactive()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
    finally:
        await launcher.shutdown()


if __name__ == "__main__":
    # Print banner
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              JARVIS PHASE 3 - INTELLIGENT AI              ║
    ║                                                           ║
    ║  🧠 Context Persistence    🔮 Predictive Pre-loading     ║
    ║  📚 Pattern Learning       💭 Episodic Memory            ║
    ║  🎯 Workflow Automation    📊 Intelligence Metrics       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Run
    asyncio.run(main())
