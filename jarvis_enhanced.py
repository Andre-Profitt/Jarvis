#!/usr/bin/env python3
"""
JARVIS Enhanced Launcher - With Full Multi-AI Support
"""
import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"jarvis_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

class EnhancedJARVIS:
    """Enhanced JARVIS with Multi-AI Integration"""
    
    def __init__(self):
        self.name = "JARVIS"
        self.version = "2.0-enhanced"
        self.active = True
        self.redis_client = None
        self.start_time = datetime.now()
        self.multi_ai = None
        self.components = {}
        
    async def initialize(self):
        """Initialize all JARVIS components"""
        print("""
╔══════════════════════════════════════════════════╗
║         🚀 JARVIS ENHANCED STARTING 🚀           ║
║          Multi-AI Integration Active             ║
╚══════════════════════════════════════════════════╝
        """)
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            self.redis_client.set("jarvis:status", "online")
            self.redis_client.set("jarvis:version", self.version)
            logger.info("✅ Redis connected")
        except:
            logger.warning("⚠️  Redis not available")
        
        # Initialize Multi-AI
        try:
            from core.updated_multi_ai_integration import multi_ai
            self.multi_ai = multi_ai
            models = self.multi_ai.get_available_models()
            logger.info(f"✅ Multi-AI initialized with {len(models)} models: {models}")
            self.components['multi_ai'] = True
        except Exception as e:
            logger.error(f"❌ Multi-AI failed: {e}")
            self.components['multi_ai'] = False
        
        # Try to load other components
        components_to_load = [
            ('consciousness', 'core.consciousness_simulation'),
            ('self_healing', 'core.self_healing_system'),
            ('neural_manager', 'core.neural_resource_manager')
        ]
        
        for comp_name, module_path in components_to_load:
            try:
                module = __import__(module_path, fromlist=[comp_name])
                self.components[comp_name] = getattr(module, comp_name, None)
                logger.info(f"✅ Loaded {comp_name}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load {comp_name}: {e}")
        
        logger.info(f"\n🎯 JARVIS {self.version} initialized successfully!")
        logger.info(f"📊 Components loaded: {sum(1 for v in self.components.values() if v)}/{len(self.components)}")
        
        return True
    
    async def process_query(self, query: str, model: str = None) -> str:
        """Process a query using multi-AI"""
        if self.multi_ai:
            try:
                response = await self.multi_ai.query(query, model=model)
                
                # Update Redis with last query
                if self.redis_client:
                    self.redis_client.set("jarvis:last_query", query)
                    self.redis_client.set("jarvis:last_response", response[:500])
                    self.redis_client.set("jarvis:last_query_time", datetime.now().isoformat())
                
                return response
            except Exception as e:
                logger.error(f"Query processing error: {e}")
                return f"Error processing query: {str(e)}"
        else:
            return "Multi-AI not available"
    
    async def heartbeat(self):
        """Maintain heartbeat"""
        while self.active:
            try:
                if self.redis_client:
                    self.redis_client.set("jarvis:heartbeat", datetime.now().isoformat())
                    self.redis_client.expire("jarvis:heartbeat", 30)
                    
                    # Store component status
                    status = {
                        "uptime": (datetime.now() - self.start_time).total_seconds(),
                        "components": {k: bool(v) for k, v in self.components.items()},
                        "version": self.version
                    }
                    self.redis_client.set("jarvis:detailed_status", str(status))
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    async def demo_multi_ai(self):
        """Demonstrate multi-AI capabilities"""
        await asyncio.sleep(2)  # Let system stabilize
        
        logger.info("\n🎭 Demonstrating Multi-AI Capabilities...")
        
        # Test queries
        test_queries = [
            ("What is photosynthesis in simple terms?", "gpt4"),
            ("Write a haiku about artificial intelligence", "gemini"),
            ("What are the benefits of Redis for caching?", None)  # Use default/fallback
        ]
        
        for query, model in test_queries:
            logger.info(f"\n📝 Query: {query}")
            logger.info(f"🤖 Model: {model or 'auto'}")
            response = await self.process_query(query, model)
            logger.info(f"💬 Response: {response[:200]}...")
            await asyncio.sleep(1)
    
    async def run(self):
        """Main run loop"""
        if not await self.initialize():
            return
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.heartbeat()),
        ]
        
        # Add demo task if in demo mode
        if os.getenv("JARVIS_DEMO", "false").lower() == "true":
            tasks.append(asyncio.create_task(self.demo_multi_ai()))
        
        logger.info("\n✅ JARVIS Enhanced is ONLINE!")
        logger.info("🤖 Multi-AI Integration: Active")
        logger.info("📝 Logs: logs/jarvis_enhanced_*.log")
        logger.info("🛑 Stop: Ctrl+C or pkill -f jarvis")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down JARVIS Enhanced...")
        self.active = False
        
        if self.redis_client:
            self.redis_client.set("jarvis:status", "offline")
            self.redis_client.set("jarvis:shutdown_time", datetime.now().isoformat())
        
        logger.info("JARVIS Enhanced shutdown complete")

def main():
    """Entry point"""
    try:
        # Kill old processes
        os.system("pkill -f jarvis_minimal")
        
        # Create PID file
        pid_file = Path("jarvis.pid")
        pid_file.write_text(str(os.getpid()))
        
        # Run JARVIS
        jarvis = EnhancedJARVIS()
        asyncio.run(jarvis.run())
        
    except Exception as e:
        logger.error(f"Failed to start JARVIS: {e}")
        sys.exit(1)
    finally:
        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink()

if __name__ == "__main__":
    # Set demo mode for first run
    os.environ["JARVIS_DEMO"] = "true"
    main()
