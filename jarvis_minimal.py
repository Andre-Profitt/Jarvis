#!/usr/bin/env python3
"""
JARVIS Minimal Launcher - Works around all syntax errors
"""
import asyncio
import logging
import sys
import os
import json
import redis
from datetime import datetime
from pathlib import Path

# Setup basic logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"jarvis_minimal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MinimalJARVIS:
    """Minimal JARVIS implementation that actually works"""
    
    def __init__(self):
        self.name = "JARVIS"
        self.version = "1.0-minimal"
        self.active = True
        self.redis_client = None
        self.start_time = datetime.now()
        
        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Connected to Redis")
        except:
            logger.warning("⚠️  Redis not available - running without persistence")
    
    async def initialize(self):
        """Initialize JARVIS components"""
        logger.info("""
╔══════════════════════════════════════════╗
║         JARVIS SYSTEM STARTING           ║
║      Minimal Mode - Core Functions       ║
╚══════════════════════════════════════════╝
        """)
        
        # Store startup info in Redis if available
        if self.redis_client:
            self.redis_client.set("jarvis:status", "online")
            self.redis_client.set("jarvis:start_time", self.start_time.isoformat())
            self.redis_client.set("jarvis:version", self.version)
        
        logger.info(f"JARVIS {self.version} initialized at {self.start_time}")
        return True
    
    async def heartbeat(self):
        """Maintain heartbeat"""
        while self.active:
            try:
                if self.redis_client:
                    self.redis_client.set("jarvis:heartbeat", datetime.now().isoformat())
                    self.redis_client.expire("jarvis:heartbeat", 30)
                
                uptime = (datetime.now() - self.start_time).total_seconds()
                logger.debug(f"Heartbeat - Uptime: {uptime:.0f}s")
                
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    async def shutdown(self):
        """Clean shutdown"""
        logger.info("Shutting down JARVIS...")
        self.active = False
        
        if self.redis_client:
            self.redis_client.set("jarvis:status", "offline")
            self.redis_client.set("jarvis:shutdown_time", datetime.now().isoformat())
        
        logger.info("JARVIS shutdown complete")
    
    async def run(self):
        """Main run loop"""
        await self.initialize()
        
        # Create tasks
        tasks = [
            asyncio.create_task(self.heartbeat()),
        ]
        
        logger.info("✅ JARVIS is ONLINE and RUNNING!")
        logger.info("📊 Status: Minimal mode - Core systems only")
        logger.info("📝 Logs: Check logs/jarvis_minimal_*.log")
        logger.info("🛑 Stop: Ctrl+C or pkill -f jarvis")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Runtime error: {e}")
        finally:
            await self.shutdown()

def main():
    """Entry point"""
    try:
        # Create PID file
        pid_file = Path("jarvis.pid")
        pid_file.write_text(str(os.getpid()))
        
        # Run JARVIS
        jarvis = MinimalJARVIS()
        asyncio.run(jarvis.run())
        
    except Exception as e:
        logger.error(f"Failed to start JARVIS: {e}")
        sys.exit(1)
    finally:
        # Clean up PID file
        if pid_file.exists():
            pid_file.unlink()

if __name__ == "__main__":
    main()
