#!/usr/bin/env python3
"""
JARVIS - Advanced AI Assistant
Main entry point with clean architecture
"""
import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

from core.assistant import Assistant
from core.config import Config
from core.logger import setup_logger

logger = setup_logger(__name__)


class JARVIS:
    """Main JARVIS application controller"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = Config(config_path or Path("config.yaml"))
        self.assistant = Assistant(self.config)
        self.running = False
        
    async def start(self):
        """Start JARVIS assistant"""
        logger.info("🚀 Starting JARVIS...")
        
        try:
            # Initialize components
            await self.assistant.initialize()
            
            # Setup signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, self._signal_handler)
            
            self.running = True
            logger.info("✅ JARVIS is ready!")
            
            # Main loop
            await self.assistant.run()
            
        except Exception as e:
            logger.error(f"Failed to start JARVIS: {e}")
            raise
            
    async def stop(self):
        """Gracefully stop JARVIS"""
        logger.info("Shutting down JARVIS...")
        self.running = False
        await self.assistant.shutdown()
        logger.info("👋 JARVIS stopped")
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        asyncio.create_task(self.stop())
        

async def main():
    """Main entry point"""
    jarvis = JARVIS()
    
    try:
        await jarvis.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await jarvis.stop()


if __name__ == "__main__":
    asyncio.run(main())