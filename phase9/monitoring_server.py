"""
JARVIS Phase 9: Performance Monitoring Server
============================================
WebSocket server for real-time performance monitoring dashboard
"""

import asyncio
import websockets
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase9.performance_optimizer import JARVISPerformanceOptimizer
from phase9.jarvis_phase9_integration import JARVISOptimizedCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitoringServer:
    """
    WebSocket server for real-time performance monitoring
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.optimizer = None
        self.jarvis_core = None
        self._running = False
        self._monitor_task = None
    
    async def initialize(self):
        """Initialize monitoring server"""
        logger.info("Initializing Performance Monitoring Server...")
        
        # Initialize JARVIS components
        self.jarvis_core = JARVISOptimizedCore()
        await self.jarvis_core.initialize()
        
        self.optimizer = self.jarvis_core.optimizer
        
        logger.info("Monitoring server initialized")
    
    async def register_client(self, websocket):
        """Register a new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send initial data
        await self.send_performance_data(websocket)
    
    async def unregister_client(self, websocket):
        """Unregister a client"""
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def send_performance_data(self, websocket=None):
        """Send performance data to client(s)"""
        try:
            # Get comprehensive performance data
            report = self.optimizer.get_performance_report()
            
            # Format data for dashboard
            dashboard_data = {
                'cache_stats': report['cache_stats'],
                'current_metrics': report['monitor_dashboard']['current_metrics'],
                'parallel_stats': report['parallel_stats'],
                'trends': report['monitor_dashboard']['trends'],
                'alerts': report['monitor_dashboard']['alerts'],
                'recommendations': report['monitor_dashboard']['recommendations'],
                'optimization_level': report['optimization_level'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Add JARVIS-specific metrics if available
            if hasattr(self.jarvis_core, 'performance_metrics'):
                dashboard_data['jarvis_metrics'] = self.jarvis_core.performance_metrics
            
            # Send to specific client or broadcast
            message = json.dumps(dashboard_data)
            
            if websocket:
                await websocket.send(message)
            else:
                # Broadcast to all clients
                if self.clients:
                    await asyncio.gather(
                        *[client.send(message) for client in self.clients],
                        return_exceptions=True
                    )
                    
        except Exception as e:
            logger.error(f"Error sending performance data: {e}")
    
    async def handle_client_message(self, websocket, message):
        """Handle messages from clients"""
        try:
            data = json.loads(message)
            action = data.get('action')
            
            if action == 'set_optimization_level':
                level = data.get('level', 'balanced')
                self.optimizer.set_optimization_level(level)
                logger.info(f"Optimization level changed to: {level}")
                
                # Send updated data
                await self.send_performance_data()
                
            elif action == 'run_auto_optimize':
                await self.jarvis_core.run_auto_optimization()
                logger.info("Auto-optimization completed")
                
                # Send updated data
                await self.send_performance_data()
                
            elif action == 'clear_cache':
                self.optimizer.cache.memory_cache.clear()
                logger.info("Cache cleared")
                
                # Send updated data
                await self.send_performance_data()
                
            elif action == 'get_report':
                # Send detailed report
                report = self.optimizer.get_performance_report()
                await websocket.send(json.dumps({
                    'type': 'detailed_report',
                    'report': report
                }))
                
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
    
    async def client_handler(self, websocket, path):
        """Handle client connections"""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def monitoring_loop(self):
        """Continuous monitoring and broadcasting"""
        while self._running:
            try:
                # Send performance data every 2 seconds
                await self.send_performance_data()
                
                # Run periodic optimizations
                if asyncio.get_event_loop().time() % 30 < 2:  # Every 30 seconds
                    await self.optimizer.auto_optimize()
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def start(self):
        """Start the monitoring server"""
        await self.initialize()
        
        self._running = True
        self._monitor_task = asyncio.create_task(self.monitoring_loop())
        
        logger.info(f"Starting WebSocket server on {self.host}:{self.port}")
        
        async with websockets.serve(self.client_handler, self.host, self.port):
            logger.info(f"Performance Monitoring Server running on ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever
    
    async def stop(self):
        """Stop the monitoring server"""
        logger.info("Stopping monitoring server...")
        
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Close all client connections
        if self.clients:
            await asyncio.gather(
                *[client.close() for client in self.clients],
                return_exceptions=True
            )
        
        # Shutdown JARVIS components
        if self.jarvis_core:
            await self.jarvis_core.shutdown()
        
        logger.info("Monitoring server stopped")


# ==================== Demo Server ====================

async def run_demo_server():
    """Run a demo monitoring server with simulated activity"""
    
    print("\n🚀 JARVIS Phase 9 Monitoring Server Demo\n")
    
    # Create server
    server = PerformanceMonitoringServer()
    
    # Create demo task to generate activity
    async def generate_demo_activity():
        """Generate demo activity for monitoring"""
        demo_inputs = [
            {'type': 'text', 'data': f'Demo message {i}'}
            for i in range(100)
        ]
        
        while True:
            try:
                # Process some inputs
                batch = demo_inputs[:10]
                await server.jarvis_core.process_batch(batch)
                
                # Rotate inputs
                demo_inputs.extend(batch)
                demo_inputs = demo_inputs[10:]
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Demo activity error: {e}")
                await asyncio.sleep(10)
    
    # Start demo activity
    demo_task = asyncio.create_task(generate_demo_activity())
    
    try:
        # Start server
        await server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        demo_task.cancel()
        await server.stop()


# ==================== Standalone Server ====================

class StandaloneMonitoringServer(PerformanceMonitoringServer):
    """
    Standalone monitoring server that doesn't require full JARVIS
    """
    
    async def initialize(self):
        """Initialize standalone monitoring"""
        logger.info("Initializing Standalone Performance Monitor...")
        
        # Create standalone optimizer
        self.optimizer = JARVISPerformanceOptimizer()
        await self.optimizer.initialize()
        
        # Create mock JARVIS core
        self.jarvis_core = type('MockCore', (), {
            'performance_metrics': {
                'pipeline_calls': 0,
                'state_updates': 0,
                'cache_hits': 0,
                'parallel_operations': 0
            },
            'shutdown': lambda: asyncio.sleep(0)
        })()
        
        # Start generating mock data
        self._mock_task = asyncio.create_task(self._generate_mock_data())
        
        logger.info("Standalone monitor initialized")
    
    async def _generate_mock_data(self):
        """Generate mock performance data"""
        import random
        
        while self._running:
            try:
                # Simulate some operations
                for _ in range(random.randint(1, 5)):
                    # Record mock response time
                    self.optimizer.monitor.record_response_time(
                        f"operation_{random.randint(1, 10)}",
                        random.uniform(10, 200)
                    )
                
                # Update mock metrics
                self.jarvis_core.performance_metrics['pipeline_calls'] += random.randint(1, 10)
                self.jarvis_core.performance_metrics['cache_hits'] += random.randint(0, 5)
                
                # Record cache performance
                self.optimizer.monitor.record_cache_performance(
                    self.optimizer.cache.stats.get_stats()
                )
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Mock data generation error: {e}")
                await asyncio.sleep(5)
    
    async def stop(self):
        """Stop standalone server"""
        if hasattr(self, '_mock_task'):
            self._mock_task.cancel()
            try:
                await self._mock_task
            except asyncio.CancelledError:
                pass
        
        await super().stop()


# ==================== Main Entry Point ====================

def main():
    """Main entry point for monitoring server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JARVIS Phase 9 Performance Monitoring Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--standalone', action='store_true', help='Run in standalone mode')
    parser.add_argument('--demo', action='store_true', help='Run with demo activity')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           JARVIS Phase 9 Performance Monitor             ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  WebSocket Server: ws://{args.host}:{args.port:<35}║
║  Mode: {'Standalone' if args.standalone else 'Full JARVIS':<49}║
║                                                          ║
║  Open performance_monitor.html in a browser to connect   ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
    """)
    
    try:
        if args.demo:
            asyncio.run(run_demo_server())
        elif args.standalone:
            server = StandaloneMonitoringServer(args.host, args.port)
            asyncio.run(server.start())
        else:
            server = PerformanceMonitoringServer(args.host, args.port)
            asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
