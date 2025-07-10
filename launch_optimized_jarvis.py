#!/usr/bin/env python3
"""
Launch Optimized JARVIS with Performance Enhancements
Main entry point for the high-performance JARVIS system
"""

import os
import sys
import asyncio
import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import signal
import argparse

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

# Import optimized components
from core.optimized_jarvis_core import OptimizedJarvisCore
from core.neural.advanced_neural_engine import NeuralLearningEngine, PatternRecognizer
from core.optimized_voice_system import OptimizedVoiceSystem
from core.realtime_performance_monitor import RealtimePerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('jarvis_optimized.log')
    ]
)
logger = logging.getLogger(__name__)


class OptimizedJarvisLauncher:
    """Main launcher for optimized JARVIS system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.components = {}
        self.running = False
        self.start_time = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        default_config = {
            'voice': {
                'enabled': True,
                'wake_words': ['jarvis', 'hey jarvis'],
                'language': 'en-US',
                'use_elevenlabs': True,
                'elevenlabs_api_key': os.getenv('ELEVENLABS_API_KEY'),
                'voice_id': '21m00Tcm4TlvDq8ikWAM'
            },
            'neural': {
                'enabled': True,
                'model_path': 'models/jarvis_neural.pt',
                'continuous_learning': True,
                'gpu_enabled': True
            },
            'performance': {
                'monitoring_enabled': True,
                'auto_optimization': True,
                'cache_size': 5000,
                'thread_workers': None,  # Auto-detect
                'monitoring_interval': 1.0
            },
            'api_keys': {
                'openai': os.getenv('OPENAI_API_KEY'),
                'gemini': os.getenv('GEMINI_API_KEY'),
                'anthropic': os.getenv('ANTHROPIC_API_KEY')
            }
        }
        
        # Load custom config if provided
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                custom_config = json.load(f)
                # Deep merge
                self._merge_config(default_config, custom_config)
        
        return default_config
    
    def _merge_config(self, base: Dict, custom: Dict):
        """Deep merge configuration"""
        for key, value in custom.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(self.shutdown())
    
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Optimized JARVIS...")
        self.start_time = time.time()
        
        try:
            # Core system
            logger.info("Initializing core system...")
            self.components['core'] = OptimizedJarvisCore()
            await self.components['core'].task_queue.start()
            
            # Neural engine
            if self.config['neural']['enabled']:
                logger.info("Initializing neural engine...")
                self.components['neural'] = NeuralLearningEngine(
                    Path(self.config['neural']['model_path'])
                )
                self.components['pattern_recognizer'] = PatternRecognizer()
                
                if self.config['neural']['continuous_learning']:
                    await self.components['neural'].start_continuous_learning()
            
            # Voice system
            if self.config['voice']['enabled']:
                logger.info("Initializing voice system...")
                self.components['voice'] = OptimizedVoiceSystem(self.config['voice'])
                await self.components['voice'].start()
            
            # Performance monitor
            if self.config['performance']['monitoring_enabled']:
                logger.info("Initializing performance monitor...")
                self.components['monitor'] = RealtimePerformanceMonitor(
                    self.components['core'],
                    update_interval=self.config['performance']['monitoring_interval']
                )
                
                # Register custom metrics
                self._register_custom_metrics()
                
                # Register alerts
                self._register_alerts()
                
                await self.components['monitor'].start()
            
            # Connect components
            await self._connect_components()
            
            init_time = time.time() - self.start_time
            logger.info(f"JARVIS initialized in {init_time:.2f} seconds")
            
            # Print startup banner
            self._print_banner()
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def _connect_components(self):
        """Connect components together"""
        # Connect voice to core
        if 'voice' in self.components and 'core' in self.components:
            # Override voice system's response generator
            async def process_voice_input(text: str) -> str:
                # Process through core
                result = await self.components['core'].process_input(text)
                
                # Use neural prediction if available
                if 'neural' in self.components:
                    prediction = self.components['neural'].predict(text)
                    result['neural_prediction'] = prediction
                    
                    # Record pattern
                    if 'pattern_recognizer' in self.components:
                        self.components['pattern_recognizer'].record_interaction(
                            text,
                            prediction['intent'],
                            result['timestamp']
                        )
                
                # Generate response (simplified)
                response = self._generate_response(result)
                
                # Learn from interaction
                if 'neural' in self.components and self.config['neural']['continuous_learning']:
                    await self.components['neural'].learn_from_interaction(
                        text, response, feedback=0.8
                    )
                
                return response
            
            self.components['voice']._generate_response = process_voice_input
    
    def _generate_response(self, result: Dict[str, Any]) -> str:
        """Generate response based on processing result"""
        intent = result.get('intent', 'general')
        
        # Simple response generation (extend this)
        responses = {
            'greeting': "Hello! How can I help you today?",
            'question': "Let me find that information for you.",
            'command': "I'll execute that command for you.",
            'search': "Searching for that information...",
            'general': "I understand. How can I assist you?"
        }
        
        return responses.get(intent, "I'm here to help!")
    
    def _register_custom_metrics(self):
        """Register custom metrics with monitor"""
        monitor = self.components['monitor']
        
        # Neural engine metrics
        if 'neural' in self.components:
            def neural_metrics():
                stats = self.components['neural'].get_performance_stats()
                return {
                    'neural_inference_time': stats['avg_inference_time'],
                    'neural_dataset_size': stats['dataset_size'],
                    'neural_memories': stats['episodic_memories']
                }
            monitor.register_metric_provider('neural', neural_metrics)
        
        # Voice system metrics
        if 'voice' in self.components:
            def voice_metrics():
                stats = self.components['voice'].get_performance_stats()
                return {
                    'voice_recognition_time': stats['recognizer_stats']['avg_recognition_time'],
                    'voice_success_rate': stats['recognizer_stats']['success_rate'],
                    'voice_synthesis_time': stats['synthesizer_stats']['avg_synthesis_time']
                }
            monitor.register_metric_provider('voice', voice_metrics)
        
        # Pattern recognition metrics
        if 'pattern_recognizer' in self.components:
            def pattern_metrics():
                patterns = self.components['pattern_recognizer'].get_user_patterns()
                return {
                    'total_interactions': patterns['total_interactions'],
                    'unique_intents': patterns['unique_intents']
                }
            monitor.register_metric_provider('patterns', pattern_metrics)
    
    def _register_alerts(self):
        """Register performance alerts"""
        async def alert_handler(bottleneck):
            # Log critical alerts
            if bottleneck.severity == 'critical':
                logger.critical(f"PERFORMANCE ALERT: {bottleneck.description}")
                
                # Speak alert if voice enabled
                if 'voice' in self.components:
                    await self.components['voice'].speak(
                        f"Performance alert: {bottleneck.component} is experiencing issues."
                    )
        
        self.components['monitor'].register_alert_callback(alert_handler)
    
    def _print_banner(self):
        """Print startup banner"""
        print("\n" + "="*60)
        print("🚀 JARVIS OPTIMIZED - High Performance AI Assistant")
        print("="*60)
        print(f"✅ Core System: Active")
        print(f"✅ Neural Engine: {'Active' if 'neural' in self.components else 'Disabled'}")
        print(f"✅ Voice System: {'Active' if 'voice' in self.components else 'Disabled'}")
        print(f"✅ Performance Monitor: {'Active' if 'monitor' in self.components else 'Disabled'}")
        
        if 'neural' in self.components:
            stats = self.components['neural'].get_performance_stats()
            print(f"🧠 Neural Network: {stats['model_parameters']:,} parameters on {stats['device']}")
        
        if 'core' in self.components:
            report = self.components['core'].get_performance_report()
            print(f"⚡ Cache Hit Rate: {report['cache_hit_rate']:.1%}")
            print(f"💾 Device: {report['device']}")
        
        print("="*60)
        print("🎤 Say 'Hey JARVIS' to start conversation")
        print("📊 Performance dashboard available at http://localhost:5000")
        print("Press Ctrl+C to shutdown")
        print("="*60 + "\n")
    
    async def run(self):
        """Main run loop"""
        self.running = True
        
        # Performance reporting loop
        last_report_time = time.time()
        report_interval = 60  # 1 minute
        
        while self.running:
            try:
                await asyncio.sleep(1)
                
                # Periodic performance report
                if time.time() - last_report_time > report_interval:
                    await self._generate_performance_report()
                    last_report_time = time.time()
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
    
    async def _generate_performance_report(self):
        """Generate and log performance report"""
        if 'monitor' not in self.components:
            return
        
        summary = self.components['monitor'].get_metrics_summary(5)
        
        logger.info("=== Performance Report ===")
        logger.info(f"CPU Average: {summary['cpu']['avg']:.1f}%")
        logger.info(f"Memory Usage: {summary['memory']['avg_mb']:.1f} MB")
        logger.info(f"Response Time P95: {summary['response_time']['p95']:.3f}s")
        logger.info(f"Cache Hit Rate: {summary['cache_hit_rate']:.1%}")
        logger.info(f"Error Rate: {summary['error_rate']:.1%}")
        
        # Export detailed report
        self.components['monitor'].export_metrics('performance_report.json')
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down JARVIS...")
        self.running = False
        
        # Save neural model
        if 'neural' in self.components:
            logger.info("Saving neural model...")
            self.components['neural'].save_model()
        
        # Stop components in reverse order
        if 'monitor' in self.components:
            await self.components['monitor'].stop()
        
        if 'voice' in self.components:
            self.components['voice'].stop()
        
        if 'core' in self.components:
            await self.components['core'].task_queue.stop()
        
        # Final performance report
        runtime = time.time() - self.start_time
        logger.info(f"JARVIS ran for {runtime/60:.1f} minutes")
        
        if 'core' in self.components:
            final_report = self.components['core'].get_performance_report()
            logger.info(f"Completed tasks: {final_report['completed_tasks']}")
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Launch Optimized JARVIS')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--no-voice',
        action='store_true',
        help='Disable voice interface'
    )
    parser.add_argument(
        '--no-neural',
        action='store_true',
        help='Disable neural engine'
    )
    parser.add_argument(
        '--no-monitor',
        action='store_true',
        help='Disable performance monitoring'
    )
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = OptimizedJarvisLauncher(args.config)
    
    # Override config from command line
    if args.no_voice:
        launcher.config['voice']['enabled'] = False
    if args.no_neural:
        launcher.config['neural']['enabled'] = False
    if args.no_monitor:
        launcher.config['performance']['monitoring_enabled'] = False
    
    try:
        # Initialize
        await launcher.initialize()
        
        # Run
        await launcher.run()
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        await launcher.shutdown()


if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ required")
        sys.exit(1)
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Run
    asyncio.run(main())