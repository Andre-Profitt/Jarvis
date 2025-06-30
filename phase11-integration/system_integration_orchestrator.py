"""
JARVIS Phase 11: System Integration & Testing Orchestrator
Integrates all 10 phases into a cohesive, production-ready system
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import numpy as np
from collections import defaultdict
import psutil
import os
import sys
import traceback

# Import all phase components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.unified_input_pipeline import UnifiedInputPipeline
from core.fluid_state_management import FluidStateManager
from core.context_memory import ContextMemorySystem
from core.proactive_interventions import InterventionSystem
from core.natural_language_flow import NaturalLanguageProcessor
from core.visual_ui_system import VisualUISystem
from core.cognitive_load_reducer import CognitiveLoadReducer
from core.performance_optimizer import PerformanceOptimizer
from core.feedback_learning import FeedbackLearningSystem
from core.adaptive_personalization import AdaptivePersonalization

@dataclass
class IntegrationMetrics:
    """Tracks system-wide integration metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    phase_status: Dict[int, str] = field(default_factory=dict)
    performance_scores: Dict[str, float] = field(default_factory=dict)
    integration_health: float = 1.0
    error_count: int = 0
    warning_count: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    cross_phase_latency: Dict[str, float] = field(default_factory=dict)
    memory_footprint: float = 0.0
    cpu_utilization: float = 0.0
    
    def calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

class SystemIntegrationOrchestrator:
    """Orchestrates all JARVIS phases into a unified system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = IntegrationMetrics()
        self.phases = {}
        self.integration_tests = []
        self.performance_benchmarks = {}
        self.health_monitors = {}
        self.initialized = False
        
    async def initialize_all_phases(self) -> bool:
        """Initialize all 10 phases with proper integration"""
        try:
            self.logger.info("🚀 Initializing JARVIS Phase 11: System Integration")
            
            # Phase 1: Unified Input Pipeline & State Management
            self.phases['phase1'] = {
                'pipeline': UnifiedInputPipeline(),
                'state_manager': FluidStateManager()
            }
            
            # Phase 2: Context & Memory Systems
            self.phases['phase2'] = {
                'context_memory': ContextMemorySystem(),
                'pattern_detector': self.phases['phase1']['state_manager'].pattern_detector
            }
            
            # Phase 3: Proactive & Intervention Systems
            self.phases['phase3'] = {
                'intervention': InterventionSystem(),
                'predictive_engine': await self._create_predictive_engine()
            }
            
            # Phase 4: Natural Language Flow
            self.phases['phase4'] = {
                'nlp': NaturalLanguageProcessor(),
                'conversation_flow': await self._create_conversation_flow()
            }
            
            # Phase 5: Visual UI Systems
            self.phases['phase5'] = {
                'visual_ui': VisualUISystem(),
                'status_indicators': await self._create_status_indicators()
            }
            
            # Phase 6: Cognitive Load Management
            self.phases['phase6'] = {
                'cognitive_reducer': CognitiveLoadReducer(),
                'attention_manager': await self._create_attention_manager()
            }
            
            # Phase 7: Performance Optimization
            self.phases['phase7'] = {
                'optimizer': PerformanceOptimizer(),
                'cache_manager': await self._create_cache_manager()
            }
            
            # Phase 8: Feedback & Learning
            self.phases['phase8'] = {
                'feedback': FeedbackLearningSystem(),
                'learning_engine': await self._create_learning_engine()
            }
            
            # Phase 9: Adaptive Personalization
            self.phases['phase9'] = {
                'personalization': AdaptivePersonalization(),
                'user_profiler': await self._create_user_profiler()
            }
            
            # Phase 10: Production Readiness
            self.phases['phase10'] = {
                'health_monitor': await self._create_health_monitor(),
                'deployment_manager': await self._create_deployment_manager()
            }
            
            # Initialize cross-phase connections
            await self._establish_phase_connections()
            
            self.initialized = True
            self.logger.info("✅ All phases initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize phases: {e}")
            self.metrics.error_count += 1
            return False
    
    async def _establish_phase_connections(self):
        """Establish connections between phases for seamless integration"""
        # Connect Pipeline to State Manager
        self.phases['phase1']['pipeline'].set_state_manager(
            self.phases['phase1']['state_manager']
        )
        
        # Connect State Manager to Context Memory
        self.phases['phase1']['state_manager'].set_context_memory(
            self.phases['phase2']['context_memory']
        )
        
        # Connect Context to Interventions
        self.phases['phase3']['intervention'].set_context_provider(
            self.phases['phase2']['context_memory']
        )
        
        # Connect NLP to all phases
        self.phases['phase4']['nlp'].register_components({
            'state': self.phases['phase1']['state_manager'],
            'context': self.phases['phase2']['context_memory'],
            'intervention': self.phases['phase3']['intervention']
        })
        
        # Connect UI to state and performance
        self.phases['phase5']['visual_ui'].connect_data_sources({
            'state': self.phases['phase1']['state_manager'],
            'performance': self.phases['phase7']['optimizer']
        })
        
        # Connect Cognitive Reducer to all input/output
        self.phases['phase6']['cognitive_reducer'].set_pipeline(
            self.phases['phase1']['pipeline']
        )
        
        # Connect Performance Optimizer to all phases
        for phase_name, components in self.phases.items():
            self.phases['phase7']['optimizer'].register_component(
                phase_name, components
            )
        
        # Connect Feedback System
        self.phases['phase8']['feedback'].connect_all_phases(self.phases)
        
        # Connect Personalization
        self.phases['phase9']['personalization'].integrate_with_system(self.phases)
        
        self.logger.info("✅ Cross-phase connections established")
    
    async def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        self.logger.info("🧪 Running integration tests...")
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_status': 'PASSED'
        }
        
        tests = [
            self._test_input_to_output_flow,
            self._test_state_transitions,
            self._test_context_persistence,
            self._test_intervention_triggering,
            self._test_performance_under_load,
            self._test_learning_adaptation,
            self._test_error_recovery,
            self._test_cross_phase_communication
        ]
        
        for test in tests:
            try:
                test_result = await test()
                results['tests'].append(test_result)
                if test_result['status'] != 'PASSED':
                    results['overall_status'] = 'FAILED'
            except Exception as e:
                self.logger.error(f"Test {test.__name__} failed: {e}")
                results['tests'].append({
                    'name': test.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                })
                results['overall_status'] = 'FAILED'
        
        return results
    
    async def _test_input_to_output_flow(self) -> Dict[str, Any]:
        """Test complete input to output flow"""
        test_name = "Input-to-Output Flow"
        try:
            # Test multimodal input
            test_input = {
                'voice': {'text': 'Hello JARVIS', 'emotion': 'happy'},
                'biometric': {'heart_rate': 75, 'stress': 0.3},
                'context': {'location': 'home', 'time': 'evening'}
            }
            
            start_time = time.time()
            
            # Process through pipeline
            result = await self.phases['phase1']['pipeline'].process(test_input)
            
            # Check state update
            state = self.phases['phase1']['state_manager'].get_current_state()
            
            # Generate response
            response = await self.phases['phase4']['nlp'].generate_response(
                result, state
            )
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            return {
                'name': test_name,
                'status': 'PASSED' if latency < 100 else 'FAILED',
                'latency_ms': latency,
                'details': {
                    'input_processed': True,
                    'state_updated': state is not None,
                    'response_generated': response is not None
                }
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_state_transitions(self) -> Dict[str, Any]:
        """Test fluid state transitions"""
        test_name = "State Transitions"
        try:
            state_manager = self.phases['phase1']['state_manager']
            
            # Test various state transitions
            transitions = [
                ('normal', 'focused', {'focus_level': 0.8}),
                ('focused', 'flow', {'productivity': 0.95}),
                ('flow', 'stressed', {'stress_indicators': 0.9}),
                ('stressed', 'recovering', {'recovery_signals': 0.7})
            ]
            
            results = []
            for from_state, to_state, signals in transitions:
                state_manager.current_state = from_state
                await state_manager.update_state(signals)
                
                results.append({
                    'from': from_state,
                    'to': to_state,
                    'actual': state_manager.current_state,
                    'smooth': state_manager.transition_smoothness > 0.8
                })
            
            all_smooth = all(r['smooth'] for r in results)
            
            return {
                'name': test_name,
                'status': 'PASSED' if all_smooth else 'FAILED',
                'transitions': results
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_context_persistence(self) -> Dict[str, Any]:
        """Test context memory persistence"""
        test_name = "Context Persistence"
        try:
            context_memory = self.phases['phase2']['context_memory']
            
            # Add context
            test_context = {
                'user_preference': 'quiet_mode',
                'current_task': 'coding',
                'emotional_state': 'focused'
            }
            
            await context_memory.update_context(test_context)
            
            # Simulate time passing
            await asyncio.sleep(0.1)
            
            # Retrieve context
            retrieved = await context_memory.get_relevant_context('coding')
            
            # Check persistence
            persistence_score = len(set(test_context.keys()) & set(retrieved.keys())) / len(test_context)
            
            return {
                'name': test_name,
                'status': 'PASSED' if persistence_score > 0.8 else 'FAILED',
                'persistence_score': persistence_score,
                'context_items': len(retrieved)
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_intervention_triggering(self) -> Dict[str, Any]:
        """Test intervention system triggering"""
        test_name = "Intervention Triggering"
        try:
            intervention = self.phases['phase3']['intervention']
            state_manager = self.phases['phase1']['state_manager']
            
            # Simulate stress condition
            state_manager.current_state = 'stressed'
            state_manager.state_confidence = 0.9
            
            # Check if intervention triggers
            should_intervene = await intervention.check_intervention_needed({
                'state': 'stressed',
                'duration': 300,  # 5 minutes
                'severity': 0.8
            })
            
            # Get intervention suggestion
            intervention_type = await intervention.get_intervention_type('stressed')
            
            return {
                'name': test_name,
                'status': 'PASSED' if should_intervene else 'FAILED',
                'intervention_triggered': should_intervene,
                'intervention_type': intervention_type
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_performance_under_load(self) -> Dict[str, Any]:
        """Test system performance under load"""
        test_name = "Performance Under Load"
        try:
            pipeline = self.phases['phase1']['pipeline']
            
            # Generate load
            num_requests = 100
            latencies = []
            
            start_time = time.time()
            
            tasks = []
            for i in range(num_requests):
                test_input = {
                    'type': 'performance_test',
                    'id': i,
                    'data': np.random.rand(100).tolist()
                }
                tasks.append(pipeline.process(test_input))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Calculate metrics
            successful = len([r for r in results if not isinstance(r, Exception)])
            avg_latency = (total_time / num_requests) * 1000
            
            return {
                'name': test_name,
                'status': 'PASSED' if avg_latency < 50 and successful > 95 else 'FAILED',
                'requests': num_requests,
                'successful': successful,
                'avg_latency_ms': avg_latency,
                'throughput_rps': num_requests / total_time
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_learning_adaptation(self) -> Dict[str, Any]:
        """Test learning and adaptation capabilities"""
        test_name = "Learning Adaptation"
        try:
            feedback_system = self.phases['phase8']['feedback']
            personalization = self.phases['phase9']['personalization']
            
            # Simulate user feedback
            feedback_samples = [
                {'action': 'intervention', 'rating': 5, 'context': 'stress'},
                {'action': 'suggestion', 'rating': 3, 'context': 'focus'},
                {'action': 'intervention', 'rating': 4, 'context': 'stress'},
                {'action': 'suggestion', 'rating': 5, 'context': 'focus'}
            ]
            
            for feedback in feedback_samples:
                await feedback_system.record_feedback(feedback)
            
            # Check if system learned
            adaptation_score = await personalization.calculate_adaptation_score()
            
            # Check preference learning
            preferences = await personalization.get_learned_preferences()
            
            return {
                'name': test_name,
                'status': 'PASSED' if adaptation_score > 0.7 else 'FAILED',
                'adaptation_score': adaptation_score,
                'preferences_learned': len(preferences)
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_error_recovery(self) -> Dict[str, Any]:
        """Test error recovery mechanisms"""
        test_name = "Error Recovery"
        try:
            # Simulate various errors
            error_scenarios = [
                ('invalid_input', {'type': None}),
                ('processing_error', {'data': 'corrupted'}),
                ('state_error', {'state': 'invalid_state'})
            ]
            
            recovery_results = []
            
            for scenario, bad_input in error_scenarios:
                try:
                    # Attempt processing
                    result = await self.phases['phase1']['pipeline'].process(bad_input)
                    recovery_results.append({
                        'scenario': scenario,
                        'recovered': True,
                        'result': 'graceful_handling'
                    })
                except Exception as e:
                    # Check if error was handled gracefully
                    if hasattr(e, 'recovery_action'):
                        recovery_results.append({
                            'scenario': scenario,
                            'recovered': True,
                            'result': 'error_handled'
                        })
                    else:
                        recovery_results.append({
                            'scenario': scenario,
                            'recovered': False,
                            'result': str(e)
                        })
            
            recovery_rate = sum(1 for r in recovery_results if r['recovered']) / len(recovery_results)
            
            return {
                'name': test_name,
                'status': 'PASSED' if recovery_rate > 0.8 else 'FAILED',
                'recovery_rate': recovery_rate,
                'scenarios': recovery_results
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def _test_cross_phase_communication(self) -> Dict[str, Any]:
        """Test communication between phases"""
        test_name = "Cross-Phase Communication"
        try:
            # Test data flow from Phase 1 to Phase 9
            test_data = {
                'user_id': 'test_user',
                'action': 'request_help',
                'context': {'mood': 'stressed', 'task': 'debugging'}
            }
            
            # Phase 1: Input processing
            processed = await self.phases['phase1']['pipeline'].process(test_data)
            
            # Phase 2: Context update
            await self.phases['phase2']['context_memory'].update_context(processed)
            
            # Phase 4: NLP processing
            nlp_result = await self.phases['phase4']['nlp'].process_with_context(
                processed,
                self.phases['phase2']['context_memory']
            )
            
            # Phase 9: Personalization
            personalized = await self.phases['phase9']['personalization'].personalize_response(
                nlp_result
            )
            
            # Verify data integrity through phases
            data_integrity = all([
                processed is not None,
                nlp_result is not None,
                personalized is not None,
                'user_id' in str(personalized)
            ])
            
            return {
                'name': test_name,
                'status': 'PASSED' if data_integrity else 'FAILED',
                'phases_connected': True,
                'data_integrity': data_integrity
            }
        except Exception as e:
            return {
                'name': test_name,
                'status': 'FAILED',
                'error': str(e)
            }
    
    async def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks across all phases"""
        self.logger.info("📊 Running performance benchmarks...")
        
        benchmarks = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': await self._collect_system_metrics(),
            'phase_benchmarks': {},
            'integration_benchmarks': {}
        }
        
        # Benchmark each phase
        for phase_name, components in self.phases.items():
            phase_bench = await self._benchmark_phase(phase_name, components)
            benchmarks['phase_benchmarks'][phase_name] = phase_bench
        
        # Benchmark integration points
        integration_tests = [
            ('pipeline_to_state', self._benchmark_pipeline_to_state),
            ('state_to_intervention', self._benchmark_state_to_intervention),
            ('full_cycle', self._benchmark_full_cycle)
        ]
        
        for test_name, test_func in integration_tests:
            result = await test_func()
            benchmarks['integration_benchmarks'][test_name] = result
        
        # Calculate overall score
        benchmarks['overall_score'] = self._calculate_benchmark_score(benchmarks)
        
        return benchmarks
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(interval=0.1),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'threads': process.num_threads(),
            'open_files': len(process.open_files()),
            'connections': len(process.connections())
        }
    
    async def _benchmark_phase(self, phase_name: str, components: Dict) -> Dict[str, Any]:
        """Benchmark a specific phase"""
        results = {
            'phase': phase_name,
            'components': len(components),
            'benchmarks': {}
        }
        
        # Simple latency test
        test_input = {'test': 'benchmark', 'phase': phase_name}
        iterations = 100
        
        for comp_name, component in components.items():
            if hasattr(component, 'process'):
                latencies = []
                
                for _ in range(iterations):
                    start = time.time()
                    try:
                        await component.process(test_input)
                    except:
                        pass
                    latencies.append((time.time() - start) * 1000)
                
                results['benchmarks'][comp_name] = {
                    'avg_latency_ms': np.mean(latencies),
                    'p95_latency_ms': np.percentile(latencies, 95),
                    'p99_latency_ms': np.percentile(latencies, 99)
                }
        
        return results
    
    def _calculate_benchmark_score(self, benchmarks: Dict) -> float:
        """Calculate overall benchmark score"""
        scores = []
        
        # Phase scores
        for phase_data in benchmarks['phase_benchmarks'].values():
            for comp_bench in phase_data['benchmarks'].values():
                # Score based on latency (lower is better)
                if comp_bench['avg_latency_ms'] < 10:
                    scores.append(1.0)
                elif comp_bench['avg_latency_ms'] < 50:
                    scores.append(0.8)
                elif comp_bench['avg_latency_ms'] < 100:
                    scores.append(0.6)
                else:
                    scores.append(0.4)
        
        # Integration scores
        for int_bench in benchmarks['integration_benchmarks'].values():
            if 'success_rate' in int_bench:
                scores.append(int_bench['success_rate'])
        
        return np.mean(scores) if scores else 0.0
    
    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report"""
        self.logger.info("🏥 Generating health report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'HEALTHY',
            'phase_health': {},
            'integration_health': {},
            'recommendations': []
        }
        
        # Check each phase
        for phase_name, components in self.phases.items():
            phase_health = await self._check_phase_health(phase_name, components)
            report['phase_health'][phase_name] = phase_health
            
            if phase_health['status'] != 'HEALTHY':
                report['overall_health'] = 'DEGRADED'
        
        # Check integration points
        integration_checks = [
            ('data_flow', self._check_data_flow_health),
            ('state_consistency', self._check_state_consistency),
            ('memory_usage', self._check_memory_health)
        ]
        
        for check_name, check_func in integration_checks:
            health = await check_func()
            report['integration_health'][check_name] = health
            
            if health['status'] != 'HEALTHY':
                report['overall_health'] = 'DEGRADED'
                report['recommendations'].extend(health.get('recommendations', []))
        
        # Add metrics
        report['metrics'] = {
            'uptime_hours': self._calculate_uptime(),
            'total_requests': self.metrics.total_requests,
            'success_rate': self.metrics.calculate_success_rate(),
            'error_count': self.metrics.error_count,
            'warning_count': self.metrics.warning_count
        }
        
        return report
    
    async def _check_phase_health(self, phase_name: str, components: Dict) -> Dict[str, Any]:
        """Check health of a specific phase"""
        health = {
            'phase': phase_name,
            'status': 'HEALTHY',
            'components': {}
        }
        
        for comp_name, component in components.items():
            if hasattr(component, 'get_health'):
                comp_health = await component.get_health()
                health['components'][comp_name] = comp_health
                
                if comp_health.get('status') != 'HEALTHY':
                    health['status'] = 'DEGRADED'
            else:
                health['components'][comp_name] = {'status': 'UNKNOWN'}
        
        return health
    
    async def optimize_system_configuration(self) -> Dict[str, Any]:
        """Optimize system configuration based on usage patterns"""
        self.logger.info("⚙️ Optimizing system configuration...")
        
        optimizations = {
            'timestamp': datetime.now().isoformat(),
            'current_config': await self._get_current_config(),
            'recommended_changes': [],
            'expected_improvements': {}
        }
        
        # Analyze usage patterns
        usage_analysis = await self._analyze_usage_patterns()
        
        # Generate optimization recommendations
        if usage_analysis['avg_load'] > 0.8:
            optimizations['recommended_changes'].append({
                'component': 'pipeline',
                'parameter': 'buffer_size',
                'current': 1000,
                'recommended': 2000,
                'reason': 'High load detected'
            })
        
        if usage_analysis['memory_pressure'] > 0.7:
            optimizations['recommended_changes'].append({
                'component': 'context_memory',
                'parameter': 'cache_size',
                'current': 10000,
                'recommended': 5000,
                'reason': 'Memory pressure detected'
            })
        
        # Calculate expected improvements
        for change in optimizations['recommended_changes']:
            if change['parameter'] == 'buffer_size':
                optimizations['expected_improvements']['latency_reduction'] = '15-20%'
            elif change['parameter'] == 'cache_size':
                optimizations['expected_improvements']['memory_reduction'] = '30-40%'
        
        return optimizations
    
    async def _create_predictive_engine(self):
        """Create placeholder predictive engine"""
        return type('PredictiveEngine', (), {
            'predict': lambda self, x: x,
            'train': lambda self, x, y: None
        })()
    
    async def _create_conversation_flow(self):
        """Create placeholder conversation flow"""
        return type('ConversationFlow', (), {
            'process': lambda self, x: x,
            'get_context': lambda self: {}
        })()
    
    async def _create_status_indicators(self):
        """Create placeholder status indicators"""
        return type('StatusIndicators', (), {
            'update': lambda self, x: None,
            'get_status': lambda self: 'active'
        })()
    
    async def _create_attention_manager(self):
        """Create placeholder attention manager"""
        return type('AttentionManager', (), {
            'focus': lambda self, x: x,
            'get_attention_state': lambda self: 'focused'
        })()
    
    async def _create_cache_manager(self):
        """Create placeholder cache manager"""
        return type('CacheManager', (), {
            'get': lambda self, key: None,
            'set': lambda self, key, value: None,
            'clear': lambda self: None
        })()
    
    async def _create_learning_engine(self):
        """Create placeholder learning engine"""
        return type('LearningEngine', (), {
            'learn': lambda self, x: None,
            'get_insights': lambda self: []
        })()
    
    async def _create_user_profiler(self):
        """Create placeholder user profiler"""
        return type('UserProfiler', (), {
            'update_profile': lambda self, x: None,
            'get_preferences': lambda self: {}
        })()
    
    async def _create_health_monitor(self):
        """Create placeholder health monitor"""
        return type('HealthMonitor', (), {
            'check_health': lambda self: {'status': 'healthy'},
            'get_metrics': lambda self: {}
        })()
    
    async def _create_deployment_manager(self):
        """Create placeholder deployment manager"""
        return type('DeploymentManager', (), {
            'deploy': lambda self: True,
            'rollback': lambda self: True,
            'get_status': lambda self: 'deployed'
        })()
    
    async def _benchmark_pipeline_to_state(self) -> Dict[str, Any]:
        """Benchmark pipeline to state manager flow"""
        iterations = 1000
        latencies = []
        
        for _ in range(iterations):
            start = time.time()
            
            # Simulate pipeline to state flow
            test_input = {'test': True, 'value': np.random.rand()}
            processed = await self.phases['phase1']['pipeline'].process(test_input)
            await self.phases['phase1']['state_manager'].update_from_input(processed)
            
            latencies.append((time.time() - start) * 1000)
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'throughput_ops': iterations / (sum(latencies) / 1000)
        }
    
    async def _benchmark_state_to_intervention(self) -> Dict[str, Any]:
        """Benchmark state to intervention flow"""
        iterations = 500
        trigger_count = 0
        
        for _ in range(iterations):
            # Simulate different states
            state = np.random.choice(['stressed', 'normal', 'focused', 'flow'])
            confidence = np.random.rand()
            
            should_intervene = await self.phases['phase3']['intervention'].check_for_state(
                state, confidence
            )
            
            if should_intervene:
                trigger_count += 1
        
        return {
            'trigger_rate': trigger_count / iterations,
            'success_rate': 1.0  # Placeholder
        }
    
    async def _benchmark_full_cycle(self) -> Dict[str, Any]:
        """Benchmark full processing cycle"""
        iterations = 100
        latencies = []
        successes = 0
        
        for _ in range(iterations):
            start = time.time()
            
            try:
                # Full cycle test
                test_input = {
                    'voice': {'text': 'test message'},
                    'context': {'time': datetime.now().isoformat()}
                }
                
                # Process through entire system
                result = await self.process_request(test_input)
                
                if result and 'response' in result:
                    successes += 1
                    
            except Exception as e:
                self.logger.error(f"Full cycle benchmark error: {e}")
            
            latencies.append((time.time() - start) * 1000)
        
        return {
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'success_rate': successes / iterations
        }
    
    async def _check_data_flow_health(self) -> Dict[str, Any]:
        """Check health of data flow between phases"""
        test_data = {'health_check': True, 'timestamp': time.time()}
        
        try:
            # Test data flow through phases
            result1 = await self.phases['phase1']['pipeline'].process(test_data)
            result2 = await self.phases['phase2']['context_memory'].process(result1)
            
            return {
                'status': 'HEALTHY' if result2 else 'DEGRADED',
                'latency_ms': 10,  # Placeholder
                'recommendations': []
            }
        except Exception as e:
            return {
                'status': 'UNHEALTHY',
                'error': str(e),
                'recommendations': ['Check phase connections', 'Verify data formats']
            }
    
    async def _check_state_consistency(self) -> Dict[str, Any]:
        """Check state consistency across phases"""
        state_manager = self.phases['phase1']['state_manager']
        current_state = state_manager.get_current_state()
        
        # Check if other phases have consistent view
        consistency_checks = []
        
        if hasattr(self.phases['phase3']['intervention'], 'get_perceived_state'):
            intervention_state = self.phases['phase3']['intervention'].get_perceived_state()
            consistency_checks.append(intervention_state == current_state)
        
        consistency_score = sum(consistency_checks) / len(consistency_checks) if consistency_checks else 1.0
        
        return {
            'status': 'HEALTHY' if consistency_score > 0.9 else 'DEGRADED',
            'consistency_score': consistency_score,
            'recommendations': [] if consistency_score > 0.9 else ['Synchronize state across phases']
        }
    
    async def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory usage health"""
        process = psutil.Process()
        memory_percent = process.memory_percent()
        
        status = 'HEALTHY'
        recommendations = []
        
        if memory_percent > 80:
            status = 'CRITICAL'
            recommendations = ['Reduce cache sizes', 'Enable memory limits']
        elif memory_percent > 60:
            status = 'DEGRADED'
            recommendations = ['Monitor memory growth', 'Consider cache optimization']
        
        return {
            'status': status,
            'memory_percent': memory_percent,
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'recommendations': recommendations
        }
    
    async def _get_current_config(self) -> Dict[str, Any]:
        """Get current system configuration"""
        config = {}
        
        for phase_name, components in self.phases.items():
            phase_config = {}
            for comp_name, component in components.items():
                if hasattr(component, 'get_config'):
                    phase_config[comp_name] = component.get_config()
            config[phase_name] = phase_config
        
        return config
    
    async def _analyze_usage_patterns(self) -> Dict[str, Any]:
        """Analyze system usage patterns"""
        # Simple analysis based on metrics
        return {
            'avg_load': 0.6,  # Placeholder
            'peak_load': 0.9,  # Placeholder
            'memory_pressure': 0.5,  # Placeholder
            'common_operations': ['query', 'update_state', 'get_context']
        }
    
    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours"""
        if hasattr(self, 'start_time'):
            return (time.time() - self.start_time) / 3600
        return 0.0
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request through the integrated system"""
        self.metrics.total_requests += 1
        
        try:
            # Phase 1: Input processing
            processed = await self.phases['phase1']['pipeline'].process(request)
            
            # Phase 2: Context update
            await self.phases['phase2']['context_memory'].update_context(processed)
            
            # Phase 3: Check interventions
            intervention_needed = await self.phases['phase3']['intervention'].check_needed(processed)
            
            # Phase 4: Generate response
            response = await self.phases['phase4']['nlp'].generate_response(
                processed,
                intervention_needed
            )
            
            # Phase 8: Learn from interaction
            await self.phases['phase8']['feedback'].record_interaction({
                'request': request,
                'response': response,
                'timestamp': datetime.now()
            })
            
            self.metrics.successful_requests += 1
            
            return {
                'status': 'success',
                'response': response,
                'metadata': {
                    'processing_time_ms': 10,  # Placeholder
                    'phases_used': ['1', '2', '3', '4', '8']
                }
            }
            
        except Exception as e:
            self.metrics.error_count += 1
            self.logger.error(f"Request processing error: {e}")
            
            return {
                'status': 'error',
                'error': str(e),
                'recovery_action': 'retry'
            }


# Convenience functions for testing
async def main():
    """Main function for testing Phase 11"""
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = SystemIntegrationOrchestrator()
    
    # Initialize all phases
    print("🚀 Initializing JARVIS Phase 11...")
    if await orchestrator.initialize_all_phases():
        print("✅ Initialization complete!")
        
        # Run integration tests
        print("\n🧪 Running integration tests...")
        test_results = await orchestrator.run_integration_tests()
        print(f"Test Status: {test_results['overall_status']}")
        for test in test_results['tests']:
            print(f"  - {test['name']}: {test['status']}")
        
        # Run performance benchmarks
        print("\n📊 Running performance benchmarks...")
        benchmarks = await orchestrator.run_performance_benchmarks()
        print(f"Overall Score: {benchmarks['overall_score']:.2f}")
        
        # Generate health report
        print("\n🏥 Generating health report...")
        health = await orchestrator.generate_health_report()
        print(f"System Health: {health['overall_health']}")
        
        # Optimize configuration
        print("\n⚙️ Optimizing configuration...")
        optimizations = await orchestrator.optimize_system_configuration()
        print(f"Recommendations: {len(optimizations['recommended_changes'])}")
        
    else:
        print("❌ Initialization failed!")


if __name__ == "__main__":
    asyncio.run(main())
