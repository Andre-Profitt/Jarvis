#!/usr/bin/env python3
"""
JARVIS Phase 1 Test Suite
Comprehensive tests for unified pipeline and fluid states
"""

import asyncio
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Phase 1 components
from core.unified_input_pipeline import (
    UnifiedInputPipeline, InputType, Priority, 
    InputDetector, PriorityCalculator
)
from core.fluid_state_management import (
    FluidStateManager, StateType, ResponseMode,
    SmoothCurve, StateCalculator
)
from core.jarvis_enhanced_core import JARVISEnhancedCore

class TestPhase1:
    """Test suite for Phase 1 components"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []
        
    async def run_all_tests(self):
        """Run all tests"""
        print("🧪 JARVIS Phase 1 Test Suite")
        print("="*60)
        
        # Test categories
        await self.test_input_detection()
        await self.test_priority_calculation()
        await self.test_pipeline_processing()
        await self.test_state_calculations()
        await self.test_smooth_curves()
        await self.test_state_management()
        await self.test_response_modes()
        await self.test_integration()
        
        # Print summary
        self._print_summary()
        
    async def test_input_detection(self):
        """Test input type detection"""
        print("\n📋 Testing Input Detection")
        print("-"*40)
        
        detector = InputDetector()
        
        # Test cases
        test_cases = [
            # (input_data, expected_type, description)
            (
                {'waveform': np.random.randn(16000), 'sample_rate': 16000},
                InputType.VOICE,
                "Voice input detection"
            ),
            (
                {'heart_rate': 72, 'hrv': 55},
                InputType.BIOMETRIC,
                "Biometric input detection"
            ),
            (
                {'text': 'Hello JARVIS', 'message': 'Test'},
                InputType.TEXT,
                "Text input detection"
            ),
            (
                {'image': np.zeros((224, 224, 3))},
                InputType.VISION,
                "Vision input detection"
            ),
            (
                {'temperature': 22.5, 'humidity': 45},
                InputType.ENVIRONMENTAL,
                "Environmental input detection"
            ),
            (
                {'cpu': 45, 'memory': 78},
                InputType.SYSTEM,
                "System input detection"
            )
        ]
        
        for input_data, expected_type, description in test_cases:
            detected_type, confidence = detector.detect(input_data)
            success = detected_type == expected_type
            self._record_test(description, success, 
                            f"Expected: {expected_type.name}, Got: {detected_type.name} (confidence: {confidence:.2f})")
            
    async def test_priority_calculation(self):
        """Test priority calculation"""
        print("\n📊 Testing Priority Calculation")
        print("-"*40)
        
        calculator = PriorityCalculator()
        
        # Test critical conditions
        test_cases = [
            # (input_type, data, context, expected_priority, description)
            (
                InputType.BIOMETRIC,
                {'heart_rate': 140, 'stress_level': 0.9},
                {},
                Priority.CRITICAL,
                "Critical biometric detection"
            ),
            (
                InputType.VOICE,
                {'features': {'pitch_variance': 0.9, 'volume': 0.9}},
                {},
                Priority.CRITICAL,
                "Panic voice detection"
            ),
            (
                InputType.TEXT,
                {'text': 'Normal message'},
                {'user_state': {'flow_state': True}},
                Priority.LOW,
                "Lower priority during flow state"
            ),
            (
                InputType.SYSTEM,
                {'error_rate': 0.7},
                {},
                Priority.CRITICAL,
                "Critical system error"
            )
        ]
        
        for input_type, data, context, expected_priority, description in test_cases:
            calculated_priority = calculator.calculate(input_type, data, context)
            success = calculated_priority == expected_priority
            self._record_test(description, success,
                            f"Expected: {expected_priority.name}, Got: {calculated_priority.name}")
            
    async def test_pipeline_processing(self):
        """Test unified input pipeline"""
        print("\n🔄 Testing Pipeline Processing")
        print("-"*40)
        
        pipeline = UnifiedInputPipeline()
        await pipeline.start()
        
        try:
            # Test immediate processing
            critical_input = {
                'heart_rate': 150,
                'stress_level': 0.95
            }
            result = await pipeline.process_input(critical_input)
            self._record_test("Critical input immediate processing", 
                            result is not None,
                            f"Result: {result}")
            
            # Test queueing
            normal_input = {
                'text': 'Normal query'
            }
            result = await pipeline.process_input(normal_input)
            self._record_test("Normal input queueing",
                            result.get('status') in ['queued', 'processed'],
                            f"Status: {result.get('status')}")
            
            # Test metrics
            await asyncio.sleep(0.1)  # Let background processing work
            metrics = pipeline.get_metrics()
            self._record_test("Pipeline metrics tracking",
                            metrics['total_processed'] > 0,
                            f"Processed: {metrics['total_processed']}")
            
        finally:
            await pipeline.stop()
            
    async def test_state_calculations(self):
        """Test state calculations"""
        print("\n🧮 Testing State Calculations")
        print("-"*40)
        
        calculator = StateCalculator()
        
        # Test stress calculation
        stress_input = {
            'biometric': {
                'heart_rate': 95,
                'hrv': 30,
                'skin_conductance': 0.7
            },
            'voice': {
                'features': {'pitch_variance': 0.7}
            }
        }
        stress = calculator.calculate_stress(stress_input)
        self._record_test("Stress calculation",
                        0 <= stress <= 1,
                        f"Stress: {stress:.2f}")
        
        # Test focus calculation
        focus_input = {
            'activity': {
                'task_switches_per_hour': 1,
                'flow_duration_minutes': 25
            },
            'eye_tracking': {'gaze_stability': 0.85}
        }
        focus = calculator.calculate_focus(focus_input)
        self._record_test("Focus calculation",
                        0 <= focus <= 1,
                        f"Focus: {focus:.2f}")
        
        # Test energy calculation
        energy_input = {
            'movement': {'activity_level': 0.6},
            'voice': {'features': {'energy': 0.7}},
            'temporal': {'hour': 14}
        }
        energy = calculator.calculate_energy(energy_input)
        self._record_test("Energy calculation",
                        0 <= energy <= 1,
                        f"Energy: {energy:.2f}")
        
    async def test_smooth_curves(self):
        """Test smoothing curves"""
        print("\n〰️ Testing Smooth Curves")
        print("-"*40)
        
        # Test different algorithms
        algorithms = ['exponential', 'kalman', 'physics', 'adaptive']
        
        for algorithm in algorithms:
            curve = SmoothCurve(smoothing=0.3, algorithm=algorithm)
            
            # Apply series of values
            values = [0.5, 0.7, 0.9, 0.3, 0.5]
            results = []
            
            for value in values:
                result = curve.apply(value)
                results.append(result)
                
            # Check smoothing effect
            value_changes = sum(abs(values[i] - values[i-1]) for i in range(1, len(values)))
            result_changes = sum(abs(results[i] - results[i-1]) for i in range(1, len(results)))
            
            is_smoother = result_changes < value_changes
            self._record_test(f"{algorithm} smoothing",
                            is_smoother,
                            f"Input variation: {value_changes:.2f}, Output variation: {result_changes:.2f}")
            
    async def test_state_management(self):
        """Test fluid state management"""
        print("\n🌊 Testing Fluid State Management")
        print("-"*40)
        
        manager = FluidStateManager()
        
        # Test state updates
        test_inputs = [
            {
                'biometric': {'heart_rate': 70, 'hrv': 55},
                'activity': {'flow_duration_minutes': 30},
                'eye_tracking': {'gaze_stability': 0.9}
            },
            {
                'biometric': {'heart_rate': 90, 'hrv': 40},
                'voice': {'features': {'pitch_variance': 0.6}}
            }
        ]
        
        states = []
        for inputs in test_inputs:
            state = await manager.update_state(inputs)
            states.append(state)
            
        # Test state vector
        self._record_test("State vector creation",
                        len(states) == 2,
                        f"Created {len(states)} states")
        
        # Test state transitions are smooth
        if len(states) == 2:
            distance = states[0].distance_to(states[1])
            self._record_test("Smooth state transitions",
                            distance < 2.0,  # Should be relatively small
                            f"State distance: {distance:.2f}")
            
        # Test trend detection
        trends = manager.get_state_trends()
        self._record_test("Trend detection",
                        all(trend in ['stable', 'increasing', 'decreasing'] for trend in trends.values()),
                        f"Trends: {list(trends.values())}")
        
    async def test_response_modes(self):
        """Test response mode selection"""
        print("\n🎭 Testing Response Modes")
        print("-"*40)
        
        manager = FluidStateManager()
        
        # Test different state scenarios
        scenarios = [
            # (name, state_values, expected_mode)
            (
                "Flow state",
                {
                    StateType.STRESS: 0.2,
                    StateType.FOCUS: 0.95,
                    StateType.ENERGY: 0.7,
                    StateType.MOOD: 0.8,
                    StateType.CREATIVITY: 0.85,
                    StateType.PRODUCTIVITY: 0.9,
                    StateType.SOCIAL: 0.3,
                    StateType.HEALTH: 0.7
                },
                ResponseMode.BACKGROUND
            ),
            (
                "High stress",
                {
                    StateType.STRESS: 0.95,
                    StateType.FOCUS: 0.3,
                    StateType.ENERGY: 0.4,
                    StateType.MOOD: 0.3,
                    StateType.CREATIVITY: 0.2,
                    StateType.PRODUCTIVITY: 0.2,
                    StateType.SOCIAL: 0.4,
                    StateType.HEALTH: 0.5
                },
                ResponseMode.EMERGENCY
            ),
            (
                "Normal state",
                {
                    StateType.STRESS: 0.4,
                    StateType.FOCUS: 0.6,
                    StateType.ENERGY: 0.6,
                    StateType.MOOD: 0.7,
                    StateType.CREATIVITY: 0.5,
                    StateType.PRODUCTIVITY: 0.6,
                    StateType.SOCIAL: 0.5,
                    StateType.HEALTH: 0.7
                },
                ResponseMode.COLLABORATIVE
            )
        ]
        
        for name, state_values, expected_mode in scenarios:
            from core.fluid_state_management import StateVector
            state = StateVector(values=state_values, timestamp=datetime.now())
            mode = manager.get_response_mode(state)
            
            # For flow state detection, BACKGROUND is correct
            success = mode == expected_mode or (expected_mode == ResponseMode.BACKGROUND and mode == ResponseMode.BACKGROUND)
            self._record_test(f"{name} response mode",
                            success,
                            f"Expected: {expected_mode.name}, Got: {mode.name}")
            
    async def test_integration(self):
        """Test integrated system"""
        print("\n🔗 Testing System Integration")
        print("-"*40)
        
        # Create enhanced core
        enhanced = JARVISEnhancedCore()
        await enhanced.initialize()
        
        try:
            # Test multi-modal input
            multi_input = {
                'voice': {'features': {'energy': 0.7}},
                'biometric': {'heart_rate': 75},
                'text': 'Testing integration'
            }
            
            result = await enhanced.process_input(multi_input, source='test')
            self._record_test("Multi-modal input processing",
                            result is not None,
                            f"Mode: {result.get('mode')}")
            
            # Test state access
            status = await enhanced.get_current_status()
            self._record_test("Status retrieval",
                            'state' in status,
                            f"States: {list(status.get('state', {}).keys())}")
            
            # Test prediction
            prediction = await enhanced.predict_future_state(30)
            self._record_test("Future state prediction",
                            'predicted_state' in prediction or prediction == {},
                            "Prediction generated")
                            
        finally:
            await enhanced.input_pipeline.stop()
            
    def _record_test(self, name: str, success: bool, details: str = ""):
        """Record test result"""
        if success:
            self.passed_tests += 1
            print(f"  ✅ {name}")
        else:
            self.failed_tests += 1
            print(f"  ❌ {name}")
            
        if details:
            print(f"     {details}")
            
        self.results.append({
            'name': name,
            'success': success,
            'details': details
        })
        
    def _print_summary(self):
        """Print test summary"""
        total = self.passed_tests + self.failed_tests
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        
        if total > 0:
            success_rate = (self.passed_tests / total) * 100
            print(f"Success Rate: {success_rate:.1f}%")
            
            if success_rate == 100:
                print("\n🎉 All tests passed! Phase 1 is working perfectly!")
            elif success_rate >= 80:
                print("\n✨ Most tests passed! Phase 1 is working well.")
            else:
                print("\n⚠️  Some tests failed. Please check the implementation.")
                
        # Show failed tests
        if self.failed_tests > 0:
            print("\n❌ Failed Tests:")
            for result in self.results:
                if not result['success']:
                    print(f"  - {result['name']}")
                    if result['details']:
                        print(f"    {result['details']}")

# ============================================
# RUN TESTS
# ============================================

async def main():
    """Run test suite"""
    tester = TestPhase1()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())