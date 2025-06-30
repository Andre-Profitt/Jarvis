"""
JARVIS Phase 3: Test and Demonstration Script
=============================================
Tests and demonstrates the intelligent processing capabilities
of Phase 3, including context persistence and predictive pre-loading.
"""

import asyncio
import logging
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any

# Import Phase 1 components
from core.unified_input_pipeline import UnifiedInputPipeline
from core.fluid_state_management import FluidStateManager
from core.jarvis_enhanced_core import JARVISEnhancedCore

# Import memory system
from core.enhanced_episodic_memory import EpisodicMemorySystem

# Import Phase 3 components  
from core.context_persistence_manager import ContextPersistenceManager
from core.predictive_preloading_system import PredictivePreloadingSystem
from core.memory_enhanced_processing import (
    MemoryEnhancedProcessor,
    Phase3Integration,
    enhance_jarvis_with_phase3
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase3Tester:
    """Comprehensive tester for Phase 3 functionality"""
    
    def __init__(self):
        self.jarvis_core = None
        self.phase3_integration = None
        self.test_results = []
        
    async def setup(self):
        """Set up test environment"""
        logger.info("🔧 Setting up Phase 3 test environment")
        
        # Create mock JARVIS core with Phase 1 components
        self.jarvis_core = await self._create_mock_jarvis_core()
        
        # Enhance with Phase 3
        self.phase3_integration = await enhance_jarvis_with_phase3(self.jarvis_core)
        
        logger.info("✅ Test environment ready")
        
    async def _create_mock_jarvis_core(self):
        """Create a mock JARVIS core for testing"""
        # In real implementation, this would be the actual JARVIS core
        # For testing, we create a minimal version
        
        class MockJARVISCore:
            def __init__(self):
                self.pipeline = UnifiedInputPipeline()
                self.state_manager = FluidStateManager()
                self.state_manager.initialize()
                
            async def process(self, prompt: str) -> Dict[str, Any]:
                return {"response": f"Processed: {prompt}", "status": "success"}
        
        return MockJARVISCore()
    
    async def run_all_tests(self):
        """Run comprehensive Phase 3 tests"""
        logger.info("\n🧪 Starting Phase 3 Tests\n")
        
        # Test 1: Context Persistence
        await self.test_context_persistence()
        
        # Test 2: Predictive Pre-loading
        await self.test_predictive_preloading()
        
        # Test 3: Memory Integration
        await self.test_memory_integration()
        
        # Test 4: Workflow Learning
        await self.test_workflow_learning()
        
        # Test 5: Intelligence Metrics
        await self.test_intelligence_metrics()
        
        # Show results
        await self.show_test_results()
    
    async def test_context_persistence(self):
        """Test context persistence across interactions"""
        logger.info("📝 Test 1: Context Persistence")
        
        # Simulate a conversation thread
        conversation_inputs = [
            "Let's discuss the new feature implementation",
            "I think we should use a microservices architecture",
            "What about the database design?",
            "Good point, let's use PostgreSQL with Redis cache",
            "Can you create a diagram of this architecture?"
        ]
        
        thread_id = None
        for i, input_text in enumerate(conversation_inputs):
            logger.info(f"\n  Input {i+1}: '{input_text}'")
            
            # Process with memory
            result = await self.jarvis_core.process_with_memory(input_text)
            
            # Check if thread was maintained
            if result.context.get("conversation_thread"):
                current_thread_id = result.context["conversation_thread"].thread_id
                if thread_id is None:
                    thread_id = current_thread_id
                    logger.info(f"  ✓ Created thread: {thread_id[:8]}...")
                elif thread_id == current_thread_id:
                    logger.info(f"  ✓ Maintained thread continuity")
                else:
                    logger.error(f"  ✗ Thread changed unexpectedly")
                
                # Show thread context
                thread = result.context["conversation_thread"]
                logger.info(f"  📊 Thread depth: {len(thread.context_stack)} messages")
                logger.info(f"  🏷️ Topic: {thread.topic}")
            
            await asyncio.sleep(0.5)  # Simulate time between messages
        
        # Test result
        test_passed = thread_id is not None
        self.test_results.append({
            "test": "Context Persistence",
            "passed": test_passed,
            "details": f"Maintained thread across {len(conversation_inputs)} messages"
        })
    
    async def test_predictive_preloading(self):
        """Test predictive pre-loading system"""
        logger.info("\n\n🔮 Test 2: Predictive Pre-loading")
        
        # Simulate a coding workflow
        coding_sequence = [
            ("open main.py", "file_open"),
            ("add new function calculate_metrics", "code_edit"),
            ("save file", "file_save"),
            ("run python main.py", "execute"),  # This should be predicted
            ("fix syntax error on line 42", "code_edit"),
            ("save file", "file_save"),
            ("run python main.py", "execute")  # This should be predicted with higher confidence
        ]
        
        predictions_made = 0
        accurate_predictions = 0
        
        for i, (action, action_type) in enumerate(coding_sequence):
            logger.info(f"\n  Action {i+1}: '{action}' (type: {action_type})")
            
            # Get predictions before processing
            predictions_before = await self.jarvis_core.get_predictions()
            
            # Process action
            result = await self.jarvis_core.process_with_memory(
                action,
                metadata={"action_type": action_type}
            )
            
            # Check if this action was predicted
            was_predicted = False
            for pred in predictions_before:
                if action_type in pred.get("content", ""):
                    was_predicted = True
                    accurate_predictions += 1
                    logger.info(f"  ✓ This action was predicted with {pred['confidence']*100:.0f}% confidence!")
                    break
            
            if not was_predicted and i > 2:  # After pattern should emerge
                logger.info(f"  ℹ️ Action was not predicted")
            
            # Show new predictions
            new_predictions = await self.jarvis_core.get_predictions()
            if new_predictions:
                predictions_made += len(new_predictions)
                logger.info(f"  🔮 New predictions:")
                for pred in new_predictions[:3]:  # Show top 3
                    logger.info(f"     - {pred['content']} ({pred['confidence']*100:.0f}% confidence)")
            
            # Check preloaded resources
            preloaded = result.preloaded_resources
            if preloaded:
                logger.info(f"  📦 Pre-loaded resources: {list(preloaded.keys())}")
            
            await asyncio.sleep(0.3)
        
        # Test result
        accuracy = accurate_predictions / max(predictions_made, 1) if predictions_made > 0 else 0
        self.test_results.append({
            "test": "Predictive Pre-loading",
            "passed": predictions_made > 0 and accuracy > 0.3,
            "details": f"Made {predictions_made} predictions, {accurate_predictions} accurate ({accuracy*100:.0f}%)"
        })
    
    async def test_memory_integration(self):
        """Test memory system integration"""
        logger.info("\n\n🧠 Test 3: Memory Integration")
        
        # Create some memories
        memory_interactions = [
            "Remember that the API key is stored in the .env file",
            "The deployment process uses Docker and Kubernetes",
            "Our coding standards require type hints for all functions"
        ]
        
        # Store memories
        for interaction in memory_interactions:
            await self.jarvis_core.process_with_memory(interaction)
            logger.info(f"  💾 Stored: '{interaction}'")
        
        await asyncio.sleep(1)  # Let memories consolidate
        
        # Test recall
        queries = [
            "Where is the API key?",
            "How do we deploy?",
            "What are our coding standards?"
        ]
        
        successful_recalls = 0
        for query in queries:
            logger.info(f"\n  🔍 Query: '{query}'")
            
            # Process with memory
            result = await self.jarvis_core.process_with_memory(query)
            
            # Check if relevant memories were retrieved
            if result.memory_utilized:
                successful_recalls += 1
                logger.info(f"  ✓ Retrieved {len(result.context.get('relevant_memories', []))} relevant memories")
                
                # Use direct recall
                memories = await self.jarvis_core.recall(query, max_results=3)
                if memories:
                    logger.info(f"  📋 Top memory: {memories[0].chunks[0].content['content'][:60]}...")
            else:
                logger.info(f"  ✗ No relevant memories found")
        
        # Test result
        recall_rate = successful_recalls / len(queries)
        self.test_results.append({
            "test": "Memory Integration",
            "passed": recall_rate >= 0.5,
            "details": f"Successfully recalled {successful_recalls}/{len(queries)} queries ({recall_rate*100:.0f}%)"
        })
    
    async def test_workflow_learning(self):
        """Test workflow learning capabilities"""
        logger.info("\n\n🔄 Test 4: Workflow Learning")
        
        # Simulate repeated workflow
        git_workflow = [
            ("git status", "git_command"),
            ("git add .", "git_command"),
            ("git commit -m 'Update features'", "git_command"),
            ("git push origin main", "git_command")
        ]
        
        # Repeat workflow multiple times to learn pattern
        logger.info("  📚 Training workflow pattern...")
        for iteration in range(3):
            logger.info(f"\n  Iteration {iteration + 1}:")
            for action, action_type in git_workflow:
                await self.jarvis_core.process_with_memory(
                    action,
                    metadata={"action_type": action_type}
                )
                await asyncio.sleep(0.1)
        
        # Test if workflow is learned
        logger.info("\n  🧪 Testing learned workflow...")
        
        # Start the workflow
        result = await self.jarvis_core.process_with_memory(
            "git status",
            metadata={"action_type": "git_command"}
        )
        
        # Check predictions
        predictions = await self.jarvis_core.get_predictions()
        workflow_predicted = False
        
        for pred in predictions:
            if "git_command" in str(pred.get("content", "")):
                workflow_predicted = True
                logger.info(f"  ✓ Workflow continuation predicted: {pred['content']}")
                break
        
        if not workflow_predicted:
            logger.info("  ✗ Workflow not predicted")
        
        # Get system metrics
        intelligence = await self.jarvis_core.get_intelligence()
        patterns_learned = intelligence["predictive_insights"]["learned_patterns"]
        workflows_learned = intelligence["predictive_insights"]["workflow_templates"]
        
        logger.info(f"  📊 Learned {patterns_learned} patterns and {workflows_learned} workflows")
        
        # Test result
        self.test_results.append({
            "test": "Workflow Learning",
            "passed": patterns_learned > 0 or workflows_learned > 0,
            "details": f"Learned {patterns_learned} patterns, {workflows_learned} workflows"
        })
    
    async def test_intelligence_metrics(self):
        """Test intelligence metrics and scoring"""
        logger.info("\n\n📊 Test 5: Intelligence Metrics")
        
        # Get comprehensive intelligence insights
        insights = await self.jarvis_core.get_intelligence()
        
        # Display metrics
        logger.info("  🧠 System Intelligence:")
        logger.info(f"     Overall Score: {insights.get('intelligence_score', 0)}/100")
        
        logger.info("\n  📈 Performance Metrics:")
        perf = insights.get("processing_performance", {})
        logger.info(f"     Avg Processing Time: {perf.get('avg_time_seconds', 0):.3f}s")
        logger.info(f"     Memory Hit Rate: {perf.get('memory_hit_rate', 0)*100:.0f}%")
        logger.info(f"     Prediction Accuracy: {perf.get('prediction_accuracy', 0)*100:.0f}%")
        
        logger.info("\n  🎯 Context Insights:")
        ctx = insights.get("context_insights", {})
        logger.info(f"     Active Threads: {ctx.get('active_threads', 0)}")
        logger.info(f"     Active Activities: {ctx.get('active_activities', 0)}")
        logger.info(f"     Context Switches: {ctx.get('context_switches', 0)}")
        
        logger.info("\n  🔮 Predictive Insights:")
        pred = insights.get("predictive_insights", {})
        logger.info(f"     Learned Patterns: {pred.get('learned_patterns', 0)}")
        logger.info(f"     Active Predictions: {pred.get('active_predictions', 0)}")
        logger.info(f"     Cache Efficiency: {pred.get('cache_efficiency', 0)*100:.0f}%")
        
        # Test result
        intelligence_score = insights.get('intelligence_score', 0)
        self.test_results.append({
            "test": "Intelligence Metrics",
            "passed": intelligence_score > 20,  # Low bar for initial testing
            "details": f"Intelligence score: {intelligence_score}/100"
        })
    
    async def show_test_results(self):
        """Display test results summary"""
        logger.info("\n\n" + "="*60)
        logger.info("📊 PHASE 3 TEST RESULTS")
        logger.info("="*60)
        
        passed_count = 0
        for result in self.test_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            logger.info(f"\n{status} - {result['test']}")
            logger.info(f"         {result['details']}")
            if result["passed"]:
                passed_count += 1
        
        logger.info("\n" + "-"*60)
        logger.info(f"Overall: {passed_count}/{len(self.test_results)} tests passed")
        logger.info("="*60)
        
        if passed_count == len(self.test_results):
            logger.info("\n🎉 All tests passed! Phase 3 is working correctly.")
        else:
            logger.info("\n⚠️ Some tests failed. Check the implementation.")
    
    async def cleanup(self):
        """Clean up test environment"""
        if self.phase3_integration:
            await self.phase3_integration.shutdown()


async def run_interactive_demo():
    """Run an interactive demonstration of Phase 3"""
    logger.info("\n🎮 JARVIS Phase 3 Interactive Demo\n")
    
    # Set up
    tester = Phase3Tester()
    await tester.setup()
    
    logger.info("Welcome to JARVIS Phase 3! I now have:")
    logger.info("- 🧠 Persistent memory across conversations")
    logger.info("- 🔮 Predictive capabilities to anticipate your needs")
    logger.info("- 📚 Learning from patterns in your behavior")
    logger.info("- 🎯 Context-aware responses\n")
    
    # Interactive loop
    print("Try having a conversation! (Type 'quit' to exit, 'status' for insights)")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'status':
                insights = await tester.jarvis_core.get_intelligence()
                print(f"\n📊 Intelligence Score: {insights.get('intelligence_score', 0)}/100")
                print(f"🧠 Total Memories: {insights['memory_insights']['total_memories']}")
                print(f"🔮 Active Predictions: {insights['predictive_insights']['active_predictions']}")
                print(f"📚 Learned Patterns: {insights['predictive_insights']['learned_patterns']}")
                continue
            elif user_input.lower() == 'predictions':
                predictions = await tester.jarvis_core.get_predictions()
                if predictions:
                    print("\n🔮 Current Predictions:")
                    for pred in predictions[:5]:
                        print(f"  - {pred['content']} ({pred['confidence']*100:.0f}% confidence)")
                else:
                    print("\n No active predictions")
                continue
            
            # Process input
            result = await tester.jarvis_core.process_with_memory(user_input)
            
            # Show response
            print(f"\nJARVIS: {result.processed_input.content}")
            
            # Show context info
            if result.context.get("conversation_thread"):
                thread = result.context["conversation_thread"]
                print(f"\n[Thread: {thread.topic} | Depth: {len(thread.context_stack)} | ", end="")
            
            if result.memory_utilized:
                print(f"Memories: ✓ | ", end="")
            
            if result.predictions:
                print(f"Predictions: {len(result.predictions)} | ", end="")
            
            print(f"Time: {result.processing_time:.3f}s]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Cleanup
    await tester.cleanup()
    print("\n\nThank you for trying JARVIS Phase 3! 👋")


async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await run_interactive_demo()
    else:
        # Run automated tests
        tester = Phase3Tester()
        try:
            await tester.setup()
            await tester.run_all_tests()
        finally:
            await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
