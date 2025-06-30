"""
JARVIS Phase 5 Performance Benchmarks
Measures natural interaction system performance
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.natural_interaction_core import NaturalInteractionCore


class Phase5Benchmarks:
    """Comprehensive benchmarks for Phase 5 components"""
    
    def __init__(self):
        self.jarvis = NaturalInteractionCore()
        self.results = {}
    
    async def run_all_benchmarks(self):
        """Run all performance benchmarks"""
        print("\n📊 JARVIS Phase 5 Performance Benchmarks")
        print("=" * 60)
        
        # Memory Performance
        await self.benchmark_memory_operations()
        
        # Emotional Processing
        await self.benchmark_emotional_processing()
        
        # Natural Language Generation
        await self.benchmark_language_generation()
        
        # End-to-End Response Time
        await self.benchmark_full_interaction()
        
        # Memory Efficiency
        await self.benchmark_memory_efficiency()
        
        # Print results
        self.print_results()
    
    async def benchmark_memory_operations(self):
        """Benchmark conversational memory performance"""
        print("\n🧠 Memory Operations Benchmark...")
        
        # Add memories
        add_times = []
        for i in range(100):
            start = time.perf_counter()
            await self.jarvis.memory.add_memory(
                f"Test memory {i}",
                {"test": True, "index": i},
                importance=0.5
            )
            add_times.append(time.perf_counter() - start)
        
        # Recall memories
        recall_times = []
        queries = ["test", "memory", "important", "conversation", "context"]
        for query in queries:
            start = time.perf_counter()
            await self.jarvis.memory.recall(query)
            recall_times.append(time.perf_counter() - start)
        
        self.results["memory_operations"] = {
            "add_avg": statistics.mean(add_times) * 1000,  # ms
            "add_p95": sorted(add_times)[95] * 1000,
            "recall_avg": statistics.mean(recall_times) * 1000,
            "recall_p95": sorted(recall_times)[int(len(recall_times) * 0.95)] * 1000
        }
    
    async def benchmark_emotional_processing(self):
        """Benchmark emotional continuity performance"""
        print("\n❤️ Emotional Processing Benchmark...")
        
        process_times = []
        test_inputs = [
            {"text": "I'm really happy!", "biometric": {"heart_rate": 75}},
            {"text": "This is frustrating", "biometric": {"heart_rate": 95}},
            {"text": "I feel calm", "biometric": {"heart_rate": 65}},
            {"text": "That's surprising!", "biometric": {"heart_rate": 85}},
            {"text": "I'm worried about this", "biometric": {"heart_rate": 90}}
        ]
        
        for inputs in test_inputs * 10:  # 50 total
            start = time.perf_counter()
            await self.jarvis.emotional_continuity.update_emotional_state(
                inputs, {}
            )
            process_times.append(time.perf_counter() - start)
        
        # Trajectory prediction
        predict_times = []
        for _ in range(20):
            start = time.perf_counter()
            await self.jarvis.emotional_continuity.predict_emotional_trajectory(
                self.jarvis.emotional_continuity.current_state
            )
            predict_times.append(time.perf_counter() - start)
        
        self.results["emotional_processing"] = {
            "update_avg": statistics.mean(process_times) * 1000,
            "update_p95": sorted(process_times)[47] * 1000,  # 95th percentile
            "predict_avg": statistics.mean(predict_times) * 1000,
            "predict_p95": sorted(predict_times)[19] * 1000
        }
    
    async def benchmark_language_generation(self):
        """Benchmark natural language flow performance"""
        print("\n💬 Language Generation Benchmark...")
        
        generation_times = []
        test_cases = [
            ("Hello JARVIS", {"type": "greeting"}, {}),
            ("Can you help me?", {"type": "request"}, {"activity": "coding"}),
            ("I don't understand", {"type": "confusion"}, {}),
            ("That worked great!", {"type": "feedback"}, {}),
            ("What about the other thing?", {"type": "reference"}, {})
        ]
        
        for user_input, intent, context in test_cases * 10:
            start = time.perf_counter()
            await self.jarvis.language_flow.generate_response(
                user_input,
                intent,
                context,
                {"valence": 0.5, "arousal": 0.5, "intensity": 0.5}
            )
            generation_times.append(time.perf_counter() - start)
        
        self.results["language_generation"] = {
            "generate_avg": statistics.mean(generation_times) * 1000,
            "generate_p95": sorted(generation_times)[47] * 1000
        }
    
    async def benchmark_full_interaction(self):
        """Benchmark complete interaction pipeline"""
        print("\n🚀 Full Interaction Benchmark...")
        
        interaction_times = []
        test_conversations = [
            "Hi JARVIS, how are you today?",
            "I need help with my project",
            "Can you remind me what we discussed?",
            "I'm feeling a bit overwhelmed",
            "Thanks for your help!"
        ]
        
        for message in test_conversations * 5:
            start = time.perf_counter()
            await self.jarvis.process_interaction(
                message,
                {"voice": {"features": {"pitch_variance": 0.5}}}
            )
            interaction_times.append(time.perf_counter() - start)
        
        self.results["full_interaction"] = {
            "total_avg": statistics.mean(interaction_times) * 1000,
            "total_p95": sorted(interaction_times)[23] * 1000,
            "total_p99": sorted(interaction_times)[24] * 1000
        }
    
    async def benchmark_memory_efficiency(self):
        """Benchmark memory usage efficiency"""
        print("\n💾 Memory Efficiency Benchmark...")
        
        # Simulate long conversation
        for i in range(200):
            await self.jarvis.memory.add_memory(
                f"Long conversation message {i}",
                {"index": i},
                importance=0.3 if i % 10 == 0 else 0.1
            )
        
        # Check memory sizes
        working_size = len(self.jarvis.memory.working_memory)
        short_term_size = len(self.jarvis.memory.short_term_memory)
        long_term_size = len(self.jarvis.memory.long_term_memory)
        
        # Test consolidation
        start = time.perf_counter()
        await self.jarvis.memory.consolidate_memories()
        consolidation_time = time.perf_counter() - start
        
        self.results["memory_efficiency"] = {
            "working_memory_size": working_size,
            "short_term_size": short_term_size,
            "long_term_size": long_term_size,
            "consolidation_time": consolidation_time * 1000
        }
    
    def print_results(self):
        """Print benchmark results"""
        print("\n\n" + "=" * 60)
        print("📊 PHASE 5 PERFORMANCE RESULTS")
        print("=" * 60)
        
        # Memory Operations
        mem_ops = self.results["memory_operations"]
        print("\n🧠 Memory Operations:")
        print(f"  Add Memory:    {mem_ops['add_avg']:.2f}ms avg, {mem_ops['add_p95']:.2f}ms p95")
        print(f"  Recall Memory: {mem_ops['recall_avg']:.2f}ms avg, {mem_ops['recall_p95']:.2f}ms p95")
        
        # Emotional Processing
        emo = self.results["emotional_processing"]
        print("\n❤️ Emotional Processing:")
        print(f"  State Update:  {emo['update_avg']:.2f}ms avg, {emo['update_p95']:.2f}ms p95")
        print(f"  Trajectory:    {emo['predict_avg']:.2f}ms avg, {emo['predict_p95']:.2f}ms p95")
        
        # Language Generation
        lang = self.results["language_generation"]
        print("\n💬 Natural Language:")
        print(f"  Generation:    {lang['generate_avg']:.2f}ms avg, {lang['generate_p95']:.2f}ms p95")
        
        # Full Interaction
        full = self.results["full_interaction"]
        print("\n🚀 End-to-End Performance:")
        print(f"  Total Time:    {full['total_avg']:.2f}ms avg")
        print(f"  95th %ile:     {full['total_p95']:.2f}ms")
        print(f"  99th %ile:     {full['total_p99']:.2f}ms")
        
        # Memory Efficiency
        mem_eff = self.results["memory_efficiency"]
        print("\n💾 Memory Efficiency:")
        print(f"  Working Mem:   {mem_eff['working_memory_size']} items")
        print(f"  Short-term:    {mem_eff['short_term_size']} items")
        print(f"  Long-term:     {mem_eff['long_term_size']} items")
        print(f"  Consolidation: {mem_eff['consolidation_time']:.2f}ms")
        
        # Summary
        print("\n" + "=" * 60)
        print("✅ PERFORMANCE SUMMARY")
        print("=" * 60)
        
        avg_response = full['total_avg']
        if avg_response < 200:
            print(f"🎯 Excellent: {avg_response:.0f}ms average response time")
            print("   Natural conversation flow achieved!")
        elif avg_response < 500:
            print(f"✓ Good: {avg_response:.0f}ms average response time")
            print("   Smooth interaction maintained")
        else:
            print(f"⚠️ Needs optimization: {avg_response:.0f}ms average response time")
        
        print(f"\n📊 Key Metrics:")
        print(f"  • Memory Recall: <{mem_ops['recall_avg']:.0f}ms")
        print(f"  • Emotional Processing: <{emo['update_avg']:.0f}ms")
        print(f"  • Language Generation: <{lang['generate_avg']:.0f}ms")
        print(f"  • 95% of responses: <{full['total_p95']:.0f}ms")


async def run_benchmarks():
    """Run all benchmarks"""
    benchmarks = Phase5Benchmarks()
    await benchmarks.run_all_benchmarks()


if __name__ == "__main__":
    print("🏃 Starting Phase 5 Performance Benchmarks...")
    print("This will take about 30 seconds to complete.\n")
    
    try:
        asyncio.run(run_benchmarks())
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmarks interrupted")
    except Exception as e:
        print(f"\n❌ Error during benchmarks: {e}")
