#!/bin/bash

# JARVIS Phase 10 - Quick Launch Script

echo "🚀 Launching JARVIS Ultra with Performance Optimizations..."
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt 2>/dev/null || echo "⚠️  Some dependencies may be missing"
fi

# Default to demo mode with monitoring
MODE=${1:-"demo"}

case $MODE in
    "demo")
        echo "🎭 Starting in Demo Mode with Performance Monitor..."
        python launch_jarvis_ultra.py --demo --monitor --benchmark
        ;;
    "interactive")
        echo "💬 Starting Interactive Mode..."
        python launch_jarvis_ultra.py --interactive --monitor
        ;;
    "benchmark")
        echo "📊 Running Performance Benchmark..."
        python launch_jarvis_ultra.py --benchmark --monitor
        ;;
    "test")
        echo "🧪 Running Phase 10 Tests..."
        python test_phase10.py
        ;;
    "realtime")
        echo "⚡ Starting in Real-time Optimized Mode..."
        python launch_jarvis_ultra.py --workload real_time --monitor
        ;;
    "batch")
        echo "📦 Starting in Batch Processing Mode..."
        python launch_jarvis_ultra.py --workload batch_processing --monitor
        ;;
    "lowmem")
        echo "💾 Starting in Memory Constrained Mode..."
        python launch_jarvis_ultra.py --workload memory_constrained --monitor
        ;;
    *)
        echo "Usage: $0 [demo|interactive|benchmark|test|realtime|batch|lowmem]"
        echo ""
        echo "Modes:"
        echo "  demo        - Run demo with performance monitor (default)"
        echo "  interactive - Interactive chat with JARVIS"
        echo "  benchmark   - Run performance benchmarks"
        echo "  test        - Run test suite"
        echo "  realtime    - Optimized for real-time responses"
        echo "  batch       - Optimized for batch processing"
        echo "  lowmem      - Optimized for low memory usage"
        exit 1
        ;;
esac