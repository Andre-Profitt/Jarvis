#!/bin/bash
# Setup script for Optimized JARVIS

echo "🚀 Setting up Optimized JARVIS..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Error: Python 3.8+ required (found $python_version)"
    exit 1
fi

echo "✅ Python $python_version detected"

# Install dependencies
echo "📦 Installing dependencies..."

# Core dependencies
pip3 install --upgrade pip
pip3 install -q \
    asyncio \
    numpy \
    psutil \
    aiofiles \
    python-dotenv \
    pyyaml \
    requests

# Neural network dependencies
echo "🧠 Installing neural network dependencies..."
pip3 install -q \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Voice dependencies
echo "🎤 Installing voice dependencies..."
pip3 install -q \
    SpeechRecognition \
    pyttsx3 \
    pyaudio \
    elevenlabs

# Performance monitoring
echo "📊 Installing monitoring dependencies..."
pip3 install -q \
    matplotlib \
    memory-profiler

# Audio processing (optional)
echo "🎵 Installing audio processing dependencies..."
pip3 install -q \
    librosa \
    soundfile

# macOS specific
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected - installing portaudio..."
    if ! brew list portaudio &>/dev/null; then
        brew install portaudio
    fi
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p models logs cache

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# API Keys
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Performance Settings
JARVIS_CACHE_SIZE=5000
JARVIS_THREAD_WORKERS=0  # 0 = auto-detect
JARVIS_MONITORING_INTERVAL=1.0

# Neural Settings
JARVIS_NEURAL_ENABLED=true
JARVIS_CONTINUOUS_LEARNING=true
JARVIS_GPU_ENABLED=true

# Voice Settings
JARVIS_VOICE_ENABLED=true
JARVIS_WAKE_WORDS=jarvis,hey jarvis
EOF
    echo "⚠️  Please edit .env and add your API keys"
fi

# Create default config
if [ ! -f config.json ]; then
    echo "📝 Creating default config..."
    cat > config.json << EOF
{
  "voice": {
    "enabled": true,
    "wake_words": ["jarvis", "hey jarvis", "ok jarvis"],
    "language": "en-US",
    "speech_rate": 180,
    "voice_id": "21m00Tcm4TlvDq8ikWAM"
  },
  "neural": {
    "enabled": true,
    "model_path": "models/jarvis_neural.pt",
    "continuous_learning": true,
    "batch_size": 32,
    "learning_rate": 0.0001
  },
  "performance": {
    "monitoring_enabled": true,
    "auto_optimization": true,
    "cache_size": 5000,
    "cache_ttl": 3600,
    "monitoring_interval": 1.0,
    "memory_threshold": 0.8,
    "cpu_threshold": 0.7
  },
  "features": {
    "pattern_recognition": true,
    "predictive_responses": true,
    "memory_consolidation": true,
    "auto_learning": true
  }
}
EOF
fi

# Test imports
echo "🧪 Testing imports..."
python3 -c "
try:
    import torch
    import speech_recognition
    import asyncio
    import psutil
    print('✅ All core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Create launcher script
echo "🚀 Creating launcher script..."
cat > start_jarvis_optimized.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Optimized JARVIS..."
python3 launch_optimized_jarvis.py "$@"
EOF
chmod +x start_jarvis_optimized.sh

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: ./start_jarvis_optimized.sh"
echo ""
echo "🎯 Launch options:"
echo "  ./start_jarvis_optimized.sh              # Full system"
echo "  ./start_jarvis_optimized.sh --no-voice   # Without voice"
echo "  ./start_jarvis_optimized.sh --no-neural  # Without neural engine"
echo "  ./start_jarvis_optimized.sh --no-monitor # Without monitoring"
echo ""
echo "📊 Performance monitoring will be available at:"
echo "  - Log file: jarvis_optimized.log"
echo "  - Metrics: performance_report.json (generated every minute)"
echo ""