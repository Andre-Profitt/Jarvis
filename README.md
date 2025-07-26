# JARVIS - Advanced AI Assistant

A production-ready AI assistant with state-of-the-art NLP, real-time voice processing, and modular architecture.

## 🚀 Features

- **Multi-Modal AI**: Seamless switching between GPT-4, Claude, Gemini, and local models
- **Advanced NLP**: Context understanding, intent recognition, and conversation memory
- **Real-Time Voice**: Wake word detection, streaming ASR/TTS, voice activity detection
- **Plugin Architecture**: Extensible system for custom functionality
- **Learning System**: Adapts to user patterns and preferences
- **Performance**: Optimized for <100ms response time

## 🏗️ Architecture

```
JARVIS/
├── core/
│   ├── assistant.py      # Main orchestrator
│   ├── nlp/             # NLP pipeline
│   ├── voice/           # Voice processing
│   ├── memory/          # Context & learning
│   └── plugins/         # Plugin system
├── models/              # ML models
├── ui/                  # Optional interfaces
└── tests/               # Comprehensive tests
```

## 🚦 Quick Start

```bash
# Install
pip install -r requirements.txt

# Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your API keys

# Run
python jarvis.py
```

## 🧠 Core Components

### 1. NLP Pipeline
- Intent classification using BERT
- Entity extraction with spaCy
- Context tracking with transformer embeddings
- Semantic search for knowledge retrieval

### 2. Voice System
- Wake word: Porcupine for offline detection
- ASR: Whisper for accuracy + WebRTC VAD
- TTS: ElevenLabs for natural voice
- Streaming: Real-time bidirectional audio

### 3. AI Integration
- Unified interface for all LLMs
- Automatic model selection based on task
- Response streaming and caching
- Cost optimization

### 4. Memory & Learning
- Short-term: Conversation context (Redis)
- Long-term: User preferences (SQLite)
- Episodic: Important interactions
- Procedural: Learned workflows

## 📊 Performance

- Wake word detection: <50ms
- Intent recognition: <30ms
- LLM first token: <500ms
- Full response: <2s average

## 🔌 Plugins

Create custom plugins by extending `BasePlugin`:

```python
from core.plugins import BasePlugin

class WeatherPlugin(BasePlugin):
    def __init__(self):
        super().__init__("weather", "Get weather information")
    
    async def execute(self, params):
        # Your implementation
        return result
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 License

MIT License