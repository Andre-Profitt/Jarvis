#!/bin/bash
# Load .env file and run Phase 12 tests

echo "🔧 Loading environment variables from .env..."

# Load .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "✅ Environment variables loaded"
else
    echo "❌ .env file not found!"
    exit 1
fi

# Show loaded keys (masked)
echo ""
echo "🔑 API Keys Loaded:"
if [ ! -z "$OPENAI_API_KEY" ]; then
    echo "  ✅ OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}...${OPENAI_API_KEY: -4}"
fi
if [ ! -z "$GEMINI_API_KEY" ]; then
    echo "  ✅ GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}...${GEMINI_API_KEY: -4}"
fi
if [ ! -z "$ELEVENLABS_API_KEY" ]; then
    echo "  ✅ ELEVENLABS_API_KEY: ${ELEVENLABS_API_KEY:0:10}...${ELEVENLABS_API_KEY: -4}"
fi

echo ""
echo "🧪 Running Phase 12 Integration Tests..."
echo ""

# Run the tests with environment loaded
python3 phase12_integration_testing.py
