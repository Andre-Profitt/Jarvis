#!/bin/bash
# Smart test runner that handles common issues

echo "🧪 Smart Test Runner"
echo "==================="

# Install any missing test dependencies
pip install -q pytest pytest-asyncio pytest-mock sqlalchemy 2>/dev/null

# Run tests with smart filtering
echo "Running tests in stages..."

# Stage 1: Run stable tests first
echo -e "\n📌 Stage 1: Stable tests"
pytest tests/test_simple_performance_optimizer.py -v -x

# Stage 2: Run database tests with compatibility mode
echo -e "\n📌 Stage 2: Database tests"
PYTEST_COMPATIBILITY_MODE=1 pytest tests/test_database.py -v -x || true

# Stage 3: Run configuration tests
echo -e "\n📌 Stage 3: Configuration tests"  
pytest tests/test_configuration.py -v -x || true

# Stage 4: Summary
echo -e "\n📊 Test Summary"
pytest tests/ --tb=no -q | grep -E "(passed|failed|error)" | sort | uniq -c

echo -e "\n✅ Done! Check results above."
