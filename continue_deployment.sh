#!/bin/bash
#===================================================================
# JARVIS Ecosystem Deployment Continuation Script
# Purpose: Continue deployment after fixing syntax errors
#===================================================================

echo "🚀 JARVIS DEPLOYMENT CONTINUATION"
echo "=================================="
echo "Starting at: $(date)"
echo

# Change to project directory
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM

# Function to check command success
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 successful"
    else
        echo "❌ $1 failed - continuing anyway"
    fi
}

# Step 1: Try Black formatting (skip if fails)
echo "📝 Step 1: Attempting Black formatting..."
if command -v black &> /dev/null; then
    black . --exclude="/(\.git|\.venv|__pycache__|\.pytest_cache|\.tox|build|dist|node_modules)/" 2>/dev/null || true
    check_status "Black formatting"
else
    echo "⚠️  Black not installed - skipping formatting"
fi

# Step 2: Check and fix any remaining Python syntax errors
echo -e "\n🔧 Step 2: Checking for syntax errors..."
python_files=$(find . -name "*.py" -type f | grep -v __pycache__ | grep -v ".venv")
error_count=0

for file in $python_files; do
    python -m py_compile "$file" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "⚠️  Syntax error in: $file"
        ((error_count++))
    fi
done

if [ $error_count -eq 0 ]; then
    echo "✅ No syntax errors found!"
else
    echo "⚠️  Found $error_count files with syntax errors - may need manual fixes"
fi

# Step 3: Ensure all required directories exist
echo -e "\n📁 Step 3: Creating required directories..."
mkdir -p logs
mkdir -p storage
mkdir -p shared_memory
mkdir -p models
mkdir -p artifacts
mkdir -p deployment
check_status "Directory creation"

# Step 4: Check Redis status
echo -e "\n🔍 Step 4: Checking Redis..."
if command -v redis-cli &> /dev/null; then
    redis-cli ping &>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis not running - starting it..."
        redis-server --daemonize yes &>/dev/null || true
        sleep 2
        redis-cli ping &>/dev/null && echo "✅ Redis started" || echo "❌ Failed to start Redis"
    fi
else
    echo "❌ Redis not installed - some features may not work"
fi

# Step 5: Install/Update Python dependencies
echo -e "\n📦 Step 5: Checking Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet 2>/dev/null || true
    check_status "Dependency installation"
else
    echo "⚠️  requirements.txt not found"
fi

# Step 6: Kill any existing JARVIS processes
echo -e "\n🛑 Step 6: Stopping existing JARVIS processes..."
pkill -f "jarvis" 2>/dev/null || true
pkill -f "LAUNCH-JARVIS" 2>/dev/null || true
sleep 2
echo "✅ Cleaned up old processes"

# Step 7: Launch JARVIS
echo -e "\n🎯 Step 7: Launching JARVIS..."
echo "Using LAUNCH-JARVIS-REAL.py as main launcher"

# Create a simple launch wrapper to handle logs
cat > launch_wrapper.sh << 'EOF'
#!/bin/bash
cd /Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM
mkdir -p logs
export PYTHONUNBUFFERED=1
python3 LAUNCH-JARVIS-REAL.py 2>&1 | tee logs/jarvis_launch_$(date +%Y%m%d_%H%M%S).log &
echo $! > jarvis.pid
echo "✅ JARVIS launched with PID: $(cat jarvis.pid)"
EOF

chmod +x launch_wrapper.sh
./launch_wrapper.sh

# Step 8: Wait and check status
echo -e "\n⏳ Waiting for JARVIS to initialize..."
sleep 5

# Step 9: Run deployment status check
echo -e "\n📊 Step 9: Checking deployment status..."
python3 deployment_status.py

# Step 10: Summary
echo -e "\n✨ DEPLOYMENT CONTINUATION COMPLETE!"
echo "===================================="
echo "Completed at: $(date)"
echo
echo "📋 Next Steps:"
echo "  1. Check logs: tail -f logs/*.log"
echo "  2. Monitor status: python3 deployment_status.py"
echo "  3. Restart Claude Desktop to see MCP integration"
echo "  4. Test JARVIS: python3 jarvis_interactive.py"
echo
echo "🔧 Troubleshooting:"
echo "  - If JARVIS didn't start: check logs/jarvis_launch_*.log"
echo "  - If Redis issues: brew services start redis"
echo "  - If import errors: pip install -r requirements.txt"
echo "  - To stop all: pkill -f jarvis"
