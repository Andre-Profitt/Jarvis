#!/bin/bash
# JARVIS Phase 11 Deployment Script
# Deploys the fully integrated JARVIS system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JARVIS_HOME="/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM"
PHASE11_DIR="$JARVIS_HOME/phase11-integration"
LOG_DIR="$JARVIS_HOME/logs"
CONFIG_TYPE="${1:-production}"

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║          JARVIS Phase 11 Deployment Script                ║"
echo "║         Complete System Integration Deployment            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

check_requirements() {
    log "Checking system requirements..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    log "Python version: $python_version"
    
    # Check required directories
    if [ ! -d "$JARVIS_HOME" ]; then
        error "JARVIS home directory not found: $JARVIS_HOME"
    fi
    
    # Check core components
    components=(
        "core/unified_input_pipeline.py"
        "core/fluid_state_management.py"
        "core/context_memory.py"
        "core/proactive_interventions.py"
        "core/natural_language_flow.py"
        "core/visual_ui_system.py"
        "core/cognitive_load_reducer.py"
        "core/performance_optimizer.py"
        "core/feedback_learning.py"
        "core/adaptive_personalization.py"
    )
    
    for component in "${components[@]}"; do
        if [ ! -f "$JARVIS_HOME/$component" ]; then
            warning "Component not found: $component"
        fi
    done
    
    log "✅ Requirements check complete"
}

setup_environment() {
    log "Setting up environment..."
    
    # Create necessary directories
    mkdir -p "$LOG_DIR"
    mkdir -p "$JARVIS_HOME/data"
    mkdir -p "$JARVIS_HOME/cache"
    mkdir -p "$JARVIS_HOME/config"
    
    # Set environment variables
    export JARVIS_HOME="$JARVIS_HOME"
    export JARVIS_ENV="$CONFIG_TYPE"
    export JARVIS_LOG_DIR="$LOG_DIR"
    export PYTHONPATH="$JARVIS_HOME:$PYTHONPATH"
    
    # Generate configuration if needed
    if [ ! -f "$JARVIS_HOME/config/jarvis_phase11_${CONFIG_TYPE}.yaml" ]; then
        log "Generating ${CONFIG_TYPE} configuration..."
        cd "$PHASE11_DIR"
        python3 production_config.py
    fi
    
    log "✅ Environment setup complete"
}

run_tests() {
    log "Running integration tests..."
    
    cd "$PHASE11_DIR"
    
    # Run test suite
    if python3 launch_phase11.py --test-only --config "$CONFIG_TYPE"; then
        log "✅ All tests passed"
    else
        error "Tests failed! Aborting deployment."
    fi
}

run_benchmarks() {
    log "Running performance benchmarks..."
    
    cd "$PHASE11_DIR"
    
    # Run benchmarks
    python3 launch_phase11.py --benchmark --config "$CONFIG_TYPE" > benchmark_results.txt
    
    # Check benchmark results
    if grep -q "Overall Score: 0.[7-9]" benchmark_results.txt || grep -q "Overall Score: 1.0" benchmark_results.txt; then
        log "✅ Performance benchmarks passed"
        cat benchmark_results.txt
    else
        warning "Performance below optimal levels"
        cat benchmark_results.txt
    fi
}

start_services() {
    log "Starting JARVIS services..."
    
    # Check if already running
    if pgrep -f "launch_phase11.py" > /dev/null; then
        warning "JARVIS is already running. Stopping existing instance..."
        pkill -f "launch_phase11.py"
        sleep 2
    fi
    
    # Start main service
    cd "$PHASE11_DIR"
    nohup python3 launch_phase11.py --config "$CONFIG_TYPE" > "$LOG_DIR/jarvis_phase11.out" 2>&1 &
    JARVIS_PID=$!
    
    # Save PID
    echo $JARVIS_PID > "$JARVIS_HOME/jarvis.pid"
    
    # Wait for startup
    log "Waiting for JARVIS to start..."
    sleep 5
    
    # Check if running
    if ps -p $JARVIS_PID > /dev/null; then
        log "✅ JARVIS Phase 11 started successfully (PID: $JARVIS_PID)"
    else
        error "Failed to start JARVIS"
    fi
}

setup_monitoring() {
    log "Setting up monitoring..."
    
    # Open dashboard in browser
    if [ -f "$PHASE11_DIR/phase11_dashboard.html" ]; then
        log "Opening monitoring dashboard..."
        if command -v open > /dev/null; then
            open "$PHASE11_DIR/phase11_dashboard.html"
        elif command -v xdg-open > /dev/null; then
            xdg-open "$PHASE11_DIR/phase11_dashboard.html"
        else
            log "Dashboard available at: file://$PHASE11_DIR/phase11_dashboard.html"
        fi
    fi
    
    log "✅ Monitoring setup complete"
}

health_check() {
    log "Performing health check..."
    
    # Simple health check
    sleep 3
    
    if pgrep -f "launch_phase11.py" > /dev/null; then
        log "✅ JARVIS is running"
        
        # Check log for errors
        if tail -n 50 "$LOG_DIR/jarvis_phase11.out" | grep -i "error" > /dev/null; then
            warning "Errors detected in logs"
        else
            log "✅ No errors in recent logs"
        fi
    else
        error "JARVIS is not running"
    fi
}

show_status() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✨ JARVIS Phase 11 Deployment Complete!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo
    echo "Configuration: $CONFIG_TYPE"
    echo "PID: $(cat $JARVIS_HOME/jarvis.pid 2>/dev/null || echo 'N/A')"
    echo "Logs: $LOG_DIR/jarvis_phase11.out"
    echo "Dashboard: file://$PHASE11_DIR/phase11_dashboard.html"
    echo
    echo -e "${YELLOW}Useful commands:${NC}"
    echo "  Stop JARVIS:    ./deploy_phase11.sh stop"
    echo "  Check status:   ./deploy_phase11.sh status"
    echo "  View logs:      tail -f $LOG_DIR/jarvis_phase11.out"
    echo "  Run tests:      ./deploy_phase11.sh test"
    echo
}

# Handle commands
case "${2:-deploy}" in
    "stop")
        log "Stopping JARVIS..."
        if [ -f "$JARVIS_HOME/jarvis.pid" ]; then
            kill $(cat "$JARVIS_HOME/jarvis.pid") 2>/dev/null || true
            rm "$JARVIS_HOME/jarvis.pid"
            log "✅ JARVIS stopped"
        else
            warning "JARVIS PID file not found"
        fi
        ;;
    
    "status")
        if [ -f "$JARVIS_HOME/jarvis.pid" ] && ps -p $(cat "$JARVIS_HOME/jarvis.pid") > /dev/null 2>&1; then
            log "JARVIS is running (PID: $(cat $JARVIS_HOME/jarvis.pid))"
        else
            warning "JARVIS is not running"
        fi
        ;;
    
    "test")
        check_requirements
        setup_environment
        run_tests
        ;;
    
    "deploy")
        # Full deployment
        log "Starting JARVIS Phase 11 deployment for $CONFIG_TYPE..."
        
        check_requirements
        setup_environment
        run_tests
        run_benchmarks
        start_services
        setup_monitoring
        health_check
        show_status
        ;;
    
    *)
        echo "Usage: $0 [production|aws|gcp|on_premise] [deploy|stop|status|test]"
        exit 1
        ;;
esac
