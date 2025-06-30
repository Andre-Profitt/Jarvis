#!/bin/bash
# JARVIS Phase 12 Runner Script

echo "🚀 JARVIS PHASE 12: INTEGRATION & TESTING"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "phase12_integration_testing.py" ]; then
    echo "❌ Error: Please run this script from the JARVIS-ECOSYSTEM directory"
    exit 1
fi

# Create necessary directories
mkdir -p test_results
mkdir -p deployment
mkdir -p logs

echo "📋 Phase 12 includes:"
echo "  1. Integration Testing - Verify all components work together"
echo "  2. Deployment Preparation - Get ready for production"
echo "  3. Performance Validation - Ensure system meets requirements"
echo ""

# Menu
echo "What would you like to do?"
echo "1) Run Integration Tests"
echo "2) Prepare for Deployment"
echo "3) Show Implementation Summary"
echo "4) Run Everything (Tests + Deployment Prep)"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🧪 Running Integration Tests..."
        echo "This will test all components from Phases 1-11"
        echo ""
        python3 phase12_integration_testing.py
        ;;
    2)
        echo ""
        echo "📦 Preparing for Deployment..."
        echo "This will create production configurations and documentation"
        echo ""
        python3 phase12_deployment_prep.py
        ;;
    3)
        echo ""
        echo "📊 Showing Implementation Summary..."
        echo ""
        python3 phase12_complete_summary.py
        ;;
    4)
        echo ""
        echo "🎯 Running Complete Phase 12..."
        echo ""
        
        # Run tests first
        echo "Step 1: Integration Tests"
        echo "========================"
        python3 phase12_integration_testing.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Integration tests completed!"
            echo ""
            echo "Step 2: Deployment Preparation"
            echo "=============================="
            python3 phase12_deployment_prep.py
            
            if [ $? -eq 0 ]; then
                echo ""
                echo "✅ Deployment preparation completed!"
                echo ""
                echo "Step 3: Summary"
                echo "==============="
                python3 phase12_complete_summary.py
            else
                echo "❌ Deployment preparation failed"
                exit 1
            fi
        else
            echo "❌ Integration tests failed"
            exit 1
        fi
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run again and select 1-5."
        exit 1
        ;;
esac

echo ""
echo "✅ Phase 12 operations completed!"
echo ""
echo "📚 Next steps:"
echo "  - Review test results in test_results/"
echo "  - Check deployment files in deployment/"
echo "  - Read deployment guide: deployment/DEPLOYMENT_GUIDE.md"
echo "  - Follow security checklist: deployment/security-checklist.md"
echo ""
echo "🎉 Congratulations on completing all 12 phases of JARVIS!"
