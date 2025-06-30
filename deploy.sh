#!/bin/bash
# JARVIS Complete Deployment Script
# Handles all deployment steps from code to production

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
SKIP_TESTS=${SKIP_TESTS:-false}
SKIP_BUILD=${SKIP_BUILD:-false}

echo -e "${GREEN}🚀 JARVIS Deployment Script${NC}"
echo -e "${GREEN}=========================${NC}"
echo "Environment: $ENVIRONMENT"
echo ""

# Function to check dependencies
check_dependencies() {
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    deps=("docker" "docker-compose" "python3" "pip" "git")
    for dep in "${deps[@]}"; do
        if ! command -v $dep &> /dev/null; then
            echo -e "${RED}❌ $dep is not installed${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ All dependencies installed${NC}"
}

# Function to run tests
run_tests() {
    if [ "$SKIP_TESTS" = "true" ]; then
        echo -e "${YELLOW}⚠️  Skipping tests${NC}"
        return
    fi
    
    echo -e "${YELLOW}Running tests...${NC}"
    
    # Install test dependencies
    pip install -r requirements-dev.txt
    
    # Run linting
    echo "Running linters..."
    black --check .
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    # Run unit tests
    echo "Running unit tests..."
    pytest tests/ -v --cov=./ --cov-report=term-missing
    
    echo -e "${GREEN}✅ All tests passed${NC}"
}

# Function to generate API documentation
generate_api_docs() {
    echo -e "${YELLOW}Generating API documentation...${NC}"
    
    cd docs/api
    python openapi_generator.py
    cd ../..
    
    echo -e "${GREEN}✅ API documentation generated${NC}"
}

# Function to setup database
setup_database() {
    echo -e "${YELLOW}Setting up database...${NC}"
    
    # Run database setup script
    python database_setup.py
    
    # Run migrations
    alembic upgrade head
    
    echo -e "${GREEN}✅ Database setup complete${NC}"
}

# Function to build Docker images
build_docker_images() {
    if [ "$SKIP_BUILD" = "true" ]; then
        echo -e "${YELLOW}⚠️  Skipping Docker build${NC}"
        return
    fi
    
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    # Build all services
    docker-compose -f docker-compose.production.yml build --parallel
    
    echo -e "${GREEN}✅ Docker images built${NC}"
}

# Function to deploy services
deploy_services() {
    echo -e "${YELLOW}Deploying services...${NC}"
    
    # Load environment variables
    if [ -f ".env.$ENVIRONMENT" ]; then
        export $(cat .env.$ENVIRONMENT | grep -v '^#' | xargs)
    fi
    
    # Start services
    docker-compose -f docker-compose.production.yml up -d
    
    # Wait for services to be healthy
    echo "Waiting for services to be healthy..."
    sleep 30
    
    # Check health
    services=("consciousness" "scheduler" "knowledge" "monitoring" "orchestrator")
    for service in "${services[@]}"; do
        if docker-compose -f docker-compose.production.yml exec $service curl -f http://localhost:8001/health &> /dev/null; then
            echo -e "${GREEN}✅ $service is healthy${NC}"
        else
            echo -e "${RED}❌ $service is not healthy${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ All services deployed${NC}"
}

# Function to run post-deployment checks
post_deployment_checks() {
    echo -e "${YELLOW}Running post-deployment checks...${NC}"
    
    # Check API endpoints
    endpoints=(
        "http://localhost:8000/health"
        "http://localhost:8001/consciousness/state"
        "http://localhost:8002/scheduler/jobs"
        "http://localhost:8003/knowledge/search"
        "http://localhost:8004/monitoring/metrics"
    )
    
    for endpoint in "${endpoints[@]}"; do
        if curl -f $endpoint &> /dev/null; then
            echo -e "${GREEN}✅ $endpoint is accessible${NC}"
        else
            echo -e "${RED}❌ $endpoint is not accessible${NC}"
        fi
    done
    
    # Check monitoring dashboards
    if curl -f http://localhost:3000 &> /dev/null; then
        echo -e "${GREEN}✅ Grafana is accessible at http://localhost:3000${NC}"
    fi
    
    if curl -f http://localhost:9090 &> /dev/null; then
        echo -e "${GREEN}✅ Prometheus is accessible at http://localhost:9090${NC}"
    fi
    
    if curl -f http://localhost:16686 &> /dev/null; then
        echo -e "${GREEN}✅ Jaeger is accessible at http://localhost:16686${NC}"
    fi
}

# Function to update JARVIS memory
update_jarvis_memory() {
    echo -e "${YELLOW}Updating JARVIS memory with deployment info...${NC}"
    
    python -c "
from jarvis_unified_memory_enhanced import add_memory
import datetime

add_memory(
    project='JARVIS-DEPLOYMENT',
    content=f'Successfully deployed JARVIS to $ENVIRONMENT environment at {datetime.datetime.now()}. All services are healthy and operational.',
    type='deployment'
)
"
    
    echo -e "${GREEN}✅ JARVIS memory updated${NC}"
}

# Main deployment flow
main() {
    echo -e "${GREEN}Starting JARVIS deployment...${NC}"
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Run tests
    run_tests
    
    # Generate documentation
    generate_api_docs
    
    # Setup database
    setup_database
    
    # Build Docker images
    build_docker_images
    
    # Deploy services
    deploy_services
    
    # Post-deployment checks
    post_deployment_checks
    
    # Update JARVIS memory
    update_jarvis_memory
    
    echo ""
    echo -e "${GREEN}🎉 JARVIS deployment complete!${NC}"
    echo ""
    echo "Access points:"
    echo "  - Main API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
    echo "  - Jaeger: http://localhost:16686"
    echo ""
    echo "To view logs: docker-compose -f docker-compose.production.yml logs -f"
    echo "To stop services: docker-compose -f docker-compose.production.yml down"
}

# Run main function
main