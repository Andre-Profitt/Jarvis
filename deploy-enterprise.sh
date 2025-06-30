#!/bin/bash

# JARVIS Enterprise Deployment Script
# Production-grade deployment with zero downtime

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
JARVIS_VERSION=${JARVIS_VERSION:-"1.0.0"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
REGION=${AWS_REGION:-"us-east-1"}
CLUSTER_NAME="jarvis-${ENVIRONMENT}"

echo -e "${BLUE}🚀 JARVIS Enterprise Deployment v${JARVIS_VERSION}${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Region: ${REGION}${NC}"
echo ""

# Pre-flight checks
pre_flight_checks() {
    echo -e "${YELLOW}Running pre-flight checks...${NC}"
    
    # Check required tools
    command -v kubectl >/dev/null 2>&1 || { echo -e "${RED}kubectl is required but not installed.${NC}" >&2; exit 1; }
    command -v helm >/dev/null 2>&1 || { echo -e "${RED}helm is required but not installed.${NC}" >&2; exit 1; }
    command -v aws >/dev/null 2>&1 || { echo -e "${RED}AWS CLI is required but not installed.${NC}" >&2; exit 1; }
    command -v docker >/dev/null 2>&1 || { echo -e "${RED}Docker is required but not installed.${NC}" >&2; exit 1; }
    
    # Check AWS credentials
    aws sts get-caller-identity >/dev/null 2>&1 || { echo -e "${RED}AWS credentials not configured.${NC}" >&2; exit 1; }
    
    # Check Kubernetes cluster
    kubectl cluster-info >/dev/null 2>&1 || { echo -e "${RED}Cannot connect to Kubernetes cluster.${NC}" >&2; exit 1; }
    
    echo -e "${GREEN}✅ Pre-flight checks passed${NC}"
}

# Build and push Docker images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    # Build main API image
    docker build -t jarvis-api:${JARVIS_VERSION} -f Dockerfile.api .
    docker tag jarvis-api:${JARVIS_VERSION} ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-api:${JARVIS_VERSION}
    
    # Build worker image
    docker build -t jarvis-worker:${JARVIS_VERSION} -f Dockerfile.worker .
    docker tag jarvis-worker:${JARVIS_VERSION} ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-worker:${JARVIS_VERSION}
    
    # Build ML serving image
    docker build -t jarvis-ml:${JARVIS_VERSION} -f Dockerfile.ml .
    docker tag jarvis-ml:${JARVIS_VERSION} ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-ml:${JARVIS_VERSION}
    
    # Push to ECR
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
    
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-api:${JARVIS_VERSION}
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-worker:${JARVIS_VERSION}
    docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-ml:${JARVIS_VERSION}
    
    echo -e "${GREEN}✅ Images built and pushed${NC}"
}

# Deploy infrastructure
deploy_infrastructure() {
    echo -e "${YELLOW}Deploying infrastructure...${NC}"
    
    # Create namespace
    kubectl create namespace jarvis-${ENVIRONMENT} --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy Redis
    helm upgrade --install redis bitnami/redis \
        --namespace jarvis-${ENVIRONMENT} \
        --set auth.password=${REDIS_PASSWORD} \
        --set replica.replicaCount=3 \
        --set sentinel.enabled=true
    
    # Deploy MongoDB
    helm upgrade --install mongodb bitnami/mongodb \
        --namespace jarvis-${ENVIRONMENT} \
        --set auth.rootPassword=${MONGO_PASSWORD} \
        --set replicaSet.enabled=true \
        --set replicaSet.replicas.secondary=2
    
    # Deploy Elasticsearch
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace jarvis-${ENVIRONMENT} \
        --set replicas=3 \
        --set minimumMasterNodes=2
    
    # Deploy Kafka
    helm upgrade --install kafka bitnami/kafka \
        --namespace jarvis-${ENVIRONMENT} \
        --set replicaCount=3 \
        --set zookeeper.replicaCount=3
    
    echo -e "${GREEN}✅ Infrastructure deployed${NC}"
}

# Deploy JARVIS services
deploy_services() {
    echo -e "${YELLOW}Deploying JARVIS services...${NC}"
    
    # Apply ConfigMaps and Secrets
    kubectl apply -f k8s/configmap.yaml -n jarvis-${ENVIRONMENT}
    kubectl apply -f k8s/secrets.yaml -n jarvis-${ENVIRONMENT}
    
    # Deploy API service with rolling update
    kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jarvis-api
  namespace: jarvis-${ENVIRONMENT}
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 0
  selector:
    matchLabels:
      app: jarvis-api
  template:
    metadata:
      labels:
        app: jarvis-api
        version: ${JARVIS_VERSION}
    spec:
      containers:
      - name: api
        image: ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/jarvis-api:${JARVIS_VERSION}
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: ${ENVIRONMENT}
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
EOF
    
    # Deploy worker service
    kubectl apply -f k8s/worker-deployment.yaml -n jarvis-${ENVIRONMENT}
    
    # Deploy ML serving
    kubectl apply -f k8s/ml-deployment.yaml -n jarvis-${ENVIRONMENT}
    
    # Deploy services
    kubectl apply -f k8s/services.yaml -n jarvis-${ENVIRONMENT}
    
    # Deploy Ingress
    kubectl apply -f k8s/ingress.yaml -n jarvis-${ENVIRONMENT}
    
    echo -e "${GREEN}✅ JARVIS services deployed${NC}"
}

# Setup autoscaling
setup_autoscaling() {
    echo -e "${YELLOW}Setting up autoscaling...${NC}"
    
    # Horizontal Pod Autoscaler
    kubectl autoscale deployment jarvis-api \
        --namespace jarvis-${ENVIRONMENT} \
        --cpu-percent=70 \
        --min=10 \
        --max=100
    
    kubectl autoscale deployment jarvis-worker \
        --namespace jarvis-${ENVIRONMENT} \
        --cpu-percent=80 \
        --min=5 \
        --max=50
    
    # Vertical Pod Autoscaler
    kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: jarvis-api-vpa
  namespace: jarvis-${ENVIRONMENT}
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind: Deployment
    name: jarvis-api
  updatePolicy:
    updateMode: "Auto"
EOF
    
    # Cluster Autoscaler
    kubectl apply -f k8s/cluster-autoscaler.yaml
    
    echo -e "${GREEN}✅ Autoscaling configured${NC}"
}

# Setup monitoring
setup_monitoring() {
    echo -e "${YELLOW}Setting up monitoring...${NC}"
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi
    
    # Deploy Grafana dashboards
    kubectl apply -f monitoring/dashboards/ -n monitoring
    
    # Deploy Jaeger for tracing
    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace monitoring \
        --set provisionDataStore.cassandra=true \
        --set cassandra.config.cluster_size=3
    
    # Deploy ELK stack
    helm upgrade --install elk elastic/elastic-stack \
        --namespace monitoring \
        --set elasticsearch.replicas=3 \
        --set kibana.enabled=true \
        --set logstash.enabled=true
    
    echo -e "${GREEN}✅ Monitoring stack deployed${NC}"
}

# Setup CDN
setup_cdn() {
    echo -e "${YELLOW}Setting up CDN...${NC}"
    
    # CloudFront distribution
    aws cloudfront create-distribution \
        --distribution-config file://cdn/cloudfront-config.json \
        --tags Key=Environment,Value=${ENVIRONMENT} Key=Service,Value=JARVIS
    
    # Cloudflare Workers
    wrangler publish --env ${ENVIRONMENT}
    
    echo -e "${GREEN}✅ CDN configured${NC}"
}

# Run health checks
health_check() {
    echo -e "${YELLOW}Running health checks...${NC}"
    
    # Wait for rollout
    kubectl rollout status deployment/jarvis-api -n jarvis-${ENVIRONMENT}
    
    # Check pod status
    kubectl get pods -n jarvis-${ENVIRONMENT}
    
    # Test endpoints
    API_URL=$(kubectl get ingress jarvis-ingress -n jarvis-${ENVIRONMENT} -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    curl -s -o /dev/null -w "%{http_code}" https://${API_URL}/health | grep -q "200" || {
        echo -e "${RED}Health check failed!${NC}"
        exit 1
    }
    
    echo -e "${GREEN}✅ All systems healthy${NC}"
}

# Setup disaster recovery
setup_disaster_recovery() {
    echo -e "${YELLOW}Setting up disaster recovery...${NC}"
    
    # Create backup CronJob
    kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: jarvis-backup
  namespace: jarvis-${ENVIRONMENT}
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: jarvis-backup:latest
            env:
            - name: S3_BUCKET
              value: jarvis-backups-${ENVIRONMENT}
            - name: AWS_REGION
              value: ${REGION}
          restartPolicy: OnFailure
EOF
    
    # Setup cross-region replication
    aws s3api put-bucket-replication \
        --bucket jarvis-data-${ENVIRONMENT} \
        --replication-configuration file://dr/replication-config.json
    
    echo -e "${GREEN}✅ Disaster recovery configured${NC}"
}

# Main deployment flow
main() {
    pre_flight_checks
    build_images
    deploy_infrastructure
    deploy_services
    setup_autoscaling
    setup_monitoring
    setup_cdn
    health_check
    setup_disaster_recovery
    
    echo ""
    echo -e "${GREEN}🎉 JARVIS ${JARVIS_VERSION} deployed successfully!${NC}"
    echo -e "${GREEN}API Endpoint: https://api.jarvis.ai${NC}"
    echo -e "${GREEN}Monitoring: https://grafana.jarvis.ai${NC}"
    echo -e "${GREEN}Logs: https://kibana.jarvis.ai${NC}"
}

# Run deployment
main
