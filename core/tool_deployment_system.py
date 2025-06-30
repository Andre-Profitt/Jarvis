"""
Tool Deployment System for JARVIS
==================================

Advanced system for deploying, managing, and scaling tools across various environments.
"""

import asyncio
import json
import yaml
import subprocess
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from pathlib import Path
import docker
import kubernetes
from kubernetes import client as k8s_client, config as k8s_config
import boto3
import aiohttp
import aiodns
import hashlib
import jwt
import redis
from structlog import get_logger
from prometheus_client import Counter, Histogram, Gauge, Summary
import consul
import etcd3
import zipfile
import tarfile
import tempfile

logger = get_logger(__name__)

# Metrics
deployments_total = Counter(
    "tool_deployments_total", "Total tool deployments", ["environment", "status"]
)
deployment_duration = Histogram(
    "tool_deployment_duration_seconds", "Deployment duration"
)
active_deployments = Gauge(
    "active_tool_deployments", "Number of active deployments", ["environment"]
)
deployment_health = Gauge(
    "deployment_health_score", "Health score of deployments", ["deployment_id"]
)


@dataclass
class DeploymentConfig:
    """Configuration for tool deployment"""

    name: str
    version: str
    environment: str  # local, docker, kubernetes, cloud
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)  # CPU, memory limits
    environment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=list)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    health_check: Optional[Dict[str, Any]] = None
    auto_scaling: Optional[Dict[str, Any]] = None
    deployment_strategy: str = "rolling"  # rolling, blue-green, canary
    monitoring_enabled: bool = True
    logging_config: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentStatus:
    """Status of a tool deployment"""

    deployment_id: str
    name: str
    version: str
    environment: str
    status: str  # pending, deploying, running, failed, terminated
    replicas_ready: int
    replicas_total: int
    endpoints: List[str]
    created_at: datetime
    updated_at: datetime
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolPackage:
    """Tool package information"""

    name: str
    version: str
    description: str
    entry_point: str
    dependencies: List[str]
    runtime: str  # python, node, go, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)


@dataclass
class DeploymentPlan:
    """Deployment execution plan"""

    steps: List[Dict[str, Any]]
    estimated_duration: float
    resource_requirements: Dict[str, Any]
    rollback_plan: Optional[List[Dict[str, Any]]] = None
    validation_checks: List[Dict[str, Any]] = field(default_factory=list)


class ToolDeploymentSystem:
    """
    Advanced tool deployment and management system.

    Features:
    - Multi-environment deployment (local, Docker, K8s, cloud)
    - Blue-green and canary deployments
    - Auto-scaling and load balancing
    - Health monitoring and self-healing
    - Configuration management
    - Secret management
    - Deployment rollback
    - Service discovery
    """

    def __init__(
        self,
        registry_url: Optional[str] = None,
        config_store: Optional[str] = None,
        monitoring_backend: Optional[str] = None,
    ):

        self.registry_url = registry_url
        self.config_store = config_store
        self.monitoring_backend = monitoring_backend

        # Initialize clients
        self._init_clients()

        # Deployment tracking
        self.deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[DeploymentStatus] = []

        # Service registry
        self.service_registry: Dict[str, List[str]] = {}

        # Configuration cache
        self.config_cache: Dict[str, Any] = {}

        logger.info("Tool Deployment System initialized")

    def _init_clients(self):
        """Initialize deployment environment clients"""
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except:
            self.docker_client = None
            self.docker_available = False
            logger.warning("Docker not available")

        # Kubernetes client
        try:
            k8s_config.load_incluster_config()
            self.k8s_available = True
        except:
            try:
                k8s_config.load_kube_config()
                self.k8s_available = True
            except:
                self.k8s_available = False
                logger.warning("Kubernetes not available")

        if self.k8s_available:
            self.k8s_v1 = k8s_client.CoreV1Api()
            self.k8s_apps_v1 = k8s_client.AppsV1Api()
            self.k8s_batch_v1 = k8s_client.BatchV1Api()

        # Cloud clients (AWS example)
        try:
            self.aws_ecs = boto3.client("ecs")
            self.aws_ec2 = boto3.client("ec2")
            self.aws_available = True
        except:
            self.aws_available = False
            logger.warning("AWS not available")

        # Service discovery
        try:
            self.consul_client = consul.Consul()
            self.consul_available = True
        except:
            self.consul_available = False
            logger.warning("Consul not available")

        # Configuration store
        if self.config_store:
            try:
                if self.config_store.startswith("redis://"):
                    self.config_client = redis.from_url(self.config_store)
                elif self.config_store.startswith("etcd://"):
                    self.config_client = etcd3.client()
                else:
                    self.config_client = None
            except:
                self.config_client = None
                logger.warning("Configuration store not available")

    async def deploy_tool(
        self, package: ToolPackage, config: DeploymentConfig, wait: bool = True
    ) -> DeploymentStatus:
        """Deploy a tool to the specified environment"""
        deployments_total.labels(environment=config.environment, status="started").inc()

        # Generate deployment ID
        deployment_id = self._generate_deployment_id(package, config)

        # Create deployment status
        status = DeploymentStatus(
            deployment_id=deployment_id,
            name=package.name,
            version=package.version,
            environment=config.environment,
            status="pending",
            replicas_ready=0,
            replicas_total=config.replicas,
            endpoints=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        self.deployments[deployment_id] = status

        try:
            # Create deployment plan
            plan = await self._create_deployment_plan(package, config)

            # Execute deployment
            if config.environment == "local":
                await self._deploy_local(package, config, status)
            elif config.environment == "docker":
                await self._deploy_docker(package, config, status)
            elif config.environment == "kubernetes":
                await self._deploy_kubernetes(package, config, status)
            elif config.environment == "cloud":
                await self._deploy_cloud(package, config, status)
            else:
                raise ValueError(f"Unsupported environment: {config.environment}")

            # Wait for deployment to be ready
            if wait:
                await self._wait_for_ready(deployment_id, timeout=300)

            # Update status
            status.status = "running"
            status.updated_at = datetime.now()

            # Register with service discovery
            await self._register_service(status)

            # Update metrics
            deployments_total.labels(
                environment=config.environment, status="success"
            ).inc()
            active_deployments.labels(environment=config.environment).inc()

        except Exception as e:
            status.status = "failed"
            status.events.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "type": "error",
                    "message": str(e),
                }
            )
            deployments_total.labels(
                environment=config.environment, status="failed"
            ).inc()
            logger.error(f"Deployment failed: {e}")
            raise

        return status

    def _generate_deployment_id(
        self, package: ToolPackage, config: DeploymentConfig
    ) -> str:
        """Generate unique deployment ID"""
        data = f"{package.name}-{package.version}-{config.environment}-{datetime.now()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    async def _create_deployment_plan(
        self, package: ToolPackage, config: DeploymentConfig
    ) -> DeploymentPlan:
        """Create deployment execution plan"""
        steps = []

        # Pre-deployment validation
        steps.append(
            {
                "name": "validate_package",
                "action": "validate",
                "params": {"package": package},
            }
        )

        # Environment-specific steps
        if config.environment == "docker":
            steps.extend(
                [
                    {
                        "name": "build_image",
                        "action": "docker_build",
                        "params": {"package": package, "config": config},
                    },
                    {
                        "name": "push_image",
                        "action": "docker_push",
                        "params": {"skip_if_local": True},
                    },
                ]
            )

        # Create resources
        steps.append(
            {
                "name": "create_resources",
                "action": "create",
                "params": {"config": config},
            }
        )

        # Deploy application
        steps.append(
            {
                "name": "deploy_application",
                "action": "deploy",
                "params": {"strategy": config.deployment_strategy},
            }
        )

        # Post-deployment
        steps.extend(
            [
                {
                    "name": "health_check",
                    "action": "health_check",
                    "params": {"config": config.health_check},
                },
                {
                    "name": "configure_monitoring",
                    "action": "setup_monitoring",
                    "params": {"enabled": config.monitoring_enabled},
                },
            ]
        )

        # Calculate resource requirements
        resource_requirements = {
            "cpu": config.resources.get("cpu", 1) * config.replicas,
            "memory": config.resources.get("memory", 512) * config.replicas,
            "storage": config.resources.get("storage", 10) * config.replicas,
        }

        # Create rollback plan
        rollback_plan = [
            {
                "name": "stop_new_version",
                "action": "stop",
                "params": {"version": package.version},
            },
            {
                "name": "restore_previous",
                "action": "restore",
                "params": {"restore_point": "pre_deployment"},
            },
        ]

        return DeploymentPlan(
            steps=steps,
            estimated_duration=len(steps) * 30,  # 30 seconds per step estimate
            resource_requirements=resource_requirements,
            rollback_plan=rollback_plan,
            validation_checks=[
                {"type": "health", "threshold": 0.8},
                {"type": "performance", "baseline": "previous_version"},
            ],
        )

    async def _deploy_local(
        self, package: ToolPackage, config: DeploymentConfig, status: DeploymentStatus
    ):
        """Deploy tool locally"""
        # Create deployment directory
        deploy_dir = Path(f"/tmp/deployments/{status.deployment_id}")
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Extract package
        for artifact in package.artifacts:
            if artifact.suffix in [".zip", ".tar.gz"]:
                if artifact.suffix == ".zip":
                    with zipfile.ZipFile(artifact, "r") as zf:
                        zf.extractall(deploy_dir)
                else:
                    with tarfile.open(artifact, "r:gz") as tf:
                        tf.extractall(deploy_dir)
            else:
                shutil.copy2(artifact, deploy_dir)

        # Install dependencies
        if package.runtime == "python":
            requirements_file = deploy_dir / "requirements.txt"
            if requirements_file.exists():
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-r",
                        str(requirements_file),
                    ],
                    check=True,
                )

        # Start processes
        processes = []
        for i in range(config.replicas):
            port = config.ports[0] + i if config.ports else 8000 + i

            # Set environment variables
            env = os.environ.copy()
            env.update(config.environment_variables)
            env["PORT"] = str(port)
            env["INSTANCE_ID"] = f"{status.deployment_id}-{i}"

            # Start process
            process = subprocess.Popen(
                [sys.executable, str(deploy_dir / package.entry_point)], env=env
            )

            processes.append(
                {
                    "pid": process.pid,
                    "port": port,
                    "instance_id": f"{status.deployment_id}-{i}",
                }
            )

            status.endpoints.append(f"http://localhost:{port}")

        # Store process info
        status.metrics["processes"] = processes
        status.replicas_ready = len(processes)

    async def _deploy_docker(
        self, package: ToolPackage, config: DeploymentConfig, status: DeploymentStatus
    ):
        """Deploy tool using Docker"""
        if not self.docker_available:
            raise RuntimeError("Docker is not available")

        # Build Docker image
        image_tag = f"{package.name}:{package.version}"

        # Create Dockerfile if not exists
        dockerfile_path = None
        for artifact in package.artifacts:
            if artifact.name == "Dockerfile":
                dockerfile_path = artifact
                break

        if not dockerfile_path:
            # Generate Dockerfile
            dockerfile_content = self._generate_dockerfile(package, config)
            dockerfile_path = Path(tempfile.mktemp())
            dockerfile_path.write_text(dockerfile_content)

        # Build image
        image, build_logs = self.docker_client.images.build(
            path=str(dockerfile_path.parent),
            tag=image_tag,
            rm=True,
            buildargs=config.environment_variables,
        )

        # Run containers
        containers = []
        for i in range(config.replicas):
            port_bindings = {}
            for port in config.ports:
                host_port = port + i
                port_bindings[f"{port}/tcp"] = host_port

            container = self.docker_client.containers.run(
                image_tag,
                detach=True,
                ports=port_bindings,
                environment=config.environment_variables,
                volumes={
                    v["host"]: {"bind": v["container"], "mode": v.get("mode", "rw")}
                    for v in config.volumes
                },
                name=f"{package.name}-{status.deployment_id}-{i}",
                labels={
                    "deployment_id": status.deployment_id,
                    "tool_name": package.name,
                    "version": package.version,
                },
            )

            containers.append(container)

            # Get container IP
            container.reload()
            ip = container.attrs["NetworkSettings"]["IPAddress"]
            port = config.ports[0] if config.ports else 80
            status.endpoints.append(f"http://{ip}:{port}")

        status.metrics["containers"] = [c.id for c in containers]
        status.replicas_ready = len(containers)

    async def _deploy_kubernetes(
        self, package: ToolPackage, config: DeploymentConfig, status: DeploymentStatus
    ):
        """Deploy tool to Kubernetes"""
        if not self.k8s_available:
            raise RuntimeError("Kubernetes is not available")

        namespace = config.environment_variables.get("K8S_NAMESPACE", "default")

        # Create ConfigMap for configuration
        config_map = k8s_client.V1ConfigMap(
            metadata=k8s_client.V1ObjectMeta(
                name=f"{package.name}-config",
                namespace=namespace,
                labels={
                    "app": package.name,
                    "version": package.version,
                    "deployment_id": status.deployment_id,
                },
            ),
            data=config.environment_variables,
        )

        self.k8s_v1.create_namespaced_config_map(namespace, config_map)

        # Create Secret for sensitive data
        if config.secrets:
            secret = k8s_client.V1Secret(
                metadata=k8s_client.V1ObjectMeta(
                    name=f"{package.name}-secrets", namespace=namespace
                ),
                string_data=config.secrets,
            )
            self.k8s_v1.create_namespaced_secret(namespace, secret)

        # Create Deployment
        deployment = k8s_client.V1Deployment(
            metadata=k8s_client.V1ObjectMeta(
                name=f"{package.name}-{package.version}", namespace=namespace
            ),
            spec=k8s_client.V1DeploymentSpec(
                replicas=config.replicas,
                selector=k8s_client.V1LabelSelector(
                    match_labels={"app": package.name, "version": package.version}
                ),
                template=k8s_client.V1PodTemplateSpec(
                    metadata=k8s_client.V1ObjectMeta(
                        labels={"app": package.name, "version": package.version}
                    ),
                    spec=k8s_client.V1PodSpec(
                        containers=[
                            k8s_client.V1Container(
                                name=package.name,
                                image=f"{package.name}:{package.version}",
                                ports=[
                                    k8s_client.V1ContainerPort(container_port=p)
                                    for p in config.ports
                                ],
                                env_from=[
                                    k8s_client.V1EnvFromSource(
                                        config_map_ref=k8s_client.V1ConfigMapEnvSource(
                                            name=f"{package.name}-config"
                                        )
                                    )
                                ],
                                resources=k8s_client.V1ResourceRequirements(
                                    requests={
                                        "cpu": str(config.resources.get("cpu", 0.1)),
                                        "memory": f"{config.resources.get('memory', 128)}Mi",
                                    },
                                    limits={
                                        "cpu": str(config.resources.get("cpu", 1)),
                                        "memory": f"{config.resources.get('memory', 512)}Mi",
                                    },
                                ),
                                liveness_probe=(
                                    self._create_k8s_probe(config.health_check)
                                    if config.health_check
                                    else None
                                ),
                                readiness_probe=(
                                    self._create_k8s_probe(config.health_check)
                                    if config.health_check
                                    else None
                                ),
                            )
                        ]
                    ),
                ),
                strategy=self._create_k8s_strategy(config.deployment_strategy),
            ),
        )

        self.k8s_apps_v1.create_namespaced_deployment(namespace, deployment)

        # Create Service
        service = k8s_client.V1Service(
            metadata=k8s_client.V1ObjectMeta(name=package.name, namespace=namespace),
            spec=k8s_client.V1ServiceSpec(
                selector={"app": package.name},
                ports=[
                    k8s_client.V1ServicePort(port=p, target_port=p, name=f"port-{p}")
                    for p in config.ports
                ],
                type=(
                    "LoadBalancer"
                    if config.environment_variables.get("EXPOSE_EXTERNAL")
                    else "ClusterIP"
                ),
            ),
        )

        self.k8s_v1.create_namespaced_service(namespace, service)

        # Create HorizontalPodAutoscaler if auto-scaling is enabled
        if config.auto_scaling:
            hpa = k8s_client.V1HorizontalPodAutoscaler(
                metadata=k8s_client.V1ObjectMeta(
                    name=f"{package.name}-hpa", namespace=namespace
                ),
                spec=k8s_client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=k8s_client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=f"{package.name}-{package.version}",
                    ),
                    min_replicas=config.auto_scaling.get("min_replicas", 1),
                    max_replicas=config.auto_scaling.get("max_replicas", 10),
                    target_cpu_utilization_percentage=config.auto_scaling.get(
                        "target_cpu", 70
                    ),
                ),
            )

            self.k8s_client.AutoscalingV1Api().create_namespaced_horizontal_pod_autoscaler(
                namespace, hpa
            )

        # Get service endpoints
        service = self.k8s_v1.read_namespaced_service(package.name, namespace)
        if service.status.load_balancer.ingress:
            for ingress in service.status.load_balancer.ingress:
                ip = ingress.ip or ingress.hostname
                for port in config.ports:
                    status.endpoints.append(f"http://{ip}:{port}")

        status.metrics["k8s_deployment"] = f"{package.name}-{package.version}"
        status.metrics["k8s_namespace"] = namespace

    def _create_k8s_probe(self, health_check: Dict[str, Any]) -> k8s_client.V1Probe:
        """Create Kubernetes health probe"""
        probe_type = health_check.get("type", "http")

        if probe_type == "http":
            return k8s_client.V1Probe(
                http_get=k8s_client.V1HTTPGetAction(
                    path=health_check.get("path", "/health"),
                    port=health_check.get("port", 80),
                ),
                initial_delay_seconds=health_check.get("initial_delay", 30),
                period_seconds=health_check.get("period", 10),
                timeout_seconds=health_check.get("timeout", 5),
                failure_threshold=health_check.get("failure_threshold", 3),
            )
        elif probe_type == "tcp":
            return k8s_client.V1Probe(
                tcp_socket=k8s_client.V1TCPSocketAction(
                    port=health_check.get("port", 80)
                ),
                initial_delay_seconds=health_check.get("initial_delay", 30),
                period_seconds=health_check.get("period", 10),
            )
        else:
            return None

    def _create_k8s_strategy(self, strategy: str) -> k8s_client.V1DeploymentStrategy:
        """Create Kubernetes deployment strategy"""
        if strategy == "rolling":
            return k8s_client.V1DeploymentStrategy(
                type="RollingUpdate",
                rolling_update=k8s_client.V1RollingUpdateDeployment(
                    max_unavailable="25%", max_surge="25%"
                ),
            )
        elif strategy == "recreate":
            return k8s_client.V1DeploymentStrategy(type="Recreate")
        else:
            return k8s_client.V1DeploymentStrategy(type="RollingUpdate")

    async def _deploy_cloud(
        self, package: ToolPackage, config: DeploymentConfig, status: DeploymentStatus
    ):
        """Deploy tool to cloud (AWS ECS example)"""
        if not self.aws_available:
            raise RuntimeError("AWS is not available")

        # Create task definition
        task_definition = {
            "family": f"{package.name}-{package.version}",
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": str(config.resources.get("cpu", 256)),
            "memory": str(config.resources.get("memory", 512)),
            "containerDefinitions": [
                {
                    "name": package.name,
                    "image": f"{package.name}:{package.version}",
                    "portMappings": [
                        {"containerPort": p, "protocol": "tcp"} for p in config.ports
                    ],
                    "environment": [
                        {"name": k, "value": v}
                        for k, v in config.environment_variables.items()
                    ],
                    "secrets": [
                        {
                            "name": k,
                            "valueFrom": f"arn:aws:secretsmanager:region:account:secret:{k}",
                        }
                        for k in config.secrets.keys()
                    ],
                    "logConfiguration": {
                        "logDriver": "awslogs",
                        "options": {
                            "awslogs-group": f"/ecs/{package.name}",
                            "awslogs-region": "us-east-1",
                            "awslogs-stream-prefix": "ecs",
                        },
                    },
                }
            ],
        }

        response = self.aws_ecs.register_task_definition(**task_definition)
        task_def_arn = response["taskDefinition"]["taskDefinitionArn"]

        # Create or update service
        service_name = f"{package.name}-service"
        cluster_name = config.environment_variables.get("ECS_CLUSTER", "default")

        try:
            # Update existing service
            self.aws_ecs.update_service(
                cluster=cluster_name,
                service=service_name,
                taskDefinition=task_def_arn,
                desiredCount=config.replicas,
            )
        except:
            # Create new service
            self.aws_ecs.create_service(
                cluster=cluster_name,
                serviceName=service_name,
                taskDefinition=task_def_arn,
                desiredCount=config.replicas,
                launchType="FARGATE",
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": config.environment_variables.get(
                            "SUBNETS", ""
                        ).split(","),
                        "securityGroups": config.environment_variables.get(
                            "SECURITY_GROUPS", ""
                        ).split(","),
                        "assignPublicIp": "ENABLED",
                    }
                },
                loadBalancers=(
                    [
                        {
                            "targetGroupArn": config.environment_variables.get(
                                "TARGET_GROUP_ARN"
                            ),
                            "containerName": package.name,
                            "containerPort": config.ports[0],
                        }
                    ]
                    if config.environment_variables.get("TARGET_GROUP_ARN")
                    else []
                ),
            )

        # Get service endpoints
        # In real implementation, would get ALB/NLB endpoints
        status.endpoints.append(f"https://{package.name}.example.com")
        status.metrics["ecs_service"] = service_name
        status.metrics["ecs_cluster"] = cluster_name

    def _generate_dockerfile(
        self, package: ToolPackage, config: DeploymentConfig
    ) -> str:
        """Generate Dockerfile for package"""
        if package.runtime == "python":
            return f"""FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE {' '.join(str(p) for p in config.ports)}

CMD ["python", "{package.entry_point}"]
"""
        elif package.runtime == "node":
            return f"""FROM node:16-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE {' '.join(str(p) for p in config.ports)}

CMD ["node", "{package.entry_point}"]
"""
        else:
            raise ValueError(f"Unsupported runtime: {package.runtime}")

    async def _wait_for_ready(self, deployment_id: str, timeout: int = 300):
        """Wait for deployment to be ready"""
        start_time = datetime.now()

        while (datetime.now() - start_time).seconds < timeout:
            status = self.deployments.get(deployment_id)
            if not status:
                raise ValueError(f"Deployment not found: {deployment_id}")

            # Check if all replicas are ready
            if status.replicas_ready >= status.replicas_total:
                # Perform health check
                if await self._check_health(status):
                    return

            await asyncio.sleep(5)

        raise TimeoutError(
            f"Deployment {deployment_id} did not become ready in {timeout} seconds"
        )

    async def _check_health(self, status: DeploymentStatus) -> bool:
        """Check deployment health"""
        healthy_endpoints = 0

        for endpoint in status.endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    health_path = "/health"  # Default health check path
                    async with session.get(
                        f"{endpoint}{health_path}", timeout=5
                    ) as resp:
                        if resp.status == 200:
                            healthy_endpoints += 1
            except:
                logger.warning(f"Health check failed for {endpoint}")

        health_ratio = (
            healthy_endpoints / len(status.endpoints) if status.endpoints else 0
        )
        status.health_status = "healthy" if health_ratio > 0.8 else "unhealthy"

        # Update metrics
        deployment_health.labels(deployment_id=status.deployment_id).set(health_ratio)

        return health_ratio > 0.8

    async def _register_service(self, status: DeploymentStatus):
        """Register service with service discovery"""
        if self.consul_available:
            # Register with Consul
            for i, endpoint in enumerate(status.endpoints):
                self.consul_client.agent.service.register(
                    name=status.name,
                    service_id=f"{status.deployment_id}-{i}",
                    address=endpoint.split("://")[1].split(":")[0],
                    port=int(endpoint.split(":")[-1]),
                    tags=[status.version, status.environment],
                    check=consul.Check.http(f"{endpoint}/health", interval="10s"),
                )

        # Update internal registry
        self.service_registry.setdefault(status.name, []).extend(status.endpoints)

    async def scale_deployment(self, deployment_id: str, replicas: int):
        """Scale a deployment"""
        status = self.deployments.get(deployment_id)
        if not status:
            raise ValueError(f"Deployment not found: {deployment_id}")

        if status.environment == "kubernetes":
            # Scale Kubernetes deployment
            self.k8s_apps_v1.patch_namespaced_deployment_scale(
                name=status.metrics["k8s_deployment"],
                namespace=status.metrics["k8s_namespace"],
                body={"spec": {"replicas": replicas}},
            )
        elif status.environment == "docker":
            # Scale Docker containers
            # Would need to start/stop containers
            pass
        elif status.environment == "cloud":
            # Scale cloud service
            self.aws_ecs.update_service(
                cluster=status.metrics["ecs_cluster"],
                service=status.metrics["ecs_service"],
                desiredCount=replicas,
            )

        status.replicas_total = replicas
        status.updated_at = datetime.now()

    async def update_deployment(
        self, deployment_id: str, new_package: ToolPackage, strategy: str = "rolling"
    ):
        """Update an existing deployment"""
        status = self.deployments.get(deployment_id)
        if not status:
            raise ValueError(f"Deployment not found: {deployment_id}")

        if strategy == "rolling":
            await self._rolling_update(status, new_package)
        elif strategy == "blue-green":
            await self._blue_green_update(status, new_package)
        elif strategy == "canary":
            await self._canary_update(status, new_package)
        else:
            raise ValueError(f"Unknown update strategy: {strategy}")

    async def _rolling_update(self, status: DeploymentStatus, new_package: ToolPackage):
        """Perform rolling update"""
        # Update one replica at a time
        for i in range(status.replicas_total):
            # Deploy new version
            # Update load balancer
            # Verify health
            # Continue to next replica
            await asyncio.sleep(10)  # Simplified

    async def _blue_green_update(
        self, status: DeploymentStatus, new_package: ToolPackage
    ):
        """Perform blue-green deployment"""
        # Deploy new version alongside old
        # Switch traffic when ready
        # Remove old version
        pass

    async def _canary_update(self, status: DeploymentStatus, new_package: ToolPackage):
        """Perform canary deployment"""
        # Deploy new version for small percentage
        # Gradually increase traffic
        # Monitor metrics
        # Rollback if issues detected
        pass

    async def rollback_deployment(self, deployment_id: str):
        """Rollback a deployment"""
        status = self.deployments.get(deployment_id)
        if not status:
            raise ValueError(f"Deployment not found: {deployment_id}")

        # Find previous version
        previous = None
        for deployment in self.deployment_history:
            if deployment.name == status.name and deployment.version != status.version:
                previous = deployment
                break

        if not previous:
            raise ValueError("No previous version found for rollback")

        # Rollback based on environment
        if status.environment == "kubernetes":
            # Kubernetes rollback
            self.k8s_apps_v1.create_namespaced_deployment_rollback(
                name=status.metrics["k8s_deployment"],
                namespace=status.metrics["k8s_namespace"],
                body=k8s_client.V1DeploymentRollback(
                    name=status.metrics["k8s_deployment"],
                    rollback_to=k8s_client.V1RollbackConfig(revision=0),
                ),
            )

    async def delete_deployment(self, deployment_id: str):
        """Delete a deployment"""
        status = self.deployments.get(deployment_id)
        if not status:
            raise ValueError(f"Deployment not found: {deployment_id}")

        # Unregister from service discovery
        if self.consul_available:
            for i in range(status.replicas_total):
                self.consul_client.agent.service.deregister(f"{deployment_id}-{i}")

        # Delete based on environment
        if status.environment == "kubernetes":
            # Delete Kubernetes resources
            namespace = status.metrics["k8s_namespace"]
            self.k8s_apps_v1.delete_namespaced_deployment(
                name=status.metrics["k8s_deployment"], namespace=namespace
            )
            self.k8s_v1.delete_namespaced_service(status.name, namespace)
            self.k8s_v1.delete_namespaced_config_map(f"{status.name}-config", namespace)

        elif status.environment == "docker":
            # Stop and remove containers
            for container_id in status.metrics.get("containers", []):
                try:
                    container = self.docker_client.containers.get(container_id)
                    container.stop()
                    container.remove()
                except:
                    pass

        elif status.environment == "cloud":
            # Delete cloud resources
            self.aws_ecs.delete_service(
                cluster=status.metrics["ecs_cluster"],
                service=status.metrics["ecs_service"],
                force=True,
            )

        # Update status
        status.status = "terminated"
        status.updated_at = datetime.now()

        # Move to history
        self.deployment_history.append(status)
        del self.deployments[deployment_id]

        # Update metrics
        active_deployments.labels(environment=status.environment).dec()

    async def get_deployment_logs(
        self, deployment_id: str, lines: int = 100, since: Optional[datetime] = None
    ) -> List[str]:
        """Get deployment logs"""
        status = self.deployments.get(deployment_id)
        if not status:
            raise ValueError(f"Deployment not found: {deployment_id}")

        logs = []

        if status.environment == "kubernetes":
            # Get Kubernetes pod logs
            pods = self.k8s_v1.list_namespaced_pod(
                namespace=status.metrics["k8s_namespace"],
                label_selector=f"app={status.name}",
            )

            for pod in pods.items:
                pod_logs = self.k8s_v1.read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=status.metrics["k8s_namespace"],
                    tail_lines=lines,
                )
                logs.extend(pod_logs.split("\n"))

        elif status.environment == "docker":
            # Get Docker container logs
            for container_id in status.metrics.get("containers", []):
                try:
                    container = self.docker_client.containers.get(container_id)
                    container_logs = container.logs(tail=lines).decode("utf-8")
                    logs.extend(container_logs.split("\n"))
                except:
                    pass

        return logs

    def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics"""
        status = self.deployments.get(deployment_id)
        if not status:
            raise ValueError(f"Deployment not found: {deployment_id}")

        metrics = {
            "deployment_id": deployment_id,
            "name": status.name,
            "version": status.version,
            "status": status.status,
            "health": status.health_status,
            "replicas": {
                "ready": status.replicas_ready,
                "total": status.replicas_total,
            },
            "uptime": (datetime.now() - status.created_at).total_seconds(),
            "endpoints": len(status.endpoints),
        }

        # Add environment-specific metrics
        if status.environment == "kubernetes":
            # Get Kubernetes metrics
            pass
        elif status.environment == "docker":
            # Get Docker metrics
            for container_id in status.metrics.get("containers", []):
                try:
                    container = self.docker_client.containers.get(container_id)
                    stats = container.stats(stream=False)
                    # Process container stats
                except:
                    pass

        return metrics


# Example usage
async def example_usage():
    """Example of using the Tool Deployment System"""
    deployment_system = ToolDeploymentSystem()

    # Create a sample tool package
    package = ToolPackage(
        name="example-api",
        version="1.0.0",
        description="Example API service",
        entry_point="main.py",
        dependencies=["fastapi", "uvicorn"],
        runtime="python",
        metadata={"author": "JARVIS"},
        artifacts=[Path("example_api.zip")],
    )

    # Create deployment configuration
    config = DeploymentConfig(
        name="example-api",
        version="1.0.0",
        environment="docker",  # or "kubernetes", "cloud"
        replicas=3,
        resources={"cpu": 0.5, "memory": 512},
        environment_variables={
            "LOG_LEVEL": "INFO",
            "DATABASE_URL": "postgresql://localhost/db",
        },
        ports=[8080],
        health_check={"type": "http", "path": "/health", "port": 8080, "interval": 30},
        auto_scaling={"min_replicas": 2, "max_replicas": 10, "target_cpu": 70},
    )

    # Deploy the tool
    print("Deploying tool...")
    deployment = await deployment_system.deploy_tool(package, config)
    print(f"Deployment ID: {deployment.deployment_id}")
    print(f"Status: {deployment.status}")
    print(f"Endpoints: {deployment.endpoints}")

    # Wait a bit
    await asyncio.sleep(5)

    # Get deployment metrics
    print("\nDeployment metrics:")
    metrics = deployment_system.get_deployment_metrics(deployment.deployment_id)
    print(json.dumps(metrics, indent=2))

    # Scale deployment
    print("\nScaling deployment to 5 replicas...")
    await deployment_system.scale_deployment(deployment.deployment_id, 5)

    # Get logs
    print("\nDeployment logs:")
    logs = await deployment_system.get_deployment_logs(
        deployment.deployment_id, lines=20
    )
    for log in logs[:10]:
        print(f"  {log}")

    # Clean up
    print("\nDeleting deployment...")
    await deployment_system.delete_deployment(deployment.deployment_id)


if __name__ == "__main__":
    asyncio.run(example_usage())
