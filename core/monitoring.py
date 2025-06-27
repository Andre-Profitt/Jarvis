#!/usr/bin/env python3
"""
Comprehensive Monitoring and Observability for JARVIS
Metrics, logging, tracing, and alerting
"""

import os
import time
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from functools import wraps
import asyncio
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    start_http_server, generate_latest
)
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
import structlog
from dataclasses import dataclass, asdict
import psutil
import aiohttp
from pathlib import Path

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus Metrics

# Counters
request_count = Counter(
    'jarvis_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

task_count = Counter(
    'jarvis_tasks_total',
    'Total number of tasks processed',
    ['task_type', 'status']
)

error_count = Counter(
    'jarvis_errors_total',
    'Total number of errors',
    ['error_type', 'component']
)

ai_query_count = Counter(
    'jarvis_ai_queries_total',
    'Total AI model queries',
    ['model', 'task_type']
)

# Histograms
request_duration = Histogram(
    'jarvis_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

task_duration = Histogram(
    'jarvis_task_duration_seconds',
    'Task processing duration',
    ['task_type']
)

model_latency = Histogram(
    'jarvis_model_latency_seconds',
    'AI model response latency',
    ['model']
)

# Gauges
active_agents = Gauge(
    'jarvis_active_agents',
    'Number of active swarm agents',
    ['agent_type']
)

memory_usage = Gauge(
    'jarvis_memory_usage_bytes',
    'Memory usage in bytes'
)

websocket_connections = Gauge(
    'jarvis_websocket_connections',
    'Active WebSocket connections'
)

model_cache_size = Gauge(
    'jarvis_model_cache_size_bytes',
    'Model cache size in bytes'
)

# Summaries
conversation_sentiment = Summary(
    'jarvis_conversation_sentiment',
    'Conversation sentiment scores'
)

learning_confidence = Summary(
    'jarvis_learning_confidence',
    'Learning confidence scores'
)


@dataclass
class MetricEvent:
    """Structured metric event"""
    timestamp: str
    event_type: str
    component: str
    metrics: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics_buffer = []
        self.start_time = time.time()
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage.set(memory.used)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics
        network = psutil.net_io_counters()
        
        return {
            'cpu': {
                'percent': cpu_percent,
                'count': cpu_count,
                'per_cpu': psutil.cpu_percent(percpu=True)
            },
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'percent': memory.percent,
                'available': memory.available
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'percent': disk.percent,
                'free': disk.free
            },
            'network': {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            },
            'uptime': time.time() - self.start_time
        }
    
    def record_event(self, event: MetricEvent):
        """Record metric event"""
        self.metrics_buffer.append(asdict(event))
        
        # Flush buffer if too large
        if len(self.metrics_buffer) > 1000:
            self.flush_metrics()
    
    def flush_metrics(self):
        """Flush metrics to storage"""
        if not self.metrics_buffer:
            return
        
        # Write to metrics log
        metrics_file = Path("logs/metrics.jsonl")
        metrics_file.parent.mkdir(exist_ok=True)
        
        with open(metrics_file, 'a') as f:
            for metric in self.metrics_buffer:
                f.write(json.dumps(metric) + '\n')
        
        self.metrics_buffer.clear()


# Tracing

def setup_tracing():
    """Setup OpenTelemetry tracing"""
    
    if os.getenv('ENABLE_TRACING', 'false').lower() == 'true':
        # Set up the tracer provider
        trace.set_tracer_provider(TracerProvider())
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=os.getenv('OTLP_ENDPOINT', 'localhost:4317'),
            insecure=True
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Auto-instrument requests
        RequestsInstrumentor().instrument()
        
        logger.info("Tracing enabled", endpoint=os.getenv('OTLP_ENDPOINT'))


# Decorators

def monitor_performance(component: str):
    """Decorator to monitor function performance"""
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Create trace span
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("component", component)
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record metrics
                    duration = time.time() - start_time
                    task_duration.labels(task_type=func.__name__).observe(duration)
                    
                    logger.info(
                        "function_executed",
                        component=component,
                        function=func.__name__,
                        duration=duration,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    error_count.labels(
                        error_type=type(e).__name__,
                        component=component
                    ).inc()
                    
                    logger.error(
                        "function_error",
                        component=component,
                        function=func.__name__,
                        error=str(e),
                        exc_info=True
                    )
                    
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("component", component)
                
                try:
                    result = func(*args, **kwargs)
                    
                    duration = time.time() - start_time
                    task_duration.labels(task_type=func.__name__).observe(duration)
                    
                    logger.info(
                        "function_executed",
                        component=component,
                        function=func.__name__,
                        duration=duration,
                        success=True
                    )
                    
                    return result
                    
                except Exception as e:
                    error_count.labels(
                        error_type=type(e).__name__,
                        component=component
                    ).inc()
                    
                    logger.error(
                        "function_error",
                        component=component,
                        function=func.__name__,
                        error=str(e),
                        exc_info=True
                    )
                    
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@contextmanager
def monitor_operation(operation_name: str, **tags):
    """Context manager for monitoring operations"""
    
    start_time = time.time()
    
    # Create trace span
    tracer = trace.get_tracer(__name__)
    span = tracer.start_span(operation_name)
    
    for key, value in tags.items():
        span.set_attribute(key, str(value))
    
    try:
        yield span
        
        # Success metrics
        duration = time.time() - start_time
        
        logger.info(
            "operation_completed",
            operation=operation_name,
            duration=duration,
            tags=tags
        )
        
    except Exception as e:
        # Error metrics
        error_count.labels(
            error_type=type(e).__name__,
            component=operation_name
        ).inc()
        
        logger.error(
            "operation_failed",
            operation=operation_name,
            error=str(e),
            tags=tags,
            exc_info=True
        )
        
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        
        raise
        
    finally:
        span.end()


# Health Checks

class HealthChecker:
    """Advanced health checking system"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_results = {}
        
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check"""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        
        results = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    check_result = await check_func()
                else:
                    check_result = check_func()
                
                results['checks'][name] = {
                    'status': check_result.get('status', 'healthy'),
                    'message': check_result.get('message', 'OK'),
                    'metrics': check_result.get('metrics', {})
                }
                
                if check_result.get('status') == 'unhealthy':
                    results['status'] = 'unhealthy'
                elif check_result.get('status') == 'degraded' and results['status'] == 'healthy':
                    results['status'] = 'degraded'
                    
            except Exception as e:
                results['checks'][name] = {
                    'status': 'unhealthy',
                    'message': str(e),
                    'error': type(e).__name__
                }
                results['status'] = 'unhealthy'
        
        self.last_check_results = results
        return results
    
    def get_status(self) -> str:
        """Get overall health status"""
        return self.last_check_results.get('status', 'unknown')


# Alerting

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv('ALERT_WEBHOOK_URL')
        self.alert_history = []
        self.alert_rules = {}
        
    def add_rule(self, 
                 name: str,
                 condition: Callable[[], bool],
                 message: str,
                 severity: str = 'warning'):
        """Add alert rule"""
        self.alert_rules[name] = {
            'condition': condition,
            'message': message,
            'severity': severity,
            'last_triggered': None
        }
    
    async def check_alerts(self):
        """Check all alert rules"""
        
        for name, rule in self.alert_rules.items():
            try:
                if rule['condition']():
                    await self.trigger_alert(
                        name,
                        rule['message'],
                        rule['severity']
                    )
            except Exception as e:
                logger.error(
                    "alert_check_failed",
                    alert=name,
                    error=str(e)
                )
    
    async def trigger_alert(self, 
                          name: str,
                          message: str,
                          severity: str = 'warning',
                          metadata: Optional[Dict[str, Any]] = None):
        """Trigger an alert"""
        
        alert = {
            'name': name,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(
            "alert_triggered",
            alert_name=name,
            severity=severity,
            message=message,
            metadata=metadata
        )
        
        # Send webhook if configured
        if self.webhook_url:
            await self._send_webhook(alert)
        
        # Update rule
        if name in self.alert_rules:
            self.alert_rules[name]['last_triggered'] = datetime.utcnow()
    
    async def _send_webhook(self, alert: Dict[str, Any]):
        """Send alert to webhook"""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json={
                        'text': f"🚨 JARVIS Alert: {alert['message']}",
                        'severity': alert['severity'],
                        'timestamp': alert['timestamp'],
                        'metadata': alert['metadata']
                    }
                ) as response:
                    if response.status != 200:
                        logger.error(
                            "webhook_failed",
                            status=response.status,
                            alert=alert['name']
                        )
        except Exception as e:
            logger.error("webhook_error", error=str(e))


# Main monitoring service

class MonitoringService:
    """Central monitoring service"""
    
    def __init__(self, metrics_port: int = 9090):
        self.metrics_port = metrics_port
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        self.running = False
        
        # Setup components
        setup_tracing()
        self._setup_default_health_checks()
        self._setup_default_alerts()
        
    def _setup_default_health_checks(self):
        """Setup default health checks"""
        
        # Database check
        async def check_database():
            try:
                from core.database import db_manager
                with db_manager.get_session() as session:
                    session.execute("SELECT 1")
                return {'status': 'healthy', 'message': 'Database connected'}
            except Exception as e:
                return {'status': 'unhealthy', 'message': str(e)}
        
        # Redis check
        async def check_redis():
            try:
                import redis
                r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'))
                r.ping()
                return {'status': 'healthy', 'message': 'Redis connected'}
            except Exception as e:
                return {'status': 'unhealthy', 'message': str(e)}
        
        # Memory check
        def check_memory():
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {'status': 'unhealthy', 'message': f'Memory usage critical: {memory.percent}%'}
            elif memory.percent > 80:
                return {'status': 'degraded', 'message': f'Memory usage high: {memory.percent}%'}
            return {'status': 'healthy', 'message': f'Memory usage: {memory.percent}%'}
        
        self.health_checker.register_check('database', check_database)
        self.health_checker.register_check('redis', check_redis)
        self.health_checker.register_check('memory', check_memory)
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        
        # High memory usage
        self.alert_manager.add_rule(
            'high_memory',
            lambda: psutil.virtual_memory().percent > 85,
            'Memory usage above 85%',
            'warning'
        )
        
        # High CPU usage
        self.alert_manager.add_rule(
            'high_cpu',
            lambda: psutil.cpu_percent(interval=1) > 90,
            'CPU usage above 90%',
            'warning'
        )
        
        # Disk space
        self.alert_manager.add_rule(
            'low_disk',
            lambda: psutil.disk_usage('/').percent > 90,
            'Disk usage above 90%',
            'critical'
        )
    
    async def start(self):
        """Start monitoring service"""
        
        # Start Prometheus metrics server
        start_http_server(self.metrics_port)
        logger.info("metrics_server_started", port=self.metrics_port)
        
        self.running = True
        
        # Start monitoring tasks
        await asyncio.gather(
            self._collect_metrics_loop(),
            self._health_check_loop(),
            self._alert_check_loop()
        )
    
    async def _collect_metrics_loop(self):
        """Continuously collect metrics"""
        
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                
                event = MetricEvent(
                    timestamp=datetime.utcnow().isoformat(),
                    event_type='system_metrics',
                    component='monitoring',
                    metrics=system_metrics
                )
                
                self.metrics_collector.record_event(event)
                
            except Exception as e:
                logger.error("metrics_collection_error", error=str(e))
            
            await asyncio.sleep(10)  # Collect every 10 seconds
    
    async def _health_check_loop(self):
        """Run periodic health checks"""
        
        while self.running:
            try:
                results = await self.health_checker.run_checks()
                
                # Log health status
                logger.info(
                    "health_check_completed",
                    status=results['status'],
                    checks=results['checks']
                )
                
            except Exception as e:
                logger.error("health_check_error", error=str(e))
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _alert_check_loop(self):
        """Check alert conditions"""
        
        while self.running:
            try:
                await self.alert_manager.check_alerts()
            except Exception as e:
                logger.error("alert_check_error", error=str(e))
            
            await asyncio.sleep(60)  # Check every minute
    
    def stop(self):
        """Stop monitoring service"""
        self.running = False
        self.metrics_collector.flush_metrics()
        logger.info("monitoring_service_stopped")


# Create global monitoring instance
monitoring_service = MonitoringService()


# Convenience functions

def log_ai_query(model: str, task_type: str, latency: float):
    """Log AI model query"""
    
    ai_query_count.labels(model=model, task_type=task_type).inc()
    model_latency.labels(model=model).observe(latency)
    
    logger.info(
        "ai_query",
        model=model,
        task_type=task_type,
        latency=latency
    )


def log_task_completion(task_type: str, status: str, duration: float):
    """Log task completion"""
    
    task_count.labels(task_type=task_type, status=status).inc()
    task_duration.labels(task_type=task_type).observe(duration)
    
    logger.info(
        "task_completed",
        task_type=task_type,
        status=status,
        duration=duration
    )


def update_agent_count(agent_type: str, count: int):
    """Update active agent count"""
    active_agents.labels(agent_type=agent_type).set(count)


if __name__ == "__main__":
    # Test monitoring
    asyncio.run(monitoring_service.start())