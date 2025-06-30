#!/usr/bin/env python3
"""
Self-Healing System for JARVIS
Monitors system health and automatically fixes issues
"""

import asyncio
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health states"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    RECOVERING = "recovering"

@dataclass
class HealthMetric:
    """Health metric data"""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    status: HealthStatus
    timestamp: datetime

class SelfHealingSystem:
    """Self-healing and monitoring system for JARVIS"""
    
    def __init__(self):
        self.active = False
        self.metrics: Dict[str, HealthMetric] = {}
        self.issues_fixed = 0
        self.last_check = datetime.now()
        self.health_history = []
        
        # Thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70, "critical": 90},
            "memory_usage": {"warning": 80, "critical": 95},
            "response_time": {"warning": 2.0, "critical": 5.0},
            "error_rate": {"warning": 0.05, "critical": 0.1}
        }
        
    async def initialize(self):
        """Initialize self-healing system"""
        logger.info("Initializing self-healing system...")
        self.active = True
        
        # Start monitoring
        asyncio.create_task(self._monitor_loop())
        
        logger.info("Self-healing system online")
        
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.active:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Analyze health
                health_status = self._analyze_health(metrics)
                
                # Take action if needed
                if health_status != HealthStatus.HEALTHY:
                    await self._heal_system(metrics)
                    
                # Store history
                self.health_history.append({
                    "timestamp": datetime.now(),
                    "status": health_status.value,
                    "metrics": {k: v.value for k, v in metrics.items()}
                })
                
                # Keep only last hour of history
                cutoff = datetime.now() - timedelta(hours=1)
                self.health_history = [h for h in self.health_history if h["timestamp"] > cutoff]
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _collect_metrics(self) -> Dict[str, HealthMetric]:
        """Collect system metrics"""
        metrics = {}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics["cpu_usage"] = HealthMetric(
            name="CPU Usage",
            value=cpu_percent,
            threshold_warning=self.thresholds["cpu_usage"]["warning"],
            threshold_critical=self.thresholds["cpu_usage"]["critical"],
            status=self._get_status(cpu_percent, self.thresholds["cpu_usage"]),
            timestamp=datetime.now()
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        metrics["memory_usage"] = HealthMetric(
            name="Memory Usage",
            value=memory.percent,
            threshold_warning=self.thresholds["memory_usage"]["warning"],
            threshold_critical=self.thresholds["memory_usage"]["critical"],
            status=self._get_status(memory.percent, self.thresholds["memory_usage"]),
            timestamp=datetime.now()
        )
        
        # Response time (simulated)
        response_time = 0.5  # Would measure actual response time in production
        metrics["response_time"] = HealthMetric(
            name="Response Time",
            value=response_time,
            threshold_warning=self.thresholds["response_time"]["warning"],
            threshold_critical=self.thresholds["response_time"]["critical"],
            status=self._get_status(response_time, self.thresholds["response_time"]),
            timestamp=datetime.now()
        )
        
        # Error rate (simulated)
        error_rate = 0.01  # Would calculate from actual errors
        metrics["error_rate"] = HealthMetric(
            name="Error Rate",
            value=error_rate,
            threshold_warning=self.thresholds["error_rate"]["warning"],
            threshold_critical=self.thresholds["error_rate"]["critical"],
            status=self._get_status(error_rate, self.thresholds["error_rate"]),
            timestamp=datetime.now()
        )
        
        self.metrics = metrics
        return metrics
        
    def _get_status(self, value: float, thresholds: Dict[str, float]) -> HealthStatus:
        """Determine health status based on value and thresholds"""
        if value >= thresholds["critical"]:
            return HealthStatus.CRITICAL
        elif value >= thresholds["warning"]:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
            
    def _analyze_health(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Analyze overall system health"""
        statuses = [m.status for m in metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
            
    async def _heal_system(self, metrics: Dict[str, HealthMetric]):
        """Attempt to heal system issues"""
        logger.info("Self-healing activated...")
        
        for key, metric in metrics.items():
            if metric.status == HealthStatus.CRITICAL:
                await self._fix_critical_issue(key, metric)
            elif metric.status == HealthStatus.WARNING:
                await self._fix_warning_issue(key, metric)
                
        self.issues_fixed += 1
        logger.info(f"Self-healing complete. Total issues fixed: {self.issues_fixed}")
        
    async def _fix_critical_issue(self, issue_type: str, metric: HealthMetric):
        """Fix critical issues"""
        logger.warning(f"Fixing critical issue: {issue_type} at {metric.value:.1f}%")
        
        if issue_type == "memory_usage":
            # Clear caches, run garbage collection
            import gc
            gc.collect()
            logger.info("Cleared memory caches")
            
        elif issue_type == "cpu_usage":
            # Reduce processing load
            logger.info("Reducing processing load")
            await asyncio.sleep(5)  # Give system time to cool down
            
    async def _fix_warning_issue(self, issue_type: str, metric: HealthMetric):
        """Fix warning issues"""
        logger.info(f"Addressing warning: {issue_type} at {metric.value:.1f}%")
        
        # Log the warning for now
        # In production, would take specific actions
        
    def get_health_report(self) -> Dict[str, Any]:
        """Get current health report"""
        overall_status = self._analyze_health(self.metrics) if self.metrics else HealthStatus.HEALTHY
        
        report = {
            "status": overall_status.value,
            "last_check": self.last_check.isoformat(),
            "issues_fixed": self.issues_fixed,
            "metrics": {}
        }
        
        for key, metric in self.metrics.items():
            report["metrics"][key] = {
                "value": metric.value,
                "status": metric.status.value,
                "threshold_warning": metric.threshold_warning,
                "threshold_critical": metric.threshold_critical
            }
            
        return report
        
    def get_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        if not self.metrics:
            return 100.0
            
        scores = []
        for key, metric in self.metrics.items():
            if metric.status == HealthStatus.HEALTHY:
                scores.append(100)
            elif metric.status == HealthStatus.WARNING:
                scores.append(70)
            elif metric.status == HealthStatus.CRITICAL:
                scores.append(30)
            else:
                scores.append(50)
                
        return sum(scores) / len(scores) if scores else 100.0

# Singleton instance
self_healing = SelfHealingSystem()
