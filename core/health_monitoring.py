#!/usr/bin/env python3
"""
JARVIS Health Monitoring System
System performance monitoring and optimization suggestions.
"""

import os
import json
import psutil
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from collections import deque
import platform
import GPUtil
import speedtest
import threading
import warnings

logger = logging.getLogger("jarvis.health")


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_percent: float
    cpu_freq: float
    cpu_temp: Optional[float]
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temp: Optional[float] = None
    process_count: int = 0
    thread_count: int = 0
    jarvis_cpu: float = 0.0
    jarvis_memory_mb: float = 0.0


@dataclass
class HealthAlert:
    """System health alert"""
    level: str  # info, warning, critical
    category: str  # cpu, memory, disk, network, etc.
    message: str
    suggestion: str
    timestamp: datetime = None
    resolved: bool = False
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class OptimizationSuggestion:
    """Performance optimization suggestion"""
    category: str
    priority: str  # low, medium, high
    title: str
    description: str
    expected_improvement: str
    implementation: List[str]
    requires_restart: bool = False


class SystemHealthMonitor:
    """Monitors system health and performance"""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1min intervals
        self.alerts: List[HealthAlert] = []
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Thresholds
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_percent': {'warning': 80, 'critical': 90},
            'cpu_temp': {'warning': 70, 'critical': 85},
            'gpu_temp': {'warning': 75, 'critical': 90},
            'response_time': {'warning': 500, 'critical': 1000}  # ms
        }
        
        # Performance baselines
        self.baselines = {}
        self.anomaly_threshold = 2.5  # standard deviations
        
        # Get JARVIS process
        self.jarvis_process = psutil.Process()
        
        # Platform-specific initialization
        self.platform = platform.system()
        self.has_gpu = len(GPUtil.getGPUs()) > 0
        
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System health monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check for issues
                self.check_system_health(metrics)
                
                # Update baselines periodically
                if len(self.metrics_history) > 60:
                    self.update_baselines()
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
            # Wait for next check
            asyncio.run(asyncio.sleep(self.check_interval))
            
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_freq = psutil.cpu_freq()
        cpu_temp = self.get_cpu_temperature()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network metrics
        net_io = psutil.net_io_counters()
        
        # Process metrics
        process_count = len(psutil.pids())
        
        # JARVIS-specific metrics
        jarvis_cpu = self.jarvis_process.cpu_percent()
        jarvis_memory = self.jarvis_process.memory_info().rss / 1024 / 1024
        
        # GPU metrics (if available)
        gpu_percent = None
        gpu_memory_percent = None
        gpu_temp = None
        
        if self.has_gpu:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_percent = gpu.load * 100
                    gpu_memory_percent = gpu.memoryUtil * 100
                    gpu_temp = gpu.temperature
            except:
                pass
                
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            cpu_freq=cpu_freq.current if cpu_freq else 0,
            cpu_temp=cpu_temp,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1024 / 1024 / 1024,
            memory_available_gb=memory.available / 1024 / 1024 / 1024,
            disk_percent=disk.percent,
            disk_read_mb=disk_io.read_bytes / 1024 / 1024 if disk_io else 0,
            disk_write_mb=disk_io.write_bytes / 1024 / 1024 if disk_io else 0,
            network_sent_mb=net_io.bytes_sent / 1024 / 1024,
            network_recv_mb=net_io.bytes_recv / 1024 / 1024,
            gpu_percent=gpu_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_temp=gpu_temp,
            process_count=process_count,
            thread_count=threading.active_count(),
            jarvis_cpu=jarvis_cpu,
            jarvis_memory_mb=jarvis_memory
        )
        
    def get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature (platform-specific)"""
        try:
            if self.platform == "Linux":
                # Try reading from thermal zone
                thermal_path = "/sys/class/thermal/thermal_zone0/temp"
                if os.path.exists(thermal_path):
                    with open(thermal_path, 'r') as f:
                        return float(f.read().strip()) / 1000
                        
            elif self.platform == "Darwin":  # macOS
                # Would need to use external tools like iStats
                pass
                
            elif self.platform == "Windows":
                # Would need WMI or OpenHardwareMonitor
                pass
                
        except:
            pass
            
        return None
        
    def check_system_health(self, metrics: SystemMetrics):
        """Check system health and generate alerts"""
        alerts = []
        
        # CPU checks
        if metrics.cpu_percent > self.thresholds['cpu_percent']['critical']:
            alerts.append(HealthAlert(
                level='critical',
                category='cpu',
                message=f'CPU usage critical: {metrics.cpu_percent:.1f}%',
                suggestion='Close unnecessary applications or upgrade CPU'
            ))
        elif metrics.cpu_percent > self.thresholds['cpu_percent']['warning']:
            alerts.append(HealthAlert(
                level='warning',
                category='cpu',
                message=f'CPU usage high: {metrics.cpu_percent:.1f}%',
                suggestion='Monitor CPU-intensive processes'
            ))
            
        # Memory checks
        if metrics.memory_percent > self.thresholds['memory_percent']['critical']:
            alerts.append(HealthAlert(
                level='critical',
                category='memory',
                message=f'Memory usage critical: {metrics.memory_percent:.1f}%',
                suggestion='Close applications or add more RAM'
            ))
        elif metrics.memory_percent > self.thresholds['memory_percent']['warning']:
            alerts.append(HealthAlert(
                level='warning',
                category='memory',
                message=f'Memory usage high: {metrics.memory_percent:.1f}%',
                suggestion='Consider closing unused applications'
            ))
            
        # Disk checks
        if metrics.disk_percent > self.thresholds['disk_percent']['critical']:
            alerts.append(HealthAlert(
                level='critical',
                category='disk',
                message=f'Disk space critical: {metrics.disk_percent:.1f}% used',
                suggestion='Free up disk space immediately'
            ))
            
        # Temperature checks
        if metrics.cpu_temp and metrics.cpu_temp > self.thresholds['cpu_temp']['critical']:
            alerts.append(HealthAlert(
                level='critical',
                category='temperature',
                message=f'CPU temperature critical: {metrics.cpu_temp:.1f}°C',
                suggestion='Check cooling system and reduce load'
            ))
            
        # JARVIS-specific checks
        if metrics.jarvis_memory_mb > 1000:  # 1GB
            alerts.append(HealthAlert(
                level='warning',
                category='jarvis',
                message=f'JARVIS memory usage high: {metrics.jarvis_memory_mb:.0f}MB',
                suggestion='Restart JARVIS to free memory'
            ))
            
        # Add new alerts
        for alert in alerts:
            if not self.is_duplicate_alert(alert):
                self.alerts.append(alert)
                logger.warning(f"Health alert: {alert.message}")
                
    def is_duplicate_alert(self, alert: HealthAlert) -> bool:
        """Check if alert is duplicate of recent alert"""
        for existing in self.alerts[-10:]:  # Check last 10 alerts
            if (existing.category == alert.category and 
                existing.level == alert.level and
                not existing.resolved and
                (alert.timestamp - existing.timestamp) < timedelta(minutes=5)):
                return True
        return False
        
    def update_baselines(self):
        """Update performance baselines"""
        if len(self.metrics_history) < 60:
            return
            
        # Calculate statistics for each metric
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        self.baselines = {
            'cpu_percent': {
                'mean': np.mean([m.cpu_percent for m in recent_metrics]),
                'std': np.std([m.cpu_percent for m in recent_metrics])
            },
            'memory_percent': {
                'mean': np.mean([m.memory_percent for m in recent_metrics]),
                'std': np.std([m.memory_percent for m in recent_metrics])
            },
            'jarvis_memory': {
                'mean': np.mean([m.jarvis_memory_mb for m in recent_metrics]),
                'std': np.std([m.jarvis_memory_mb for m in recent_metrics])
            }
        }
        
    def detect_anomalies(self, metrics: SystemMetrics) -> List[str]:
        """Detect performance anomalies"""
        anomalies = []
        
        if not self.baselines:
            return anomalies
            
        # Check CPU anomaly
        if 'cpu_percent' in self.baselines:
            baseline = self.baselines['cpu_percent']
            if abs(metrics.cpu_percent - baseline['mean']) > self.anomaly_threshold * baseline['std']:
                anomalies.append(f"CPU usage anomaly: {metrics.cpu_percent:.1f}% (baseline: {baseline['mean']:.1f}%)")
                
        # Check memory anomaly
        if 'memory_percent' in self.baselines:
            baseline = self.baselines['memory_percent']
            if abs(metrics.memory_percent - baseline['mean']) > self.anomaly_threshold * baseline['std']:
                anomalies.append(f"Memory usage anomaly: {metrics.memory_percent:.1f}% (baseline: {baseline['mean']:.1f}%)")
                
        return anomalies
        
    def get_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on metrics"""
        suggestions = []
        
        if not self.metrics_history:
            return suggestions
            
        # Analyze recent metrics
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        # High CPU usage
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        if avg_cpu > 60:
            suggestions.append(OptimizationSuggestion(
                category='cpu',
                priority='high' if avg_cpu > 80 else 'medium',
                title='Reduce CPU Usage',
                description='CPU usage has been consistently high',
                expected_improvement='20-30% CPU reduction',
                implementation=[
                    'Identify CPU-intensive processes',
                    'Disable unnecessary background services',
                    'Consider upgrading CPU or optimizing code'
                ]
            ))
            
        # High memory usage
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        if avg_memory > 70:
            suggestions.append(OptimizationSuggestion(
                category='memory',
                priority='high' if avg_memory > 85 else 'medium',
                title='Optimize Memory Usage',
                description='Memory usage is consistently high',
                expected_improvement='15-25% memory reduction',
                implementation=[
                    'Close unused applications',
                    'Disable memory-intensive features',
                    'Add more RAM if needed'
                ]
            ))
            
        # JARVIS-specific optimizations
        avg_jarvis_memory = np.mean([m.jarvis_memory_mb for m in recent_metrics])
        if avg_jarvis_memory > 500:
            suggestions.append(OptimizationSuggestion(
                category='jarvis',
                priority='medium',
                title='Optimize JARVIS Memory',
                description='JARVIS is using significant memory',
                expected_improvement='30-40% memory reduction',
                implementation=[
                    'Restart JARVIS to clear memory',
                    'Disable unused plugins',
                    'Reduce conversation history size',
                    'Enable memory optimization mode'
                ],
                requires_restart=True
            ))
            
        # Disk space
        if self.metrics_history[-1].disk_percent > 70:
            suggestions.append(OptimizationSuggestion(
                category='disk',
                priority='medium',
                title='Free Up Disk Space',
                description='Disk space is running low',
                expected_improvement='10-20% free space',
                implementation=[
                    'Clear JARVIS logs and cache',
                    'Remove old recordings',
                    'Clean temporary files',
                    'Move data to external storage'
                ]
            ))
            
        return suggestions
        
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        if not self.metrics_history:
            return {'status': 'No data available'}
            
        current = self.metrics_history[-1]
        recent_metrics = list(self.metrics_history)[-60:]  # Last hour
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_jarvis_cpu = np.mean([m.jarvis_cpu for m in recent_metrics])
        avg_jarvis_memory = np.mean([m.jarvis_memory_mb for m in recent_metrics])
        
        # Determine overall health
        health_score = 100
        if avg_cpu > 80:
            health_score -= 20
        elif avg_cpu > 60:
            health_score -= 10
            
        if avg_memory > 85:
            health_score -= 20
        elif avg_memory > 70:
            health_score -= 10
            
        if current.disk_percent > 85:
            health_score -= 15
        elif current.disk_percent > 70:
            health_score -= 5
            
        # Get active alerts
        active_alerts = [a for a in self.alerts if not a.resolved]
        critical_alerts = [a for a in active_alerts if a.level == 'critical']
        warning_alerts = [a for a in active_alerts if a.level == 'warning']
        
        return {
            'timestamp': current.timestamp.isoformat(),
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score > 70 else 'degraded' if health_score > 40 else 'unhealthy',
            'current_metrics': {
                'cpu_percent': current.cpu_percent,
                'memory_percent': current.memory_percent,
                'disk_percent': current.disk_percent,
                'jarvis_memory_mb': current.jarvis_memory_mb,
                'gpu_percent': current.gpu_percent,
                'cpu_temp': current.cpu_temp
            },
            'averages': {
                'cpu_percent': round(avg_cpu, 1),
                'memory_percent': round(avg_memory, 1),
                'jarvis_cpu_percent': round(avg_jarvis_cpu, 1),
                'jarvis_memory_mb': round(avg_jarvis_memory, 1)
            },
            'alerts': {
                'total': len(active_alerts),
                'critical': len(critical_alerts),
                'warning': len(warning_alerts),
                'recent': [asdict(a) for a in active_alerts[-5:]]
            },
            'suggestions': [asdict(s) for s in self.get_optimization_suggestions()],
            'anomalies': self.detect_anomalies(current)
        }


class NetworkHealthMonitor:
    """Monitors network connectivity and performance"""
    
    def __init__(self):
        self.last_speed_test = None
        self.speed_test_interval = timedelta(hours=6)
        self.connectivity_checks = deque(maxlen=100)
        
    async def check_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        import aiohttp
        
        checks = {
            'dns': False,
            'internet': False,
            'latency_ms': None,
            'packet_loss': 0
        }
        
        # DNS check
        try:
            import socket
            socket.gethostbyname('google.com')
            checks['dns'] = True
        except:
            pass
            
        # Internet connectivity
        try:
            async with aiohttp.ClientSession() as session:
                start = datetime.now()
                async with session.get('https://www.google.com', timeout=5) as resp:
                    if resp.status == 200:
                        checks['internet'] = True
                        checks['latency_ms'] = (datetime.now() - start).total_seconds() * 1000
        except:
            pass
            
        self.connectivity_checks.append({
            'timestamp': datetime.now(),
            'connected': checks['internet'],
            'latency': checks['latency_ms']
        })
        
        return checks
        
    def run_speed_test(self) -> Optional[Dict[str, float]]:
        """Run internet speed test"""
        try:
            if (self.last_speed_test and 
                datetime.now() - self.last_speed_test < self.speed_test_interval):
                return None
                
            st = speedtest.Speedtest()
            st.get_best_server()
            
            download = st.download() / 1024 / 1024  # Mbps
            upload = st.upload() / 1024 / 1024  # Mbps
            ping = st.results.ping
            
            self.last_speed_test = datetime.now()
            
            return {
                'download_mbps': round(download, 2),
                'upload_mbps': round(upload, 2),
                'ping_ms': round(ping, 2)
            }
            
        except Exception as e:
            logger.error(f"Speed test failed: {e}")
            return None


class HealthMonitoringCommandProcessor:
    """Processes health monitoring commands"""
    
    def __init__(self, health_monitor: SystemHealthMonitor):
        self.monitor = health_monitor
        self.network_monitor = NetworkHealthMonitor()
        
    async def process_command(self, command: str) -> Tuple[bool, str]:
        """Process health monitoring commands"""
        command_lower = command.lower()
        
        # System status
        if "system status" in command_lower or "health check" in command_lower:
            report = self.monitor.get_health_report()
            
            status = report['status']
            score = report['health_score']
            alerts = report['alerts']['total']
            
            response = f"System health: {status} (score: {score}/100)\n"
            
            if alerts > 0:
                response += f"Active alerts: {alerts} ({report['alerts']['critical']} critical)\n"
                
            metrics = report['current_metrics']
            response += f"CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory_percent']:.1f}%, Disk: {metrics['disk_percent']:.1f}%"
            
            return True, response
            
        # Performance metrics
        if "performance" in command_lower or "metrics" in command_lower:
            current = self.monitor.metrics_history[-1] if self.monitor.metrics_history else None
            
            if not current:
                return True, "No performance data available yet"
                
            response = f"Current Performance:\n"
            response += f"• CPU: {current.cpu_percent:.1f}% @ {current.cpu_freq:.0f} MHz\n"
            response += f"• Memory: {current.memory_used_gb:.1f}/{current.memory_used_gb + current.memory_available_gb:.1f} GB\n"
            response += f"• JARVIS: {current.jarvis_cpu:.1f}% CPU, {current.jarvis_memory_mb:.0f} MB RAM\n"
            
            if current.gpu_percent is not None:
                response += f"• GPU: {current.gpu_percent:.1f}% usage"
                
            return True, response
            
        # Optimization suggestions
        if "optimize" in command_lower or "suggestions" in command_lower:
            suggestions = self.monitor.get_optimization_suggestions()
            
            if not suggestions:
                return True, "System is running optimally. No suggestions at this time."
                
            response = "Optimization suggestions:\n"
            for i, sug in enumerate(suggestions[:3], 1):
                response += f"{i}. {sug.title} ({sug.priority} priority)\n"
                response += f"   {sug.description}\n"
                
            return True, response
            
        # Network check
        if "network" in command_lower or "connectivity" in command_lower:
            checks = await self.network_monitor.check_connectivity()
            
            if checks['internet']:
                response = f"Network connected. Latency: {checks['latency_ms']:.0f}ms"
                
                # Run speed test if requested
                if "speed" in command_lower:
                    response += "\nRunning speed test..."
                    speed = self.network_monitor.run_speed_test()
                    if speed:
                        response = f"Network speed: ↓{speed['download_mbps']} Mbps, ↑{speed['upload_mbps']} Mbps, ping: {speed['ping_ms']}ms"
            else:
                response = "Network connectivity issues detected"
                
            return True, response
            
        # Clear alerts
        if "clear alerts" in command_lower:
            cleared = len([a for a in self.monitor.alerts if not a.resolved])
            for alert in self.monitor.alerts:
                alert.resolved = True
                
            return True, f"Cleared {cleared} alerts"
            
        return False, ""


def integrate_health_monitoring_with_jarvis(jarvis_instance) -> SystemHealthMonitor:
    """Integrate health monitoring with JARVIS"""
    
    # Create health monitor
    monitor = SystemHealthMonitor()
    
    # Create command processor
    processor = HealthMonitoringCommandProcessor(monitor)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Add to JARVIS if available
    if hasattr(jarvis_instance, 'health_monitor'):
        jarvis_instance.health_monitor = monitor
        jarvis_instance.health_processor = processor
        
        logger.info("Health monitoring integrated with JARVIS")
        
    return monitor