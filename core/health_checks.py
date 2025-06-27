#!/usr/bin/env python3
"""
JARVIS Health Check System
"""

import asyncio
from typing import Dict, Any, List
import psutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class JARVISHealthCheck:
    """Comprehensive health monitoring"""
    
    def __init__(self):
        self.checks = {
            "cpu": self.check_cpu,
            "memory": self.check_memory,
            "disk": self.check_disk,
            "network": self.check_network,
            "services": self.check_services,
            "ai_models": self.check_ai_models
        }
        
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "healthy",
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                results["checks"][name] = await check_func()
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["overall_health"] = "degraded"
        
        return results
    
    async def check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "status": "healthy" if cpu_percent < 80 else "warning",
            "usage": cpu_percent,
            "cores": psutil.cpu_count()
        }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy" if memory.percent < 80 else "warning",
            "usage": memory.percent,
            "available": memory.available,
            "total": memory.total
        }
    
    async def check_disk(self) -> Dict[str, Any]:
        """Check disk usage"""
        
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy" if disk.percent < 90 else "warning",
            "usage": disk.percent,
            "free": disk.free,
            "total": disk.total
        }
    
    async def check_network(self) -> Dict[str, Any]:
        """Check network connectivity"""
        
        # Simple connectivity check
        import socket
        
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {"status": "healthy", "connectivity": True}
        except:
            return {"status": "error", "connectivity": False}
    
    async def check_services(self) -> Dict[str, Any]:
        """Check JARVIS services"""
        
        services_status = {}
        
        # Check Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)
            r.ping()
            services_status["redis"] = "running"
        except:
            services_status["redis"] = "stopped"
        
        # Add more service checks
        
        return {
            "status": "healthy" if all(v == "running" for v in services_status.values()) else "degraded",
            "services": services_status
        }
    
    async def check_ai_models(self) -> Dict[str, Any]:
        """Check AI model availability"""
        
        from core.updated_multi_ai_integration import multi_ai
        
        return {
            "status": "healthy" if multi_ai.available_models else "error",
            "available_models": list(multi_ai.available_models.keys())
        }

# Create singleton
health_checker = JARVISHealthCheck()
