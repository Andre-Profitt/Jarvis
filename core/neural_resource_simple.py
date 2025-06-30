#!/usr/bin/env python3
"""
Neural Resource Manager for JARVIS
Intelligent resource allocation inspired by neural networks
"""

import asyncio
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)

@dataclass
class Resource:
    """Resource allocation unit"""
    name: str
    allocated: float  # 0-100%
    priority: int     # 1-10
    task: Optional[str] = None
    
@dataclass 
class Task:
    """Task requiring resources"""
    id: str
    name: str
    priority: int
    required_cpu: float
    required_memory: float
    required_time: float
    status: str = "pending"

class NeuralResourceManager:
    """Brain-inspired resource management"""
    
    def __init__(self):
        self.active = False
        self.resources: Dict[str, Resource] = {}
        self.tasks: List[Task] = []
        self.completed_tasks = 0
        self.optimization_cycles = 0
        
        # Neural-inspired parameters
        self.learning_rate = 0.1
        self.efficiency_score = 1.0
        self.adaptation_history = []
        
    async def initialize(self):
        """Initialize neural resource manager"""
        logger.info("Initializing Neural Resource Manager...")
        
        # Initialize resource pools
        self.resources = {
            "cpu": Resource("CPU", 0, 5),
            "memory": Resource("Memory", 0, 5),
            "network": Resource("Network", 0, 3),
            "storage": Resource("Storage", 0, 2)
        }
        
        self.active = True
        
        # Start optimization loop
        asyncio.create_task(self._optimization_loop())
        
        logger.info("Neural Resource Manager online")
        
    async def _optimization_loop(self):
        """Main optimization loop"""
        while self.active:
            try:
                # Monitor current usage
                current_usage = self._get_current_usage()
                
                # Optimize allocations
                await self._optimize_resources(current_usage)
                
                # Process pending tasks
                await self._process_tasks()
                
                # Learn from patterns
                self._adapt_parameters()
                
                self.optimization_cycles += 1
                
                await asyncio.sleep(10)  # Optimize every 10 seconds
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                await asyncio.sleep(30)
                
    def _get_current_usage(self) -> Dict[str, float]:
        """Get current system resource usage"""
        return {
            "cpu": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory().percent,
            "network": random.uniform(10, 50),  # Simulated
            "storage": psutil.disk_usage('/').percent
        }
        
    async def _optimize_resources(self, current_usage: Dict[str, float]):
        """Optimize resource allocation using neural-inspired algorithm"""
        
        for resource_name, usage in current_usage.items():
            resource = self.resources[resource_name]
            
            # Neural-inspired allocation adjustment
            # Higher priority resources get more aggressive allocation
            target_allocation = min(100, usage * (1 + resource.priority * 0.1))
            
            # Smooth adjustment using learning rate
            adjustment = (target_allocation - resource.allocated) * self.learning_rate
            resource.allocated = max(0, min(100, resource.allocated + adjustment))
            
        # Update efficiency score
        total_usage = sum(current_usage.values()) / len(current_usage)
        total_allocated = sum(r.allocated for r in self.resources.values()) / len(self.resources)
        
        if total_allocated > 0:
            self.efficiency_score = min(1.5, total_usage / total_allocated)
        else:
            self.efficiency_score = 1.0
            
    async def _process_tasks(self):
        """Process pending tasks with intelligent scheduling"""
        
        # Sort tasks by priority
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        for task in pending_tasks:
            # Check if resources available
            cpu_available = 100 - self.resources["cpu"].allocated
            memory_available = 100 - self.resources["memory"].allocated
            
            if (task.required_cpu <= cpu_available and 
                task.required_memory <= memory_available):
                
                # Allocate resources
                task.status = "running"
                self.resources["cpu"].allocated += task.required_cpu
                self.resources["memory"].allocated += task.required_memory
                
                # Simulate task execution
                asyncio.create_task(self._execute_task(task))
                
    async def _execute_task(self, task: Task):
        """Execute a task"""
        logger.info(f"Executing task: {task.name}")
        
        # Simulate task execution
        await asyncio.sleep(task.required_time)
        
        # Free resources
        self.resources["cpu"].allocated -= task.required_cpu
        self.resources["memory"].allocated -= task.required_memory
        
        task.status = "completed"
        self.completed_tasks += 1
        
        logger.info(f"Task completed: {task.name}")
        
    def _adapt_parameters(self):
        """Adapt parameters based on performance"""
        
        # Adjust learning rate based on efficiency
        if self.efficiency_score > 1.2:
            self.learning_rate = min(0.3, self.learning_rate * 1.05)
        elif self.efficiency_score < 0.8:
            self.learning_rate = max(0.05, self.learning_rate * 0.95)
            
        # Store adaptation history
        self.adaptation_history.append({
            "timestamp": datetime.now(),
            "efficiency": self.efficiency_score,
            "learning_rate": self.learning_rate,
            "completed_tasks": self.completed_tasks
        })
        
        # Keep only last 100 entries
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
            
    async def allocate_for_task(
        self, 
        task_name: str, 
        cpu_needed: float = 10, 
        memory_needed: float = 10,
        priority: int = 5,
        duration: float = 5
    ) -> bool:
        """Request resources for a task"""
        
        task = Task(
            id=f"task_{len(self.tasks)}",
            name=task_name,
            priority=priority,
            required_cpu=cpu_needed,
            required_memory=memory_needed,
            required_time=duration
        )
        
        self.tasks.append(task)
        
        logger.info(f"Task queued: {task_name} (Priority: {priority})")
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """Get resource manager status"""
        return {
            "active": self.active,
            "efficiency_score": round(self.efficiency_score, 2),
            "completed_tasks": self.completed_tasks,
            "optimization_cycles": self.optimization_cycles,
            "learning_rate": round(self.learning_rate, 3),
            "resources": {
                name: {
                    "allocated": round(res.allocated, 1),
                    "priority": res.priority
                }
                for name, res in self.resources.items()
            },
            "pending_tasks": len([t for t in self.tasks if t.status == "pending"]),
            "running_tasks": len([t for t in self.tasks if t.status == "running"])
        }
        
    def get_efficiency_report(self) -> str:
        """Get efficiency report"""
        status = self.get_status()
        
        report = f"""
Neural Resource Manager Report
=============================
Status: {'Active' if status['active'] else 'Inactive'}
Efficiency Score: {status['efficiency_score']:.2f}x
Completed Tasks: {status['completed_tasks']}
Optimization Cycles: {status['optimization_cycles']}

Resource Allocation:
"""
        
        for name, info in status['resources'].items():
            report += f"  {name.upper()}: {info['allocated']:.1f}% (Priority: {info['priority']})\n"
            
        report += f"\nTask Queue: {status['pending_tasks']} pending, {status['running_tasks']} running"
        
        return report

# Singleton instance
neural_manager = NeuralResourceManager()
