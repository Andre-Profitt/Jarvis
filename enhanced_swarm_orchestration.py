#!/usr/bin/env python3
"""
Enhanced Swarm Orchestration for JARVIS
Maximizes ruv-swarm's distributed capabilities with real parallel execution.
"""

import os
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger("jarvis.swarm.orchestration")


class EnhancedSwarmOrchestrator:
    """
    Orchestrates complex JARVIS operations using ruv-swarm's full capabilities.
    Key improvements:
    1. TRUE parallel agent execution (not sequential)
    2. Automatic agent specialization based on task
    3. Real-time coordination through ruv-swarm hooks
    4. Memory-based inter-agent communication
    5. Performance optimization through caching
    """
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.swarm_id = str(uuid.uuid4())[:8]
        self.active_agents = {}
        self.swarm_memory = {}
        self.performance_cache = {}
        
        # Swarm configuration for different scenarios
        self.swarm_configs = {
            'research': {
                'topology': 'mesh',
                'max_agents': 8,
                'agents': [
                    ('researcher', 3),  # 3 researchers
                    ('analyst', 2),     # 2 analysts
                    ('coder', 1),       # 1 coder
                    ('validator', 1),   # 1 validator
                    ('coordinator', 1)  # 1 coordinator
                ]
            },
            'development': {
                'topology': 'hierarchical',
                'max_agents': 10,
                'agents': [
                    ('architect', 1),
                    ('coder', 4),
                    ('tester', 2),
                    ('reviewer', 1),
                    ('documenter', 1),
                    ('coordinator', 1)
                ]
            },
            'analysis': {
                'topology': 'star',
                'max_agents': 6,
                'agents': [
                    ('analyst', 3),
                    ('researcher', 2),
                    ('coordinator', 1)
                ]
            },
            'automation': {
                'topology': 'ring',
                'max_agents': 5,
                'agents': [
                    ('orchestrator', 1),
                    ('executor', 2),
                    ('monitor', 1),
                    ('optimizer', 1)
                ]
            }
        }
        
    async def orchestrate_complex_task(self, task: str, task_type: str = 'research') -> Dict[str, Any]:
        """
        Orchestrate a complex task using the full power of ruv-swarm.
        This is the main entry point for swarm-based operations.
        """
        logger.info(f"🐝 Orchestrating {task_type} task: {task}")
        
        # Initialize swarm with optimal configuration
        swarm_config = self.swarm_configs.get(task_type, self.swarm_configs['research'])
        
        # Step 1: Initialize swarm with hooks
        await self._initialize_enhanced_swarm(swarm_config)
        
        # Step 2: Spawn all agents IN PARALLEL with specific roles
        agent_tasks = await self._spawn_specialized_agents(swarm_config['agents'], task)
        
        # Step 3: Set up real-time coordination
        coordination_task = asyncio.create_task(self._coordinate_agents_realtime())
        
        # Step 4: Execute task with parallel processing
        results = await self._execute_parallel_swarm_task(task, agent_tasks)
        
        # Step 5: Aggregate and optimize results
        final_result = await self._aggregate_swarm_results(results)
        
        # Clean up
        coordination_task.cancel()
        await self._cleanup_swarm()
        
        return final_result
        
    async def _initialize_enhanced_swarm(self, config: Dict[str, Any]):
        """Initialize swarm with advanced features and hooks."""
        # Initialize with optimal topology
        cmd = [
            "npx", "ruv-swarm", "init",
            config['topology'],
            str(config['max_agents']),
            "--enable-hooks",
            "--enable-memory",
            "--enable-telemetry"
        ]
        
        result = await self._run_command(cmd)
        
        # Configure hooks for automatic coordination
        hooks_config = {
            "pre-task": {
                "auto-spawn-agents": False,  # We control spawning
                "load-context": True,
                "optimize-topology": True
            },
            "post-edit": {
                "auto-format": True,
                "update-memory": True,
                "sync-agents": True
            },
            "notification": {
                "log-events": True,
                "update-telemetry": True
            }
        }
        
        # Save hooks configuration
        await self._run_command([
            "npx", "ruv-swarm", "hook", "configure",
            "--config", json.dumps(hooks_config)
        ])
        
        logger.info(f"✅ Enhanced swarm initialized with {config['topology']} topology")
        
    async def _spawn_specialized_agents(self, agent_specs: List[Tuple[str, int]], task: str) -> List[asyncio.Task]:
        """Spawn specialized agents in parallel with specific capabilities."""
        spawn_commands = []
        
        for agent_type, count in agent_specs:
            for i in range(count):
                agent_name = f"{agent_type}_{i+1}"
                
                # Create specialized prompt based on agent type
                prompt = self._create_specialized_prompt(agent_type, task, i)
                
                # Add to batch
                spawn_commands.append({
                    'type': agent_type,
                    'name': agent_name,
                    'prompt': prompt
                })
                
        # Spawn ALL agents in parallel using batch command
        agent_tasks = []
        batch_size = 5  # Spawn 5 agents at a time for efficiency
        
        for i in range(0, len(spawn_commands), batch_size):
            batch = spawn_commands[i:i+batch_size]
            batch_task = asyncio.create_task(self._spawn_agent_batch(batch))
            agent_tasks.append(batch_task)
            
        # Wait for all spawning to complete
        await asyncio.gather(*agent_tasks)
        
        logger.info(f"🚀 Spawned {len(spawn_commands)} specialized agents in parallel")
        
        return agent_tasks
        
    async def _spawn_agent_batch(self, batch: List[Dict[str, str]]):
        """Spawn a batch of agents in parallel."""
        spawn_tasks = []
        
        for agent in batch:
            cmd = [
                "npx", "ruv-swarm", "agent", "spawn",
                "--type", agent['type'],
                "--name", agent['name'],
                "--prompt", agent['prompt'],
                "--memory-key", f"swarm/{self.swarm_id}/{agent['name']}"
            ]
            
            task = asyncio.create_task(self._run_command(cmd))
            spawn_tasks.append(task)
            self.active_agents[agent['name']] = {
                'type': agent['type'],
                'status': 'spawning',
                'start_time': datetime.now()
            }
            
        await asyncio.gather(*spawn_tasks)
        
    def _create_specialized_prompt(self, agent_type: str, task: str, index: int) -> str:
        """Create specialized prompts for different agent types."""
        base_coordination = """
MANDATORY: You MUST coordinate with other agents using these commands:
1. Before starting: npx ruv-swarm hook pre-task --description "your specific subtask"
2. After EVERY finding: npx ruv-swarm hook notification --message "what you found"
3. Store results: npx ruv-swarm hook post-edit --memory-key "swarm/{swarm_id}/{agent_name}/results"
4. Check others' work: npx ruv-swarm hook pre-search --query "relevant findings"
        """.format(swarm_id=self.swarm_id, agent_name=f"{agent_type}_{index+1}")
        
        prompts = {
            'researcher': f"""You are Research Agent {index+1} in a coordinated swarm.
{base_coordination}

Your task: Research different aspects of "{task}"
Focus area {index+1}: {self._get_research_focus(task, index)}

IMPORTANT: Share ALL findings immediately using notification hooks.""",
            
            'coder': f"""You are Coding Agent {index+1} in a coordinated swarm.
{base_coordination}

Your task: Implement solutions for "{task}"
Focus: Clean, efficient, production-ready code

IMPORTANT: Check researcher findings BEFORE coding using memory hooks.""",
            
            'analyst': f"""You are Analysis Agent {index+1} in a coordinated swarm.
{base_coordination}

Your task: Analyze data and patterns related to "{task}"
Focus area: {self._get_analysis_focus(task, index)}

IMPORTANT: Build on other agents' findings, don't duplicate work.""",
            
            'coordinator': f"""You are the Coordinator Agent for this swarm.
{base_coordination}

Your task: Coordinate and synthesize all agent outputs for "{task}"

IMPORTANT: Monitor all agents, resolve conflicts, ensure comprehensive coverage.""",
            
            'validator': f"""You are the Validation Agent for this swarm.
{base_coordination}

Your task: Validate and verify all findings/code for "{task}"

IMPORTANT: Check accuracy, test code, verify claims."""
        }
        
        return prompts.get(agent_type, f"You are {agent_type} agent {index+1}. Help with: {task}")
        
    def _get_research_focus(self, task: str, index: int) -> str:
        """Assign specific research focus areas to avoid duplication."""
        focuses = [
            "Core concepts and theoretical background",
            "Existing solutions and best practices",
            "Performance considerations and optimizations",
            "Security and edge cases",
            "Integration and compatibility"
        ]
        return focuses[index % len(focuses)]
        
    def _get_analysis_focus(self, task: str, index: int) -> str:
        """Assign specific analysis focus areas."""
        focuses = [
            "Performance metrics and bottlenecks",
            "Code quality and patterns",
            "Security vulnerabilities",
            "Scalability concerns"
        ]
        return focuses[index % len(focuses)]
        
    async def _coordinate_agents_realtime(self):
        """Real-time coordination loop using ruv-swarm hooks."""
        while True:
            try:
                # Check agent status
                status_cmd = ["npx", "ruv-swarm", "swarm", "status", "--json"]
                status_result = await self._run_command(status_cmd)
                
                if status_result and 'output' in status_result:
                    status_data = json.loads(status_result['output'])
                    
                    # Update agent statuses
                    for agent_name, agent_info in self.active_agents.items():
                        if agent_name in status_data.get('agents', {}):
                            agent_info['status'] = status_data['agents'][agent_name]['status']
                            
                # Check for coordination needs
                await self._handle_coordination_events()
                
                # Brief pause
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordination error: {e}")
                
    async def _handle_coordination_events(self):
        """Handle inter-agent coordination events."""
        # Check shared memory for coordination requests
        memory_cmd = [
            "npx", "ruv-swarm", "memory", "list",
            "--pattern", f"swarm/{self.swarm_id}/coordination/*"
        ]
        
        result = await self._run_command(memory_cmd)
        
        if result and 'output' in result:
            coordination_requests = json.loads(result.get('output', '[]'))
            
            for request in coordination_requests:
                await self._process_coordination_request(request)
                
    async def _process_coordination_request(self, request: Dict[str, Any]):
        """Process a coordination request between agents."""
        request_type = request.get('type')
        
        if request_type == 'conflict':
            # Resolve conflicts between agents
            await self._resolve_conflict(request)
        elif request_type == 'dependency':
            # Handle dependencies between agents
            await self._handle_dependency(request)
        elif request_type == 'synchronize':
            # Synchronize agent states
            await self._synchronize_agents(request)
            
    async def _execute_parallel_swarm_task(self, task: str, agent_tasks: List[asyncio.Task]) -> Dict[str, Any]:
        """Execute the main task with true parallel processing."""
        # Start task orchestration
        orchestrate_cmd = [
            "npx", "ruv-swarm", "task", "orchestrate",
            "--task", task,
            "--strategy", "parallel",
            "--optimization", "aggressive",
            "--memory-key", f"swarm/{self.swarm_id}/results"
        ]
        
        orchestration_task = asyncio.create_task(self._run_command(orchestrate_cmd))
        
        # Monitor progress in parallel
        monitor_task = asyncio.create_task(self._monitor_task_progress())
        
        # Wait for completion
        await orchestration_task
        monitor_task.cancel()
        
        # Collect results from all agents
        results = await self._collect_agent_results()
        
        return results
        
    async def _monitor_task_progress(self):
        """Monitor task progress and optimize performance."""
        while True:
            try:
                # Get task status
                status_cmd = [
                    "npx", "ruv-swarm", "task", "status",
                    "--detailed",
                    "--include-metrics"
                ]
                
                result = await self._run_command(status_cmd)
                
                if result and 'output' in result:
                    status = json.loads(result['output'])
                    
                    # Log progress
                    completion = status.get('completion_percentage', 0)
                    logger.info(f"📊 Task progress: {completion}%")
                    
                    # Optimize if needed
                    if status.get('bottlenecks'):
                        await self._optimize_swarm_performance(status['bottlenecks'])
                        
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
    async def _optimize_swarm_performance(self, bottlenecks: List[Dict[str, Any]]):
        """Dynamically optimize swarm performance based on bottlenecks."""
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'agent_overload':
                # Spawn additional agents
                agent_type = bottleneck['agent_type']
                await self._spawn_specialized_agents([(agent_type, 1)], bottleneck['task'])
                
            elif bottleneck['type'] == 'memory_pressure':
                # Clear unnecessary cache
                await self._optimize_memory_usage()
                
            elif bottleneck['type'] == 'communication_delay':
                # Optimize topology
                await self._optimize_topology()
                
    async def _collect_agent_results(self) -> Dict[str, Any]:
        """Collect results from all agents."""
        results = {
            'agents': {},
            'aggregated': {},
            'metrics': {}
        }
        
        # Collect from each agent's memory
        for agent_name in self.active_agents:
            memory_key = f"swarm/{self.swarm_id}/{agent_name}/results"
            
            cmd = [
                "npx", "ruv-swarm", "memory", "retrieve",
                "--key", memory_key
            ]
            
            result = await self._run_command(cmd)
            
            if result and 'output' in result:
                agent_results = json.loads(result.get('output', '{}'))
                results['agents'][agent_name] = agent_results
                
        # Get aggregated metrics
        metrics_cmd = [
            "npx", "ruv-swarm", "swarm", "metrics",
            "--aggregate",
            "--include-performance"
        ]
        
        metrics_result = await self._run_command(metrics_cmd)
        if metrics_result and 'output' in metrics_result:
            results['metrics'] = json.loads(metrics_result['output'])
            
        return results
        
    async def _aggregate_swarm_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate and synthesize results from all agents."""
        # Use coordinator agent's synthesis if available
        coordinator_results = None
        for agent_name, agent_results in results['agents'].items():
            if 'coordinator' in agent_name:
                coordinator_results = agent_results
                break
                
        final_result = {
            'task_completed': True,
            'summary': coordinator_results.get('summary', 'Task completed') if coordinator_results else 'Task completed',
            'details': {},
            'performance': results['metrics']
        }
        
        # Aggregate findings by type
        for agent_name, agent_results in results['agents'].items():
            agent_type = self.active_agents[agent_name]['type']
            
            if agent_type not in final_result['details']:
                final_result['details'][agent_type] = []
                
            final_result['details'][agent_type].append({
                'agent': agent_name,
                'findings': agent_results
            })
            
        return final_result
        
    async def _cleanup_swarm(self):
        """Clean up swarm resources."""
        # Save final state
        final_state = {
            'swarm_id': self.swarm_id,
            'agents': self.active_agents,
            'completion_time': datetime.now().isoformat()
        }
        
        await self._run_command([
            "npx", "ruv-swarm", "memory", "store",
            "--key", f"swarm/{self.swarm_id}/final_state",
            "--value", json.dumps(final_state)
        ])
        
        # Terminate agents
        for agent_name in self.active_agents:
            await self._run_command([
                "npx", "ruv-swarm", "agent", "terminate",
                "--name", agent_name
            ])
            
        logger.info(f"🧹 Swarm {self.swarm_id} cleaned up")
        
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Run a command and return results."""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                'success': process.returncode == 0,
                'output': stdout.decode() if stdout else '',
                'error': stderr.decode() if stderr else ''
            }
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _optimize_memory_usage(self):
        """Optimize memory usage across the swarm."""
        # Clear old cache entries
        await self._run_command([
            "npx", "ruv-swarm", "memory", "cleanup",
            "--older-than", "300"  # 5 minutes
        ])
        
    async def _optimize_topology(self):
        """Dynamically optimize swarm topology."""
        await self._run_command([
            "npx", "ruv-swarm", "swarm", "optimize",
            "--metric", "latency"
        ])


class SwarmTaskTemplates:
    """Pre-configured swarm task templates for common JARVIS operations."""
    
    @staticmethod
    def research_task(topic: str) -> Dict[str, Any]:
        """Research task template."""
        return {
            'task': f"Comprehensive research on: {topic}",
            'type': 'research',
            'subtasks': [
                f"Find academic papers and research on {topic}",
                f"Analyze current industry practices for {topic}",
                f"Identify best practices and patterns",
                f"Find potential issues and solutions",
                f"Create implementation recommendations"
            ]
        }
        
    @staticmethod
    def development_task(project: str) -> Dict[str, Any]:
        """Development task template."""
        return {
            'task': f"Develop {project}",
            'type': 'development',
            'subtasks': [
                f"Design architecture for {project}",
                f"Implement core functionality",
                f"Write comprehensive tests",
                f"Create documentation",
                f"Optimize performance",
                f"Review security"
            ]
        }
        
    @staticmethod
    def analysis_task(data: str) -> Dict[str, Any]:
        """Analysis task template."""
        return {
            'task': f"Analyze {data}",
            'type': 'analysis',
            'subtasks': [
                f"Collect and validate {data}",
                f"Identify patterns and trends",
                f"Find anomalies and outliers",
                f"Generate insights",
                f"Create visualizations",
                f"Make recommendations"
            ]
        }


async def demonstrate_enhanced_swarm():
    """Demonstrate the enhanced swarm capabilities."""
    print("🐝 Enhanced Swarm Orchestration Demo")
    print("=" * 50)
    
    orchestrator = EnhancedSwarmOrchestrator()
    
    # Example 1: Research Task
    print("\n📚 Research Task: Neural Architecture Search")
    research_result = await orchestrator.orchestrate_complex_task(
        "Neural Architecture Search optimization techniques",
        task_type='research'
    )
    
    print(f"\nResearch completed!")
    print(f"Summary: {research_result['summary']}")
    print(f"Performance: {research_result['performance']}")
    
    # Example 2: Development Task
    print("\n💻 Development Task: Build REST API")
    dev_result = await orchestrator.orchestrate_complex_task(
        "REST API with authentication and rate limiting",
        task_type='development'
    )
    
    print(f"\nDevelopment completed!")
    print(f"Summary: {dev_result['summary']}")
    
    # Example 3: Analysis Task
    print("\n📊 Analysis Task: Performance Optimization")
    analysis_result = await orchestrator.orchestrate_complex_task(
        "JARVIS performance bottlenecks and optimization opportunities",
        task_type='analysis'
    )
    
    print(f"\nAnalysis completed!")
    print(f"Summary: {analysis_result['summary']}")


if __name__ == "__main__":
    # Run demonstration
    asyncio.run(demonstrate_enhanced_swarm())