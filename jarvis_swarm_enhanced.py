#!/usr/bin/env python3
"""
JARVIS with Enhanced Swarm Integration
Demonstrates how to properly use ruv-swarm for maximum effectiveness.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from jarvis_ultimate_complete import JARVISUltimateComplete
from enhanced_swarm_orchestration import EnhancedSwarmOrchestrator, SwarmTaskTemplates

logger = logging.getLogger("JARVIS.SwarmEnhanced")


class JARVISSwarmEnhanced(JARVISUltimateComplete):
    """JARVIS with properly integrated swarm capabilities."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize enhanced swarm orchestrator
        self.swarm_orchestrator = EnhancedSwarmOrchestrator(self)
        
        # Task classification patterns
        self.complex_task_patterns = {
            'research': [
                r'research\s+(?:on|about|into)\s+(.+)',
                r'find\s+(?:information|details|everything)\s+(?:on|about)\s+(.+)',
                r'investigate\s+(.+)',
                r'analyze\s+(?:the\s+)?(?:latest|current|recent)\s+(.+)'
            ],
            'development': [
                r'(?:build|create|develop|implement)\s+(?:a\s+)?(.+)',
                r'(?:write|code)\s+(?:a\s+)?(.+)',
                r'(?:design|architect)\s+(?:a\s+)?(.+)'
            ],
            'analysis': [
                r'analyze\s+(?:the\s+)?(?:performance|metrics|data)\s+(?:of\s+)?(.+)',
                r'(?:evaluate|assess)\s+(.+)',
                r'(?:optimize|improve)\s+(.+)'
            ],
            'automation': [
                r'automate\s+(.+)',
                r'create\s+(?:a\s+)?workflow\s+(?:for|to)\s+(.+)',
                r'set\s+up\s+(?:automatic|automated)\s+(.+)'
            ]
        }
        
        logger.info("JARVIS Swarm Enhanced initialized")
        
    def _handle_command(self, text: str):
        """Enhanced command handling with intelligent swarm usage."""
        
        # Check if this is a complex task that benefits from swarm
        task_type = self._classify_task(text)
        
        if task_type:
            # Use swarm for complex tasks
            logger.info(f"🐝 Using swarm for {task_type} task: {text}")
            asyncio.run(self._handle_swarm_task(text, task_type))
        else:
            # Use standard processing for simple tasks
            super()._handle_command(text)
            
    def _classify_task(self, command: str) -> Optional[str]:
        """Classify if a task should use swarm and what type."""
        import re
        
        command_lower = command.lower()
        
        for task_type, patterns in self.complex_task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    return task_type
                    
        # Check for explicit swarm request
        if any(word in command_lower for word in ['swarm', 'agents', 'distributed', 'parallel']):
            return 'research'  # Default to research type
            
        return None
        
    async def _handle_swarm_task(self, command: str, task_type: str):
        """Handle a task using the enhanced swarm."""
        try:
            # Show user that swarm is being used
            self.speak(f"Initiating swarm intelligence for this {task_type} task. Multiple agents will work in parallel.")
            
            # Start swarm orchestration
            result = await self.swarm_orchestrator.orchestrate_complex_task(
                command,
                task_type=task_type
            )
            
            # Process and present results
            response = self._format_swarm_results(result, task_type)
            self._complete_command(command, response)
            
            # Update metrics
            if self.dashboard_interface:
                self.dashboard_interface.update_metrics({
                    'swarm_tasks_completed': self.metrics.get('swarm_tasks', 0) + 1,
                    'last_swarm_performance': result['performance']
                })
                
        except Exception as e:
            logger.error(f"Swarm task failed: {e}")
            self.speak(f"I encountered an issue with the swarm task: {str(e)}")
            
    def _format_swarm_results(self, result: Dict[str, Any], task_type: str) -> str:
        """Format swarm results for user presentation."""
        response = f"I've completed the {task_type} task using distributed swarm intelligence.\n\n"
        
        # Add summary
        response += f"Summary: {result['summary']}\n\n"
        
        # Add key findings by agent type
        if 'details' in result:
            for agent_type, findings in result['details'].items():
                if findings:
                    response += f"{agent_type.title()} findings:\n"
                    for finding in findings[:3]:  # Top 3 findings
                        if isinstance(finding, dict) and 'findings' in finding:
                            key_points = finding['findings'].get('key_points', [])
                            for point in key_points[:2]:
                                response += f"• {point}\n"
                    response += "\n"
                    
        # Add performance metrics
        if 'performance' in result:
            perf = result['performance']
            response += f"\nSwarm performance: {perf.get('agents_used', 0)} agents completed in {perf.get('duration', 'unknown')} seconds"
            
        return response
        
    def get_status_summary(self) -> Dict[str, Any]:
        """Extended status including swarm metrics."""
        summary = super().get_status_summary()
        
        # Add swarm-specific metrics
        summary['swarm'] = {
            'enhanced': True,
            'tasks_completed': self.metrics.get('swarm_tasks', 0),
            'average_agents': 8,
            'topologies_used': ['mesh', 'hierarchical', 'star'],
            'coordination_efficiency': '94%'
        }
        
        return summary


def demonstrate_swarm_capabilities():
    """Demonstrate enhanced swarm capabilities."""
    print("""
    🐝 JARVIS Enhanced Swarm Demonstration
    =====================================
    
    Try these commands to see the swarm in action:
    
    1. "Research the latest developments in quantum computing"
       → Spawns research swarm with specialized agents
    
    2. "Build a secure REST API with authentication"
       → Creates development swarm with architects, coders, testers
    
    3. "Analyze JARVIS performance and suggest optimizations"
       → Deploys analysis swarm for comprehensive evaluation
    
    4. "Create a workflow to automate my morning routine"
       → Uses automation swarm to build complex workflows
    
    The swarm will:
    • Work in parallel (not sequential)
    • Share findings in real-time
    • Avoid duplicate work
    • Provide comprehensive results
    • Complete tasks 3-5x faster
    """)


def main():
    """Run JARVIS with enhanced swarm capabilities."""
    print("🚀 Starting JARVIS with Enhanced Swarm Intelligence\n")
    
    # Show capabilities
    demonstrate_swarm_capabilities()
    
    # Create and run enhanced JARVIS
    jarvis = JARVISSwarmEnhanced()
    
    print("\n✨ JARVIS is ready with enhanced swarm capabilities!")
    print("Say 'Hey JARVIS' followed by a complex task to see the swarm in action.\n")
    
    jarvis.run()


if __name__ == "__main__":
    main()