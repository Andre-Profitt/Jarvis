#!/usr/bin/env python3
"""
JARVIS Workflow Automation System
Visual workflow builder and automation engine.
"""

import os
import json
import logging
import asyncio
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, time
from pathlib import Path
import yaml
import schedule
import threading
from enum import Enum
import inspect
import re

logger = logging.getLogger("jarvis.workflows")


class TriggerType(Enum):
    """Types of workflow triggers"""
    MANUAL = "manual"
    SCHEDULE = "schedule"
    EVENT = "event"
    CONDITION = "condition"
    VOICE_COMMAND = "voice_command"
    WEBHOOK = "webhook"
    FILE_CHANGE = "file_change"
    DEVICE_STATE = "device_state"


class ActionType(Enum):
    """Types of workflow actions"""
    JARVIS_COMMAND = "jarvis_command"
    SMART_HOME = "smart_home"
    NOTIFICATION = "notification"
    EMAIL = "email"
    HTTP_REQUEST = "http_request"
    SCRIPT = "script"
    WAIT = "wait"
    CONDITION = "condition"
    LOOP = "loop"
    VARIABLE = "variable"


@dataclass
class WorkflowTrigger:
    """Workflow trigger definition"""
    trigger_id: str
    trigger_type: TriggerType
    config: Dict[str, Any]
    enabled: bool = True
    
    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if trigger matches an event"""
        if not self.enabled:
            return False
            
        if self.trigger_type == TriggerType.EVENT:
            return event.get('type') == self.config.get('event_type')
            
        elif self.trigger_type == TriggerType.VOICE_COMMAND:
            pattern = self.config.get('pattern', '')
            command = event.get('command', '')
            return bool(re.match(pattern, command, re.IGNORECASE))
            
        elif self.trigger_type == TriggerType.DEVICE_STATE:
            device_id = self.config.get('device_id')
            state_key = self.config.get('state_key')
            expected_value = self.config.get('value')
            
            return (event.get('device_id') == device_id and
                   event.get('state', {}).get(state_key) == expected_value)
                   
        return False


@dataclass
class WorkflowAction:
    """Workflow action definition"""
    action_id: str
    action_type: ActionType
    config: Dict[str, Any]
    next_action: Optional[str] = None
    error_action: Optional[str] = None
    
    async def execute(self, context: Dict[str, Any], executor: 'WorkflowExecutor') -> Tuple[bool, Any]:
        """Execute the action"""
        try:
            if self.action_type == ActionType.JARVIS_COMMAND:
                return await executor.execute_jarvis_command(self.config, context)
                
            elif self.action_type == ActionType.SMART_HOME:
                return await executor.execute_smart_home_action(self.config, context)
                
            elif self.action_type == ActionType.NOTIFICATION:
                return await executor.execute_notification(self.config, context)
                
            elif self.action_type == ActionType.WAIT:
                duration = self.config.get('duration', 1)
                await asyncio.sleep(duration)
                return True, None
                
            elif self.action_type == ActionType.VARIABLE:
                return await executor.execute_variable_action(self.config, context)
                
            elif self.action_type == ActionType.CONDITION:
                return await executor.evaluate_condition(self.config, context)
                
            else:
                logger.warning(f"Unknown action type: {self.action_type}")
                return False, f"Unknown action type: {self.action_type}"
                
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            return False, str(e)


@dataclass
class Workflow:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    triggers: List[WorkflowTrigger]
    actions: Dict[str, WorkflowAction]
    entry_action: str
    variables: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    run_count: int = 0
    
    def get_action_sequence(self) -> List[str]:
        """Get the sequence of action IDs"""
        sequence = []
        current = self.entry_action
        
        while current and current not in sequence:
            sequence.append(current)
            action = self.actions.get(current)
            if action:
                current = action.next_action
            else:
                break
                
        return sequence


class WorkflowExecutor:
    """Executes workflow actions"""
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.running_workflows: Dict[str, asyncio.Task] = {}
        
    async def execute_workflow(self, workflow: Workflow, trigger_context: Dict[str, Any] = None) -> bool:
        """Execute a complete workflow"""
        logger.info(f"Executing workflow: {workflow.name}")
        
        # Initialize context
        context = {
            'workflow_id': workflow.workflow_id,
            'workflow_name': workflow.name,
            'trigger': trigger_context or {},
            'variables': workflow.variables.copy(),
            'start_time': datetime.now()
        }
        
        # Update workflow stats
        workflow.last_run = datetime.now()
        workflow.run_count += 1
        
        # Execute actions in sequence
        current_action_id = workflow.entry_action
        success = True
        
        while current_action_id:
            action = workflow.actions.get(current_action_id)
            if not action:
                logger.error(f"Action not found: {current_action_id}")
                break
                
            # Execute action
            action_success, result = await action.execute(context, self)
            
            # Update context
            context[f'action_{current_action_id}_result'] = result
            
            if action_success:
                # Handle conditional branching
                if action.action_type == ActionType.CONDITION and isinstance(result, str):
                    # Result is the next action ID for conditions
                    current_action_id = result
                else:
                    current_action_id = action.next_action
            else:
                # Handle error
                if action.error_action:
                    current_action_id = action.error_action
                else:
                    success = False
                    break
                    
        logger.info(f"Workflow {workflow.name} completed: {'success' if success else 'failed'}")
        return success
        
    async def execute_jarvis_command(self, config: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute a JARVIS command"""
        command = self.interpolate_variables(config.get('command', ''), context)
        
        if self.jarvis:
            result = await self.jarvis.process_voice_command(command)
            return result.get('success', False), result.get('response')
        else:
            logger.warning("JARVIS instance not available")
            return False, "JARVIS not available"
            
    async def execute_smart_home_action(self, config: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute smart home action"""
        device_id = config.get('device_id')
        action = config.get('action')
        params = config.get('params', {})
        
        # Interpolate variables in params
        for key, value in params.items():
            if isinstance(value, str):
                params[key] = self.interpolate_variables(value, context)
                
        if self.jarvis and hasattr(self.jarvis, 'smart_home'):
            try:
                result = await self.jarvis.smart_home.control_device(device_id, action, **params)
                return True, result
            except Exception as e:
                return False, str(e)
        else:
            return False, "Smart home not available"
            
    async def execute_notification(self, config: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Send notification"""
        title = self.interpolate_variables(config.get('title', 'JARVIS'), context)
        message = self.interpolate_variables(config.get('message', ''), context)
        
        # Send via JARVIS speech
        if self.jarvis:
            self.jarvis.speak(message)
            
        # Could also send push notifications, desktop notifications, etc.
        logger.info(f"Notification: {title} - {message}")
        return True, message
        
    async def execute_variable_action(self, config: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Set or modify variables"""
        operation = config.get('operation', 'set')
        var_name = config.get('name')
        value = config.get('value')
        
        if operation == 'set':
            context['variables'][var_name] = self.interpolate_variables(str(value), context)
            
        elif operation == 'increment':
            current = context['variables'].get(var_name, 0)
            context['variables'][var_name] = current + value
            
        elif operation == 'append':
            current = context['variables'].get(var_name, [])
            if not isinstance(current, list):
                current = [current]
            current.append(self.interpolate_variables(str(value), context))
            context['variables'][var_name] = current
            
        return True, context['variables'].get(var_name)
        
    async def evaluate_condition(self, config: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Evaluate condition and return next action"""
        condition_type = config.get('type', 'simple')
        
        if condition_type == 'simple':
            left = self.interpolate_variables(str(config.get('left', '')), context)
            operator = config.get('operator', '==')
            right = self.interpolate_variables(str(config.get('right', '')), context)
            
            # Evaluate condition
            result = False
            if operator == '==':
                result = left == right
            elif operator == '!=':
                result = left != right
            elif operator == '>':
                result = float(left) > float(right)
            elif operator == '<':
                result = float(left) < float(right)
            elif operator == 'contains':
                result = right in left
            elif operator == 'matches':
                result = bool(re.match(right, left))
                
            # Return next action based on result
            if result:
                return True, config.get('true_action')
            else:
                return True, config.get('false_action')
                
        return False, None
        
    def interpolate_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Replace variables in text with their values"""
        if not isinstance(text, str):
            return text
            
        # Replace {{variable}} with values from context
        pattern = r'\{\{(\w+)\}\}'
        
        def replacer(match):
            var_name = match.group(1)
            
            # Check in variables first
            if var_name in context.get('variables', {}):
                return str(context['variables'][var_name])
                
            # Check in context
            if var_name in context:
                return str(context[var_name])
                
            # Check nested paths (e.g., trigger.event_type)
            parts = var_name.split('.')
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return match.group(0)  # Return original if not found
                    
            return str(value)
            
        return re.sub(pattern, replacer, text)


class WorkflowEngine:
    """Main workflow automation engine"""
    
    def __init__(self, jarvis_instance=None):
        self.jarvis = jarvis_instance
        self.workflows: Dict[str, Workflow] = {}
        self.executor = WorkflowExecutor(jarvis_instance)
        self.scheduler_thread = None
        self.is_running = False
        
        # Storage
        self.workflows_dir = Path.home() / ".jarvis" / "workflows"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        
        # Event listeners
        self.event_listeners: Dict[str, List[str]] = {}  # event_type -> workflow_ids
        
        # Load workflows
        self.load_workflows()
        
    def create_workflow(self, name: str, description: str = "") -> Workflow:
        """Create a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            triggers=[],
            actions={},
            entry_action=""
        )
        
        self.workflows[workflow_id] = workflow
        self.save_workflow(workflow)
        
        logger.info(f"Created workflow: {name} ({workflow_id})")
        return workflow
        
    def add_trigger(self, workflow_id: str, trigger_type: TriggerType, config: Dict[str, Any]) -> Optional[WorkflowTrigger]:
        """Add trigger to workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
            
        trigger = WorkflowTrigger(
            trigger_id=str(uuid.uuid4()),
            trigger_type=trigger_type,
            config=config
        )
        
        workflow.triggers.append(trigger)
        
        # Register event listeners
        if trigger_type == TriggerType.EVENT:
            event_type = config.get('event_type')
            if event_type:
                if event_type not in self.event_listeners:
                    self.event_listeners[event_type] = []
                self.event_listeners[event_type].append(workflow_id)
                
        self.save_workflow(workflow)
        return trigger
        
    def add_action(self, workflow_id: str, action_type: ActionType, config: Dict[str, Any],
                   previous_action_id: Optional[str] = None) -> Optional[WorkflowAction]:
        """Add action to workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
            
        action = WorkflowAction(
            action_id=str(uuid.uuid4()),
            action_type=action_type,
            config=config
        )
        
        workflow.actions[action.action_id] = action
        
        # Link to previous action
        if previous_action_id and previous_action_id in workflow.actions:
            workflow.actions[previous_action_id].next_action = action.action_id
        elif not workflow.entry_action:
            workflow.entry_action = action.action_id
            
        self.save_workflow(workflow)
        return action
        
    def save_workflow(self, workflow: Workflow):
        """Save workflow to disk"""
        workflow_file = self.workflows_dir / f"{workflow.workflow_id}.json"
        
        # Convert to dict
        data = asdict(workflow)
        
        # Convert enums
        for trigger in data['triggers']:
            trigger['trigger_type'] = trigger['trigger_type'].value
            
        for action in data['actions'].values():
            action['action_type'] = action['action_type'].value
            
        with open(workflow_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
    def load_workflows(self):
        """Load all workflows from disk"""
        for workflow_file in self.workflows_dir.glob("*.json"):
            try:
                with open(workflow_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert enums back
                for trigger in data['triggers']:
                    trigger['trigger_type'] = TriggerType(trigger['trigger_type'])
                    
                for action in data['actions'].values():
                    action['action_type'] = ActionType(action['action_type'])
                    
                # Reconstruct objects
                triggers = [WorkflowTrigger(**t) for t in data['triggers']]
                actions = {aid: WorkflowAction(**a) for aid, a in data['actions'].items()}
                
                workflow = Workflow(
                    workflow_id=data['workflow_id'],
                    name=data['name'],
                    description=data['description'],
                    triggers=triggers,
                    actions=actions,
                    entry_action=data['entry_action'],
                    variables=data.get('variables', {}),
                    enabled=data.get('enabled', True),
                    created_at=datetime.fromisoformat(data['created_at']),
                    last_run=datetime.fromisoformat(data['last_run']) if data.get('last_run') else None,
                    run_count=data.get('run_count', 0)
                )
                
                self.workflows[workflow.workflow_id] = workflow
                
                # Register event listeners
                for trigger in workflow.triggers:
                    if trigger.trigger_type == TriggerType.EVENT:
                        event_type = trigger.config.get('event_type')
                        if event_type:
                            if event_type not in self.event_listeners:
                                self.event_listeners[event_type] = []
                            self.event_listeners[event_type].append(workflow.workflow_id)
                            
            except Exception as e:
                logger.error(f"Failed to load workflow {workflow_file}: {e}")
                
    def start(self):
        """Start the workflow engine"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Workflow engine started")
        
    def stop(self):
        """Stop the workflow engine"""
        self.is_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join()
            
        logger.info("Workflow engine stopped")
        
    def _scheduler_loop(self):
        """Main scheduler loop for time-based triggers"""
        while self.is_running:
            # Check scheduled workflows
            for workflow in self.workflows.values():
                if not workflow.enabled:
                    continue
                    
                for trigger in workflow.triggers:
                    if trigger.trigger_type == TriggerType.SCHEDULE and trigger.enabled:
                        # Check if it's time to run
                        schedule_config = trigger.config
                        
                        # Simple implementation - in production, use more robust scheduling
                        if self._should_run_scheduled(workflow, schedule_config):
                            asyncio.run(self.trigger_workflow(workflow.workflow_id, {
                                'trigger_type': 'schedule',
                                'schedule': schedule_config
                            }))
                            
            # Sleep for a bit
            asyncio.run(asyncio.sleep(30))  # Check every 30 seconds
            
    def _should_run_scheduled(self, workflow: Workflow, schedule_config: Dict[str, Any]) -> bool:
        """Check if scheduled workflow should run"""
        # Simple implementation - check if enough time has passed
        interval_minutes = schedule_config.get('interval_minutes')
        
        if interval_minutes and workflow.last_run:
            time_since_last = datetime.now() - workflow.last_run
            return time_since_last.total_seconds() >= interval_minutes * 60
            
        return False
        
    async def trigger_workflow(self, workflow_id: str, context: Dict[str, Any] = None) -> bool:
        """Manually trigger a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow or not workflow.enabled:
            return False
            
        # Create task for workflow execution
        task = asyncio.create_task(self.executor.execute_workflow(workflow, context))
        self.executor.running_workflows[workflow_id] = task
        
        # Clean up when done
        def cleanup(t):
            del self.executor.running_workflows[workflow_id]
            
        task.add_done_callback(cleanup)
        
        return True
        
    async def handle_event(self, event: Dict[str, Any]):
        """Handle an event that might trigger workflows"""
        event_type = event.get('type')
        
        # Get workflows listening for this event
        workflow_ids = self.event_listeners.get(event_type, [])
        
        for workflow_id in workflow_ids:
            workflow = self.workflows.get(workflow_id)
            if not workflow or not workflow.enabled:
                continue
                
            # Check if any trigger matches
            for trigger in workflow.triggers:
                if trigger.matches(event):
                    await self.trigger_workflow(workflow_id, {
                        'trigger': trigger.trigger_id,
                        'event': event
                    })
                    break


class WorkflowBuilder:
    """Visual workflow builder helper"""
    
    @staticmethod
    def create_morning_routine() -> Dict[str, Any]:
        """Create a morning routine workflow"""
        return {
            'name': 'Morning Routine',
            'description': 'Automated morning routine',
            'triggers': [{
                'type': TriggerType.SCHEDULE,
                'config': {
                    'time': '07:00',
                    'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
                }
            }],
            'actions': [
                {
                    'type': ActionType.SMART_HOME,
                    'config': {
                        'device_id': 'bedroom_lights',
                        'action': 'turn_on',
                        'params': {'brightness': 30}
                    }
                },
                {
                    'type': ActionType.WAIT,
                    'config': {'duration': 300}  # 5 minutes
                },
                {
                    'type': ActionType.SMART_HOME,
                    'config': {
                        'device_id': 'bedroom_lights',
                        'action': 'set_brightness',
                        'params': {'brightness': 70}
                    }
                },
                {
                    'type': ActionType.JARVIS_COMMAND,
                    'config': {'command': 'Good morning! Check my calendar and weather'}
                },
                {
                    'type': ActionType.SMART_HOME,
                    'config': {
                        'device_id': 'coffee_maker',
                        'action': 'turn_on'
                    }
                }
            ]
        }
        
    @staticmethod
    def create_leaving_home() -> Dict[str, Any]:
        """Create leaving home workflow"""
        return {
            'name': 'Leaving Home',
            'description': 'Actions when leaving home',
            'triggers': [{
                'type': TriggerType.VOICE_COMMAND,
                'config': {
                    'pattern': r'(leaving|going out|bye)'
                }
            }],
            'actions': [
                {
                    'type': ActionType.SMART_HOME,
                    'config': {
                        'device_id': 'all_lights',
                        'action': 'turn_off'
                    }
                },
                {
                    'type': ActionType.SMART_HOME,
                    'config': {
                        'device_id': 'thermostat',
                        'action': 'set_mode',
                        'params': {'mode': 'away'}
                    }
                },
                {
                    'type': ActionType.NOTIFICATION,
                    'config': {
                        'title': 'Home Secured',
                        'message': 'All lights off, thermostat set to away mode'
                    }
                }
            ]
        }


class WorkflowCommandProcessor:
    """Process workflow-related commands"""
    
    def __init__(self, workflow_engine: WorkflowEngine):
        self.engine = workflow_engine
        
    async def process_command(self, command: str) -> Tuple[bool, str]:
        """Process workflow commands"""
        command_lower = command.lower()
        
        # List workflows
        if "list workflows" in command_lower or "show workflows" in command_lower:
            workflows = list(self.engine.workflows.values())
            
            if not workflows:
                return True, "No workflows configured yet"
                
            response = f"I have {len(workflows)} workflows:\n"
            for wf in workflows[:5]:  # Show first 5
                status = "enabled" if wf.enabled else "disabled"
                response += f"• {wf.name} ({status})\n"
                
            return True, response
            
        # Create workflow
        if "create workflow" in command_lower:
            # Extract name if provided
            match = re.search(r'workflow\s+(?:named?|called?)\s+(.+)', command_lower)
            if match:
                name = match.group(1).strip()
                workflow = self.engine.create_workflow(name)
                return True, f"Created workflow '{name}'. Add triggers and actions to complete it."
            else:
                return True, "What would you like to name the workflow?"
                
        # Run workflow
        if "run workflow" in command_lower or "execute workflow" in command_lower:
            # Extract workflow name
            words = command_lower.split()
            idx = words.index("workflow") + 1
            if idx < len(words):
                name = ' '.join(words[idx:])
                
                # Find workflow by name
                for workflow in self.engine.workflows.values():
                    if workflow.name.lower() == name:
                        await self.engine.trigger_workflow(workflow.workflow_id)
                        return True, f"Running workflow: {workflow.name}"
                        
                return True, f"I couldn't find a workflow named '{name}'"
                
        # Enable/disable workflow
        if "enable workflow" in command_lower or "disable workflow" in command_lower:
            enable = "enable" in command_lower
            
            # Extract workflow name
            pattern = r'(?:enable|disable)\s+workflow\s+(.+)'
            match = re.search(pattern, command_lower)
            
            if match:
                name = match.group(1).strip()
                
                for workflow in self.engine.workflows.values():
                    if workflow.name.lower() == name:
                        workflow.enabled = enable
                        self.engine.save_workflow(workflow)
                        status = "enabled" if enable else "disabled"
                        return True, f"Workflow '{workflow.name}' is now {status}"
                        
        # Create common workflows
        if "morning routine" in command_lower:
            template = WorkflowBuilder.create_morning_routine()
            workflow = self.engine.create_workflow(
                template['name'],
                template['description']
            )
            
            # Add triggers and actions from template
            # (simplified for example)
            
            return True, "Created morning routine workflow. Customize the schedule and actions as needed."
            
        return False, ""


def integrate_workflow_automation_with_jarvis(jarvis_instance) -> WorkflowEngine:
    """Integrate workflow automation with JARVIS"""
    
    # Create workflow engine
    engine = WorkflowEngine(jarvis_instance)
    
    # Create command processor
    processor = WorkflowCommandProcessor(engine)
    
    # Start engine
    engine.start()
    
    # Add to JARVIS
    if hasattr(jarvis_instance, 'workflow_engine'):
        jarvis_instance.workflow_engine = engine
        jarvis_instance.workflow_processor = processor
        
        # Register event handler
        if hasattr(jarvis_instance, 'event_system'):
            jarvis_instance.event_system.subscribe('*', engine.handle_event)
            
        logger.info("Workflow automation integrated with JARVIS")
        
    return engine