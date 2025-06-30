"""
Scheduler Tool for JARVIS
========================

Provides advanced task scheduling capabilities with cron-like functionality,
recurring tasks, and intelligent scheduling based on system load.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
import croniter
import heapq
from enum import Enum
import pickle
import aiofiles
from pathlib import Path

from .base import BaseTool, ToolMetadata, ToolCategory, ToolResult

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of scheduled tasks"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class RecurrenceType(Enum):
    """Types of task recurrence"""

    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CRON = "cron"
    INTERVAL = "interval"


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""

    id: str
    name: str
    function: Union[Callable, str]  # Function or function name
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    schedule_time: datetime = field(default_factory=datetime.now)
    recurrence_type: RecurrenceType = RecurrenceType.ONCE
    recurrence_pattern: Optional[str] = None  # Cron expression or interval
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: Optional[int] = None
    priority: int = 5  # 1-10, higher is more important
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    enabled: bool = True


class SchedulerTool(BaseTool):
    """
    Advanced task scheduling tool with cron-like capabilities

    Features:
    - Cron expression support
    - Recurring tasks (daily, weekly, monthly)
    - Task priorities and dependencies
    - Retry mechanisms
    - Task persistence
    - Load-based scheduling
    - Task history and analytics
    """

    def __init__(self):
        metadata = ToolMetadata(
            name="scheduler",
            description="Advanced task scheduling with cron-like functionality",
            category=ToolCategory.SYSTEM,
            version="2.0.0",
            tags=["scheduler", "cron", "tasks", "automation", "recurring"],
            required_permissions=["task_execution", "system_time"],
            rate_limit=100,
            timeout=300,
            examples=[
                {
                    "description": "Schedule a one-time task",
                    "params": {
                        "action": "schedule",
                        "task_name": "backup_database",
                        "function": "backup_function",
                        "schedule_time": "2024-01-01T10:00:00",
                        "args": ["production_db"],
                    },
                },
                {
                    "description": "Create a recurring task with cron",
                    "params": {
                        "action": "schedule",
                        "task_name": "daily_report",
                        "function": "generate_report",
                        "recurrence_type": "cron",
                        "cron_expression": "0 9 * * *",
                    },
                },
            ],
        )
        super().__init__(metadata)

        # Task storage
        self.tasks: Dict[str, ScheduledTask] = {}
        self.task_queue: List[tuple] = []  # Min heap of (next_run_time, task_id)
        self.task_history: List[Dict[str, Any]] = []

        # Function registry for serializable tasks
        self.function_registry: Dict[str, Callable] = {}

        # Scheduler state
        self.scheduler_running = False
        self.scheduler_task = None
        self.max_concurrent_tasks = 10
        self.current_tasks = 0

        # Persistence
        self.storage_path = Path("./storage/scheduler")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load persisted tasks
        asyncio.create_task(self._load_tasks())

    async def _execute(self, **kwargs) -> Any:
        """Execute scheduler operations"""
        action = kwargs.get("action", "").lower()

        if action == "schedule":
            return await self._schedule_task(**kwargs)
        elif action == "cancel":
            return await self._cancel_task(kwargs.get("task_id"))
        elif action == "list":
            return await self._list_tasks(
                status=kwargs.get("status"), tags=kwargs.get("tags")
            )
        elif action == "get":
            return await self._get_task(kwargs.get("task_id"))
        elif action == "pause":
            return await self._pause_task(kwargs.get("task_id"))
        elif action == "resume":
            return await self._resume_task(kwargs.get("task_id"))
        elif action == "update":
            return await self._update_task(kwargs.get("task_id"), kwargs)
        elif action == "history":
            return await self._get_history(
                task_id=kwargs.get("task_id"), limit=kwargs.get("limit", 100)
            )
        elif action == "stats":
            return await self._get_statistics()
        elif action == "start_scheduler":
            return await self._start_scheduler()
        elif action == "stop_scheduler":
            return await self._stop_scheduler()
        else:
            raise ValueError(f"Unknown action: {action}")

    def validate_inputs(self, **kwargs) -> tuple[bool, Optional[str]]:
        """Validate scheduler inputs"""
        action = kwargs.get("action")

        if not action:
            return False, "Action is required"

        if action == "schedule":
            if not kwargs.get("task_name"):
                return False, "task_name is required for scheduling"
            if not kwargs.get("function"):
                return False, "function is required for scheduling"

        elif action in ["cancel", "get", "pause", "resume", "update"]:
            if not kwargs.get("task_id"):
                return False, f"task_id is required for {action}"

        return True, None

    async def _schedule_task(self, **kwargs) -> Dict[str, Any]:
        """Schedule a new task"""
        import uuid

        # Create task ID
        task_id = kwargs.get("task_id", str(uuid.uuid4()))

        # Parse schedule time
        schedule_time = kwargs.get("schedule_time", datetime.now())
        if isinstance(schedule_time, str):
            schedule_time = datetime.fromisoformat(schedule_time)

        # Parse recurrence
        recurrence_type = RecurrenceType(kwargs.get("recurrence_type", "once").lower())

        # Create task
        task = ScheduledTask(
            id=task_id,
            name=kwargs.get("task_name"),
            function=kwargs.get("function"),
            args=kwargs.get("args", []),
            kwargs=kwargs.get("kwargs", {}),
            schedule_time=schedule_time,
            recurrence_type=recurrence_type,
            recurrence_pattern=kwargs.get("cron_expression") or kwargs.get("interval"),
            max_retries=kwargs.get("max_retries", 3),
            retry_delay=kwargs.get("retry_delay", 60),
            timeout=kwargs.get("timeout"),
            priority=kwargs.get("priority", 5),
            tags=kwargs.get("tags", []),
            metadata=kwargs.get("metadata", {}),
        )

        # Calculate next run time
        task.next_run = self._calculate_next_run(task)

        # Store task
        self.tasks[task_id] = task

        # Add to queue
        if task.next_run and task.enabled:
            heapq.heappush(self.task_queue, (task.next_run, task_id))

        # Persist
        await self._save_task(task)

        # Start scheduler if not running
        if not self.scheduler_running:
            await self._start_scheduler()

        return {
            "task_id": task_id,
            "name": task.name,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "status": task.status.value,
        }

    async def _cancel_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel a scheduled task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.status = TaskStatus.CANCELLED
        task.enabled = False

        # Remove from queue
        self.task_queue = [(t, tid) for t, tid in self.task_queue if tid != task_id]
        heapq.heapify(self.task_queue)

        # Persist
        await self._save_task(task)

        return {"task_id": task_id, "status": "cancelled", "name": task.name}

    async def _list_tasks(
        self, status: Optional[str] = None, tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List scheduled tasks with optional filtering"""
        tasks = []

        for task in self.tasks.values():
            # Filter by status
            if status and task.status.value != status:
                continue

            # Filter by tags
            if tags and not any(tag in task.tags for tag in tags):
                continue

            tasks.append(
                {
                    "task_id": task.id,
                    "name": task.name,
                    "status": task.status.value,
                    "next_run": task.next_run.isoformat() if task.next_run else None,
                    "last_run": task.last_run.isoformat() if task.last_run else None,
                    "run_count": task.run_count,
                    "priority": task.priority,
                    "tags": task.tags,
                    "enabled": task.enabled,
                }
            )

        # Sort by next run time
        tasks.sort(key=lambda x: x["next_run"] or "9999")

        return tasks

    async def _get_task(self, task_id: str) -> Dict[str, Any]:
        """Get detailed information about a task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        return {
            "task_id": task.id,
            "name": task.name,
            "function": str(task.function),
            "args": task.args,
            "kwargs": task.kwargs,
            "schedule_time": task.schedule_time.isoformat(),
            "recurrence_type": task.recurrence_type.value,
            "recurrence_pattern": task.recurrence_pattern,
            "status": task.status.value,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "run_count": task.run_count,
            "error_count": task.error_count,
            "last_error": task.last_error,
            "priority": task.priority,
            "tags": task.tags,
            "metadata": task.metadata,
            "enabled": task.enabled,
            "created_at": task.created_at.isoformat(),
        }

    async def _start_scheduler(self) -> Dict[str, Any]:
        """Start the task scheduler"""
        if self.scheduler_running:
            return {"status": "already_running"}

        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())

        return {
            "status": "started",
            "tasks_count": len(self.tasks),
            "pending_tasks": len(
                [t for t in self.tasks.values() if t.status == TaskStatus.PENDING]
            ),
        }

    async def _stop_scheduler(self) -> Dict[str, Any]:
        """Stop the task scheduler"""
        if not self.scheduler_running:
            return {"status": "not_running"}

        self.scheduler_running = False

        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass

        return {"status": "stopped"}

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Scheduler started")

        while self.scheduler_running:
            try:
                # Check for tasks to run
                now = datetime.now()
                tasks_to_run = []

                # Get all tasks that should run now
                while self.task_queue and self.task_queue[0][0] <= now:
                    _, task_id = heapq.heappop(self.task_queue)

                    if task_id in self.tasks and self.tasks[task_id].enabled:
                        tasks_to_run.append(self.tasks[task_id])

                # Execute tasks concurrently with limit
                if tasks_to_run:
                    # Sort by priority
                    tasks_to_run.sort(key=lambda x: x.priority, reverse=True)

                    # Execute with concurrency limit
                    for i in range(0, len(tasks_to_run), self.max_concurrent_tasks):
                        batch = tasks_to_run[i : i + self.max_concurrent_tasks]
                        await asyncio.gather(
                            *[self._run_task(task) for task in batch],
                            return_exceptions=True,
                        )

                # Sleep for a short interval
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(5)

        logger.info("Scheduler stopped")

    async def _run_task(self, task: ScheduledTask):
        """Run a single task"""
        logger.info(f"Running task: {task.name} (ID: {task.id})")

        # Update status
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now()
        self.current_tasks += 1

        try:
            # Get function
            if isinstance(task.function, str):
                if task.function not in self.function_registry:
                    raise ValueError(f"Function {task.function} not registered")
                func = self.function_registry[task.function]
            else:
                func = task.function

            # Execute with timeout
            if task.timeout:
                result = await asyncio.wait_for(
                    self._execute_function(func, task.args, task.kwargs),
                    timeout=task.timeout,
                )
            else:
                result = await self._execute_function(func, task.args, task.kwargs)

            # Success
            task.status = TaskStatus.COMPLETED
            task.run_count += 1
            task.error_count = 0
            task.last_error = None

            # Log to history
            await self._log_execution(task, True, result)

            # Schedule next run for recurring tasks
            if task.recurrence_type != RecurrenceType.ONCE:
                task.next_run = self._calculate_next_run(task)
                if task.next_run:
                    heapq.heappush(self.task_queue, (task.next_run, task.id))
                    task.status = TaskStatus.PENDING

        except Exception as e:
            # Handle failure
            logger.error(f"Task {task.name} failed: {e}")
            task.error_count += 1
            task.last_error = str(e)

            # Log to history
            await self._log_execution(task, False, str(e))

            # Retry logic
            if task.error_count < task.max_retries:
                task.status = TaskStatus.PENDING
                retry_time = datetime.now() + timedelta(seconds=task.retry_delay)
                heapq.heappush(self.task_queue, (retry_time, task.id))
                logger.info(f"Task {task.name} scheduled for retry at {retry_time}")
            else:
                task.status = TaskStatus.FAILED

        finally:
            self.current_tasks -= 1
            await self._save_task(task)

    async def _execute_function(
        self, func: Callable, args: List[Any], kwargs: Dict[str, Any]
    ) -> Any:
        """Execute a function, handling both sync and async"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _calculate_next_run(self, task: ScheduledTask) -> Optional[datetime]:
        """Calculate the next run time for a task"""
        if task.recurrence_type == RecurrenceType.ONCE:
            return None

        base_time = task.last_run or task.schedule_time

        if task.recurrence_type == RecurrenceType.DAILY:
            return base_time + timedelta(days=1)

        elif task.recurrence_type == RecurrenceType.WEEKLY:
            return base_time + timedelta(weeks=1)

        elif task.recurrence_type == RecurrenceType.MONTHLY:
            # Handle month boundaries
            next_month = base_time.replace(day=1) + timedelta(days=32)
            return next_month.replace(day=min(base_time.day, 28))

        elif task.recurrence_type == RecurrenceType.CRON:
            if task.recurrence_pattern:
                cron = croniter.croniter(task.recurrence_pattern, base_time)
                return cron.get_next(datetime)

        elif task.recurrence_type == RecurrenceType.INTERVAL:
            if task.recurrence_pattern:
                # Parse interval (e.g., "5m", "1h", "30s")
                interval = self._parse_interval(task.recurrence_pattern)
                return base_time + interval

        return None

    def _parse_interval(self, interval_str: str) -> timedelta:
        """Parse interval string to timedelta"""
        import re

        # Match patterns like "5m", "1h", "30s", "2d"
        match = re.match(r"(\d+)([smhd])", interval_str.lower())
        if not match:
            raise ValueError(f"Invalid interval format: {interval_str}")

        value, unit = match.groups()
        value = int(value)

        if unit == "s":
            return timedelta(seconds=value)
        elif unit == "m":
            return timedelta(minutes=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        else:
            raise ValueError(f"Unknown interval unit: {unit}")

    async def _save_task(self, task: ScheduledTask):
        """Persist task to storage"""
        task_file = self.storage_path / f"{task.id}.json"

        # Convert task to dict
        task_dict = {
            "id": task.id,
            "name": task.name,
            "function": str(task.function),
            "args": task.args,
            "kwargs": task.kwargs,
            "schedule_time": task.schedule_time.isoformat(),
            "recurrence_type": task.recurrence_type.value,
            "recurrence_pattern": task.recurrence_pattern,
            "max_retries": task.max_retries,
            "retry_delay": task.retry_delay,
            "timeout": task.timeout,
            "priority": task.priority,
            "tags": task.tags,
            "metadata": task.metadata,
            "status": task.status.value,
            "created_at": task.created_at.isoformat(),
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "run_count": task.run_count,
            "error_count": task.error_count,
            "last_error": task.last_error,
            "enabled": task.enabled,
        }

        async with aiofiles.open(task_file, "w") as f:
            await f.write(json.dumps(task_dict, indent=2))

    async def _load_tasks(self):
        """Load persisted tasks"""
        try:
            for task_file in self.storage_path.glob("*.json"):
                async with aiofiles.open(task_file, "r") as f:
                    task_dict = json.loads(await f.read())

                # Reconstruct task
                task = ScheduledTask(
                    id=task_dict["id"],
                    name=task_dict["name"],
                    function=task_dict["function"],
                    args=task_dict["args"],
                    kwargs=task_dict["kwargs"],
                    schedule_time=datetime.fromisoformat(task_dict["schedule_time"]),
                    recurrence_type=RecurrenceType(task_dict["recurrence_type"]),
                    recurrence_pattern=task_dict["recurrence_pattern"],
                    max_retries=task_dict["max_retries"],
                    retry_delay=task_dict["retry_delay"],
                    timeout=task_dict["timeout"],
                    priority=task_dict["priority"],
                    tags=task_dict["tags"],
                    metadata=task_dict["metadata"],
                    status=TaskStatus(task_dict["status"]),
                    created_at=datetime.fromisoformat(task_dict["created_at"]),
                    run_count=task_dict["run_count"],
                    error_count=task_dict["error_count"],
                    last_error=task_dict["last_error"],
                    enabled=task_dict["enabled"],
                )

                if task_dict["last_run"]:
                    task.last_run = datetime.fromisoformat(task_dict["last_run"])
                if task_dict["next_run"]:
                    task.next_run = datetime.fromisoformat(task_dict["next_run"])

                # Add to tasks
                self.tasks[task.id] = task

                # Add to queue if pending
                if task.enabled and task.next_run and task.status == TaskStatus.PENDING:
                    heapq.heappush(self.task_queue, (task.next_run, task.id))

            logger.info(f"Loaded {len(self.tasks)} tasks from storage")

        except Exception as e:
            logger.error(f"Error loading tasks: {e}")

    async def _log_execution(self, task: ScheduledTask, success: bool, result: Any):
        """Log task execution to history"""
        log_entry = {
            "task_id": task.id,
            "task_name": task.name,
            "execution_time": datetime.now().isoformat(),
            "success": success,
            "result": str(result)[:1000],  # Limit result size
            "run_count": task.run_count,
            "error_count": task.error_count,
        }

        self.task_history.append(log_entry)

        # Keep only recent history
        if len(self.task_history) > 10000:
            self.task_history = self.task_history[-5000:]

        # Also save to file for persistence
        history_file = self.storage_path / "history.jsonl"
        async with aiofiles.open(history_file, "a") as f:
            await f.write(json.dumps(log_entry) + "\n")

    async def _get_history(
        self, task_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get execution history"""
        history = self.task_history

        if task_id:
            history = [h for h in history if h["task_id"] == task_id]

        # Return most recent entries
        return history[-limit:]

    async def _get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        total_tasks = len(self.tasks)

        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len(
                [t for t in self.tasks.values() if t.status == status]
            )

        # Calculate success rate
        total_runs = sum(t.run_count for t in self.tasks.values())
        total_errors = sum(t.error_count for t in self.tasks.values())
        success_rate = (total_runs - total_errors) / total_runs if total_runs > 0 else 0

        return {
            "total_tasks": total_tasks,
            "status_breakdown": status_counts,
            "active_tasks": self.current_tasks,
            "scheduler_running": self.scheduler_running,
            "total_executions": total_runs,
            "total_errors": total_errors,
            "success_rate": success_rate,
            "queue_size": len(self.task_queue),
            "history_size": len(self.task_history),
        }

    async def _pause_task(self, task_id: str) -> Dict[str, Any]:
        """Pause a scheduled task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.enabled = False
        task.status = TaskStatus.PAUSED

        # Remove from queue
        self.task_queue = [(t, tid) for t, tid in self.task_queue if tid != task_id]
        heapq.heapify(self.task_queue)

        await self._save_task(task)

        return {"task_id": task_id, "name": task.name, "status": "paused"}

    async def _resume_task(self, task_id: str) -> Dict[str, Any]:
        """Resume a paused task"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]
        task.enabled = True
        task.status = TaskStatus.PENDING

        # Calculate next run and add to queue
        if not task.next_run or task.next_run <= datetime.now():
            task.next_run = self._calculate_next_run(task) or datetime.now()

        heapq.heappush(self.task_queue, (task.next_run, task_id))

        await self._save_task(task)

        return {
            "task_id": task_id,
            "name": task.name,
            "status": "resumed",
            "next_run": task.next_run.isoformat(),
        }

    async def _update_task(
        self, task_id: str, updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update task properties"""
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.tasks[task_id]

        # Update allowed fields
        allowed_updates = {
            "name",
            "priority",
            "tags",
            "metadata",
            "max_retries",
            "retry_delay",
            "timeout",
            "recurrence_pattern",
        }

        for field, value in updates.items():
            if field in allowed_updates and hasattr(task, field):
                setattr(task, field, value)

        # Recalculate next run if pattern changed
        if "recurrence_pattern" in updates:
            task.next_run = self._calculate_next_run(task)

            # Update queue
            self.task_queue = [(t, tid) for t, tid in self.task_queue if tid != task_id]
            if task.next_run and task.enabled:
                heapq.heappush(self.task_queue, (task.next_run, task_id))
            heapq.heapify(self.task_queue)

        await self._save_task(task)

        return {
            "task_id": task_id,
            "name": task.name,
            "updated_fields": list(updates.keys()),
        }

    def register_function(self, name: str, function: Callable):
        """Register a function for use in scheduled tasks"""
        self.function_registry[name] = function
        logger.info(f"Registered function: {name}")

    def _get_parameter_documentation(self) -> Dict[str, Any]:
        """Get parameter documentation for the scheduler"""
        return {
            "action": {
                "type": "string",
                "required": True,
                "enum": [
                    "schedule",
                    "cancel",
                    "list",
                    "get",
                    "pause",
                    "resume",
                    "update",
                    "history",
                    "stats",
                    "start_scheduler",
                    "stop_scheduler",
                ],
                "description": "Action to perform",
            },
            "task_name": {
                "type": "string",
                "required": "for schedule action",
                "description": "Name of the task",
            },
            "function": {
                "type": "string or callable",
                "required": "for schedule action",
                "description": "Function to execute",
            },
            "schedule_time": {
                "type": "datetime or ISO string",
                "required": False,
                "description": "When to first run the task (default: now)",
            },
            "recurrence_type": {
                "type": "string",
                "required": False,
                "enum": ["once", "daily", "weekly", "monthly", "cron", "interval"],
                "description": "Type of recurrence",
            },
            "cron_expression": {
                "type": "string",
                "required": "for cron recurrence",
                "description": "Cron expression (e.g., '0 9 * * *' for daily at 9am)",
            },
            "interval": {
                "type": "string",
                "required": "for interval recurrence",
                "description": "Interval string (e.g., '5m', '1h', '30s')",
            },
            "args": {
                "type": "list",
                "required": False,
                "description": "Positional arguments for the function",
            },
            "kwargs": {
                "type": "dict",
                "required": False,
                "description": "Keyword arguments for the function",
            },
            "priority": {
                "type": "integer",
                "required": False,
                "description": "Task priority (1-10, default: 5)",
            },
            "tags": {
                "type": "list",
                "required": False,
                "description": "Tags for categorizing tasks",
            },
            "task_id": {
                "type": "string",
                "required": "for cancel, get, pause, resume, update actions",
                "description": "ID of the task",
            },
        }
