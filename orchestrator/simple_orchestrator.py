#!/usr/bin/env python3
"""
Simple Claude Orchestrator
=========================

A practical file-based orchestrator that multiple Claude instances
can use to coordinate work without complex infrastructure.
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import hashlib
import fcntl
import os


class SimpleClaudeOrchestrator:
    def __init__(self, orchestration_dir: str = "./orchestration"):
        self.base_dir = Path(orchestration_dir)

        # Create directory structure
        self.tasks_dir = self.base_dir / "tasks"
        self.active_dir = self.base_dir / "active"
        self.completed_dir = self.base_dir / "completed"
        self.agents_dir = self.base_dir / "agents"

        for dir in [
            self.tasks_dir,
            self.active_dir,
            self.completed_dir,
            self.agents_dir,
        ]:
            dir.mkdir(parents=True, exist_ok=True)

        # Create instruction file
        self._create_instructions()

    def _create_instructions(self):
        """Create instructions for Claude instances"""
        instructions = """# Claude Orchestration Instructions

## For Claude Instances:

1. **Check for tasks**: Look in tasks/ directory
2. **Claim a task**: Move from tasks/ to active/ with your agent_id
3. **Complete task**: Move from active/ to completed/ with results
4. **Check agent status**: Look in agents/ directory

## Task Format:
```json
{
    "id": "task_123",
    "type": "implement|design|review|test",
    "priority": 1-5,
    "description": "What to do",
    "context": "Any additional context",
    "memory_refs": ["mcp://memory/ref1", "mcp://memory/ref2"]
}
```

## Workflow:
1. Producer creates task in tasks/
2. Claude claims by moving to active/task_123_claude-xxx.json
3. Claude works on task (using MCP for memory/context)
4. Claude completes by moving to completed/ with results
"""

        with open(self.base_dir / "README.md", "w") as f:
            f.write(instructions)

    def submit_task(
        self, task_type: str, description: str, context: Dict = None, priority: int = 3
    ) -> str:
        """Submit a new task for any Claude to pick up"""
        task_id = f"task_{int(time.time())}_{hashlib.md5(description.encode()).hexdigest()[:8]}"

        task = {
            "id": task_id,
            "type": task_type,
            "priority": priority,
            "description": description,
            "context": context or {},
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }

        # Add memory references if using MCP
        if "memory_refs" in context:
            task["memory_refs"] = context["memory_refs"]

        # Write task file
        task_file = self.tasks_dir / f"{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task, f, indent=2)

        print(f"📋 Task submitted: {task_id}")
        return task_id

    def claim_task(
        self, agent_id: str, preferred_types: List[str] = None
    ) -> Optional[Dict]:
        """Claim an available task (atomic operation)"""
        # Get all pending tasks
        task_files = sorted(
            self.tasks_dir.glob("*.json"), key=lambda x: x.stat().st_mtime
        )

        for task_file in task_files:
            try:
                # Try to claim the task atomically
                task_data = None

                # Read task
                with open(task_file, "r") as f:
                    task_data = json.load(f)

                # Check if preferred type
                if preferred_types and task_data["type"] not in preferred_types:
                    continue

                # Try to move file (atomic on same filesystem)
                active_file = self.active_dir / f"{task_data['id']}_{agent_id}.json"

                try:
                    task_file.rename(active_file)

                    # Update task data
                    task_data["claimed_by"] = agent_id
                    task_data["claimed_at"] = datetime.now().isoformat()
                    task_data["status"] = "active"

                    # Write updated data
                    with open(active_file, "w") as f:
                        json.dump(task_data, f, indent=2)

                    print(f"✅ {agent_id} claimed task: {task_data['id']}")
                    return task_data

                except FileExistsError:
                    # Another agent got it first
                    continue

            except Exception as e:
                continue

        return None

    def complete_task(
        self, task_id: str, agent_id: str, result: Dict, output_files: List[str] = None
    ) -> bool:
        """Complete a task with results"""
        # Find active task file
        active_pattern = f"{task_id}_{agent_id}.json"
        active_files = list(self.active_dir.glob(active_pattern))

        if not active_files:
            print(f"❌ No active task found: {task_id} for {agent_id}")
            return False

        active_file = active_files[0]

        # Read task data
        with open(active_file, "r") as f:
            task_data = json.load(f)

        # Update with results
        task_data["completed_at"] = datetime.now().isoformat()
        task_data["status"] = "completed"
        task_data["result"] = result
        task_data["output_files"] = output_files or []
        task_data["duration_seconds"] = (
            datetime.fromisoformat(task_data["completed_at"])
            - datetime.fromisoformat(task_data["claimed_at"])
        ).total_seconds()

        # Move to completed
        completed_file = self.completed_dir / f"{task_id}_{agent_id}_completed.json"
        with open(completed_file, "w") as f:
            json.dump(task_data, f, indent=2)

        # Remove active file
        active_file.unlink()

        print(f"✅ {agent_id} completed task: {task_id}")
        return True

    def register_agent(
        self, agent_id: str, agent_type: str, capabilities: List[str]
    ) -> None:
        """Register a Claude agent"""
        agent_file = self.agents_dir / f"{agent_id}.json"

        agent_data = {
            "id": agent_id,
            "type": agent_type,
            "capabilities": capabilities,
            "registered_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "status": "online",
        }

        with open(agent_file, "w") as f:
            json.dump(agent_data, f, indent=2)

        print(f"🤖 Registered agent: {agent_id}")

    def update_agent_status(self, agent_id: str, status: str = "online") -> None:
        """Update agent status"""
        agent_file = self.agents_dir / f"{agent_id}.json"

        if agent_file.exists():
            with open(agent_file, "r") as f:
                agent_data = json.load(f)

            agent_data["status"] = status
            agent_data["last_seen"] = datetime.now().isoformat()

            with open(agent_file, "w") as f:
                json.dump(agent_data, f, indent=2)

    def get_status(self) -> Dict:
        """Get orchestrator status"""
        pending_tasks = len(list(self.tasks_dir.glob("*.json")))
        active_tasks = len(list(self.active_dir.glob("*.json")))
        completed_tasks = len(list(self.completed_dir.glob("*.json")))

        # Get agent statuses
        agents = []
        for agent_file in self.agents_dir.glob("*.json"):
            with open(agent_file, "r") as f:
                agent_data = json.load(f)

            # Check if agent is stale (not seen in 5 minutes)
            last_seen = datetime.fromisoformat(agent_data["last_seen"])
            if (datetime.now() - last_seen).total_seconds() > 300:
                agent_data["status"] = "offline"

            agents.append(agent_data)

        return {
            "tasks": {
                "pending": pending_tasks,
                "active": active_tasks,
                "completed": completed_tasks,
            },
            "agents": agents,
            "orchestration_dir": str(self.base_dir),
        }


# Helper class for Claude instances
class ClaudeAgent:
    """Helper for Claude instances to interact with orchestrator"""

    def __init__(self, agent_id: str, agent_type: str = "claude-code"):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.orchestrator = SimpleClaudeOrchestrator()

        # Register self
        capabilities = self._get_capabilities()
        self.orchestrator.register_agent(agent_id, agent_type, capabilities)

    def _get_capabilities(self) -> List[str]:
        if self.agent_type == "claude-code":
            return ["implement", "refactor", "debug", "test"]
        elif self.agent_type == "claude-desktop":
            return ["design", "architect", "prototype", "document"]
        else:
            return ["general"]

    async def work_loop(self):
        """Main work loop for agent"""
        print(f"🤖 {self.agent_id} starting work loop...")

        while True:
            try:
                # Update status
                self.orchestrator.update_agent_status(self.agent_id, "online")

                # Try to claim a task
                task = self.orchestrator.claim_task(
                    self.agent_id, preferred_types=self._get_capabilities()
                )

                if task:
                    print(f"📋 Working on: {task['description'][:50]}...")

                    # Process task (simulate work)
                    result = await self.process_task(task)

                    # Complete task
                    self.orchestrator.complete_task(task["id"], self.agent_id, result)
                else:
                    # No tasks available, wait
                    await asyncio.sleep(5)

            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(10)

    async def process_task(self, task: Dict) -> Dict:
        """Process a task (override in actual implementation)"""
        # Simulate work
        await asyncio.sleep(2)

        return {
            "status": "success",
            "message": f"Completed {task['type']} task",
            "details": f"Processed: {task['description']}",
        }


# Example usage
if __name__ == "__main__":
    import asyncio

    # Create orchestrator
    orch = SimpleClaudeOrchestrator()

    # Submit some tasks
    orch.submit_task(
        "implement",
        "Create neural cache module with Redis backend",
        {"memory_refs": ["mcp://jarvis-memory/design-neural-cache"]},
    )

    orch.submit_task(
        "design",
        "Design authentication system for JARVIS",
        {"requirements": "OAuth2, JWT, role-based"},
    )

    orch.submit_task("test", "Write tests for quantum swarm module")

    # Show status
    status = orch.get_status()
    print(f"\n📊 Orchestrator Status:")
    print(f"Pending tasks: {status['tasks']['pending']}")
    print(f"Active tasks: {status['tasks']['active']}")
    print(f"Completed tasks: {status['tasks']['completed']}")

    print("\n✨ Orchestrator ready! Claude instances can now:")
    print("1. Check orchestration/tasks/ for work")
    print("2. Claim tasks by moving them to orchestration/active/")
    print("3. Complete tasks by moving to orchestration/completed/")

    # Example: Simulate an agent
    async def simulate_agent():
        agent = ClaudeAgent("claude-code-001", "claude-code")
        await agent.work_loop()

    # Uncomment to run simulation
    # asyncio.run(simulate_agent())
