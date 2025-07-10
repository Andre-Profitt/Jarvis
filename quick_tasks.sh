#!/bin/bash

# Quick task assignment examples for the JARVIS Orchestra

echo "🎼 JARVIS Orchestra Quick Tasks"
echo "==============================="

# Function to assign task
assign_task() {
    local agent=$1
    local task=$2
    echo "📋 Assigning to $agent: $task"
    npx ruv-swarm task assign "$agent" "$task"
}

# Example 1: Assign specific tasks
echo -e "\n1️⃣ Specific Agent Tasks:"
assign_task "Violin1" "Implement voice recognition module"
assign_task "Trumpet1" "Optimize memory usage in core system"
assign_task "Flute" "Research best NLP libraries"
assign_task "Timpani" "Create integration tests for voice module"

# Example 2: Broadcast to all
echo -e "\n2️⃣ Broadcast Task:"
echo "📢 Broadcasting: Prepare for JARVIS v2.0 sprint"
npx ruv-swarm task broadcast "Prepare for JARVIS v2.0 sprint planning"

# Example 3: Section coordination
echo -e "\n3️⃣ Section Coordination:"
npx ruv-swarm memory store "orchestra/strings/task" '{"task": "Build conversation management system", "priority": "high"}'
npx ruv-swarm memory store "orchestra/brass/task" '{"task": "Optimize real-time response latency", "priority": "high"}'

# Example 4: Check task status
echo -e "\n4️⃣ Task Status:"
npx ruv-swarm task list

echo -e "\n✅ Tasks assigned! Use 'npx ruv-swarm task list' to monitor progress"