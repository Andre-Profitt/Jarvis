#!/bin/bash

# Optimize Claude Code's ruv-swarm Performance
# This script configures the swarm for maximum performance

echo "🚀 Optimizing Claude Code's ruv-swarm Configuration"
echo "=================================================="

# 1. Configure High-Performance Hooks
echo "⚡ Configuring high-performance hooks..."
cat > ~/.claude/hooks/swarm-performance.json << 'EOF'
{
  "pre-task": {
    "auto-spawn-agents": true,
    "spawn-count": 3,
    "load-context": true,
    "optimize-topology": true,
    "cache-searches": true,
    "parallel-execution": true
  },
  "post-edit": {
    "auto-format": true,
    "update-memory": true,
    "sync-agents": true,
    "neural-train": true,
    "cache-results": true
  },
  "notification": {
    "batch-size": 10,
    "async-process": true,
    "compress-data": true
  },
  "session-end": {
    "save-patterns": true,
    "optimize-cache": true,
    "generate-metrics": true
  }
}
EOF

# 2. Create Performance Monitoring Script
echo "📊 Creating performance monitoring..."
cat > monitor_swarm_performance.js << 'EOF'
#!/usr/bin/env node

const { exec } = require('child_process');
const fs = require('fs');

async function monitorSwarm() {
  console.log('🔍 Monitoring Swarm Performance...\n');
  
  // Check current swarm status
  exec('npx ruv-swarm swarm status --json', (err, stdout) => {
    if (!err) {
      const status = JSON.parse(stdout);
      console.log(`✅ Active Agents: ${status.agents.length}`);
      console.log(`📈 Tasks Completed: ${status.tasksCompleted}`);
      console.log(`⚡ Average Speed: ${status.avgTaskTime}ms`);
      console.log(`🧠 Memory Usage: ${status.memoryUsage}MB`);
    }
  });
  
  // Check for bottlenecks
  exec('npx ruv-swarm benchmark current', (err, stdout) => {
    if (!err && stdout.includes('bottleneck')) {
      console.log('\n⚠️  Performance Bottlenecks Detected!');
      console.log('Recommended actions:');
      console.log('- Spawn more agents: npx ruv-swarm agent spawn --type optimizer');
      console.log('- Clear cache: npx ruv-swarm memory cleanup --old');
      console.log('- Optimize topology: npx ruv-swarm swarm optimize');
    }
  });
}

// Monitor every 30 seconds
setInterval(monitorSwarm, 30000);
monitorSwarm();
EOF

# 3. Create Swarm Optimization Commands
echo "🛠️  Creating optimization commands..."
mkdir -p ~/.claude/commands

cat > ~/.claude/commands/swarm-boost.sh << 'EOF'
#!/bin/bash
# Boost swarm performance for heavy tasks

echo "🚀 Boosting Swarm Performance..."

# Increase agent count
npx ruv-swarm swarm scale --agents 15

# Enable all optimizations
npx ruv-swarm features enable --all

# Optimize memory
npx ruv-swarm memory optimize --aggressive

# Set high-performance mode
npx ruv-swarm swarm configure --mode "performance"

echo "✅ Swarm boosted! Agents: 15, Mode: Performance"
EOF

cat > ~/.claude/commands/swarm-parallel.sh << 'EOF'
#!/bin/bash
# Enable maximum parallelization

echo "⚡ Enabling Maximum Parallelization..."

# Switch to mesh topology for max parallel
npx ruv-swarm swarm reconfigure --topology mesh

# Enable parallel features
npx ruv-swarm features enable parallel-execution
npx ruv-swarm features enable async-coordination
npx ruv-swarm features enable batch-processing

# Configure parallel settings
npx ruv-swarm config set --parallel-tasks 20
npx ruv-swarm config set --batch-size 10

echo "✅ Parallel mode enabled!"
EOF

# 4. Create Smart Agent Templates
echo "🤖 Creating smart agent templates..."
cat > ~/.claude/agents/specialist-templates.json << 'EOF'
{
  "performance-specialist": {
    "type": "optimizer",
    "capabilities": ["profiling", "caching", "parallel-optimization"],
    "memory-access": "full",
    "priority": "high"
  },
  "security-specialist": {
    "type": "validator",
    "capabilities": ["vulnerability-scan", "code-audit", "dependency-check"],
    "memory-access": "read",
    "priority": "critical"
  },
  "architecture-specialist": {
    "type": "architect",
    "capabilities": ["design-patterns", "scalability", "integration"],
    "memory-access": "full",
    "priority": "high"
  },
  "test-specialist": {
    "type": "tester",
    "capabilities": ["unit-testing", "integration-testing", "performance-testing"],
    "memory-access": "full",
    "priority": "high"
  }
}
EOF

# 5. Create Swarm Presets for Common Tasks
echo "📋 Creating task presets..."
cat > ~/.claude/swarm-presets.json << 'EOF'
{
  "research-heavy": {
    "topology": "mesh",
    "agents": 12,
    "distribution": {
      "researcher": 5,
      "analyst": 3,
      "validator": 2,
      "coordinator": 2
    }
  },
  "development-sprint": {
    "topology": "hierarchical",
    "agents": 15,
    "distribution": {
      "architect": 1,
      "coder": 6,
      "tester": 4,
      "optimizer": 2,
      "coordinator": 2
    }
  },
  "bug-hunt": {
    "topology": "star",
    "agents": 10,
    "distribution": {
      "debugger": 4,
      "analyst": 3,
      "tester": 2,
      "coordinator": 1
    }
  },
  "performance-tune": {
    "topology": "ring",
    "agents": 8,
    "distribution": {
      "profiler": 3,
      "optimizer": 3,
      "validator": 1,
      "coordinator": 1
    }
  }
}
EOF

# 6. Enable Auto-Optimization
echo "🔧 Enabling auto-optimization..."
npx ruv-swarm config set --auto-optimize true
npx ruv-swarm config set --auto-scale true
npx ruv-swarm config set --neural-learning true
npx ruv-swarm config set --cache-size 1024

# 7. Make scripts executable
chmod +x ~/.claude/commands/*.sh
chmod +x monitor_swarm_performance.js

echo ""
echo "✅ Claude Swarm Optimization Complete!"
echo ""
echo "🎯 Quick Commands:"
echo "  ~/.claude/commands/swarm-boost.sh     - Boost performance"
echo "  ~/.claude/commands/swarm-parallel.sh  - Max parallelization"
echo "  ./monitor_swarm_performance.js        - Monitor performance"
echo ""
echo "📊 Current Configuration:"
npx ruv-swarm config list

echo ""
echo "💡 Tips for Maximum Performance:"
echo "1. Use 'swarm-boost' before complex tasks"
echo "2. Monitor performance during long operations"
echo "3. Use presets for specific task types"
echo "4. Enable neural learning for repeated tasks"
echo "5. Clear cache if memory usage is high"