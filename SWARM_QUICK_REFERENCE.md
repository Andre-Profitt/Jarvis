# 🚀 Claude Code Swarm Quick Reference

## Immediate Performance Boosts

### 1. **Start Every Session With More Agents**
```javascript
// Instead of default 6, use 12-15 agents
mcp__ruv-swarm__swarm_init { 
  topology: "hierarchical", 
  maxAgents: 15,
  strategy: "adaptive"
}
```

### 2. **Spawn Agents in Batches**
```javascript
// BAD: One at a time ❌
mcp__ruv-swarm__agent_spawn { type: "researcher" }
mcp__ruv-swarm__agent_spawn { type: "coder" }

// GOOD: All at once ✅
[BatchTool]:
  mcp__ruv-swarm__agent_spawn { type: "researcher", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "coder", count: 4 }
  mcp__ruv-swarm__agent_spawn { type: "tester", count: 2 }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", count: 1 }
```

### 3. **Use Memory for Coordination**
```javascript
// Store findings immediately
mcp__ruv-swarm__memory_usage {
  action: "store",
  key: "findings/critical/security-issue",
  value: { issue: "SQL injection", location: "api.js:45" }
}

// Check before duplicating work
mcp__ruv-swarm__memory_usage {
  action: "list",
  pattern: "findings/*"
}
```

### 4. **Enable Performance Features**
```javascript
mcp__ruv-swarm__features_detect { enable: ["parallel", "neural", "cache", "optimize"] }
```

## Task-Specific Configurations

### For Research Tasks
```javascript
// Mesh topology = all agents can talk to each other
mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 12 }

// Spawn research-focused team
[BatchTool]:
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "DeepDiver", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "PatternSeeker", count: 2 }
  mcp__ruv-swarm__agent_spawn { type: "validator", name: "FactChecker", count: 1 }
```

### For Development Tasks
```javascript
// Hierarchical = clear chain of command
mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 15 }

// Spawn development team
[BatchTool]:
  mcp__ruv-swarm__agent_spawn { type: "architect", name: "Designer", count: 1 }
  mcp__ruv-swarm__agent_spawn { type: "coder", name: "Builder", count: 5 }
  mcp__ruv-swarm__agent_spawn { type: "tester", name: "Validator", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "optimizer", name: "Tuner", count: 1 }
```

### For Analysis Tasks
```javascript
// Star topology = central coordinator
mcp__ruv-swarm__swarm_init { topology: "star", maxAgents: 10 }

// Spawn analysis team
[BatchTool]:
  mcp__ruv-swarm__agent_spawn { type: "analyst", name: "DataExpert", count: 4 }
  mcp__ruv-swarm__agent_spawn { type: "researcher", name: "ContextFinder", count: 2 }
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "Synthesizer", count: 1 }
```

## Performance Monitoring

### Check Swarm Health
```javascript
// Real-time monitoring
mcp__ruv-swarm__swarm_monitor { 
  metrics: ["agents", "tasks", "memory", "latency"],
  realtime: true 
}

// Get performance report
mcp__ruv-swarm__benchmark_run { test: "current_performance" }
```

### Identify Bottlenecks
```javascript
// Find slow points
mcp__ruv-swarm__swarm_status { detailed: true, showBottlenecks: true }

// Auto-fix bottlenecks
mcp__ruv-swarm__swarm_optimize { target: "bottlenecks", aggressive: true }
```

## Advanced Techniques

### 1. **Neural Pattern Learning**
```javascript
// Train on successful patterns
mcp__ruv-swarm__neural_train {
  pattern: "jarvis_success",
  data: { config: currentSwarmConfig, result: "excellent" }
}

// Apply learned patterns
mcp__ruv-swarm__neural_predict { task: "similar_project" }
```

### 2. **Dynamic Scaling**
```javascript
// Auto-scale based on load
mcp__ruv-swarm__swarm_configure {
  autoScale: true,
  minAgents: 8,
  maxAgents: 20,
  scaleThreshold: 0.8
}
```

### 3. **Parallel Task Chains**
```javascript
mcp__ruv-swarm__task_orchestrate {
  tasks: [
    { id: "analyze", parallel: ["scan", "review", "audit"] },
    { id: "implement", parallel: ["core", "api", "ui", "tests"] },
    { id: "optimize", parallel: ["performance", "security", "ux"] }
  ],
  strategy: "maximum-parallel"
}
```

## Common Patterns for JARVIS

### Pattern 1: Full System Analysis
```javascript
// Initialize power swarm
mcp__ruv-swarm__swarm_init { topology: "mesh", maxAgents: 15 }

// Spawn specialized teams
[BatchTool]:
  // Analysis team
  mcp__ruv-swarm__agent_spawn { type: "analyst", specialization: "code", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "analyst", specialization: "performance", count: 2 }
  mcp__ruv-swarm__agent_spawn { type: "analyst", specialization: "security", count: 2 }
  
  // Research team
  mcp__ruv-swarm__agent_spawn { type: "researcher", specialization: "best_practices", count: 2 }
  mcp__ruv-swarm__agent_spawn { type: "researcher", specialization: "similar_projects", count: 2 }
  
  // Coordination
  mcp__ruv-swarm__agent_spawn { type: "coordinator", name: "MasterCoordinator", count: 1 }

// Execute parallel analysis
mcp__ruv-swarm__task_orchestrate {
  task: "Complete JARVIS system analysis",
  strategy: "parallel-aggressive",
  coordination: "continuous"
}
```

### Pattern 2: Rapid Feature Development
```javascript
// Initialize dev swarm
mcp__ruv-swarm__swarm_init { topology: "hierarchical", maxAgents: 18 }

// Spawn feature teams
[BatchTool]:
  // Architecture
  mcp__ruv-swarm__agent_spawn { type: "architect", count: 2 }
  
  // Feature teams (3 features in parallel)
  mcp__ruv-swarm__agent_spawn { type: "coder", team: "voice", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "coder", team: "ai", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "coder", team: "ui", count: 3 }
  
  // Quality
  mcp__ruv-swarm__agent_spawn { type: "tester", count: 3 }
  mcp__ruv-swarm__agent_spawn { type: "reviewer", count: 2 }
  
  // Coordination
  mcp__ruv-swarm__agent_spawn { type: "coordinator", count: 2 }
```

## Quick Wins Checklist

- [ ] Always use 12+ agents for complex tasks
- [ ] Spawn agents in batches, not individually  
- [ ] Use memory for inter-agent communication
- [ ] Enable neural learning for repeated tasks
- [ ] Monitor performance during execution
- [ ] Use appropriate topology for task type
- [ ] Cache results aggressively
- [ ] Parallelize everything possible
- [ ] Train patterns on successful runs
- [ ] Auto-scale based on workload

## Performance Expectations

With optimized swarm:
- **Simple tasks**: 1.5-2x faster
- **Complex tasks**: 3-5x faster
- **Research tasks**: 4-6x more comprehensive
- **Development**: 3-4x faster with better quality
- **Bug finding**: 2-3x more issues detected

Remember: The swarm is most effective when agents work as a coordinated organism, not just parallel workers!