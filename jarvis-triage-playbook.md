# Jarvis Repository Triage & Restore Playbook
## Orchestra-Style Multi-Agent Execution Guide

### 🎭 Active Swarm Configuration
- **Swarm ID**: `swarm_1753477092608_bavfq6423`
- **Topology**: Hierarchical
- **Total Agents**: 9 specialized agents
- **Commander**: Jarvis Triage Commander

### 👥 Agent Roster & Responsibilities

| Agent | Type | Primary Responsibilities |
|-------|------|-------------------------|
| **Jarvis Triage Commander** | Coordinator | Overall orchestration, progress monitoring |
| **Git Surgeon** | Specialist | History rewriting, git-filter-repo operations |
| **Structure Architect** | Architect | Repository restructuring, service design |
| **Python Modernizer** | Specialist | Poetry setup, dependency management |
| **CI/CD Engineer** | Specialist | GitHub Actions, quality gates |
| **Docker Captain** | Specialist | Container strategy, multi-stage builds |
| **Documentation Curator** | Documenter | MkDocs setup, doc organization |
| **Repository Analyzer** | Analyst | Artifact identification, size analysis |
| **Quality Guardian** | Tester | Test execution, build verification |

## 📋 10-Step Execution Plan

### Step 0: Create Safe Snapshot
**Agent**: Git Surgeon
**Command**:
```bash
/agents orchestrate --task "Create mirror backup of Jarvis repository" --agent "Git Surgeon"
```

**Actions**:
```bash
git clone --mirror https://github.com/Andre-Profitt/Jarvis jarvis-backup.git
tar -czf jarvis-backup-$(date +%Y%m%d).tgz jarvis-backup.git
```

### Step 1: Identify & Ignore Artifacts
**Agent**: Repository Analyzer
**Command**:
```bash
/agents orchestrate --task "Scan and identify all artifacts that don't belong in Git" --agent "Repository Analyzer"
```

**Artifacts to Remove**:
- `artifacts/`, `training_data/`, `JARVIS-KNOWLEDGE/`
- `.ruv-swarm/`, `__pycache__/`, `*.py[cod]`
- `node_modules/`, `build/`, `dist/`
- Any `*.pt`, `*.h5`, `*.onnx` files

### Step 2: Purge Heavy History
**Agent**: Git Surgeon
**Command**:
```bash
/agents orchestrate --task "Execute git-filter-repo to remove heavy artifacts from history" --agent "Git Surgeon"
```

**Filter Commands**:
```bash
pip install git-filter-repo
git filter-repo --path artifacts/ --path training_data/ --path-glob '*.zip' --invert-paths
git gc --prune=now --aggressive
```

### Step 3: Restructure Repository
**Agent**: Structure Architect
**Command**:
```bash
/agents orchestrate --task "Restructure Jarvis into three logical layers" --agent "Structure Architect"
```

**New Structure**:
```
/
├─ docs/                 → Documentation site
├─ services/             → Runtime artifacts
│   ├─ orchestrator/     → FastAPI service
│   ├─ core/             → Pure Python library
│   ├─ plugins/          → Plugin system
│   ├─ mobile_app/       → React Native app
│   └─ ui/               → Next.js frontend
├─ infra/                → Infrastructure as code
└─ tools/                → Maintenance scripts
```

### Step 4: Standardize Python Packaging
**Agent**: Python Modernizer
**Command**:
```bash
/agents orchestrate --task "Convert all Python services to Poetry packaging" --agent "Python Modernizer"
```

**Poetry Setup**:
```bash
cd services/orchestrator
poetry init --name jarvis-orchestrator --python '>=3.12,<3.13'
poetry add fastapi uvicorn pydantic-settings
poetry add --group dev black ruff pytest pytest-cov
```

### Step 5: Introduce Quality Gates
**Agent**: CI/CD Engineer
**Command**:
```bash
/agents orchestrate --task "Setup GitHub Actions CI/CD pipeline with quality gates" --agent "CI/CD Engineer"
```

**GitHub Actions Config**:
```yaml
name: CI
on: [push, pull_request]
jobs:
  python-quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [orchestrator, core, plugins]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.12' }
      - run: pip install poetry
      - run: cd services/${{ matrix.service }} && poetry install
      - run: cd services/${{ matrix.service }} && poetry run ruff check .
      - run: cd services/${{ matrix.service }} && poetry run pytest
```

### Step 6: Single Configuration Source
**Agent**: Python Modernizer + Structure Architect
**Command**:
```bash
/agents orchestrate --task "Create unified Pydantic settings model" --agents "Python Modernizer,Structure Architect"
```

### Step 7: Curate Documentation
**Agent**: Documentation Curator
**Command**:
```bash
/agents orchestrate --task "Organize and curate all documentation with MkDocs" --agent "Documentation Curator"
```

**Documentation Plan**:
- Keep: Architecture, API docs, setup guides
- Archive: Vision documents to `docs/archive/`
- Delete: Empty stubs, duplicates

### Step 8: Docker Strategy
**Agent**: Docker Captain
**Command**:
```bash
/agents orchestrate --task "Create multi-stage Dockerfiles for all services" --agent "Docker Captain"
```

### Step 9: Execute Roadmap
**Agent**: Jarvis Triage Commander
**Command**:
```bash
/agents orchestrate --task "Execute 10-day roadmap with daily progress reports" --agent "Jarvis Triage Commander"
```

## 🚀 Quick Execution Commands

### Initialize Triage
```bash
# Start the triage process
/agents orchestrate --workflow jarvis-triage --auto-spawn true

# Monitor progress
/agents monitor --swarm-id swarm_1753477092608_bavfq6423 --interval 5

# Check agent status
/agents status --detailed
```

### Execute Specific Steps
```bash
# Step 0: Backup
/agents task --step 0 --agent "Git Surgeon" --priority high

# Step 1-2: Cleanup
/agents task --steps 1,2 --parallel --agents "Repository Analyzer,Git Surgeon"

# Step 3-5: Restructure & Modernize
/agents task --steps 3,4,5 --agents "Structure Architect,Python Modernizer,CI/CD Engineer"
```

## 📊 Progress Tracking

### Daily Milestones
| Day | Agent | Deliverable |
|-----|-------|-------------|
| D-1 | Git Surgeon | Snapshot & branch protection |
| D-2 | Repository Analyzer | .gitignore & artifact identification |
| D-3 | Git Surgeon | History rewrite complete |
| D-4 | Structure Architect | /services structure created |
| D-5 | Python Modernizer | Poetry migration complete |
| D-6 | CI/CD Engineer | Green CI pipeline |
| D-7 | Docker Captain | Container images in GHCR |
| D-8 | Quality Guardian | All tests passing |
| D-9 | Documentation Curator | MkDocs deployed |
| D-10 | Jarvis Triage Commander | Final review & handoff |

## 🔄 Coordination Patterns

### Parallel Execution
```bash
# Agents work simultaneously on independent tasks
/agents orchestrate --task "Parallel cleanup" --strategy parallel \
  --agents "Repository Analyzer,Documentation Curator,Quality Guardian"
```

### Sequential Dependencies
```bash
# Enforced order for dependent tasks
/agents orchestrate --task "Sequential restructure" --strategy sequential \
  --steps "backup,cleanup,restructure,modernize"
```

### Adaptive Coordination
```bash
# Swarm adapts based on findings
/agents orchestrate --task "Adaptive triage" --strategy adaptive \
  --threshold "500MB" --fallback "aggressive-cleanup"
```

## 🧠 Memory & Learning

### Store Progress
```bash
/agents memory store --key "jarvis/progress/day-1" \
  --value '{"completed": ["snapshot", "gitignore"], "size_reduction": "40%"}'
```

### Query Status
```bash
/agents memory search --pattern "jarvis/progress/*" --format table
```

### Train Patterns
```bash
/agents train --pattern "repository-triage" --from "jarvis-success"
```

## ✅ Success Criteria

- [ ] Repository size < 500 MB
- [ ] `poetry install` succeeds in all services
- [ ] CI pipeline green on main branch
- [ ] Docker images build successfully
- [ ] Documentation accessible via GitHub Pages
- [ ] Pre-commit hooks pass on all files
- [ ] Health endpoints respond on all services

## 🚨 Troubleshooting

### If History Rewrite Fails
```bash
/agents recover --agent "Git Surgeon" --from-backup jarvis-backup.git
```

### If Restructure Breaks Imports
```bash
/agents analyze --agent "Repository Analyzer" --check-imports
/agents fix --agent "Python Modernizer" --update-imports
```

### If CI Fails
```bash
/agents debug --agent "CI/CD Engineer" --workflow-run-id [ID]
```

## 🎯 Final Command

Execute the complete triage:
```bash
/agents orchestrate --workflow jarvis-complete-triage \
  --swarm-id swarm_1753477092608_bavfq6423 \
  --parallel-limit 4 \
  --checkpoint-every step \
  --notify-on-complete
```

This playbook transforms Jarvis from an "un-navigable dumping ground" into a clean, professional, build-able project ready for contributors!