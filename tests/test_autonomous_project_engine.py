import asyncio
import pytest
from datetime import datetime, timedelta
from core.autonomous_project_engine import AutonomousProjectEngine, ProjectContext, Task, AgentRole, AgentCapability, BaseAgent, OrchestratorAgent, AnalyzerAgent, QAAgent, ContinuousLearningSystem

# Mock Agents for testing purposes
class MockAnalyzerAgent(AnalyzerAgent):
    async def _analyze_requirements(self, task, context):
        return {"functional_requirements": ["mock_req1"], "non_functional_requirements": [], "acceptance_criteria": {}}
    async def _assess_risks(self, task, context):
        return []
    async def _estimate_complexity(self, task, context):
        return {"technical_complexity": 0.5}
    async def _generate_recommendations(self, task, context):
        return []

class MockQAAgent(QAAgent):
    async def _run_test_suite(self, test_type, patterns, task, context):
        return {"test_type": test_type, "tests_run": 10, "passed": 10, "failed": 0, "coverage": 1.0}
    async def _perform_self_healing(self, issues):
        return []

@pytest.fixture
def mock_engine():
    engine = AutonomousProjectEngine(storage_path="mock_storage")
    # Replace default agents with mock agents for predictable behavior
    engine.orchestrator.agent_registry = {} # Clear existing agents
    engine.orchestrator.register_agent(MockAnalyzerAgent("mock_analyzer_01"))
    engine.orchestrator.register_agent(MockQAAgent("mock_qa_01"))
    return engine

@pytest.mark.asyncio
async def test_project_execution_success(mock_engine):
    project_description = "Develop a simple web application"
    
    result = await mock_engine.execute_project(
        project_description=project_description,
        objectives=["Create user authentication", "Display data"],
        success_criteria={"auth_working": True, "data_displayed": True},
        domain="web_development",
        priority=7
    )
    
    assert result is not None
    assert result["status"] == "completed"
    assert "project_id" in result
    assert result["description"] == project_description
    assert result["quality_metrics"]["overall_score"] > 0
    assert len(result["learning_insights"]["agent_performances"]) > 0
    assert len(result["recommendations"]) >= 0 # Can be 0 if no issues

@pytest.mark.asyncio
async def test_agent_registration(mock_engine):
    assert "mock_analyzer_01" in mock_engine.orchestrator.agent_registry
    assert "mock_qa_01" in mock_engine.orchestrator.agent_registry
    assert isinstance(mock_engine.orchestrator.agent_registry["mock_analyzer_01"], MockAnalyzerAgent)
    assert isinstance(mock_engine.orchestrator.agent_registry["mock_qa_01"], MockQAAgent)

@pytest.mark.asyncio
async def test_analyzer_agent_execution(mock_engine):
    analyzer_agent = mock_engine.orchestrator.agent_registry["mock_analyzer_01"]
    test_task = Task(
        id="test_analysis_task",
        name="Test Analysis",
        description="Analyze a test project",
        dependencies=[],
        estimated_effort=1.0,
        required_capabilities=["requirements_analysis"]
    )
    test_context = ProjectContext(
        description="Test Project", objectives=[], constraints={}, stakeholders=[],
        success_criteria={}, domain="test", priority=5
    )
    
    analysis_result = await analyzer_agent.execute(test_task, test_context)
    
    assert "breakdown" in analysis_result
    assert "risks" in analysis_result
    assert "complexity" in analysis_result
    assert "recommendations" in analysis_result
    assert analysis_result["breakdown"]["functional_requirements"] == ["mock_req1"]

@pytest.mark.asyncio
async def test_qa_agent_execution(mock_engine):
    qa_agent = mock_engine.orchestrator.agent_registry["mock_qa_01"]
    test_task = Task(
        id="test_qa_task",
        name="Test QA",
        description="Perform QA on a test module",
        dependencies=[],
        estimated_effort=1.0,
        required_capabilities=["automated_testing"]
    )
    test_context = ProjectContext(
        description="Test QA Project", objectives=[], constraints={}, stakeholders=[],
        success_criteria={}, domain="test", priority=5, quality_standards={"code_coverage": 0.8}
    )
    
    qa_result = await qa_agent.execute(test_task, test_context)
    
    assert "test_suites" in qa_result
    assert "quality_metrics" in qa_result
    assert qa_result["quality_metrics"]["overall_score"] == 1.0
    assert qa_result["meets_standards"] == True
