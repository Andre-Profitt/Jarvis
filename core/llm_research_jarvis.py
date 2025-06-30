"""
LLM Research Integration with JARVIS Core Systems
Provides advanced research capabilities using Claude and Gemini CLI
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import hashlib

from .llm_research_integration import (
    LLMEnhancedResearcher,
    LLMProvider,
    ClaudeCLI,
    GeminiCLI,
    DualLLMValidator as DualLLM,
    # ArxivPlugin,  # Not implemented yet
    # SemanticScholarPlugin,  # Not implemented yet
    # CrossRefPlugin,  # Not implemented yet
    # APIConfig  # Not implemented yet
)
from .world_class_swarm import WorldClassSwarmSystem
from .neural_integration import neural_jarvis
from .self_healing_integration import self_healing_jarvis
from .database import DatabaseManager as Database
from .monitoring import MonitoringService as MonitoringSystem

logger = logging.getLogger(__name__)


@dataclass
class ResearchTask:
    """Research task with metadata"""

    id: str
    topic: str
    type: str = "standard"  # standard, deep, comparative, hypothesis, tool
    priority: float = 1.0
    depth: str = "standard"  # quick, standard, comprehensive
    custom_questions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class LLMResearchJARVIS:
    """Integrates LLM Research capabilities with JARVIS ecosystem"""

    def __init__(self):
        self.researcher = None
        self.swarm_system = None
        self.db = Database()
        self.monitoring = MonitoringSystem()
        self._initialized = False
        self._research_cache = {}
        self._active_tasks = {}

    async def initialize(self, api_config: Optional[Dict[str, str]] = None):
        """Initialize LLM research integration"""
        if self._initialized:
            return

        logger.info("Initializing LLM Research JARVIS Integration...")

        # Initialize API config
        if not api_config:
            api_config = {
                "arxiv_api_key": "",  # ArXiv doesn't require key
                "s2_api_key": "",  # Get from Semantic Scholar
                "crossref_email": "",  # Your email for CrossRef
            }

        # Initialize LLM interfaces
        claude_cli = ClaudeCLI()
        gemini_cli = GeminiCLI()
        dual_llm = DualLLM(claude_cli, gemini_cli)

        # Initialize source plugins
        source_plugins = [
            ArxivPlugin(APIConfig(**api_config)),
            SemanticScholarPlugin(APIConfig(**api_config)),
            CrossRefPlugin(APIConfig(**api_config)),
        ]

        # Create researcher
        self.researcher = LLMEnhancedResearcher(
            source_plugins=source_plugins, llm_provider=LLMProvider.BOTH
        )
        self.researcher.llm = dual_llm

        # Connect to swarm system if available
        try:
            self.swarm_system = WorldClassSwarmSystem()
            await self.swarm_system.initialize()
            await self._register_research_agents()
        except Exception as e:
            logger.warning(f"Swarm system unavailable: {e}")

        # Setup monitoring
        self._setup_monitoring()

        self._initialized = True
        logger.info("LLM Research JARVIS Integration initialized successfully")

    def _setup_monitoring(self):
        """Setup monitoring for research activities"""
        self.monitoring.register_metric(
            "research_tasks_active", lambda: len(self._active_tasks)
        )
        self.monitoring.register_metric(
            "research_cache_size", lambda: len(self._research_cache)
        )
        self.monitoring.register_metric(
            "research_tasks_completed",
            lambda: sum(
                1 for t in self._active_tasks.values() if t.status == "completed"
            ),
        )

    async def _register_research_agents(self):
        """Register specialized research agents with swarm"""
        if not self.swarm_system:
            return

        # Create research specialist agent
        research_agent = await self.swarm_system.create_agent(
            "research_specialist",
            capabilities={"deep_research", "llm_analysis", "source_synthesis"},
        )

        # Create validation agent
        validation_agent = await self.swarm_system.create_agent(
            "validation_specialist",
            capabilities={"cross_validation", "fact_checking", "consensus_building"},
        )

        logger.info("Research agents registered with swarm system")

    async def research(self, topic: str, **kwargs) -> Dict[str, Any]:
        """Conduct research on a topic with neural resource allocation"""

        # Create research task
        task = ResearchTask(
            id=self._generate_task_id(topic),
            topic=topic,
            type=kwargs.get("type", "standard"),
            depth=kwargs.get("depth", "standard"),
            custom_questions=kwargs.get("custom_questions", []),
            context=kwargs.get("context", {}),
        )

        # Check cache
        cache_key = self._get_cache_key(task)
        if cache_key in self._research_cache:
            logger.info(f"Returning cached research for: {topic}")
            return self._research_cache[cache_key]

        # Allocate neural resources
        neural_allocation = await self._allocate_neural_resources(task)

        try:
            # Track active task
            self._active_tasks[task.id] = task

            # Execute research based on type
            if task.type == "deep":
                result = await self._deep_research(task)
            elif task.type == "comparative":
                result = await self._comparative_research(task)
            elif task.type == "hypothesis":
                result = await self._hypothesis_research(task)
            elif task.type == "tool":
                result = await self._tool_research(task)
            else:
                result = await self._standard_research(task)

            # Update task
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = result

            # Cache result
            self._research_cache[cache_key] = result

            # Store in database
            await self._store_research(task, result)

            # Monitor for anomalies
            await self._check_research_quality(result)

            return result

        except Exception as e:
            logger.error(f"Research failed for {topic}: {e}")
            task.status = "failed"
            raise
        finally:
            # Release neural resources
            if neural_allocation:
                await neural_jarvis.neural_manager.release_resources(
                    neural_allocation["allocated_resources"]
                )
            # Remove from active tasks
            self._active_tasks.pop(task.id, None)

    async def _allocate_neural_resources(self, task: ResearchTask) -> Dict[str, Any]:
        """Allocate neural resources for research task"""

        # Determine resource requirements based on task
        requirements = {
            "task_type": f"research_{task.type}",
            "priority": task.priority,
            "language_complexity": 0.9,  # High for research
            "reasoning_depth": 0.8 if task.depth == "comprehensive" else 0.5,
            "memory_requirements": 0.7,
            "attention_heads": 0.8,
        }

        # Add specific requirements based on research type
        if task.type == "comparative":
            requirements["spatial_processing"] = 0.7  # For comparing concepts
        elif task.type == "hypothesis":
            requirements["creativity_level"] = 0.8  # For generating hypotheses

        # Allocate resources
        allocation = await neural_jarvis.neural_manager.allocate_resources(requirements)

        logger.info(
            f"Allocated {len(allocation['allocated_resources'])} neural resources for research"
        )
        return allocation

    async def _standard_research(self, task: ResearchTask) -> Dict[str, Any]:
        """Standard research using LLM enhancement"""

        result = await self.researcher.research_with_llm(
            topic=task.topic, depth=task.depth, custom_questions=task.custom_questions
        )

        # Enhance with JARVIS context
        result["jarvis_context"] = {
            "task_id": task.id,
            "neural_resources_used": True,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    async def _deep_research(self, task: ResearchTask) -> Dict[str, Any]:
        """Deep comprehensive research with iterative refinement"""

        # Initial research
        initial = await self.researcher.research_with_llm(
            topic=task.topic, depth="comprehensive"
        )

        # Identify gaps
        gaps = initial["synthesis"].get("research_gaps", [])

        # Deep dive into gaps
        gap_research = []
        for gap in gaps[:3]:  # Limit to top 3 gaps
            gap_result = await self.researcher.research_with_llm(
                topic=gap, depth="standard"
            )
            gap_research.append(gap_result)

        # Synthesize all findings
        combined_findings = initial["key_findings"] + [
            f for gr in gap_research for f in gr.get("key_findings", [])
        ]

        # Final synthesis with all data
        final_synthesis = await self.researcher.llm.synthesize(
            combined_findings,
            f"Provide a comprehensive synthesis of research on {task.topic}",
        )

        return {
            "initial_research": initial,
            "gap_research": gap_research,
            "combined_findings": combined_findings,
            "final_synthesis": final_synthesis.content,
            "research_depth": "comprehensive_with_gaps",
        }

    async def _comparative_research(self, task: ResearchTask) -> Dict[str, Any]:
        """Compare multiple topics or approaches"""

        topics = task.context.get("topics", [task.topic])

        # Research each topic
        results = []
        for topic in topics:
            result = await self.researcher.research_with_llm(
                topic=topic, depth=task.depth
            )
            results.append(result)

        # Compare findings
        comparison = await self.researcher.compare_findings_across_topics(
            [(r["topic"], r["key_findings"]) for r in results]
        )

        return {
            "individual_research": results,
            "comparison": comparison,
            "topics_compared": topics,
        }

    async def _hypothesis_research(self, task: ResearchTask) -> Dict[str, Any]:
        """Generate and validate hypotheses"""

        # Initial research
        research = await self.researcher.research_with_llm(
            topic=task.topic, depth=task.depth
        )

        # Generate hypotheses
        hypothesis_prompt = f"""Based on research about {task.topic}, 
        generate 3 testable hypotheses that could advance the field.
        
        Current findings: {json.dumps(research['key_findings'][:5])}
        
        Format as JSON with:
        - statement: The hypothesis
        - testable_prediction: What we would observe if true
        - required_evidence: What evidence would validate/invalidate
        """

        hypotheses_response = await self.researcher.llm.analyze(hypothesis_prompt)
        hypotheses = json.loads(hypotheses_response.content)

        # Validate each hypothesis
        validated_hypotheses = []
        for hyp in hypotheses.get("hypotheses", []):
            # Search for evidence
            evidence_sources = await self.researcher._gather_sources_for_question(
                hyp["statement"]
            )

            # Validate
            validation = await self.researcher.llm.validate(
                hyp["statement"], [s["abstract"] for s in evidence_sources[:5]]
            )

            hyp["validation"] = validation.content
            hyp["evidence_count"] = len(evidence_sources)
            validated_hypotheses.append(hyp)

        return {
            "base_research": research,
            "hypotheses": validated_hypotheses,
            "methodology": "hypothesis_generation_and_validation",
        }

    async def _tool_research(self, task: ResearchTask) -> Dict[str, Any]:
        """Research for autonomous tool creation"""

        tool_name = task.context.get("tool_name", "unknown_tool")
        tool_purpose = task.context.get("tool_purpose", task.topic)

        # Research existing approaches
        existing_research = await self.researcher.research_with_llm(
            topic=f"existing tools and methods for {tool_purpose}", depth="standard"
        )

        # Research implementation techniques
        implementation_research = await self.researcher.research_with_llm(
            topic=f"implementation techniques for {tool_purpose}", depth="standard"
        )

        # Generate tool design recommendations
        design_prompt = f"""Based on research about {tool_purpose}, 
        provide design recommendations for creating {tool_name}.
        
        Existing approaches: {json.dumps(existing_research['key_findings'][:3])}
        Implementation insights: {json.dumps(implementation_research['key_findings'][:3])}
        
        Provide:
        1. Architecture recommendations
        2. Key algorithms to implement
        3. Best practices
        4. Potential pitfalls to avoid
        """

        design_recommendations = await self.researcher.llm.analyze(design_prompt)

        return {
            "tool_name": tool_name,
            "tool_purpose": tool_purpose,
            "existing_approaches": existing_research,
            "implementation_techniques": implementation_research,
            "design_recommendations": design_recommendations.content,
            "research_type": "tool_creation_research",
        }

    async def _check_research_quality(self, result: Dict[str, Any]):
        """Check research quality and trigger healing if needed"""

        # Quality metrics
        quality_score = 0.0

        # Check for key components
        if "key_findings" in result and len(result["key_findings"]) > 0:
            quality_score += 0.3
        if "synthesis" in result and result["synthesis"]:
            quality_score += 0.3
        if "sources" in result and len(result["sources"]) >= 5:
            quality_score += 0.2
        if "validation" in result:
            quality_score += 0.2

        # Check with self-healing if quality is low
        if quality_score < 0.6:
            anomaly = {
                "type": "research_quality",
                "severity": 1.0 - quality_score,
                "component": "llm_research",
                "metrics": {"quality_score": quality_score},
            }

            # Report to self-healing system
            logger.warning(f"Low quality research detected: {quality_score}")
            # self_healing_jarvis would handle this

    async def _store_research(self, task: ResearchTask, result: Dict[str, Any]):
        """Store research results in database"""

        record = {
            "task_id": task.id,
            "topic": task.topic,
            "type": task.type,
            "depth": task.depth,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": (
                task.completed_at.isoformat() if task.completed_at else None
            ),
            "result_summary": {
                "findings_count": len(result.get("key_findings", [])),
                "sources_count": len(result.get("sources", [])),
                "has_synthesis": "synthesis" in result,
                "has_validation": "validation" in result,
            },
            "full_result": result,
        }

        await self.db.store("research_results", record)

    def _generate_task_id(self, topic: str) -> str:
        """Generate unique task ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(f"{topic}:{timestamp}".encode()).hexdigest()[:12]

    def _get_cache_key(self, task: ResearchTask) -> str:
        """Generate cache key for research task"""
        key_parts = [
            task.topic,
            task.type,
            task.depth,
            str(sorted(task.custom_questions)),
        ]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()

    async def monitor_topic(self, topic: str, callback=None):
        """Monitor a topic for new developments"""

        logger.info(f"Starting monitoring for topic: {topic}")

        while True:
            try:
                # Get recent sources
                recent_sources = await self.researcher._get_recent_sources(topic)

                if recent_sources:
                    # Analyze new developments
                    analysis = await self.researcher.llm.analyze(
                        f"What are the key developments in these new papers about {topic}?",
                        context=json.dumps(recent_sources[:5]),
                    )

                    # Check for breakthroughs
                    if any(
                        word in analysis.content.lower()
                        for word in ["breakthrough", "novel", "significant"]
                    ):

                        alert = {
                            "topic": topic,
                            "timestamp": datetime.now().isoformat(),
                            "analysis": analysis.content,
                            "sources": recent_sources[:3],
                        }

                        # Store alert
                        await self.db.store("research_alerts", alert)

                        # Callback if provided
                        if callback:
                            await callback(alert)

                        logger.info(f"Significant development detected in {topic}")

                # Wait before next check
                await asyncio.sleep(3600)  # Check hourly

            except Exception as e:
                logger.error(f"Error monitoring topic {topic}: {e}")
                await asyncio.sleep(300)  # Retry in 5 minutes

    async def get_research_status(self) -> Dict[str, Any]:
        """Get current research system status"""

        active_tasks = [
            {
                "id": t.id,
                "topic": t.topic,
                "type": t.type,
                "status": t.status,
                "created_at": t.created_at.isoformat(),
            }
            for t in self._active_tasks.values()
        ]

        return {
            "initialized": self._initialized,
            "active_tasks": active_tasks,
            "cache_size": len(self._research_cache),
            "capabilities": {
                "llm_providers": ["claude", "gemini"],
                "source_apis": ["arxiv", "semantic_scholar", "crossref"],
                "research_types": [
                    "standard",
                    "deep",
                    "comparative",
                    "hypothesis",
                    "tool",
                ],
            },
            "timestamp": datetime.now().isoformat(),
        }


# Global instance
llm_research_jarvis = LLMResearchJARVIS()


async def initialize_llm_research(api_config: Optional[Dict[str, str]] = None):
    """Initialize the global LLM research integration"""
    await llm_research_jarvis.initialize(api_config)
    return llm_research_jarvis
