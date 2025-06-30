#!/usr/bin/env python3
"""
LLM-Enhanced Autonomous Research Agent
Integrates Claude and Gemini CLI for advanced research capabilities
"""

import asyncio
import subprocess
import json
import aiohttp
from typing import Dict, List, Any, Optional, Protocol, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import hashlib
from abc import ABC, abstractmethod
from enum import Enum
import tempfile
import os

logger = logging.getLogger(__name__)

# ==================== LLM INTERFACES ====================


class LLMProvider(Enum):
    """Available LLM providers"""

    CLAUDE = "claude"
    GEMINI = "gemini"
    BOTH = "both"  # Use both for validation


@dataclass
class LLMResponse:
    """Response from LLM"""

    provider: LLMProvider
    content: str
    confidence: float
    reasoning: Optional[str] = None
    sources_cited: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class LLMInterface(ABC):
    """Abstract interface for LLM interactions"""

    @abstractmethod
    async def analyze(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Analyze content using LLM"""
        pass

    @abstractmethod
    async def synthesize(self, sources: List[Dict], question: str) -> LLMResponse:
        """Synthesize insights from sources"""
        pass

    @abstractmethod
    async def validate(self, claim: str, evidence: List[str]) -> LLMResponse:
        """Validate a claim against evidence"""
        pass


class ClaudeCLI(LLMInterface):
    """Claude CLI interface for research"""

    def __init__(self, model: str = "claude-3-opus-20240229"):
        self.model = model
        self.cli_command = "claude"  # Assumes claude CLI is installed

    async def analyze(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Use Claude for analysis"""
        full_prompt = self._build_analysis_prompt(prompt, context)

        try:
            result = await self._run_claude(full_prompt)
            return self._parse_response(result, LLMProvider.CLAUDE)
        except Exception as e:
            logger.error(f"Claude analysis failed: {e}")
            raise

    async def synthesize(self, sources: List[Dict], question: str) -> LLMResponse:
        """Use Claude to synthesize insights"""
        prompt = self._build_synthesis_prompt(sources, question)

        try:
            result = await self._run_claude(prompt)
            return self._parse_response(result, LLMProvider.CLAUDE)
        except Exception as e:
            logger.error(f"Claude synthesis failed: {e}")
            raise

    async def validate(self, claim: str, evidence: List[str]) -> LLMResponse:
        """Use Claude to validate claims"""
        prompt = self._build_validation_prompt(claim, evidence)

        try:
            result = await self._run_claude(prompt)
            return self._parse_response(result, LLMProvider.CLAUDE)
        except Exception as e:
            logger.error(f"Claude validation failed: {e}")
            raise

    async def _run_claude(self, prompt: str) -> str:
        """Execute Claude CLI command"""
        # Save prompt to temporary file (CLI might have length limits)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            temp_file = f.name

        try:
            # Run Claude CLI
            cmd = [
                self.cli_command,
                "--model",
                self.model,
                "--max-tokens",
                "4000",
                "--temperature",
                "0.7",
                "--file",
                temp_file,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"Claude CLI error: {stderr.decode()}")

            return stdout.decode()

        finally:
            # Clean up temp file
            os.unlink(temp_file)

    def _build_analysis_prompt(self, prompt: str, context: Optional[str]) -> str:
        """Build analysis prompt for Claude"""
        return f"""You are an expert research analyst. Analyze the following and provide deep insights.

Context: {context if context else 'General research analysis'}

Task: {prompt}

Provide your analysis in the following JSON format:
{{
    "main_insights": ["insight1", "insight2", ...],
    "confidence": 0.0-1.0,
    "reasoning": "Your detailed reasoning",
    "key_findings": ["finding1", "finding2", ...],
    "limitations": ["limitation1", "limitation2", ...],
    "recommendations": ["recommendation1", "recommendation2", ...]
}}

Be thorough, critical, and evidence-based in your analysis."""

    def _build_synthesis_prompt(self, sources: List[Dict], question: str) -> str:
        """Build synthesis prompt"""
        sources_text = "\n\n".join(
            [
                f"Source {i+1} ({s.get('type', 'unknown')}):\n"
                f"Title: {s.get('title', 'N/A')}\n"
                f"Content: {s.get('abstract', s.get('content', 'N/A'))[:500]}..."
                for i, s in enumerate(sources[:10])  # Limit to 10 sources
            ]
        )

        return f"""You are a PhD-level researcher synthesizing findings from multiple sources.

Research Question: {question}

Sources:
{sources_text}

Synthesize these sources to answer the research question. Provide:

1. A comprehensive answer combining insights from all sources
2. Areas of consensus among sources
3. Areas of disagreement or contradiction
4. Confidence level (0.0-1.0) in your synthesis
5. Gaps in the current research
6. Recommendations for further research

Format your response as JSON:
{{
    "synthesis": "Your synthesized answer",
    "consensus_points": ["point1", "point2", ...],
    "contradictions": ["contradiction1", "contradiction2", ...],
    "confidence": 0.0-1.0,
    "research_gaps": ["gap1", "gap2", ...],
    "future_research": ["suggestion1", "suggestion2", ...],
    "key_insight": "The most important takeaway"
}}"""

    def _build_validation_prompt(self, claim: str, evidence: List[str]) -> str:
        """Build validation prompt"""
        evidence_text = "\n".join([f"- {e}" for e in evidence[:10]])

        return f"""You are a critical research validator. Evaluate the following claim against the provided evidence.

Claim: {claim}

Evidence:
{evidence_text}

Critically evaluate:
1. Is the claim supported by the evidence?
2. What is the strength of the evidence?
3. Are there any logical fallacies or biases?
4. What additional evidence would strengthen/weaken the claim?

Provide your validation in JSON format:
{{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Detailed reasoning",
    "evidence_strength": "strong/moderate/weak",
    "issues_found": ["issue1", "issue2", ...],
    "additional_evidence_needed": ["evidence1", "evidence2", ...]
}}"""

    def _parse_response(self, response: str, provider: LLMProvider) -> LLMResponse:
        """Parse LLM response"""
        try:
            # Try to parse JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                return LLMResponse(
                    provider=provider,
                    content=json.dumps(data),
                    confidence=data.get("confidence", 0.7),
                    reasoning=data.get("reasoning", ""),
                )
            else:
                # Fallback for non-JSON responses
                return LLMResponse(
                    provider=provider,
                    content=response,
                    confidence=0.7,
                    reasoning="Response parsed as plain text",
                )
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return LLMResponse(
                provider=provider,
                content=response,
                confidence=0.5,
                reasoning="Failed to parse structured response",
            )


class GeminiCLI(LLMInterface):
    """Gemini CLI interface for research"""

    def __init__(self, model: str = "gemini-pro"):
        self.model = model
        self.cli_command = "gemini"  # Assumes gemini CLI is installed

    async def analyze(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Use Gemini for analysis"""
        full_prompt = self._build_analysis_prompt(prompt, context)

        try:
            result = await self._run_gemini(full_prompt)
            return self._parse_response(result, LLMProvider.GEMINI)
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise

    async def synthesize(self, sources: List[Dict], question: str) -> LLMResponse:
        """Use Gemini to synthesize insights"""
        # Similar to Claude implementation
        prompt = self._build_synthesis_prompt(sources, question)
        result = await self._run_gemini(prompt)
        return self._parse_response(result, LLMProvider.GEMINI)

    async def validate(self, claim: str, evidence: List[str]) -> LLMResponse:
        """Use Gemini to validate claims"""
        prompt = self._build_validation_prompt(claim, evidence)
        result = await self._run_gemini(prompt)
        return self._parse_response(result, LLMProvider.GEMINI)

    async def _run_gemini(self, prompt: str) -> str:
        """Execute Gemini CLI command"""
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(prompt)
            temp_file = f.name

        try:
            # Run Gemini CLI (adjust command based on actual CLI interface)
            cmd = [
                self.cli_command,
                "--model",
                self.model,
                "--temperature",
                "0.7",
                "--file",
                temp_file,
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise Exception(f"Gemini CLI error: {stderr.decode()}")

            return stdout.decode()

        finally:
            os.unlink(temp_file)

    # Use similar prompt builders as Claude
    _build_analysis_prompt = ClaudeCLI._build_analysis_prompt
    _build_synthesis_prompt = ClaudeCLI._build_synthesis_prompt
    _build_validation_prompt = ClaudeCLI._build_validation_prompt
    _parse_response = ClaudeCLI._parse_response


class DualLLMValidator(LLMInterface):
    """Use both Claude and Gemini for validation and consensus"""

    def __init__(self):
        self.claude = ClaudeCLI()
        self.gemini = GeminiCLI()

    async def analyze(self, prompt: str, context: Optional[str] = None) -> LLMResponse:
        """Analyze using both LLMs and combine results"""
        # Run both in parallel
        claude_task = self.claude.analyze(prompt, context)
        gemini_task = self.gemini.analyze(prompt, context)

        claude_response, gemini_response = await asyncio.gather(
            claude_task, gemini_task
        )

        # Combine responses
        return self._combine_responses(claude_response, gemini_response)

    async def synthesize(self, sources: List[Dict], question: str) -> LLMResponse:
        """Synthesize using both LLMs"""
        claude_response, gemini_response = await asyncio.gather(
            self.claude.synthesize(sources, question),
            self.gemini.synthesize(sources, question),
        )

        return self._combine_responses(claude_response, gemini_response)

    async def validate(self, claim: str, evidence: List[str]) -> LLMResponse:
        """Validate using both LLMs for higher confidence"""
        claude_response, gemini_response = await asyncio.gather(
            self.claude.validate(claim, evidence), self.gemini.validate(claim, evidence)
        )

        return self._combine_responses(claude_response, gemini_response)

    def _combine_responses(
        self, response1: LLMResponse, response2: LLMResponse
    ) -> LLMResponse:
        """Combine responses from both LLMs"""
        # Average confidence
        combined_confidence = (response1.confidence + response2.confidence) / 2

        # Combine content
        try:
            data1 = json.loads(response1.content)
            data2 = json.loads(response2.content)

            # Merge insights, findings, etc.
            combined_data = {
                "claude_analysis": data1,
                "gemini_analysis": data2,
                "consensus": self._find_consensus(data1, data2),
                "disagreements": self._find_disagreements(data1, data2),
                "combined_confidence": combined_confidence,
            }

            return LLMResponse(
                provider=LLMProvider.BOTH,
                content=json.dumps(combined_data),
                confidence=combined_confidence,
                reasoning="Combined analysis from Claude and Gemini",
            )
        except:
            # Fallback for non-JSON responses
            return LLMResponse(
                provider=LLMProvider.BOTH,
                content=f"Claude: {response1.content}\n\nGemini: {response2.content}",
                confidence=combined_confidence,
                reasoning="Combined responses (non-structured)",
            )

    def _find_consensus(self, data1: Dict, data2: Dict) -> List[str]:
        """Find consensus points between two analyses"""
        consensus = []

        # Compare main insights/findings
        if "main_insights" in data1 and "main_insights" in data2:
            # Simple similarity check (in practice, use embeddings)
            for insight1 in data1["main_insights"]:
                for insight2 in data2["main_insights"]:
                    if self._similar_text(insight1, insight2):
                        consensus.append(insight1)

        return consensus

    def _find_disagreements(self, data1: Dict, data2: Dict) -> List[str]:
        """Find disagreements between analyses"""
        disagreements = []

        # Check for conflicting validations
        if "is_valid" in data1 and "is_valid" in data2:
            if data1["is_valid"] != data2["is_valid"]:
                disagreements.append("Validation result differs between models")

        return disagreements

    def _similar_text(self, text1: str, text2: str) -> bool:
        """Simple similarity check (enhance with embeddings in production)"""
        # Simple word overlap check
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2) / max(len(words1), len(words2))
        return overlap > 0.5


# ==================== ENHANCED RESEARCHER ====================


class LLMEnhancedResearcher:
    """
    Autonomous researcher enhanced with LLM capabilities
    Combines API sources with LLM analysis
    """

    def __init__(
        self,
        source_plugins: List[Any],
        llm_provider: LLMProvider = LLMProvider.BOTH,
        config: Optional[Dict] = None,
    ):
        self.source_plugins = source_plugins
        self.config = config or {}

        # Initialize LLM interface
        if llm_provider == LLMProvider.CLAUDE:
            self.llm = ClaudeCLI()
        elif llm_provider == LLMProvider.GEMINI:
            self.llm = GeminiCLI()
        else:
            self.llm = DualLLMValidator()

        self.research_history = []

    async def research_with_llm(
        self,
        topic: str,
        depth: str = "comprehensive",
        custom_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Conduct research using both APIs and LLMs
        """
        logger.info(f"Starting LLM-enhanced research on: {topic}")

        # Phase 1: Generate research questions using LLM
        questions = await self._generate_questions_with_llm(
            topic, depth, custom_questions
        )

        # Phase 2: Gather sources from APIs
        all_sources = []
        for question in questions[:5]:  # Limit questions
            sources = await self._gather_sources_for_question(question)
            all_sources.extend(sources)

        # Deduplicate sources
        unique_sources = self._deduplicate_sources(all_sources)

        # Phase 3: LLM Analysis of sources
        analysis_results = await self._analyze_sources_with_llm(unique_sources, topic)

        # Phase 4: Synthesize findings
        synthesis = await self._synthesize_with_llm(
            unique_sources, analysis_results, questions, topic
        )

        # Phase 5: Validate key findings
        validated_findings = await self._validate_findings(
            synthesis.get("key_findings", []), unique_sources
        )

        # Phase 6: Generate research output
        output = await self._generate_research_output(
            topic=topic,
            questions=questions,
            sources=unique_sources,
            analysis=analysis_results,
            synthesis=synthesis,
            validated_findings=validated_findings,
        )

        # Store in history
        self.research_history.append(
            {"topic": topic, "timestamp": datetime.now(), "output": output}
        )

        return output

    async def _generate_questions_with_llm(
        self, topic: str, depth: str, custom_questions: Optional[List[str]]
    ) -> List[str]:
        """Generate research questions using LLM"""
        prompt = f"""Generate {10 if depth == 'comprehensive' else 5} research questions for the topic: "{topic}"

The questions should:
1. Cover different aspects (theoretical, practical, limitations, future)
2. Be specific and answerable through research
3. Build upon each other for comprehensive understanding

Depth level: {depth}

Format as JSON:
{{
    "questions": ["question1", "question2", ...],
    "rationale": "Why these questions matter"
}}"""

        response = await self.llm.analyze(prompt, f"Research depth: {depth}")

        try:
            data = json.loads(response.content)
            questions = data.get("questions", [])

            # Add custom questions if provided
            if custom_questions:
                questions.extend(custom_questions)

            return questions
        except:
            # Fallback questions
            return [
                f"What is the current state of {topic}?",
                f"What are the main challenges in {topic}?",
                f"What are future directions for {topic}?",
            ]

    async def _gather_sources_for_question(self, question: str) -> List[Dict]:
        """Gather sources for a specific question"""
        all_sources = []

        # Search each plugin
        search_tasks = []
        for plugin in self.source_plugins:
            task = plugin.search(question, limit=5)
            search_tasks.append(task)

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_sources.extend(result)

        return all_sources

    def _deduplicate_sources(self, sources: List[Dict]) -> List[Dict]:
        """Remove duplicate sources"""
        seen = set()
        unique = []

        for source in sources:
            source_id = source.get("id") or source.get("url", "")
            if source_id and source_id not in seen:
                seen.add(source_id)
                unique.append(source)

        return unique

    async def _analyze_sources_with_llm(
        self, sources: List[Dict], topic: str
    ) -> Dict[str, Any]:
        """Use LLM to analyze sources"""
        # Batch sources for analysis (LLMs have context limits)
        batch_size = 5
        all_analyses = []

        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]

            prompt = f"""Analyze these sources related to "{topic}" and extract key insights:

Sources:
{json.dumps(batch, indent=2)}

Provide:
1. Key insights from each source
2. Quality assessment of each source
3. Connections between sources
4. Gaps or contradictions

Format as JSON with source IDs as keys."""

            response = await self.llm.analyze(prompt, f"Analyzing sources for: {topic}")
            all_analyses.append(response)

        return {"batch_analyses": all_analyses, "total_sources": len(sources)}

    async def _synthesize_with_llm(
        self, sources: List[Dict], analysis: Dict, questions: List[str], topic: str
    ) -> Dict[str, Any]:
        """Synthesize findings using LLM"""
        # Prepare synthesis context
        context = {
            "topic": topic,
            "num_sources": len(sources),
            "questions": questions[:5],  # Top 5 questions
        }

        synthesis_response = await self.llm.synthesize(sources[:10], topic)

        try:
            synthesis_data = json.loads(synthesis_response.content)
            return synthesis_data
        except:
            return {
                "synthesis": synthesis_response.content,
                "confidence": synthesis_response.confidence,
            }

    async def _validate_findings(
        self, findings: List[str], sources: List[Dict]
    ) -> List[Dict]:
        """Validate key findings against sources"""
        validated = []

        for finding in findings[:5]:  # Validate top 5 findings
            # Extract evidence from sources
            evidence = self._extract_evidence_for_finding(finding, sources)

            validation_response = await self.llm.validate(finding, evidence)

            try:
                validation_data = json.loads(validation_response.content)
                validated.append(
                    {
                        "finding": finding,
                        "is_valid": validation_data.get("is_valid", False),
                        "confidence": validation_data.get("confidence", 0),
                        "evidence_strength": validation_data.get(
                            "evidence_strength", "unknown"
                        ),
                    }
                )
            except:
                validated.append(
                    {
                        "finding": finding,
                        "is_valid": None,
                        "confidence": 0.5,
                        "evidence_strength": "unclear",
                    }
                )

        return validated

    def _extract_evidence_for_finding(
        self, finding: str, sources: List[Dict]
    ) -> List[str]:
        """Extract relevant evidence for a finding"""
        evidence = []

        # Simple keyword matching (enhance with embeddings)
        finding_words = set(finding.lower().split())

        for source in sources[:10]:  # Check top 10 sources
            abstract = source.get("abstract", "").lower()
            title = source.get("title", "").lower()

            # Check relevance
            text = f"{title} {abstract}"
            text_words = set(text.split())

            if len(finding_words & text_words) > 2:  # At least 2 words match
                evidence.append(
                    f"{source.get('title', 'Unknown')}: {abstract[:200]}..."
                )

        return evidence

    async def _generate_research_output(
        self,
        topic: str,
        questions: List[str],
        sources: List[Dict],
        analysis: Dict,
        synthesis: Dict,
        validated_findings: List[Dict],
    ) -> Dict[str, Any]:
        """Generate comprehensive research output"""
        # Create research summary using LLM
        summary_prompt = f"""Create an executive summary for research on "{topic}"

Key findings: {json.dumps(validated_findings[:3], indent=2)}
Number of sources: {len(sources)}
Confidence level: {synthesis.get('confidence', 0.7)}

Provide a 2-3 paragraph executive summary."""

        summary_response = await self.llm.analyze(summary_prompt)

        return {
            "topic": topic,
            "executive_summary": summary_response.content,
            "research_questions": questions,
            "sources": {
                "total": len(sources),
                "by_type": self._count_sources_by_type(sources),
                "top_sources": sources[:10],
            },
            "key_findings": validated_findings,
            "synthesis": synthesis,
            "confidence": synthesis.get("confidence", 0.7),
            "llm_provider": self.llm.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
        }

    def _count_sources_by_type(self, sources: List[Dict]) -> Dict[str, int]:
        """Count sources by type"""
        counts = {}
        for source in sources:
            source_type = source.get("source_type", "unknown")
            counts[source_type] = counts.get(source_type, 0) + 1
        return counts

    async def compare_findings_across_topics(self, topics: List[str]) -> Dict[str, Any]:
        """Compare research findings across multiple topics"""
        comparison_prompt = f"""Compare research findings across these topics: {topics}

Previous research summaries:
{json.dumps([r['output']['executive_summary'] for r in self.research_history[-len(topics):]], indent=2)}

Identify:
1. Common themes
2. Contradictions
3. Complementary insights
4. Research gaps

Format as structured analysis."""

        comparison = await self.llm.analyze(comparison_prompt)

        return {
            "topics": topics,
            "comparison": comparison.content,
            "timestamp": datetime.now().isoformat(),
        }


# ==================== USAGE EXAMPLE ====================


async def demo_llm_enhanced_research():
    """Demonstrate LLM-enhanced research"""

    # Mock source plugins for demo
    class MockArxivPlugin:
        async def search(self, query: str, limit: int = 5) -> List[Dict]:
            # Simulate API response
            return [
                {
                    "id": f"arxiv_{i}",
                    "title": f"Paper on {query} - Study {i}",
                    "abstract": f"This paper investigates {query} using advanced methods...",
                    "authors": ["Author A", "Author B"],
                    "source_type": "arxiv",
                    "url": f"https://arxiv.org/abs/2024.{i:04d}",
                }
                for i in range(1, min(limit + 1, 4))
            ]

    # Initialize researcher with LLM
    researcher = LLMEnhancedResearcher(
        source_plugins=[MockArxivPlugin()],
        llm_provider=LLMProvider.BOTH,  # Use both Claude and Gemini
    )

    # Conduct research
    print("🔬 Starting LLM-Enhanced Research")
    print("=" * 60)

    topic = "Improving LLM Reasoning through Multi-Agent Debate"

    result = await researcher.research_with_llm(
        topic=topic,
        depth="comprehensive",
        custom_questions=[
            "How does multi-agent debate reduce hallucinations?",
            "What are the computational costs of multi-agent systems?",
        ],
    )

    # Display results
    print(f"\n📊 Research Complete!")
    print(f"Topic: {result['topic']}")
    print(f"\n📝 Executive Summary:")
    print(result["executive_summary"])

    print(f"\n🔍 Key Validated Findings:")
    for finding in result["key_findings"]:
        status = "✓" if finding["is_valid"] else "✗"
        print(f"{status} {finding['finding']}")
        print(f"   Confidence: {finding['confidence']:.2%}")
        print(f"   Evidence: {finding['evidence_strength']}")

    print(f"\n📚 Sources:")
    print(f"Total sources analyzed: {result['sources']['total']}")
    print(f"By type: {result['sources']['by_type']}")

    print(f"\n🤖 LLM Analysis:")
    print(f"Provider: {result['llm_provider']}")
    print(f"Overall confidence: {result['confidence']:.2%}")

    # Compare with another topic
    print("\n\n🔄 Comparing with related topic...")

    result2 = await researcher.research_with_llm(
        topic="Self-Reflection in Large Language Models", depth="standard"
    )

    comparison = await researcher.compare_findings_across_topics(
        [topic, "Self-Reflection in Large Language Models"]
    )

    print("\n📊 Cross-Topic Analysis:")
    print(comparison["comparison"])


if __name__ == "__main__":
    asyncio.run(demo_llm_enhanced_research())
