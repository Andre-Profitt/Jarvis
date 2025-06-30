"""
Quick Start Module for LLM Research
Provides simple interface for immediate research capabilities
"""

import asyncio
from typing import Dict, List, Optional
import json
import subprocess
import tempfile
import os
from datetime import datetime

from .llm_research_jarvis import llm_research_jarvis


class QuickResearch:
    """Simple interface for quick LLM-enhanced research"""

    def __init__(self):
        self.initialized = False

    async def setup(self):
        """Quick setup with minimal configuration"""
        if not self.initialized:
            await llm_research_jarvis.initialize()
            self.initialized = True

    async def research(self, topic: str, depth: str = "standard") -> Dict:
        """
        Quick research on any topic

        Args:
            topic: What to research
            depth: "quick", "standard", or "comprehensive"

        Returns:
            Research results with summary, findings, and sources
        """
        await self.setup()

        print(f"\n🔬 Researching: {topic}")
        print("=" * 60)

        result = await llm_research_jarvis.research(
            topic=topic, depth=depth, type="standard"
        )

        # Print summary
        if "summary" in result:
            print(f"\n📄 Summary:\n{result['summary']}")

        # Print key findings
        if "key_findings" in result:
            print(f"\n💡 Key Findings:")
            for i, finding in enumerate(result["key_findings"][:5], 1):
                print(f"{i}. {finding['finding']}")
                print(f"   Confidence: {finding['confidence']:.2f}")

        return result

    async def compare(self, topics: List[str]) -> Dict:
        """
        Compare multiple topics

        Args:
            topics: List of topics to compare

        Returns:
            Comparative analysis results
        """
        await self.setup()

        print(f"\n📊 Comparing: {', '.join(topics)}")
        print("=" * 60)

        result = await llm_research_jarvis.research(
            topic=f"Comparison of {' vs '.join(topics)}",
            type="comparative",
            context={"topics": topics},
        )

        return result

    async def hypothesize(self, topic: str) -> Dict:
        """
        Generate and validate hypotheses about a topic

        Args:
            topic: Topic to generate hypotheses for

        Returns:
            Generated hypotheses with validation
        """
        await self.setup()

        print(f"\n🧪 Generating hypotheses for: {topic}")
        print("=" * 60)

        result = await llm_research_jarvis.research(topic=topic, type="hypothesis")

        # Print hypotheses
        if "hypotheses" in result:
            print("\n📋 Generated Hypotheses:")
            for i, hyp in enumerate(result["hypotheses"], 1):
                print(f"\n{i}. {hyp['statement']}")
                print(f"   Testable: {hyp.get('testable_prediction', 'N/A')}")
                if "validation" in hyp:
                    print(f"   Validation: {hyp['validation'][:100]}...")

        return result

    async def research_for_tool(self, tool_name: str, tool_purpose: str) -> Dict:
        """
        Research to help create a new tool

        Args:
            tool_name: Name of the tool to create
            tool_purpose: What the tool should do

        Returns:
            Research results with design recommendations
        """
        await self.setup()

        print(f"\n🔧 Researching for tool creation: {tool_name}")
        print(f"Purpose: {tool_purpose}")
        print("=" * 60)

        result = await llm_research_jarvis.research(
            topic=tool_purpose,
            type="tool",
            context={"tool_name": tool_name, "tool_purpose": tool_purpose},
        )

        # Print recommendations
        if "design_recommendations" in result:
            print(f"\n🏗️ Design Recommendations:")
            print(result["design_recommendations"])

        return result


# Convenience functions for immediate use


async def quick_research(topic: str) -> Dict:
    """One-line research function"""
    qr = QuickResearch()
    return await qr.research(topic)


async def deep_research(topic: str) -> Dict:
    """One-line deep research function"""
    qr = QuickResearch()
    return await qr.research(topic, depth="comprehensive")


async def compare_topics(*topics) -> Dict:
    """One-line topic comparison"""
    qr = QuickResearch()
    return await qr.compare(list(topics))


async def generate_hypotheses(topic: str) -> Dict:
    """One-line hypothesis generation"""
    qr = QuickResearch()
    return await qr.hypothesize(topic)


async def research_tool(name: str, purpose: str) -> Dict:
    """One-line tool research"""
    qr = QuickResearch()
    return await qr.research_for_tool(name, purpose)


# Example usage functions


async def demo_research():
    """Demo basic research capabilities"""
    print("🚀 LLM Research Quick Start Demo")
    print("================================\n")

    # Basic research
    print("1️⃣ Basic Research Example:")
    result = await quick_research("Latest advances in quantum computing")

    # Deep research
    print("\n2️⃣ Deep Research Example:")
    deep_result = await deep_research("AI safety measures")

    # Comparison
    print("\n3️⃣ Comparison Example:")
    comparison = await compare_topics(
        "Transformer architecture",
        "Recurrent neural networks",
        "Convolutional neural networks",
    )

    # Hypothesis generation
    print("\n4️⃣ Hypothesis Generation Example:")
    hypotheses = await generate_hypotheses("Brain-computer interfaces")

    # Tool research
    print("\n5️⃣ Tool Research Example:")
    tool_research = await research_tool(
        "AutoCodeReviewer", "Automatically review code for bugs and improvements"
    )

    print("\n✅ Demo complete! Research results saved.")


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_research_quickstart.py <topic>")
        print("Or: python llm_research_quickstart.py demo")
        sys.exit(1)

    if sys.argv[1] == "demo":
        asyncio.run(demo_research())
    else:
        topic = " ".join(sys.argv[1:])
        asyncio.run(quick_research(topic))
