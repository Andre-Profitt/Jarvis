#!/usr/bin/env python3
"""
Demo Script for LLM Research Capabilities
Shows how to use JARVIS's advanced research features
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.llm_research_quickstart import (
    quick_research,
    deep_research,
    compare_topics,
    generate_hypotheses,
    research_tool,
)


async def main():
    """Run research demos"""

    print(
        """
    ╔══════════════════════════════════════════════════╗
    ║     🔬 JARVIS LLM Research Demo 🔬               ║
    ║  Advanced Research with Claude & Gemini CLI      ║
    ╚══════════════════════════════════════════════════╝
    """
    )

    # Demo 1: Quick Research
    print("\n" + "=" * 60)
    print("DEMO 1: Quick Research on Quantum Computing")
    print("=" * 60)

    quantum_result = await quick_research("Latest quantum computing breakthroughs 2024")

    # Demo 2: Deep Research
    print("\n" + "=" * 60)
    print("DEMO 2: Deep Research on AI Safety")
    print("=" * 60)

    safety_result = await deep_research("AI alignment and safety measures")

    # Demo 3: Comparative Analysis
    print("\n" + "=" * 60)
    print("DEMO 3: Comparing ML Architectures")
    print("=" * 60)

    comparison = await compare_topics(
        "Transformer models", "Graph Neural Networks", "Diffusion models"
    )

    # Demo 4: Hypothesis Generation
    print("\n" + "=" * 60)
    print("DEMO 4: Generating Research Hypotheses")
    print("=" * 60)

    hypotheses = await generate_hypotheses("Neuromorphic computing applications")

    # Demo 5: Tool Research
    print("\n" + "=" * 60)
    print("DEMO 5: Research for Tool Creation")
    print("=" * 60)

    tool_research_result = await research_tool(
        "SmartCodeOptimizer",
        "Automatically optimize Python code for performance and readability",
    )

    # Summary
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)
    print(
        """
    The LLM Research system can:
    
    1. 🔍 Conduct quick or deep research on any topic
    2. 📊 Compare multiple concepts or approaches
    3. 🧪 Generate and validate research hypotheses
    4. 🔧 Research best practices for tool creation
    5. 📚 Access ArXiv, Semantic Scholar, and CrossRef
    6. 🤖 Use Claude and Gemini for PhD-level analysis
    7. ✅ Cross-validate findings with multiple LLMs
    8. 🧠 Integrate with neural resource allocation
    
    All research is stored in JARVIS's knowledge base for future use!
    """
    )


if __name__ == "__main__":
    asyncio.run(main())
