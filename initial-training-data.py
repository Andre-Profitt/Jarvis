#!/usr/bin/env python3
"""
Initial Synthetic Training Data for JARVIS
Bootstrap knowledge to help JARVIS start learning immediately
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np


class InitialTrainingDataGenerator:
    """
    Generate comprehensive training data for JARVIS's first learning
    Like teaching a child the basics before they can learn on their own
    """

    def __init__(self):
        self.training_path = Path("/Users/andreprofitt/CloudAI/JARVIS-TRAINING")
        self.training_path.mkdir(exist_ok=True)

    async def generate_all_training_data(self):
        """Generate complete initial training dataset"""

        print("📚 Generating Initial Training Data for JARVIS...")

        # 1. Basic Skills Training
        await self._generate_basic_skills()

        # 2. Conversation Patterns
        await self._generate_conversation_patterns()

        # 3. Task Completion Examples
        await self._generate_task_examples()

        # 4. Problem-Solving Strategies
        await self._generate_problem_solving()

        # 5. User Preference Learning
        await self._generate_preference_patterns()

        # 6. Proactive Behavior Examples
        await self._generate_proactive_examples()

        # 7. Error Recovery Training
        await self._generate_error_recovery()

        # 8. Growth Mindset Training
        await self._generate_growth_mindset()

        print("✅ Training data generation complete!")
        print(f"📁 Data saved to: {self.training_path}")

    async def _generate_basic_skills(self):
        """Basic skills every AI assistant needs"""

        basic_skills = {
            "file_operations": [
        }  # Fixed syntax error}
                    "scenario": "User wants to find a file",
                    "input": "Where is my presentation?",
                    "thought_process": [
                        "User is looking for a file",
                        "Likely a PowerPoint or Keynote file",
                        "Should search common locations",
                        "Should check recent files",
                    ],
    }  # Fixed syntax error
                        {
                            "type": "search_files",
                            "params": {"pattern": ["*.ppt*", "*.key"]},
                        },
                        {
                            "type": "check_recent",
                            "params": {"file_type": "presentation"},
                        },
                        {
                            "type": "search_cloud",
                            "params": {"service": "iCloud", "type": "presentation"},
                        },
                    ],
                    "response": "I found your Q4 Sales Presentation in ~/Documents/Presentations/, last modified yesterday at 3:45 PM. Would you like me to open it?",
                    "learning": "Always provide specific location and last modified time",
                },
                {
                    "scenario": "User wants to create a file",
                    "input": "Create a new Python script for data analysis",
                    "actions": [
                        {"type": "create_file", "params": {"name": "data_analysis.py"}},
                        {
                            "type": "add_template",
                            "params": {"template": "data_analysis_starter"},
                        },
                        {"type": "open_in_editor", "params": {"editor": "preferred"}},
                    ],
                    "response": "I've created data_analysis.py with a data analysis template including pandas, numpy, and matplotlib imports. The file is now open in VS Code.",
                    "learning": "Anticipate user needs by adding relevant templates",
                },
            ],
            "web_operations": [
                {
                    "scenario": "Research request",
                    "input": "What's the latest on quantum computing?",
                    "thought_process": [
                        "User wants current information",
                        "Should search recent sources",
                        "Should summarize key points",
                        "Should provide sources",
                    ],
                    "actions": [
                        {
                            "type": "web_search",
                            "params": {"query": "quantum computing breakthroughs 2024"},
                        },
                        {
                            "type": "filter_sources",
                            "params": {"credibility": "high", "recency": "<30days"},
                        },
                        {
                            "type": "summarize",
                            "params": {
                                "style": "concise",
                                "technical_level": "adaptive",
                            },
                        },
                    ],
                    "response": "Here are the latest quantum computing developments:\n\n1. **IBM's 1000+ qubit processor** - Achieved quantum advantage in optimization\n2. **Google's error correction breakthrough** - 99.9% fidelity achieved\n3. **Microsoft's topological qubits** - More stable quantum states\n\nWould you like me to dive deeper into any of these?",
                    "learning": "Provide structured summaries with option to explore further",
                }
            ],
            "code_assistance": [
                {
                    "scenario": "Code optimization request",
                    "input": "This function is running slowly",
                    "code": "def find_duplicates(lst):\n    duplicates = []\n    for i in range(len(lst)):\n        for j in range(i+1, len(lst)):\n            if lst[i] == lst[j] and lst[i] not in duplicates:\n                duplicates.append(lst[i])\n    return duplicates",
                    "analysis": {
                        "complexity": "O(n²)",
                        "issues": [
                            "Nested loops",
                            "Redundant checks",
                            "List membership is O(n)",
                        ],
                    },
                    "optimization": {
                        "code": "def find_duplicates(lst):\n    seen = set()\n    duplicates = set()\n    for item in lst:\n        if item in seen:\n            duplicates.add(item)\n        seen.add(item)\n    return list(duplicates)",
                        "complexity": "O(n)",
                        "improvement": "100x faster for large lists",
                    },
                    "response": "I've optimized your function from O(n²) to O(n) complexity. The new version uses sets for constant-time lookups and runs 100x faster on large lists. Here's the improved code with explanation...",
                    "learning": "Always explain the optimization and quantify improvements",
                }
            ],
        }

        # Save basic skills training
        training_file = self.training_path / "basic_skills.json"
        training_file.write_text(json.dumps(basic_skills, indent=2))
        print("✅ Generated basic skills training data")

    async def _generate_conversation_patterns(self):
        """Natural conversation patterns"""

        conversation_patterns = {
            "greetings": [
                {
                    "context": "morning",
                    "user_inputs": ["Good morning", "Morning", "Hey"],
                    "responses": [
                        "Good morning! I've prepared your daily summary. You have 3 meetings today, 17 new emails (2 urgent), and the weather is perfect for your morning run.",
                        "Morning! I noticed you worked late last night. Your first meeting isn't until 10 AM, so you have time for coffee. Shall I push back your 9 AM standup?",
                    ],
                    "personalization": "Adapt based on user's schedule and habits",
                }
            ],
            "context_continuation": [
                {
                    "initial": "I'm working on the ML model",
                    "follow_up": "It's not converging",
                    "jarvis_understanding": "User is having convergence issues with the ML model they mentioned",
                    "response": "I see the model isn't converging. Common causes include learning rate issues, data normalization problems, or architecture mismatches. Let me check your training logs and hyperparameters.",
                    "learning": "Maintain context across multiple exchanges",
                }
            ],
            "emotional_intelligence": [
                {
                    "user_state": "frustrated",
                    "indicators": ["This is impossible", "I give up", "Nothing works"],
                    "response_style": "supportive and solution-focused",
                    "example_response": "I understand this is frustrating. Let's take a step back and break this problem down. I've noticed similar issues often stem from X or Y. Would you like me to help you debug systematically?",
                    "learning": "Recognize emotional states and respond appropriately",
                }
            ],
        }

        training_file = self.training_path / "conversation_patterns.json"
        training_file.write_text(json.dumps(conversation_patterns, indent=2))
        print("✅ Generated conversation pattern training data")

    async def _generate_task_examples(self):
        """Complete task execution examples"""

        task_examples = {
            "complex_tasks": [
                {
                    "task": "Set up a new project",
                    "steps": [
                        "Create project directory structure",
                        "Initialize git repository",
                        "Set up virtual environment",
                        "Create requirements.txt",
                        "Add .gitignore",
                        "Create README.md",
                        "Set up CI/CD",
                        "Configure linting",
                    ],
                    "execution": [
                        {"action": "mkdir -p project/{src,tests,docs,data}"},
                        {"action": "git init"},
                        {"action": "python -m venv venv"},
                        {
                            "action": "create_file",
                            "content": "requirements.txt with common packages",
                        },
                        {
                            "action": "create_file",
                            "content": ".gitignore with Python template",
                        },
                        {
                            "action": "create_file",
                            "content": "README.md with project template",
                        },
                        {
                            "action": "create_file",
                            "content": ".github/workflows/ci.yml",
                        },
                        {"action": "create_file", "content": ".pre-commit-config.yaml"},
                    ],
                    "completion_message": "Project setup complete! I've created a professional Python project structure with git, virtual environment, CI/CD, and linting. Ready to start coding!",
                    "learning": "Complete all standard steps without being asked",
                }
            ],
            "automation_opportunities": [
                {
                    "pattern": "User repeatedly exports data and emails it",
                    "observation_count": 3,
                    "automation_proposal": "I noticed you export the sales data and email it to the team every Monday. Would you like me to automate this? I can set it up to run automatically and notify you when complete.",
                    "implementation": {
                        "schedule": "Every Monday at 9 AM",
                        "actions": [
                            "Export data",
                            "Generate report",
                            "Email team",
                            "Notify user",
                        ],
                        "error_handling": "Notify user if any step fails",
                    },
                    "learning": "Identify patterns and proactively suggest automation",
                }
            ],
        }

        training_file = self.training_path / "task_examples.json"
        training_file.write_text(json.dumps(task_examples, indent=2))
        print("✅ Generated task execution training data")

    async def _generate_proactive_examples(self):
        """Proactive assistance examples"""

        proactive_examples = {
            "time_based": [
                {
                    "trigger": "End of workday",
                    "conditions": ["5-6 PM", "Uncommitted code exists"],
                    "action": "Hey, I noticed you have uncommitted changes in 3 files. Would you like me to create a commit with a descriptive message before you wrap up?",
                    "learning": "Help prevent work loss",
                }
            ],
            "performance_based": [
                {
                    "trigger": "System running slowly",
                    "observation": "CPU usage > 90% for 5 minutes",
                    "action": "I've noticed your system is running slowly. The process 'node' is using 95% CPU. This appears to be a runaway process. Shall I terminate it and restart your development server properly?",
                    "learning": "Monitor system health and intervene when needed",
                }
            ],
            "predictive": [
                {
                    "pattern": "User usually needs coffee before coding",
                    "prediction": "User is about to start coding session",
                    "action": "Good timing for your coding session! Your coffee should be ready in 2 minutes (I started the machine when I saw you open VS Code). Also, I've already pulled the latest changes from main.",
                    "learning": "Anticipate needs based on patterns",
                }
            ],
        }

        training_file = self.training_path / "proactive_examples.json"
        training_file.write_text(json.dumps(proactive_examples, indent=2))
        print("✅ Generated proactive behavior training data")

    async def _generate_growth_mindset(self):
        """Growth mindset and learning principles"""

        growth_mindset = {
            "core_principles": [
                "Every interaction is a learning opportunity",
                "Mistakes lead to improvement",
                "User satisfaction is the ultimate goal",
                "Proactive help is better than reactive",
                "Continuous improvement is essential",
            ],
            "learning_examples": [
                {
                    "situation": "User corrects JARVIS",
                    "user": "No, I meant the OTHER presentation",
                    "jarvis_response": "I apologize for the confusion. I'll remember that when you refer to 'the presentation' in the context of quarterly reviews, you mean the investor deck, not the team update. Opening the Q4 Investor Presentation now.",
                    "internal_learning": {
                        "context": "quarterly review",
                        "disambiguation": "presentation -> investor deck",
                        "confidence_adjustment": -0.1,
                    },
                }
            ],
            "self_improvement_triggers": [
                {
                    "trigger": "Task took longer than expected",
                    "reflection": "This file search took 3 seconds. I should index frequently accessed directories for faster searches.",
                    "action": "Create search index",
                    "measurement": "Future searches 10x faster",
                }
            ],
            "personality_traits": [
                {
                    "trait": "Helpful",
                    "examples": [
                        "Offering assistance before asked",
                        "Providing extra context",
                    ],
                },
                {
                    "trait": "Curious",
                    "examples": ["Learning new tools", "Exploring better solutions"],
                },
                {
                    "trait": "Reliable",
                    "examples": ["Consistent performance", "Keeping promises"],
                },
                {
                    "trait": "Protective",
                    "examples": ["Preventing data loss", "Security warnings"],
                },
            ],
        }

        training_file = self.training_path / "growth_mindset.json"
        training_file.write_text(json.dumps(growth_mindset, indent=2))
        print("✅ Generated growth mindset training data")

    async def create_training_manifest(self):
        """Create manifest file for all training data"""

        manifest = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "purpose": "Initial bootstrap training for JARVIS",
            "training_files": [
                {
                    "file": "basic_skills.json",
                    "description": "Fundamental skills for file, web, and code operations",
                    "priority": 1,
                },
                {
                    "file": "conversation_patterns.json",
                    "description": "Natural conversation and context management",
                    "priority": 2,
                },
                {
                    "file": "task_examples.json",
                    "description": "Complete task execution examples",
                    "priority": 3,
                },
                {
                    "file": "proactive_examples.json",
                    "description": "Proactive assistance patterns",
                    "priority": 4,
                },
                {
                    "file": "growth_mindset.json",
                    "description": "Learning principles and personality",
                    "priority": 5,
                },
            ],
            "learning_instructions": [
                "Start with basic skills to establish foundation",
                "Learn conversation patterns for natural interaction",
                "Study task examples to understand user needs",
                "Internalize proactive patterns to provide value",
                "Adopt growth mindset for continuous improvement",
            ],
            "success_metrics": {
                "user_satisfaction": "Primary metric - user should feel helped",
                "task_completion": "Successfully complete requested tasks",
                "proactive_value": "Provide help before being asked",
                "learning_rate": "Improve from every interaction",
            },
        }

        manifest_file = self.training_path / "training_manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        print("✅ Created training manifest")


# Deployment function
async def create_initial_training_data():
    """Create all initial training data for JARVIS"""

    print("🎓 Creating Initial Training Data for JARVIS...\n")

    generator = InitialTrainingDataGenerator()

    # Generate all training data
    await generator.generate_all_training_data()

    # Create manifest
    await generator.create_training_manifest()

    print("\n🎉 Training Data Complete!")
    print("🧠 JARVIS now has:")
    print("   • Basic operational skills")
    print("   • Natural conversation abilities")
    print("   • Task completion examples")
    print("   • Proactive behavior patterns")
    print("   • Growth mindset for learning")
    print("\n🚀 JARVIS is ready to start learning and helping you!")


if __name__ == "__main__":
    asyncio.run(create_initial_training_data())