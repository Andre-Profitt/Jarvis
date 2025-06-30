#!/usr/bin/env python3
"""
Updated Multi-AI Integration for JARVIS
Real implementations for all AI models
"""

import asyncio
import subprocess
import json
import os
import sys
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import aiohttp
import tempfile
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import real integrations
from core.real_claude_integration import claude_integration
from core.real_openai_integration import openai_integration
from core.real_elevenlabs_integration import elevenlabs_integration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedMultiAIIntegration:
    """
    Real integration with multiple AI models
    """

    def __init__(self):
        self.available_models = {}
        self.model_capabilities = {
            "claude_desktop": {
                "strengths": [
                    "general_intelligence",
                    "reasoning",
                    "creativity",
                    "analysis",
                ],
                "context_window": 200000,
                "multimodal": True,
                "integration": claude_integration,
            },
            "gpt4": {
                "strengths": [
                    "general_tasks",
                    "creativity",
                    "problem_solving",
                    "coding",
                ],
                "context_window": 128000,
                "requires_api_key": True,
                "integration": openai_integration,
            },
            "gemini_cli": {
                "strengths": ["multimodal", "long_context", "reasoning", "analysis"],
                "context_window": 2000000,  # 2M tokens!
                "models": ["gemini-1.5-pro-latest", "gemini-1.5-flash"],
                "integration": None,  # Will be initialized
            },
        }

        self.task_routing = {
            "code_generation": ["gpt4", "claude_desktop"],
            "code_review": ["gpt4", "claude_desktop"],
            "creative_writing": ["claude_desktop", "gpt4"],
            "data_analysis": ["gpt4", "gemini_cli"],
            "general_question": ["claude_desktop", "gpt4", "gemini_cli"],
            "vision_tasks": ["gemini_cli", "claude_desktop", "gpt4"],
            "long_document": ["gemini_cli", "claude_desktop"],
            "reasoning": ["claude_desktop", "gemini_cli", "gpt4"],
        }

        self.gemini_cli = None

    async def initialize(self):
        """Initialize all AI integrations with real connections"""

        logger.info("Initializing Multi-AI Integration...")

        # Test Claude Desktop via MCP
        try:
            await claude_integration.setup_mcp_server()
            if await claude_integration.test_integration():
                self.available_models["claude_desktop"] = True
                logger.info("✅ Claude Desktop (MCP) initialized")
            else:
                logger.warning("⚠️ Claude Desktop not available")
        except Exception as e:
            logger.error(f"Claude Desktop init error: {e}")

        # Test OpenAI GPT-4
        try:
            if await openai_integration.test_connection():
                self.available_models["gpt4"] = True
                logger.info("✅ GPT-4 initialized")
            else:
                logger.warning("⚠️ GPT-4 not available")
        except Exception as e:
            logger.error(f"GPT-4 init error: {e}")

        # Initialize Gemini CLI
        try:
            self.gemini_cli = GeminiCLIIntegration()
            if await self.gemini_cli.test_connection():
                self.available_models["gemini_cli"] = True
                self.model_capabilities["gemini_cli"]["integration"] = self.gemini_cli
                logger.info("✅ Gemini CLI initialized")
            else:
                logger.warning("⚠️ Gemini CLI not available")
        except Exception as e:
            logger.error(f"Gemini CLI init error: {e}")

        # Test ElevenLabs
        try:
            if await elevenlabs_integration.test_connection():
                logger.info("✅ ElevenLabs voice initialized")
            else:
                logger.warning("⚠️ ElevenLabs not available")
        except Exception as e:
            logger.error(f"ElevenLabs init error: {e}")

        logger.info(f"Available models: {list(self.available_models.keys())}")

    async def query(
        self,
        prompt: str,
        task_type: str = "general_question",
        context: Optional[Dict[str, Any]] = None,
        preferred_model: Optional[str] = None,
        multimodal_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Route query to best available model"""

        # Select model based on task and availability
        model = self._select_model(task_type, preferred_model)

        if not model:
            return {
                "error": "No available models for this task",
                "available_models": list(self.available_models.keys()),
            }

        try:
            # Route to appropriate model
            if model == "claude_desktop":
                response = await claude_integration.query_claude(prompt, context)

            elif model == "gpt4":
                response = await openai_integration.query(
                    prompt, context=context, temperature=0.7)

            elif model == "gemini_cli":
                response = await self.gemini_cli.query(
                    prompt, context=context, multimodal_data=multimodal_data
                )

            else:
                response = f"Model {model} not implemented"

            return {
                "response": response,
                "model_used": model,
                "task_type": task_type,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Query error with {model}: {e}")

            # Try fallback model
            fallback = self._get_fallback_model(model, task_type)
            if fallback:
                logger.info(f"Trying fallback model: {fallback}")
                return await self.query(
                    prompt,
                    task_type,
                    context,
                    preferred_model=fallback,
                    multimodal_data=multimodal_data,
                )

            return {"error": str(e), "model_attempted": model, "success": False}

    def _select_model(
        self, task_type: str, preferred_model: Optional[str]
    ) -> Optional[str]:
        """Select best available model for task"""

        # Use preferred if available
        if preferred_model and preferred_model in self.available_models:
            return preferred_model

        # Get recommended models for task
        recommended = self.task_routing.get(
            task_type, ["claude_desktop", "gpt4", "gemini_cli"]
        )

        # Return first available
        for model in recommended:
            if model in self.available_models:
                return model

        # Return any available model
        return list(self.available_models.keys())[0] if self.available_models else None

    def _get_fallback_model(self, failed_model: str, task_type: str) -> Optional[str]:
        """Get fallback model when primary fails"""

        recommended = self.task_routing.get(task_type, [])

        for model in recommended:
            if model != failed_model and model in self.available_models:
                return model

        # Return any other available model
        for model in self.available_models:
            if model != failed_model:
                return model

        return None

    async def parallel_query(
        self, prompt: str, models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query multiple models in parallel for comparison"""

        if not models:
            models = list(self.available_models.keys())

        tasks = []
        for model in models:
            if model in self.available_models:
                task = asyncio.create_task(self.query(prompt, preferred_model=model))
                tasks.append((model, task))

        results = {}
        for model, task in tasks:
            try:
                result = await task
                results[model] = result
            except Exception:
            pass
                results[model] = {"error": str(e)}

        return results

    async def specialized_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle specialized requests requiring specific models"""

        request_type = request.get("type")

        if request_type == "code_generation":
            return await self._handle_code_generation(request)

        elif request_type == "vision_analysis":
            return await self._handle_vision_analysis(request)

        elif request_type == "long_document_analysis":
            return await self._handle_long_document(request)

        else:
            return await self.query(
                request.get("prompt", ""), task_type=request_type or "general_question"
            )

    async def _handle_code_generation(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Special handling for code generation"""

        language = request.get("language", "python")
        description = request.get("description", "")
        context = request.get("context")

        # Prefer GPT-4 for code generation
        if "gpt4" in self.available_models:
            return await openai_integration.generate_code(
                description, language=language, context=context
            )

        # Fallback to Claude
        return await self.query(
            f"Generate {language} code for: {description}", task_type="code_generation"
        )

    async def _handle_vision_analysis(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle image/vision analysis"""

        image_path = request.get("image_path")
        prompt = request.get("prompt", "Describe this image")

        # Prefer Gemini for vision tasks (2M context)
        if "gemini_cli" in self.available_models:
            return await self.gemini_cli.analyze_image(image_path, prompt)

        # Fallback to other multimodal models
        return await self.query(
            prompt, task_type="vision_tasks", multimodal_data={"image": image_path}
        )

    async def _handle_long_document(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle long document analysis"""

        document = request.get("document", "")
        prompt = request.get("prompt", "Summarize this document")

        # Check document length
        token_count = len(document.split()) * 1.3  # Rough estimate

        # Use Gemini for very long documents
        if token_count > 100000 and "gemini_cli" in self.available_models:
            return await self.gemini_cli.query(
                f"{prompt}\n\nDocument:\n{document}",
                context={"task": "long_document_analysis"},
            )

        # Use Claude or GPT-4 for shorter documents
        return await self.query(
            f"{prompt}\n\nDocument:\n{document}", task_type="long_document"
        )


class GeminiCLIIntegration:
    """Real Google Gemini CLI integration"""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.cli_path = self._find_gemini_cli()

    def _find_gemini_cli(self) -> Optional[Path]:
        """Find Gemini CLI installation"""

        # Check common locations
        possible_paths = [
            Path.home() / ".local/bin/gemini",
            Path("/usr/local/bin/gemini"),
            Path("/opt/homebrew/bin/gemini"),
        ]

        for path in possible_paths:
            if path.exists():
                return path

        # Try which command
        try:
            result = subprocess.run(["which", "gemini"], capture_output=True, text=True)
            if result.returncode == 0:
                return Path(result.stdout.strip())
        except Exception:
            pass
            pass

        return None

    async def test_connection(self) -> bool:
        """Test Gemini CLI availability"""

        if not self.cli_path:
            # Try to install if not found
            await self._install_gemini_cli()
            self.cli_path = self._find_gemini_cli()

        if not self.cli_path or not self.api_key:
            return False

        try:
            # Test with simple query
            result = subprocess.run(
                [str(self.cli_path), "--version"], capture_output=True, text=True
            )
            return result.returncode == 0
        except Exception:
            pass
            return False

    async def _install_gemini_cli(self):
        """Install Gemini CLI if not present"""

        try:
            logger.info("Installing Gemini CLI...")

            # Install via go
            subprocess.run(
                ["go", "install", "github.com/google-gemini/gemini-cli/gemini@latest"],
                check=True,
            )

        except Exception as e:
            logger.error(f"Failed to install Gemini CLI: {e}")

    async def query(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        multimodal_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Query Gemini via CLI"""

        if not self.cli_path:
            raise Exception("Gemini CLI not available")

        try:
            # Build command
            cmd = [
                str(self.cli_path),
                "chat",
                "--api-key",
                self.api_key,
                "--model",
                "gemini-1.5-pro-latest",
            ]

            # Add multimodal data if present
            if multimodal_data and "image" in multimodal_data:
                cmd.extend(["--image", multimodal_data["image"]])

            # Run query
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate(prompt.encode())

            if process.returncode != 0:
                raise Exception(f"Gemini error: {stderr.decode()}")

            return stdout.decode().strip()

        except Exception as e:
            logger.error(f"Gemini query error: {e}")
            raise

    async def analyze_image(self, image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze image with Gemini"""

        response = await self.query(prompt, multimodal_data={"image": image_path})

        return {
            "response": response,
            "model": "gemini-1.5-pro",
            "image_analyzed": image_path,
        }


# Create singleton instance
multi_ai = EnhancedMultiAIIntegration()