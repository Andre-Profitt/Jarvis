#!/usr/bin/env python3
"""
Real OpenAI Integration
Actual GPT-4 API implementation with proper error handling
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
import openai
from openai import AsyncOpenAI
import json
import logging
from datetime import datetime
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealOpenAIIntegration:
    """Real OpenAI GPT-4 integration with all features"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.api_key)

        # Model configuration
        self.model = "gpt-4-turbo-preview"  # Latest GPT-4 model
        self.fallback_model = "gpt-3.5-turbo"  # Fallback for rate limits

        # Token management
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.max_tokens = 128000  # GPT-4 Turbo context window
        self.max_response_tokens = 4096

        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)  # 10 concurrent requests
        self.request_count = 0
        self.last_reset = datetime.now()

        # Conversation management
        self.conversations = {}

    async def query(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """Query GPT-4 with full feature support"""

        async with self.rate_limiter:
            try:
                # Prepare messages
                messages = self._prepare_messages(
                    prompt, context, conversation_id, system_prompt
                )

                # Check token count
                token_count = self._count_tokens(messages)
                if token_count > self.max_tokens - self.max_response_tokens:
                    messages = self._truncate_messages(messages)

                # Make API call
                if stream:
                    return await self._stream_completion(messages, temperature)
                else:
                    return await self._get_completion(messages, temperature)

            except openai.RateLimitError:
                logger.warning("Rate limit hit, falling back to GPT-3.5")
                return await self._fallback_query(prompt, context)

            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    def _prepare_messages(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        conversation_id: Optional[str],
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """Prepare messages for the API call"""

        messages = []

        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append(
                {
                    "role": "system",
                    "content": "You are JARVIS, an advanced AI assistant with deep reasoning capabilities. "
                    "You're helpful, protective, and always learning to better serve your family.",
                }
            )

        # Add conversation history if exists
        if conversation_id and conversation_id in self.conversations:
            messages.extend(
                self.conversations[conversation_id][-10:]
            )  # Last 10 messages

        # Add context if provided
        if context:
            context_str = f"Context: {json.dumps(context, indent=2)}"
            messages.append({"role": "system", "content": context_str})

        # Add user prompt
        messages.append({"role": "user", "content": prompt})

        return messages

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in messages"""

        total_tokens = 0
        for message in messages:
            total_tokens += len(self.encoding.encode(message["content"]))
            total_tokens += 4  # Message overhead

        return total_tokens

    def _truncate_messages(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Truncate messages to fit within token limit"""

        # Keep system prompt and last user message
        system_messages = [m for m in messages if m["role"] == "system"]
        user_message = messages[-1]

        # Add middle messages until we hit limit
        middle_messages = messages[len(system_messages) : -1]
        truncated = system_messages + middle_messages[-5:] + [user_message]

        return truncated

    async def _get_completion(
        self, messages: List[Dict[str, str]], temperature: float
    ) -> str:
        """Get completion from OpenAI"""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_response_tokens,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )

        return response.choices[0].message.content

    async def _stream_completion(
        self, messages: List[Dict[str, str]], temperature: float
    ) -> str:
        """Stream completion from OpenAI"""

        full_response = ""

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_response_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                # You could yield content here for real-time streaming

        return full_response

    async def _fallback_query(
        self, prompt: str, context: Optional[Dict[str, Any]]
    ) -> str:
        """Fallback to GPT-3.5 when rate limited"""

        try:
            response = await self.client.chat.completions.create(
                model=self.fallback_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are JARVIS, a helpful AI assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=2048,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Fallback query failed: {e}")
            return "I'm experiencing technical difficulties. Please try again."

    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for semantic search"""

        try:
            response = await self.client.embeddings.create(
                model="text-embedding-3-small", input=text
            )

            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    async def analyze_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Analyze code for improvements and issues"""

        prompt = f"""Analyze this {language} code and provide:
1. Potential bugs or issues
2. Performance improvements
3. Code style suggestions
4. Security concerns
5. Refactoring opportunities

Code:
```{language}
{code}
```
"""

        response = await self.query(
            prompt,
            system_prompt="You are an expert code reviewer with deep knowledge of best practices.",
            temperature=0.3,  # Lower temperature for more focused analysis
        )

        return {"analysis": response}

    async def generate_code(
        self, description: str, language: str = "python", context: Optional[str] = None
    ) -> str:
        """Generate code based on description"""

        prompt = f"Generate {language} code for: {description}"

        if context:
            prompt += f"\n\nContext:\n{context}"

        system_prompt = (
            f"You are an expert {language} developer. "
            "Generate clean, efficient, well-documented code. "
            "Include error handling and follow best practices."
        )

        return await self.query(prompt, system_prompt=system_prompt, temperature=0.5)

    def save_conversation(self, conversation_id: str, messages: List[Dict[str, str]]):
        """Save conversation for context"""

        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].extend(messages)

        # Keep only last 50 messages per conversation
        if len(self.conversations[conversation_id]) > 50:
            self.conversations[conversation_id] = self.conversations[conversation_id][
                -50:
            ]

    async def test_connection(self) -> bool:
        """Test OpenAI API connection"""

        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5,
            )
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Create singleton instance lazily
_openai_integration = None

def get_openai_integration():
    """Get or create OpenAI integration instance"""
    global _openai_integration
    if _openai_integration is None:
        _openai_integration = RealOpenAIIntegration()
    return _openai_integration

# For backward compatibility
try:
    openai_integration = RealOpenAIIntegration()
except ValueError:
    # If API key not found, create a dummy instance
    openai_integration = None
