import { aiModelService, AIMessage } from '../services/aiModelService';
import * as readline from 'readline';

// Demo: Real AI Integration with JARVIS
async function demoAIIntegration() {
  console.log('🤖 JARVIS AI Model Service Demo\n');

  // Check available providers
  console.log('Checking available AI providers...');
  const availableProviders = await aiModelService.getAvailableProviders();
  console.log('Available providers:', availableProviders);
  console.log('');

  // Set up event listeners for monitoring
  aiModelService.on('request:start', ({ provider }) => {
    console.log(`📡 Sending request to ${provider}...`);
  });

  aiModelService.on('request:complete', ({ provider, response }) => {
    console.log(`✅ ${provider} responded (${response.usage?.totalTokens || 'N/A'} tokens used)`);
  });

  aiModelService.on('fallback:attempt', ({ from, to }) => {
    console.log(`⚠️ ${from} failed, trying ${to}...`);
  });

  aiModelService.on('stream:chunk', ({ chunk }) => {
    process.stdout.write(chunk.content);
  });

  // Demo 1: Code Analysis (Routes to Anthropic)
  console.log('=== Demo 1: Code Analysis ===');
  const codeMessages: AIMessage[] = [
    {
      role: 'system',
      content: 'You are a code analysis expert. Be concise and technical.'
    },
    {
      role: 'user',
      content: 'Analyze this TypeScript code and suggest improvements:\n\nfunction getData(id) {\n  return fetch(`/api/data/${id}`).then(res => res.json())\n}'
    }
  ];

  try {
    const codeResponse = await aiModelService.complete(codeMessages, {
      taskType: 'code',
      config: {
        maxTokens: 500,
        temperature: 0.3
      }
    });
    console.log('\nResponse:', codeResponse.content);
    console.log(`\n(Used ${codeResponse.provider} model: ${codeResponse.model})\n`);
  } catch (error) {
    console.error('Code analysis failed:', error);
  }

  // Demo 2: Creative Writing (Routes to OpenAI)
  console.log('\n=== Demo 2: Creative Writing ===');
  const creativeMessages: AIMessage[] = [
    {
      role: 'system',
      content: 'You are a creative storyteller. Be imaginative and engaging.'
    },
    {
      role: 'user',
      content: 'Write a short story about a robot learning to paint in exactly 3 sentences.'
    }
  ];

  try {
    const creativeResponse = await aiModelService.complete(creativeMessages, {
      taskType: 'creative',
      config: {
        maxTokens: 200,
        temperature: 0.9
      }
    });
    console.log('Response:', creativeResponse.content);
    console.log(`\n(Used ${creativeResponse.provider} model: ${creativeResponse.model})\n`);
  } catch (error) {
    console.error('Creative writing failed:', error);
  }

  // Demo 3: Streaming Response
  console.log('\n=== Demo 3: Streaming Response ===');
  const streamMessages: AIMessage[] = [
    {
      role: 'user',
      content: 'Explain how neural networks work in simple terms.'
    }
  ];

  console.log('Streaming response:\n');
  try {
    for await (const chunk of aiModelService.stream(streamMessages, {
      taskType: 'general',
      config: {
        maxTokens: 300,
        temperature: 0.7
      }
    })) {
      // Chunks are handled by event listener above
      if (chunk.done) {
        console.log('\n\n✅ Stream complete');
      }
    }
  } catch (error) {
    console.error('\nStreaming failed:', error);
  }

  // Demo 4: Fallback Mechanism
  console.log('\n=== Demo 4: Fallback Mechanism ===');
  const fallbackMessages: AIMessage[] = [
    {
      role: 'user',
      content: 'What is 2 + 2?'
    }
  ];

  try {
    // Force a specific provider that might not be available
    const fallbackResponse = await aiModelService.complete(fallbackMessages, {
      preferredProvider: 'ollama', // This might fail if Ollama isn't running
      config: {
        maxTokens: 50
      }
    });
    console.log('Response:', fallbackResponse.content);
    console.log(`(Used ${fallbackResponse.provider})\n`);
  } catch (error) {
    console.error('All providers failed:', error);
  }

  // Demo 5: Interactive Chat
  console.log('\n=== Demo 5: Interactive Chat ===');
  console.log('Type your questions (or "quit" to exit):\n');

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  const chatHistory: AIMessage[] = [
    {
      role: 'system',
      content: 'You are JARVIS, an AI assistant. Be helpful, concise, and friendly.'
    }
  ];

  const askQuestion = () => {
    rl.question('You: ', async (input) => {
      if (input.toLowerCase() === 'quit') {
        rl.close();
        return;
      }

      chatHistory.push({ role: 'user', content: input });

      try {
        console.log('\nJARVIS: ');
        let fullResponse = '';
        
        for await (const chunk of aiModelService.stream(chatHistory, {
          taskType: 'general',
          config: {
            maxTokens: 500,
            temperature: 0.7
          }
        })) {
          fullResponse += chunk.content;
        }

        chatHistory.push({ role: 'assistant', content: fullResponse });
        console.log('\n');
        askQuestion();
      } catch (error) {
        console.error('\nError:', error);
        askQuestion();
      }
    });
  };

  askQuestion();
}

// Integration example for JARVIS
export class JARVISAIIntegration {
  async processCommand(command: string): Promise<string> {
    const messages: AIMessage[] = [
      {
        role: 'system',
        content: 'You are JARVIS, an AI assistant integrated into a development environment. Help with coding, analysis, and automation tasks.'
      },
      {
        role: 'user',
        content: command
      }
    ];

    const response = await aiModelService.complete(messages, {
      taskType: this.determineTaskType(command),
      config: {
        maxTokens: 1000,
        temperature: 0.7
      }
    });

    return response.content;
  }

  private determineTaskType(command: string): string {
    const lowerCommand = command.toLowerCase();
    
    if (lowerCommand.includes('code') || lowerCommand.includes('debug') || lowerCommand.includes('analyze')) {
      return 'code';
    } else if (lowerCommand.includes('write') || lowerCommand.includes('create') || lowerCommand.includes('story')) {
      return 'creative';
    } else if (lowerCommand.includes('local') || lowerCommand.includes('private')) {
      return 'local';
    } else {
      return 'general';
    }
  }

  async streamResponse(command: string, onChunk: (content: string) => void): Promise<void> {
    const messages: AIMessage[] = [
      {
        role: 'system',
        content: 'You are JARVIS. Provide helpful, accurate responses.'
      },
      {
        role: 'user',
        content: command
      }
    ];

    for await (const chunk of aiModelService.stream(messages, {
      taskType: this.determineTaskType(command)
    })) {
      onChunk(chunk.content);
    }
  }
}

// Run demo if this file is executed directly
if (require.main === module) {
  demoAIIntegration().catch(console.error);
}