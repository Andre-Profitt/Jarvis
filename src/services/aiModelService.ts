import { EventEmitter } from 'events';
import OpenAI from 'openai';
import Anthropic from 'anthropic';
import { pipeline } from '@xenova/transformers';
import { Ollama } from 'ollama';

// Types
export interface AIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface AIStreamChunk {
  content: string;
  done: boolean;
}

export interface AIModelConfig {
  provider: 'openai' | 'anthropic' | 'ollama' | 'transformers';
  model?: string;
  apiKey?: string;
  baseUrl?: string;
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
}

export interface AIResponse {
  content: string;
  model: string;
  provider: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

// Base Provider Interface
interface AIProvider {
  name: string;
  complete(messages: AIMessage[], config: AIModelConfig): Promise<AIResponse>;
  stream(messages: AIMessage[], config: AIModelConfig): AsyncGenerator<AIStreamChunk>;
  isAvailable(): Promise<boolean>;
}

// OpenAI Provider
class OpenAIProvider implements AIProvider {
  name = 'openai';
  private client: OpenAI | null = null;

  private getClient(config: AIModelConfig): OpenAI {
    if (!this.client || config.apiKey) {
      this.client = new OpenAI({
        apiKey: config.apiKey || process.env.OPENAI_API_KEY,
        baseURL: config.baseUrl
      });
    }
    return this.client;
  }

  async complete(messages: AIMessage[], config: AIModelConfig): Promise<AIResponse> {
    const client = this.getClient(config);
    const completion = await client.chat.completions.create({
      model: config.model || 'gpt-4-turbo-preview',
      messages: messages as any,
      max_tokens: config.maxTokens || 1000,
      temperature: config.temperature || 0.7,
      stream: false
    });

    return {
      content: completion.choices[0].message.content || '',
      model: completion.model,
      provider: this.name,
      usage: completion.usage ? {
        promptTokens: completion.usage.prompt_tokens,
        completionTokens: completion.usage.completion_tokens,
        totalTokens: completion.usage.total_tokens
      } : undefined
    };
  }

  async *stream(messages: AIMessage[], config: AIModelConfig): AsyncGenerator<AIStreamChunk> {
    const client = this.getClient(config);
    const stream = await client.chat.completions.create({
      model: config.model || 'gpt-4-turbo-preview',
      messages: messages as any,
      max_tokens: config.maxTokens || 1000,
      temperature: config.temperature || 0.7,
      stream: true
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      yield {
        content,
        done: chunk.choices[0]?.finish_reason !== null
      };
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      const client = this.getClient({});
      await client.models.list();
      return true;
    } catch {
      return false;
    }
  }
}

// Anthropic Provider
class AnthropicProvider implements AIProvider {
  name = 'anthropic';
  private client: Anthropic | null = null;

  private getClient(config: AIModelConfig): Anthropic {
    if (!this.client || config.apiKey) {
      this.client = new Anthropic({
        apiKey: config.apiKey || process.env.ANTHROPIC_API_KEY,
        baseURL: config.baseUrl
      });
    }
    return this.client;
  }

  async complete(messages: AIMessage[], config: AIModelConfig): Promise<AIResponse> {
    const client = this.getClient(config);
    
    // Convert messages to Anthropic format
    const systemMessage = messages.find(m => m.role === 'system')?.content || '';
    const userMessages = messages.filter(m => m.role !== 'system');

    const response = await client.messages.create({
      model: config.model || 'claude-3-opus-20240229',
      messages: userMessages.map(m => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content
      })),
      system: systemMessage,
      max_tokens: config.maxTokens || 1000,
      temperature: config.temperature || 0.7
    });

    const content = response.content[0].type === 'text' ? response.content[0].text : '';

    return {
      content,
      model: response.model,
      provider: this.name,
      usage: {
        promptTokens: response.usage.input_tokens,
        completionTokens: response.usage.output_tokens,
        totalTokens: response.usage.input_tokens + response.usage.output_tokens
      }
    };
  }

  async *stream(messages: AIMessage[], config: AIModelConfig): AsyncGenerator<AIStreamChunk> {
    const client = this.getClient(config);
    
    const systemMessage = messages.find(m => m.role === 'system')?.content || '';
    const userMessages = messages.filter(m => m.role !== 'system');

    const stream = await client.messages.create({
      model: config.model || 'claude-3-opus-20240229',
      messages: userMessages.map(m => ({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.content
      })),
      system: systemMessage,
      max_tokens: config.maxTokens || 1000,
      temperature: config.temperature || 0.7,
      stream: true
    });

    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
        yield {
          content: chunk.delta.text,
          done: false
        };
      } else if (chunk.type === 'message_stop') {
        yield {
          content: '',
          done: true
        };
      }
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      const client = this.getClient({});
      // Simple test to check if API key is valid
      await client.messages.create({
        model: 'claude-3-haiku-20240307',
        messages: [{ role: 'user', content: 'test' }],
        max_tokens: 1
      });
      return true;
    } catch {
      return false;
    }
  }
}

// Ollama Provider (Local)
class OllamaProvider implements AIProvider {
  name = 'ollama';
  private client: Ollama;

  constructor() {
    this.client = new Ollama({
      host: process.env.OLLAMA_HOST || 'http://localhost:11434'
    });
  }

  async complete(messages: AIMessage[], config: AIModelConfig): Promise<AIResponse> {
    const response = await this.client.chat({
      model: config.model || 'llama2',
      messages: messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      options: {
        temperature: config.temperature || 0.7,
        num_predict: config.maxTokens || 1000
      }
    });

    return {
      content: response.message.content,
      model: config.model || 'llama2',
      provider: this.name,
      usage: {
        promptTokens: response.prompt_eval_count || 0,
        completionTokens: response.eval_count || 0,
        totalTokens: (response.prompt_eval_count || 0) + (response.eval_count || 0)
      }
    };
  }

  async *stream(messages: AIMessage[], config: AIModelConfig): AsyncGenerator<AIStreamChunk> {
    const response = await this.client.chat({
      model: config.model || 'llama2',
      messages: messages.map(m => ({
        role: m.role,
        content: m.content
      })),
      options: {
        temperature: config.temperature || 0.7,
        num_predict: config.maxTokens || 1000
      },
      stream: true
    });

    for await (const chunk of response) {
      yield {
        content: chunk.message.content,
        done: chunk.done || false
      };
    }
  }

  async isAvailable(): Promise<boolean> {
    try {
      await this.client.list();
      return true;
    } catch {
      return false;
    }
  }
}

// Transformers.js Provider (Browser-compatible)
class TransformersProvider implements AIProvider {
  name = 'transformers';
  private generator: any = null;

  async complete(messages: AIMessage[], config: AIModelConfig): Promise<AIResponse> {
    if (!this.generator) {
      this.generator = await pipeline('text-generation', config.model || 'Xenova/distilgpt2');
    }

    const prompt = messages.map(m => `${m.role}: ${m.content}`).join('\n');
    const result = await this.generator(prompt, {
      max_new_tokens: config.maxTokens || 100,
      temperature: config.temperature || 0.7
    });

    return {
      content: result[0].generated_text.replace(prompt, '').trim(),
      model: config.model || 'Xenova/distilgpt2',
      provider: this.name
    };
  }

  async *stream(messages: AIMessage[], config: AIModelConfig): AsyncGenerator<AIStreamChunk> {
    // Transformers.js doesn't support streaming natively, so we'll simulate it
    const response = await this.complete(messages, config);
    const words = response.content.split(' ');
    
    for (const word of words) {
      yield {
        content: word + ' ',
        done: false
      };
      await new Promise(resolve => setTimeout(resolve, 50)); // Simulate streaming delay
    }
    
    yield {
      content: '',
      done: true
    };
  }

  async isAvailable(): Promise<boolean> {
    try {
      // Check if we're in a browser environment
      return typeof window !== 'undefined';
    } catch {
      return true; // Node environment
    }
  }
}

// Main AI Model Service
export class AIModelService extends EventEmitter {
  private providers: Map<string, AIProvider> = new Map();
  private fallbackOrder: string[] = ['openai', 'anthropic', 'ollama', 'transformers'];

  constructor() {
    super();
    this.initializeProviders();
  }

  private initializeProviders() {
    this.providers.set('openai', new OpenAIProvider());
    this.providers.set('anthropic', new AnthropicProvider());
    this.providers.set('ollama', new OllamaProvider());
    this.providers.set('transformers', new TransformersProvider());
  }

  // Intelligent routing based on task type
  private selectProvider(taskType: string, preferredProvider?: string): string {
    if (preferredProvider && this.providers.has(preferredProvider)) {
      return preferredProvider;
    }

    // Task-based routing logic
    switch (taskType) {
      case 'code':
      case 'analysis':
        return 'anthropic'; // Claude is excellent for code
      case 'creative':
      case 'general':
        return 'openai'; // GPT-4 for creative tasks
      case 'local':
      case 'private':
        return 'ollama'; // Local for privacy
      case 'browser':
      case 'lightweight':
        return 'transformers'; // Browser-compatible
      default:
        return 'openai';
    }
  }

  async complete(
    messages: AIMessage[],
    options: {
      taskType?: string;
      preferredProvider?: string;
      config?: Partial<AIModelConfig>;
    } = {}
  ): Promise<AIResponse> {
    const providerName = this.selectProvider(options.taskType || 'general', options.preferredProvider);
    const provider = this.providers.get(providerName);
    
    if (!provider) {
      throw new Error(`Provider ${providerName} not found`);
    }

    const config: AIModelConfig = {
      provider: providerName as any,
      ...options.config
    };

    try {
      // Check if provider is available
      const isAvailable = await provider.isAvailable();
      if (!isAvailable) {
        throw new Error(`Provider ${providerName} is not available`);
      }

      this.emit('request:start', { provider: providerName, messages });
      const response = await provider.complete(messages, config);
      this.emit('request:complete', { provider: providerName, response });
      
      return response;
    } catch (error) {
      this.emit('request:error', { provider: providerName, error });
      
      // Fallback mechanism
      for (const fallbackName of this.fallbackOrder) {
        if (fallbackName === providerName) continue;
        
        const fallbackProvider = this.providers.get(fallbackName);
        if (!fallbackProvider) continue;
        
        try {
          const isAvailable = await fallbackProvider.isAvailable();
          if (!isAvailable) continue;
          
          this.emit('fallback:attempt', { from: providerName, to: fallbackName });
          const response = await fallbackProvider.complete(messages, config);
          this.emit('fallback:success', { provider: fallbackName, response });
          
          return response;
        } catch (fallbackError) {
          this.emit('fallback:error', { provider: fallbackName, error: fallbackError });
          continue;
        }
      }
      
      throw new Error(`All AI providers failed. Original error: ${error}`);
    }
  }

  async *stream(
    messages: AIMessage[],
    options: {
      taskType?: string;
      preferredProvider?: string;
      config?: Partial<AIModelConfig>;
    } = {}
  ): AsyncGenerator<AIStreamChunk> {
    const providerName = this.selectProvider(options.taskType || 'general', options.preferredProvider);
    const provider = this.providers.get(providerName);
    
    if (!provider) {
      throw new Error(`Provider ${providerName} not found`);
    }

    const config: AIModelConfig = {
      provider: providerName as any,
      stream: true,
      ...options.config
    };

    try {
      this.emit('stream:start', { provider: providerName, messages });
      
      for await (const chunk of provider.stream(messages, config)) {
        this.emit('stream:chunk', { provider: providerName, chunk });
        yield chunk;
      }
      
      this.emit('stream:complete', { provider: providerName });
    } catch (error) {
      this.emit('stream:error', { provider: providerName, error });
      throw error;
    }
  }

  // Get available providers
  async getAvailableProviders(): Promise<string[]> {
    const available: string[] = [];
    
    for (const [name, provider] of this.providers) {
      if (await provider.isAvailable()) {
        available.push(name);
      }
    }
    
    return available;
  }

  // Add custom provider
  addProvider(name: string, provider: AIProvider) {
    this.providers.set(name, provider);
    this.emit('provider:added', { name });
  }

  // Remove provider
  removeProvider(name: string) {
    this.providers.delete(name);
    this.emit('provider:removed', { name });
  }
}

// Export singleton instance
export const aiModelService = new AIModelService();