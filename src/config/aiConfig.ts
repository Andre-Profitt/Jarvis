// AI Model Configuration
// This file manages API keys and model settings for JARVIS AI integration

export interface AIProviderConfig {
  enabled: boolean;
  apiKey?: string;
  models: {
    default: string;
    code?: string;
    creative?: string;
    fast?: string;
  };
  baseUrl?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface AIConfig {
  providers: {
    openai: AIProviderConfig;
    anthropic: AIProviderConfig;
    ollama: AIProviderConfig;
    transformers: AIProviderConfig;
  };
  routing: {
    taskTypes: {
      [key: string]: string; // task type -> provider mapping
    };
    fallbackChain: string[];
  };
  defaults: {
    maxTokens: number;
    temperature: number;
    streamingEnabled: boolean;
  };
}

// Default configuration
export const defaultAIConfig: AIConfig = {
  providers: {
    openai: {
      enabled: true,
      apiKey: process.env.OPENAI_API_KEY,
      models: {
        default: 'gpt-4-turbo-preview',
        code: 'gpt-4-turbo-preview',
        creative: 'gpt-4-turbo-preview',
        fast: 'gpt-3.5-turbo'
      },
      maxTokens: 2000,
      temperature: 0.7
    },
    anthropic: {
      enabled: true,
      apiKey: process.env.ANTHROPIC_API_KEY,
      models: {
        default: 'claude-3-opus-20240229',
        code: 'claude-3-opus-20240229',
        creative: 'claude-3-sonnet-20240229',
        fast: 'claude-3-haiku-20240307'
      },
      maxTokens: 2000,
      temperature: 0.7
    },
    ollama: {
      enabled: true,
      baseUrl: process.env.OLLAMA_HOST || 'http://localhost:11434',
      models: {
        default: 'llama2',
        code: 'codellama',
        creative: 'llama2',
        fast: 'mistral'
      },
      maxTokens: 2000,
      temperature: 0.7
    },
    transformers: {
      enabled: true,
      models: {
        default: 'Xenova/distilgpt2',
        code: 'Xenova/codegen-350M-mono',
        creative: 'Xenova/gpt2',
        fast: 'Xenova/distilgpt2'
      },
      maxTokens: 500,
      temperature: 0.7
    }
  },
  routing: {
    taskTypes: {
      code: 'anthropic',
      analysis: 'anthropic',
      debug: 'anthropic',
      creative: 'openai',
      general: 'openai',
      chat: 'openai',
      local: 'ollama',
      private: 'ollama',
      offline: 'ollama',
      browser: 'transformers',
      lightweight: 'transformers'
    },
    fallbackChain: ['openai', 'anthropic', 'ollama', 'transformers']
  },
  defaults: {
    maxTokens: 1500,
    temperature: 0.7,
    streamingEnabled: true
  }
};

// Configuration loader
export class AIConfigManager {
  private config: AIConfig;

  constructor(customConfig?: Partial<AIConfig>) {
    this.config = this.mergeConfigs(defaultAIConfig, customConfig || {});
  }

  private mergeConfigs(base: AIConfig, custom: Partial<AIConfig>): AIConfig {
    return {
      providers: {
        ...base.providers,
        ...(custom.providers || {})
      },
      routing: {
        ...base.routing,
        ...(custom.routing || {})
      },
      defaults: {
        ...base.defaults,
        ...(custom.defaults || {})
      }
    };
  }

  getProviderConfig(provider: string): AIProviderConfig | undefined {
    return this.config.providers[provider as keyof typeof this.config.providers];
  }

  getProviderForTask(taskType: string): string {
    return this.config.routing.taskTypes[taskType] || 'openai';
  }

  getFallbackChain(): string[] {
    return this.config.routing.fallbackChain;
  }

  getDefaults() {
    return this.config.defaults;
  }

  // Update configuration at runtime
  updateProviderConfig(provider: string, config: Partial<AIProviderConfig>) {
    const providerKey = provider as keyof typeof this.config.providers;
    if (this.config.providers[providerKey]) {
      this.config.providers[providerKey] = {
        ...this.config.providers[providerKey],
        ...config
      };
    }
  }

  // Enable/disable providers
  setProviderEnabled(provider: string, enabled: boolean) {
    const providerKey = provider as keyof typeof this.config.providers;
    if (this.config.providers[providerKey]) {
      this.config.providers[providerKey].enabled = enabled;
    }
  }

  // Get enabled providers
  getEnabledProviders(): string[] {
    return Object.entries(this.config.providers)
      .filter(([_, config]) => config.enabled)
      .map(([name, _]) => name);
  }

  // Validate configuration
  validate(): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check if at least one provider is enabled
    const enabledProviders = this.getEnabledProviders();
    if (enabledProviders.length === 0) {
      errors.push('No AI providers are enabled');
    }

    // Check API keys for enabled providers
    for (const [name, config] of Object.entries(this.config.providers)) {
      if (config.enabled && !config.apiKey && name !== 'ollama' && name !== 'transformers') {
        errors.push(`${name} is enabled but no API key is provided`);
      }
    }

    // Validate fallback chain
    for (const provider of this.config.routing.fallbackChain) {
      if (!this.config.providers[provider as keyof typeof this.config.providers]) {
        errors.push(`Unknown provider in fallback chain: ${provider}`);
      }
    }

    return {
      valid: errors.length === 0,
      errors
    };
  }
}

// Export singleton instance
export const aiConfig = new AIConfigManager();