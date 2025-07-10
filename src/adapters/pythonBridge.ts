import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import { aiModelService, AIMessage } from '../services/aiModelService';
import { JARVISAIIntegration } from '../demo/aiModelDemo';

interface PythonMessage {
  type: 'command' | 'response' | 'error' | 'stream';
  id: string;
  data: any;
}

export class PythonAIBridge extends EventEmitter {
  private pythonProcess: ChildProcess | null = null;
  private jarvisAI: JARVISAIIntegration;
  private messageQueue: Map<string, (response: any) => void> = new Map();

  constructor() {
    super();
    this.jarvisAI = new JARVISAIIntegration();
  }

  async initialize(pythonScriptPath: string = 'jarvis_enhanced.py') {
    return new Promise((resolve, reject) => {
      try {
        // Start Python JARVIS process
        this.pythonProcess = spawn('python3', [pythonScriptPath], {
          env: { ...process.env, JARVIS_AI_BRIDGE: 'true' }
        });

        this.pythonProcess.stdout?.on('data', (data) => {
          try {
            const messages = data.toString().split('\n').filter(Boolean);
            messages.forEach(msg => {
              const parsed = JSON.parse(msg) as PythonMessage;
              this.handlePythonMessage(parsed);
            });
          } catch (error) {
            console.error('Failed to parse Python message:', error);
          }
        });

        this.pythonProcess.stderr?.on('data', (data) => {
          console.error('Python error:', data.toString());
        });

        this.pythonProcess.on('close', (code) => {
          this.emit('close', code);
          this.pythonProcess = null;
        });

        // Give Python process time to initialize
        setTimeout(() => {
          this.emit('ready');
          resolve(true);
        }, 2000);

      } catch (error) {
        reject(error);
      }
    });
  }

  private async handlePythonMessage(message: PythonMessage) {
    switch (message.type) {
      case 'command':
        await this.processAICommand(message);
        break;
      case 'response':
        const callback = this.messageQueue.get(message.id);
        if (callback) {
          callback(message.data);
          this.messageQueue.delete(message.id);
        }
        break;
      case 'error':
        this.emit('error', message.data);
        break;
      case 'stream':
        this.emit('stream', message.data);
        break;
    }
  }

  private async processAICommand(message: PythonMessage) {
    const { command, context, options } = message.data;

    try {
      if (options?.stream) {
        // Handle streaming response
        let fullResponse = '';
        for await (const chunk of this.jarvisAI.streamResponse(command, (content) => {
          this.sendToPython({
            type: 'stream',
            id: message.id,
            data: { content, done: false }
          });
        })) {
          fullResponse += chunk;
        }
        
        this.sendToPython({
          type: 'response',
          id: message.id,
          data: { content: fullResponse, done: true }
        });
      } else {
        // Handle regular completion
        const response = await this.jarvisAI.processCommand(command);
        this.sendToPython({
          type: 'response',
          id: message.id,
          data: { content: response }
        });
      }
    } catch (error) {
      this.sendToPython({
        type: 'error',
        id: message.id,
        data: { error: error.message }
      });
    }
  }

  private sendToPython(message: PythonMessage) {
    if (this.pythonProcess?.stdin) {
      this.pythonProcess.stdin.write(JSON.stringify(message) + '\n');
    }
  }

  // Public API for Python integration
  async sendCommand(command: string, options?: any): Promise<string> {
    const messageId = Math.random().toString(36).substr(2, 9);
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.messageQueue.delete(messageId);
        reject(new Error('Command timeout'));
      }, 30000);

      this.messageQueue.set(messageId, (response) => {
        clearTimeout(timeout);
        resolve(response.content);
      });

      this.sendToPython({
        type: 'command',
        id: messageId,
        data: { command, options }
      });
    });
  }

  // Direct AI access for TypeScript components
  async processAIRequest(messages: AIMessage[], taskType?: string): Promise<string> {
    const response = await aiModelService.complete(messages, {
      taskType,
      config: {
        maxTokens: 1500,
        temperature: 0.7
      }
    });
    return response.content;
  }

  async *streamAIResponse(messages: AIMessage[], taskType?: string) {
    for await (const chunk of aiModelService.stream(messages, {
      taskType,
      config: {
        maxTokens: 1500,
        temperature: 0.7
      }
    })) {
      yield chunk.content;
    }
  }

  // Get AI service status
  async getAIStatus() {
    const providers = await aiModelService.getAvailableProviders();
    return {
      available: providers.length > 0,
      providers,
      primaryProvider: providers[0] || 'none'
    };
  }

  shutdown() {
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
  }
}

// Export singleton instance
export const pythonAIBridge = new PythonAIBridge();