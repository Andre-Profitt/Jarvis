#!/usr/bin/env ts-node
import { pythonAIBridge } from './src/adapters/pythonBridge';
import { aiModelService } from './src/services/aiModelService';
import { aiConfig } from './src/config/aiConfig';
import * as dotenv from 'dotenv';
import * as readline from 'readline';

// Load environment variables
dotenv.config();

async function startJARVISWithAI() {
  console.log(`
╔═══════════════════════════════════════════════════════════════╗
║                  JARVIS AI INTEGRATION SYSTEM                  ║
║                    With Real AI Intelligence                   ║
╚═══════════════════════════════════════════════════════════════╝
  `);

  // Validate AI configuration
  console.log('🔍 Validating AI configuration...');
  const validation = aiConfig.validate();
  if (!validation.valid) {
    console.error('❌ Configuration errors:');
    validation.errors.forEach(err => console.error(`   - ${err}`));
    console.log('\n💡 Please set up your API keys in .env file');
    process.exit(1);
  }

  // Check available AI providers
  console.log('\n🤖 Checking AI providers...');
  const providers = await aiModelService.getAvailableProviders();
  console.log(`✅ Available providers: ${providers.join(', ') || 'None'}`);

  if (providers.length === 0) {
    console.error('\n❌ No AI providers available!');
    console.log('Please ensure:');
    console.log('1. You have set API keys in .env file');
    console.log('2. For Ollama: Run "ollama serve" in another terminal');
    console.log('3. Check your internet connection for cloud providers');
    process.exit(1);
  }

  // Set up event monitoring
  aiModelService.on('request:start', ({ provider }) => {
    console.log(`\n🔄 [${new Date().toISOString()}] Request to ${provider}...`);
  });

  aiModelService.on('request:complete', ({ provider, response }) => {
    console.log(`✅ [${new Date().toISOString()}] ${provider} completed (${response.usage?.totalTokens || 'N/A'} tokens)`);
  });

  aiModelService.on('fallback:attempt', ({ from, to }) => {
    console.log(`⚠️  [${new Date().toISOString()}] Fallback: ${from} → ${to}`);
  });

  // Initialize Python bridge
  console.log('\n🌉 Initializing Python bridge...');
  try {
    await pythonAIBridge.initialize();
    console.log('✅ Python bridge ready');
  } catch (error) {
    console.log('⚠️  Running in standalone AI mode (Python integration unavailable)');
  }

  // Interactive CLI
  console.log('\n✨ JARVIS AI is ready! Type your commands or "help" for options.\n');
  
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: 'JARVIS> '
  });

  const commands = {
    help: () => {
      console.log(`
Available commands:
  help          - Show this help message
  status        - Show AI system status
  providers     - List available AI providers
  test          - Run AI integration test
  analyze <file> - Analyze a code file
  chat          - Start interactive chat mode
  quit/exit     - Exit JARVIS
      `);
    },

    status: async () => {
      const status = await pythonAIBridge.getAIStatus();
      console.log('\n📊 AI System Status:');
      console.log(`   Available: ${status.available ? 'Yes' : 'No'}`);
      console.log(`   Providers: ${status.providers.join(', ') || 'None'}`);
      console.log(`   Primary: ${status.primaryProvider}`);
    },

    providers: async () => {
      const providers = await aiModelService.getAvailableProviders();
      console.log('\n🤖 Available AI Providers:');
      providers.forEach(p => {
        const config = aiConfig.getProviderConfig(p);
        console.log(`   - ${p}: ${config?.models.default || 'N/A'}`);
      });
    },

    test: async () => {
      console.log('\n🧪 Running AI test...');
      const response = await aiModelService.complete([
        { role: 'user', content: 'Briefly explain what JARVIS is.' }
      ], { taskType: 'general' });
      console.log('Response:', response.content);
    },

    analyze: async (filePath: string) => {
      if (!filePath) {
        console.log('Usage: analyze <file_path>');
        return;
      }
      console.log(`\n📝 Analyzing ${filePath}...`);
      // In a real implementation, you'd read the file first
      const response = await aiModelService.complete([
        {
          role: 'system',
          content: 'You are a code analysis expert. Analyze the given file path and provide insights.'
        },
        {
          role: 'user',
          content: `Analyze this file: ${filePath}`
        }
      ], { taskType: 'code' });
      console.log('\nAnalysis:', response.content);
    },

    chat: () => {
      console.log('\n💬 Entering chat mode (type "back" to return to command mode)');
      rl.setPrompt('You> ');
      // Chat mode would be handled in the main loop
    }
  };

  let chatMode = false;
  const chatHistory: any[] = [
    { role: 'system', content: 'You are JARVIS, an advanced AI assistant. Be helpful, concise, and intelligent.' }
  ];

  rl.prompt();

  rl.on('line', async (line) => {
    const input = line.trim();

    if (input.toLowerCase() === 'quit' || input.toLowerCase() === 'exit') {
      console.log('\n👋 JARVIS shutting down. Goodbye!');
      pythonAIBridge.shutdown();
      rl.close();
      process.exit(0);
    }

    if (chatMode) {
      if (input.toLowerCase() === 'back') {
        chatMode = false;
        rl.setPrompt('JARVIS> ');
        console.log('Returned to command mode.');
      } else {
        chatHistory.push({ role: 'user', content: input });
        
        try {
          process.stdout.write('\nJARVIS: ');
          let fullResponse = '';
          
          for await (const chunk of aiModelService.stream(chatHistory, { taskType: 'general' })) {
            process.stdout.write(chunk.content);
            fullResponse += chunk.content;
          }
          
          chatHistory.push({ role: 'assistant', content: fullResponse });
          console.log('\n');
        } catch (error) {
          console.error('\nError:', error.message);
        }
      }
    } else {
      const [command, ...args] = input.split(' ');
      const cmd = commands[command as keyof typeof commands];
      
      if (cmd) {
        if (command === 'analyze') {
          await cmd(args.join(' '));
        } else if (command === 'chat') {
          chatMode = true;
          cmd();
        } else {
          await cmd();
        }
      } else if (input) {
        // Process as AI command
        try {
          console.log('\nProcessing...');
          const response = await pythonAIBridge.processAIRequest([
            { role: 'user', content: input }
          ], 'general');
          console.log('\n' + response);
        } catch (error) {
          console.error('Error:', error.message);
        }
      }
    }

    rl.prompt();
  });

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    console.log('\n\n👋 JARVIS shutting down gracefully...');
    pythonAIBridge.shutdown();
    process.exit(0);
  });
}

// Start JARVIS with AI
startJARVISWithAI().catch(error => {
  console.error('❌ Failed to start JARVIS:', error);
  process.exit(1);
});