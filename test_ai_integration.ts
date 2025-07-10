import { aiModelService } from './src/services/aiModelService';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

async function testAIIntegration() {
  console.log('🧪 Testing JARVIS AI Integration...\n');

  // Test 1: Check available providers
  console.log('1️⃣ Checking available providers...');
  const providers = await aiModelService.getAvailableProviders();
  console.log('Available:', providers);
  
  if (providers.length === 0) {
    console.log('\n❌ No AI providers available. Please check your API keys in .env file');
    return;
  }

  // Test 2: Simple completion
  console.log('\n2️⃣ Testing simple completion...');
  try {
    const response = await aiModelService.complete([
      { role: 'user', content: 'Say "JARVIS AI is online!" in a cool way.' }
    ], {
      config: { maxTokens: 50, temperature: 0.8 }
    });
    console.log('Response:', response.content);
    console.log(`Provider: ${response.provider}, Model: ${response.model}`);
  } catch (error) {
    console.error('Completion failed:', error);
  }

  // Test 3: Code analysis
  console.log('\n3️⃣ Testing code analysis...');
  try {
    const codeResponse = await aiModelService.complete([
      {
        role: 'system',
        content: 'You are a TypeScript expert. Analyze code concisely.'
      },
      {
        role: 'user',
        content: 'What\'s wrong with this code?\n\nconst data = await fetch(url).json()'
      }
    ], {
      taskType: 'code',
      config: { maxTokens: 200, temperature: 0.3 }
    });
    console.log('Analysis:', codeResponse.content);
  } catch (error) {
    console.error('Code analysis failed:', error);
  }

  // Test 4: Streaming
  console.log('\n4️⃣ Testing streaming response...');
  try {
    process.stdout.write('Streaming: ');
    for await (const chunk of aiModelService.stream([
      { role: 'user', content: 'Count from 1 to 5 slowly.' }
    ], {
      config: { maxTokens: 100 }
    })) {
      process.stdout.write(chunk.content);
    }
    console.log('\n✅ Streaming complete');
  } catch (error) {
    console.error('\nStreaming failed:', error);
  }

  console.log('\n✨ AI Integration test complete!');
}

// Run the test
testAIIntegration().catch(console.error);