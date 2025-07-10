# Real-time Voice Streaming Setup

This implementation provides a complete WebSocket-based voice streaming system with ultra-low latency (<50ms target).

## Features

- ✅ WebSocket audio streaming using Socket.io
- ✅ Binary audio frame transmission
- ✅ Opus codec support (placeholder for actual implementation)
- ✅ Voice Activity Detection (VAD) using Web Audio API
- ✅ Real-time latency monitoring
- ✅ Automatic reconnection and error handling
- ✅ Jitter buffer for smooth playback

## Architecture

### Client-Side Components

1. **VoiceStreamingService** (`/src/services/voiceStreaming.ts`)
   - Manages WebSocket connection
   - Handles audio capture and playback
   - Implements VAD and latency monitoring
   - Encodes/decodes audio frames

2. **useVoiceStreaming Hook** (`/src/hooks/useVoiceStreaming.ts`)
   - React hook for easy integration
   - Manages streaming state
   - Provides latency statistics
   - Handles errors and reconnection

3. **VAD Processor** (`/public/vad-processor.js`)
   - AudioWorklet for efficient voice detection
   - Energy-based speech detection
   - Dynamic noise floor estimation

### Server-Side Components

1. **Socket.io Server** (`/src/app/api/socket/route.ts`)
   - WebSocket server implementation
   - Audio frame routing
   - Session management
   - Latency probe handling

2. **Custom Next.js Server** (`/server.js`)
   - Integrates Socket.io with Next.js
   - Handles WebSocket upgrades
   - Manages voice namespace

## Usage

### Basic Implementation

```tsx
import { useVoiceStreaming } from '@/hooks/useVoiceStreaming';

function VoiceChat() {
  const {
    isConnected,
    isStreaming,
    latencyStats,
    toggleStreaming
  } = useVoiceStreaming({
    latencyTarget: 50,
    vadThreshold: 0.3
  });

  return (
    <button onClick={toggleStreaming}>
      {isStreaming ? 'Stop' : 'Start'} Voice Chat
    </button>
  );
}
```

### Demo Component

See `/src/components/VoiceStreamingDemo.tsx` for a complete example with:
- Connection status
- Voice activity visualization
- Latency monitoring
- Error handling

## Configuration

### Environment Variables

```env
# Enable voice loopback for testing
VOICE_LOOPBACK=true

# Production URL
NEXT_PUBLIC_APP_URL=https://your-app.com
```

### Voice Streaming Config

```typescript
{
  sampleRate: 48000,      // Audio sample rate
  channels: 1,            // Mono audio
  frameSize: 960,         // 20ms frames at 48kHz
  vadThreshold: 0.5,      // Voice detection threshold
  latencyTarget: 50       // Target latency in ms
}
```

## Running the Application

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server with Socket.io:
   ```bash
   npm run dev
   ```

3. Access the application at `http://localhost:3000`

## Performance Optimization

1. **Low Latency Tips**:
   - Use WebSocket transport only
   - Minimize audio buffer sizes
   - Implement efficient opus encoding
   - Use AudioWorklet for processing

2. **Network Optimization**:
   - Binary frame transmission
   - Compression with Opus codec
   - Jitter buffer management
   - Adaptive bitrate

3. **CPU Optimization**:
   - VAD to reduce unnecessary processing
   - Efficient audio resampling
   - Web Workers for heavy computation

## Next Steps

1. Implement actual Opus.js codec integration
2. Add WebRTC fallback for peer-to-peer
3. Implement audio mixing for multi-party
4. Add echo cancellation processing
5. Create production-ready error handling