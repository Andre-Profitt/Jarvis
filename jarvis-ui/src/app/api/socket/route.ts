import { Server as SocketIOServer } from 'socket.io';
import { NextRequest } from 'next/server';
import { Server as NetServer } from 'http';
import { Socket as NetSocket } from 'net';

interface SocketServer extends NetServer {
  io?: SocketIOServer;
}

interface SocketWithIO extends NetSocket {
  server: SocketServer;
}

interface AudioFrame {
  data: ArrayBuffer;
  timestamp: number;
  sequenceId: number;
  metadata?: {
    isSpeech?: boolean;
    energy?: number;
  };
}

interface ClientSession {
  id: string;
  config: any;
  isStreaming: boolean;
  lastActivity: number;
  latencyProbes: Map<number, number>;
}

const sessions = new Map<string, ClientSession>();

// Global Socket.IO server instance
let io: SocketIOServer | null = null;

function initializeSocketServer(server: SocketServer) {
  if (!server.io) {
    console.log('Initializing Socket.IO server...');
    
    const socketIO = new SocketIOServer(server, {
      path: '/api/socket',
      cors: {
        origin: process.env.NODE_ENV === 'production' 
          ? process.env.NEXT_PUBLIC_APP_URL 
          : 'http://localhost:3000',
        methods: ['GET', 'POST']
      },
      transports: ['websocket'],
      pingInterval: 10000,
      pingTimeout: 5000,
      connectionStateRecovery: {
        maxDisconnectionDuration: 2 * 60 * 1000, // 2 minutes
        skipMiddlewares: true
      }
    });

    // Voice namespace for audio streaming
    const voiceNamespace = socketIO.of('/voice');
    
    voiceNamespace.on('connection', (socket) => {
      console.log(`Client connected: ${socket.id}`);
      
      // Create session
      const session: ClientSession = {
        id: socket.id,
        config: {},
        isStreaming: false,
        lastActivity: Date.now(),
        latencyProbes: new Map()
      };
      sessions.set(socket.id, session);

      // Handle configuration
      socket.on('configure', (config) => {
        session.config = config;
        console.log(`Client ${socket.id} configured:`, config);
        socket.emit('configured', { success: true });
      });

      // Handle streaming start
      socket.on('start-streaming', () => {
        session.isStreaming = true;
        session.lastActivity = Date.now();
        console.log(`Client ${socket.id} started streaming`);
        
        // Join streaming room
        socket.join('streaming');
        socket.emit('streaming-started');
      });

      // Handle streaming stop
      socket.on('stop-streaming', () => {
        session.isStreaming = false;
        console.log(`Client ${socket.id} stopped streaming`);
        
        // Leave streaming room
        socket.leave('streaming');
        socket.emit('streaming-stopped');
      });

      // Handle incoming audio frames
      socket.on('audio-frame', async (frame: AudioFrame) => {
        if (!session.isStreaming) return;
        
        session.lastActivity = Date.now();
        
        // Process audio frame (placeholder for actual processing)
        const processedFrame = await processAudioFrame(frame, session);
        
        // Broadcast to other clients in the room (for multi-party scenarios)
        socket.to('streaming').emit('audio-frame', processedFrame);
        
        // Echo back to sender (for testing/loopback)
        if (process.env.VOICE_LOOPBACK === 'true') {
          socket.emit('audio-frame', processedFrame);
        }
      });

      // Handle latency monitoring
      socket.on('latency-ping', (data: { id: number; timestamp: number }) => {
        session.latencyProbes.set(data.id, Date.now());
        socket.emit('latency-probe', { id: data.id });
      });

      socket.on('latency-pong', (data: { id: number }) => {
        const startTime = session.latencyProbes.get(data.id);
        if (startTime) {
          const rtt = Date.now() - startTime;
          session.latencyProbes.delete(data.id);
          
          // Send RTT back to client
          socket.emit('latency-result', { rtt, timestamp: Date.now() });
          
          // Log if RTT exceeds target
          if (rtt > 50) {
            console.warn(`High RTT for ${socket.id}: ${rtt}ms`);
          }
        }
      });

      // Handle disconnection
      socket.on('disconnect', (reason) => {
        console.log(`Client ${socket.id} disconnected: ${reason}`);
        sessions.delete(socket.id);
      });

      // Handle errors
      socket.on('error', (error) => {
        console.error(`Socket error for ${socket.id}:`, error);
      });
    });

    // Periodic cleanup of inactive sessions
    setInterval(() => {
      const now = Date.now();
      const timeout = 5 * 60 * 1000; // 5 minutes
      
      for (const [id, session] of sessions) {
        if (now - session.lastActivity > timeout) {
          console.log(`Cleaning up inactive session: ${id}`);
          sessions.delete(id);
          voiceNamespace.sockets.get(id)?.disconnect();
        }
      }
    }, 60000); // Check every minute

    server.io = socketIO;
    io = socketIO;
  }
  
  return server.io;
}

// Audio processing function (placeholder)
async function processAudioFrame(frame: AudioFrame, session: ClientSession): Promise<AudioFrame> {
  // In a real implementation, this would:
  // 1. Decode Opus if needed
  // 2. Apply audio processing (noise reduction, echo cancellation)
  // 3. Run speech-to-text if needed
  // 4. Apply any transformations
  // 5. Re-encode if needed
  
  // For now, just add processing timestamp
  return {
    ...frame,
    metadata: {
      ...frame.metadata,
      processedAt: Date.now(),
      sessionId: session.id
    }
  };
}

// WebSocket upgrade handler
export async function GET(request: NextRequest) {
  // This is needed for WebSocket upgrade in Next.js App Router
  const upgradeHeader = request.headers.get('upgrade');
  
  if (upgradeHeader !== 'websocket') {
    return new Response('Expected WebSocket upgrade', { status: 426 });
  }

  // Get server instance
  const res = new Response(null, {
    status: 101,
    headers: {
      'Upgrade': 'websocket',
      'Connection': 'Upgrade'
    }
  });

  // Initialize Socket.IO on the server if not already done
  // Note: In production, this should be done in server initialization
  try {
    // @ts-ignore - accessing internal server
    const server = (global as any).server;
    if (server && !server.io) {
      initializeSocketServer(server);
    }
  } catch (error) {
    console.error('Failed to initialize Socket.IO:', error);
  }

  return res;
}

// HTTP POST handler for non-WebSocket fallback
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Handle non-WebSocket audio submission (fallback)
    if (body.type === 'audio-frame' && body.data) {
      // Process audio frame
      const result = await processAudioFrame(body.data, {
        id: 'http-client',
        config: {},
        isStreaming: true,
        lastActivity: Date.now(),
        latencyProbes: new Map()
      });
      
      return Response.json({ success: true, frame: result });
    }
    
    return Response.json({ error: 'Invalid request' }, { status: 400 });
  } catch (error) {
    console.error('POST handler error:', error);
    return Response.json({ error: 'Server error' }, { status: 500 });
  }
}

// Export Socket.IO instance getter
export function getIO(): SocketIOServer | null {
  return io;
}