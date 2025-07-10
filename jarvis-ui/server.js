const { createServer } = require('http');
const { parse } = require('url');
const next = require('next');
const { Server } = require('socket.io');

const dev = process.env.NODE_ENV !== 'production';
const hostname = 'localhost';
const port = process.env.PORT || 3000;

// Create Next.js app
const app = next({ dev, hostname, port });
const handle = app.getRequestHandler();

app.prepare().then(() => {
  const server = createServer(async (req, res) => {
    try {
      const parsedUrl = parse(req.url, true);
      await handle(req, res, parsedUrl);
    } catch (err) {
      console.error('Error occurred handling', req.url, err);
      res.statusCode = 500;
      res.end('internal server error');
    }
  });

  // Initialize Socket.IO
  const io = new Server(server, {
    path: '/api/socket/io',
    cors: {
      origin: dev ? ['http://localhost:3000', 'http://127.0.0.1:3000'] : process.env.NEXT_PUBLIC_APP_URL,
      methods: ['GET', 'POST'],
      credentials: true
    },
    transports: ['websocket', 'polling']
  });

  // Voice namespace
  const voiceNamespace = io.of('/voice');
  
  // Store active sessions
  const sessions = new Map();

  voiceNamespace.on('connection', (socket) => {
    console.log(`Voice client connected: ${socket.id}`);
    
    // Create session
    const session = {
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
      socket.join('streaming');
      socket.emit('streaming-started');
    });

    // Handle streaming stop
    socket.on('stop-streaming', () => {
      session.isStreaming = false;
      console.log(`Client ${socket.id} stopped streaming`);
      socket.leave('streaming');
      socket.emit('streaming-stopped');
    });

    // Handle audio frames
    socket.on('audio-frame', async (frame) => {
      if (!session.isStreaming) return;
      
      session.lastActivity = Date.now();
      
      // Echo back for testing (remove in production)
      if (process.env.VOICE_LOOPBACK === 'true') {
        socket.emit('audio-frame', frame);
      }
      
      // Broadcast to other clients
      socket.to('streaming').emit('audio-frame', frame);
    });

    // Handle latency monitoring
    socket.on('latency-ping', (data) => {
      session.latencyProbes.set(data.id, Date.now());
      socket.emit('latency-probe', { id: data.id });
    });

    socket.on('latency-pong', (data) => {
      const startTime = session.latencyProbes.get(data.id);
      if (startTime) {
        const rtt = Date.now() - startTime;
        session.latencyProbes.delete(data.id);
        socket.emit('latency-result', { rtt, timestamp: Date.now() });
      }
    });

    // Handle disconnection
    socket.on('disconnect', (reason) => {
      console.log(`Client ${socket.id} disconnected: ${reason}`);
      sessions.delete(socket.id);
    });
  });

  // Periodic cleanup
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
  }, 60000);

  server.once('error', (err) => {
    console.error(err);
    process.exit(1);
  });

  server.listen(port, () => {
    console.log(`> Ready on http://${hostname}:${port}`);
    console.log('> Socket.IO server initialized');
  });
});