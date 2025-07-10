import { Server as HTTPServer } from 'http';
import { Server as SocketIOServer } from 'socket.io';
import { NextApiRequest } from 'next';
import { Socket as NetSocket } from 'net';

export interface SocketServer extends HTTPServer {
  io?: SocketIOServer;
}

export interface SocketWithIO extends NetSocket {
  server: SocketServer;
}

export interface NextApiResponseWithSocket extends NextApiRequest {
  socket: SocketWithIO;
}

let io: SocketIOServer | undefined;

export function getSocketIO(): SocketIOServer | undefined {
  return io;
}

export function initializeSocketIO(server: SocketServer): SocketIOServer {
  if (!server.io) {
    console.log('Setting up Socket.IO server...');
    
    io = new SocketIOServer(server, {
      path: '/api/socket/io',
      cors: {
        origin: process.env.NODE_ENV === 'production' 
          ? [process.env.NEXT_PUBLIC_APP_URL || ''] 
          : ['http://localhost:3000', 'http://127.0.0.1:3000'],
        methods: ['GET', 'POST'],
        credentials: true
      },
      transports: ['websocket', 'polling'],
      allowEIO3: true,
      pingTimeout: 60000,
      pingInterval: 25000
    });

    server.io = io;

    // Setup voice namespace handlers
    const voiceNamespace = io.of('/voice');
    
    voiceNamespace.use((socket, next) => {
      // Authentication middleware can go here
      next();
    });

    // Global error handler
    io.engine.on('connection_error', (err) => {
      console.error('Socket.IO connection error:', err);
    });
  }

  return server.io;
}