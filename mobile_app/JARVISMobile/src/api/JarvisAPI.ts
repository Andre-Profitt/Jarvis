import io from 'socket.io-client';

interface DeviceData {
  device_id: string;
  platform: string;
  model: string;
  app_version: string;
  os_version: string;
  device_name: string;
  capabilities: {
    voice: boolean;
    push_notifications: boolean;
    background_mode: boolean;
  };
}

interface CommandResponse {
  success: boolean;
  response: string;
  confidence: number;
  suggestions?: string[];
}

interface SmartDevice {
  id: string;
  name: string;
  type: string;
  state: any;
  room: string;
  capabilities: string[];
}

export default class JarvisAPI {
  private baseUrl: string;
  private authToken: string | null = null;
  private socket: any = null;
  private listeners: Map<string, Function[]> = new Map();

  constructor(baseUrl: string = 'http://localhost:5001') {
    this.baseUrl = baseUrl;
  }

  setAuthToken(token: string) {
    this.authToken = token;
    if (this.socket) {
      this.socket.emit('authenticate', { token });
    }
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    };
    
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    
    return headers;
  }

  async registerDevice(deviceData: DeviceData) {
    const response = await fetch(`${this.baseUrl}/api/v1/register`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify(deviceData),
    });

    if (!response.ok) {
      throw new Error(`Registration failed: ${response.statusText}`);
    }

    return response.json();
  }

  async sendCommand(command: string): Promise<CommandResponse> {
    const response = await fetch(`${this.baseUrl}/api/v1/command`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({ command }),
    });

    if (!response.ok) {
      throw new Error(`Command failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getStatus() {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/status`, {
        headers: this.getHeaders(),
      });

      if (!response.ok) {
        throw new Error(`Status failed: ${response.statusText}`);
      }

      return response.json();
    } catch (error) {
      console.error('Status error:', error);
      return null;
    }
  }

  async getSmartDevices(): Promise<SmartDevice[]> {
    const response = await fetch(`${this.baseUrl}/api/v1/smart-home/devices`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Failed to get devices: ${response.statusText}`);
    }

    const data = await response.json();
    return data.devices || [];
  }

  async controlSmartDevice(deviceId: string, action: string, params: any = {}) {
    const response = await fetch(`${this.baseUrl}/api/v1/smart-home/control`, {
      method: 'POST',
      headers: this.getHeaders(),
      body: JSON.stringify({
        device_id: deviceId,
        action,
        params,
      }),
    });

    if (!response.ok) {
      throw new Error(`Control failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getCalendarEvents(days: number = 7) {
    const response = await fetch(
      `${this.baseUrl}/api/v1/calendar/events?days=${days}`,
      {
        headers: this.getHeaders(),
      }
    );

    if (!response.ok) {
      throw new Error(`Calendar failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getShortcuts() {
    const response = await fetch(`${this.baseUrl}/api/v1/shortcuts`, {
      headers: this.getHeaders(),
    });

    if (!response.ok) {
      throw new Error(`Shortcuts failed: ${response.statusText}`);
    }

    return response.json();
  }

  // WebSocket connection for real-time updates
  connectWebSocket() {
    if (this.socket) {
      return;
    }

    this.socket = io(this.baseUrl, {
      transports: ['websocket'],
      autoConnect: true,
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      if (this.authToken) {
        this.socket.emit('authenticate', { token: this.authToken });
      }
      this.emit('connected', true);
    });

    this.socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      this.emit('connected', false);
    });

    this.socket.on('authenticated', (data: any) => {
      console.log('WebSocket authenticated:', data);
    });

    this.socket.on('status_update', (data: any) => {
      this.emit('status_update', data);
    });

    this.socket.on('command_update', (data: any) => {
      this.emit('command_update', data);
    });

    this.socket.on('voice_activity', (data: any) => {
      this.emit('voice_activity', data);
    });
  }

  disconnectWebSocket() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  // Event handling
  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)!.push(callback);
  }

  off(event: string, callback: Function) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  // Voice streaming (for future implementation)
  startVoiceStream() {
    if (this.socket) {
      this.socket.emit('start_voice_stream');
    }
  }

  sendVoiceData(audioData: ArrayBuffer) {
    if (this.socket) {
      this.socket.emit('voice_stream', audioData);
    }
  }

  stopVoiceStream() {
    if (this.socket) {
      this.socket.emit('stop_voice_stream');
    }
  }
}