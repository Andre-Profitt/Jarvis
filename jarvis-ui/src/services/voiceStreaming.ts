import { io, Socket } from 'socket.io-client';

interface VoiceStreamingConfig {
  sampleRate?: number;
  channels?: number;
  frameSize?: number;
  vadThreshold?: number;
  latencyTarget?: number;
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

export class VoiceStreamingService {
  private socket: Socket | null = null;
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private encoder: any = null; // Opus encoder
  private decoder: any = null; // Opus decoder
  private config: VoiceStreamingConfig;
  private sequenceId = 0;
  private receiveBuffer: Map<number, AudioFrame> = new Map();
  private playbackQueue: AudioFrame[] = [];
  private isStreaming = false;
  private vadProcessor: AudioWorkletNode | null = null;
  private latencyMonitor: { sent: Map<number, number>; rtts: number[] } = {
    sent: new Map(),
    rtts: []
  };

  constructor(config: VoiceStreamingConfig = {}) {
    this.config = {
      sampleRate: config.sampleRate || 48000,
      channels: config.channels || 1,
      frameSize: config.frameSize || 960, // 20ms at 48kHz
      vadThreshold: config.vadThreshold || 0.5,
      latencyTarget: config.latencyTarget || 50
    };
  }

  async initialize(): Promise<void> {
    // Initialize WebSocket connection
    this.socket = io('/voice', {
      transports: ['websocket'],
      upgrade: false,
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000
    });

    // Set up socket event handlers
    this.setupSocketHandlers();

    // Initialize audio context
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
      sampleRate: this.config.sampleRate,
      latencyHint: 'interactive'
    });

    // Load Opus encoder/decoder (using opus.js or similar)
    await this.loadOpusCodec();

    // Initialize VAD (Voice Activity Detection)
    await this.initializeVAD();
  }

  private setupSocketHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('Connected to voice streaming server');
      this.socket?.emit('configure', this.config);
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from voice streaming server');
      this.stopStreaming();
    });

    this.socket.on('audio-frame', (frame: AudioFrame) => {
      this.handleIncomingAudioFrame(frame);
    });

    this.socket.on('latency-probe', (data: { id: number }) => {
      // Respond to latency probe for RTT measurement
      this.socket?.emit('latency-pong', { id: data.id });
    });

    this.socket.on('error', (error: Error) => {
      console.error('Socket error:', error);
    });
  }

  async startStreaming(): Promise<void> {
    if (this.isStreaming) return;

    try {
      // Get user media
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      // Create audio processing chain
      const source = this.audioContext!.createMediaStreamSource(this.mediaStream);
      
      // Create script processor for capturing audio
      this.processor = this.audioContext!.createScriptProcessor(
        this.config.frameSize!,
        this.config.channels!,
        this.config.channels!
      );

      this.processor.onaudioprocess = (event) => {
        if (!this.isStreaming) return;
        this.processOutgoingAudio(event.inputBuffer);
      };

      // Connect audio chain with VAD
      if (this.vadProcessor) {
        source.connect(this.vadProcessor);
        this.vadProcessor.connect(this.processor);
      } else {
        source.connect(this.processor);
      }
      
      this.processor.connect(this.audioContext!.destination);

      this.isStreaming = true;
      this.socket?.emit('start-streaming');

      // Start latency monitoring
      this.startLatencyMonitoring();

    } catch (error) {
      console.error('Failed to start streaming:', error);
      throw error;
    }
  }

  stopStreaming(): void {
    if (!this.isStreaming) return;

    this.isStreaming = false;

    // Clean up audio nodes
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.vadProcessor) {
      this.vadProcessor.disconnect();
      this.vadProcessor = null;
    }

    // Stop media stream
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach(track => track.stop());
      this.mediaStream = null;
    }

    // Notify server
    this.socket?.emit('stop-streaming');

    // Clear buffers
    this.receiveBuffer.clear();
    this.playbackQueue = [];
  }

  private async processOutgoingAudio(audioBuffer: AudioBuffer): Promise<void> {
    // Extract PCM data
    const pcmData = audioBuffer.getChannelData(0);
    
    // Check VAD
    const isSpeech = await this.checkVoiceActivity(pcmData);
    
    // Only send if voice is detected or we're in continuous mode
    if (isSpeech || !this.config.vadThreshold) {
      // Encode to Opus
      const encodedData = await this.encodeOpus(pcmData);
      
      // Create audio frame
      const frame: AudioFrame = {
        data: encodedData,
        timestamp: performance.now(),
        sequenceId: this.sequenceId++,
        metadata: {
          isSpeech,
          energy: this.calculateEnergy(pcmData)
        }
      };

      // Track for latency monitoring
      this.latencyMonitor.sent.set(frame.sequenceId, frame.timestamp);

      // Send via WebSocket
      this.socket?.emit('audio-frame', frame);
    }
  }

  private handleIncomingAudioFrame(frame: AudioFrame): void {
    // Add to receive buffer for jitter compensation
    this.receiveBuffer.set(frame.sequenceId, frame);

    // Process buffer if we have enough frames
    if (this.receiveBuffer.size >= 2) {
      this.processReceiveBuffer();
    }
  }

  private async processReceiveBuffer(): Promise<void> {
    // Sort frames by sequence ID
    const sortedFrames = Array.from(this.receiveBuffer.entries())
      .sort((a, b) => a[0] - b[0])
      .map(entry => entry[1]);

    // Process frames in order
    for (const frame of sortedFrames) {
      // Decode Opus
      const pcmData = await this.decodeOpus(frame.data);
      
      // Play audio
      await this.playAudioData(pcmData);
      
      // Remove from buffer
      this.receiveBuffer.delete(frame.sequenceId);
    }
  }

  private async playAudioData(pcmData: Float32Array): Promise<void> {
    if (!this.audioContext) return;

    // Create audio buffer
    const audioBuffer = this.audioContext.createBuffer(
      this.config.channels!,
      pcmData.length,
      this.config.sampleRate!
    );
    
    audioBuffer.getChannelData(0).set(pcmData);

    // Create buffer source
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.audioContext.destination);
    
    // Schedule playback with minimal latency
    const playTime = this.audioContext.currentTime + 0.005; // 5ms buffer
    source.start(playTime);
  }

  private async loadOpusCodec(): Promise<void> {
    // This would load the actual Opus.js library
    // For now, we'll use placeholder encoding/decoding
    console.log('Loading Opus codec...');
    // In production, load opus.js or similar library
  }

  private async initializeVAD(): Promise<void> {
    if (!this.audioContext) return;

    try {
      // Load VAD AudioWorklet
      await this.audioContext.audioWorklet.addModule('/vad-processor.js');
      
      this.vadProcessor = new AudioWorkletNode(this.audioContext, 'vad-processor', {
        parameterData: { threshold: this.config.vadThreshold }
      });

      this.vadProcessor.port.onmessage = (event) => {
        if (event.data.type === 'vad-result') {
          // Handle VAD results
          console.log('VAD:', event.data.isSpeech);
        }
      };
    } catch (error) {
      console.warn('VAD initialization failed, continuing without VAD:', error);
    }
  }

  private async checkVoiceActivity(pcmData: Float32Array): Promise<boolean> {
    // Simple energy-based VAD
    const energy = this.calculateEnergy(pcmData);
    return energy > this.config.vadThreshold!;
  }

  private calculateEnergy(pcmData: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < pcmData.length; i++) {
      sum += pcmData[i] * pcmData[i];
    }
    return Math.sqrt(sum / pcmData.length);
  }

  private async encodeOpus(pcmData: Float32Array): Promise<ArrayBuffer> {
    // Placeholder - in production, use actual Opus encoder
    // For now, return PCM data as ArrayBuffer
    return pcmData.buffer.slice(pcmData.byteOffset, pcmData.byteOffset + pcmData.byteLength);
  }

  private async decodeOpus(data: ArrayBuffer): Promise<Float32Array> {
    // Placeholder - in production, use actual Opus decoder
    return new Float32Array(data);
  }

  private startLatencyMonitoring(): void {
    setInterval(() => {
      // Send latency probe
      const probeId = Date.now();
      this.socket?.emit('latency-ping', { id: probeId, timestamp: performance.now() });
      
      // Clean old entries
      const cutoff = performance.now() - 5000;
      for (const [id, timestamp] of this.latencyMonitor.sent) {
        if (timestamp < cutoff) {
          this.latencyMonitor.sent.delete(id);
        }
      }
      
      // Calculate average RTT
      if (this.latencyMonitor.rtts.length > 0) {
        const avgRtt = this.latencyMonitor.rtts.reduce((a, b) => a + b) / this.latencyMonitor.rtts.length;
        console.log(`Average RTT: ${avgRtt.toFixed(2)}ms`);
        
        // Keep only recent RTTs
        this.latencyMonitor.rtts = this.latencyMonitor.rtts.slice(-10);
      }
    }, 1000);
  }

  getLatencyStats(): { avgRtt: number; minRtt: number; maxRtt: number } {
    if (this.latencyMonitor.rtts.length === 0) {
      return { avgRtt: 0, minRtt: 0, maxRtt: 0 };
    }

    const rtts = this.latencyMonitor.rtts;
    return {
      avgRtt: rtts.reduce((a, b) => a + b) / rtts.length,
      minRtt: Math.min(...rtts),
      maxRtt: Math.max(...rtts)
    };
  }

  disconnect(): void {
    this.stopStreaming();
    this.socket?.disconnect();
    this.socket = null;
    this.audioContext?.close();
    this.audioContext = null;
  }
}

// Singleton instance
let voiceStreamingInstance: VoiceStreamingService | null = null;

export function getVoiceStreamingService(config?: VoiceStreamingConfig): VoiceStreamingService {
  if (!voiceStreamingInstance) {
    voiceStreamingInstance = new VoiceStreamingService(config);
  }
  return voiceStreamingInstance;
}