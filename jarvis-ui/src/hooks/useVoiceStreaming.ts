import { useState, useEffect, useCallback, useRef } from 'react';
import { getVoiceStreamingService, VoiceStreamingService } from '@/services/voiceStreaming';

interface UseVoiceStreamingOptions {
  autoConnect?: boolean;
  sampleRate?: number;
  vadThreshold?: number;
  latencyTarget?: number;
  onStreamStart?: () => void;
  onStreamEnd?: () => void;
  onError?: (error: Error) => void;
  onLatencyUpdate?: (stats: { avgRtt: number; minRtt: number; maxRtt: number }) => void;
}

interface VoiceStreamingState {
  isConnected: boolean;
  isStreaming: boolean;
  isInitializing: boolean;
  error: Error | null;
  latencyStats: {
    avgRtt: number;
    minRtt: number;
    maxRtt: number;
  };
}

export function useVoiceStreaming(options: UseVoiceStreamingOptions = {}) {
  const [state, setState] = useState<VoiceStreamingState>({
    isConnected: false,
    isStreaming: false,
    isInitializing: false,
    error: null,
    latencyStats: {
      avgRtt: 0,
      minRtt: 0,
      maxRtt: 0
    }
  });

  const serviceRef = useRef<VoiceStreamingService | null>(null);
  const latencyIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize service
  useEffect(() => {
    if (options.autoConnect !== false) {
      initializeService();
    }

    return () => {
      cleanup();
    };
  }, []);

  // Update latency stats periodically
  useEffect(() => {
    if (state.isStreaming && serviceRef.current) {
      latencyIntervalRef.current = setInterval(() => {
        const stats = serviceRef.current!.getLatencyStats();
        setState(prev => ({
          ...prev,
          latencyStats: stats
        }));
        options.onLatencyUpdate?.(stats);
      }, 1000);
    } else {
      if (latencyIntervalRef.current) {
        clearInterval(latencyIntervalRef.current);
        latencyIntervalRef.current = null;
      }
    }

    return () => {
      if (latencyIntervalRef.current) {
        clearInterval(latencyIntervalRef.current);
      }
    };
  }, [state.isStreaming, options.onLatencyUpdate]);

  const initializeService = useCallback(async () => {
    if (serviceRef.current || state.isInitializing) return;

    setState(prev => ({ ...prev, isInitializing: true, error: null }));

    try {
      const service = getVoiceStreamingService({
        sampleRate: options.sampleRate,
        vadThreshold: options.vadThreshold,
        latencyTarget: options.latencyTarget
      });

      await service.initialize();
      serviceRef.current = service;

      setState(prev => ({
        ...prev,
        isConnected: true,
        isInitializing: false
      }));
    } catch (error) {
      const err = error as Error;
      setState(prev => ({
        ...prev,
        error: err,
        isInitializing: false
      }));
      options.onError?.(err);
    }
  }, [options.sampleRate, options.vadThreshold, options.latencyTarget, options.onError]);

  const startStreaming = useCallback(async () => {
    if (!serviceRef.current || state.isStreaming) return;

    try {
      await serviceRef.current.startStreaming();
      setState(prev => ({ ...prev, isStreaming: true, error: null }));
      options.onStreamStart?.();
    } catch (error) {
      const err = error as Error;
      setState(prev => ({ ...prev, error: err }));
      options.onError?.(err);
    }
  }, [state.isStreaming, options.onStreamStart, options.onError]);

  const stopStreaming = useCallback(() => {
    if (!serviceRef.current || !state.isStreaming) return;

    serviceRef.current.stopStreaming();
    setState(prev => ({ ...prev, isStreaming: false }));
    options.onStreamEnd?.();
  }, [state.isStreaming, options.onStreamEnd]);

  const toggleStreaming = useCallback(async () => {
    if (state.isStreaming) {
      stopStreaming();
    } else {
      await startStreaming();
    }
  }, [state.isStreaming, startStreaming, stopStreaming]);

  const disconnect = useCallback(() => {
    cleanup();
    setState({
      isConnected: false,
      isStreaming: false,
      isInitializing: false,
      error: null,
      latencyStats: {
        avgRtt: 0,
        minRtt: 0,
        maxRtt: 0
      }
    });
  }, []);

  const cleanup = useCallback(() => {
    if (latencyIntervalRef.current) {
      clearInterval(latencyIntervalRef.current);
      latencyIntervalRef.current = null;
    }

    if (serviceRef.current) {
      serviceRef.current.disconnect();
      serviceRef.current = null;
    }
  }, []);

  return {
    // State
    isConnected: state.isConnected,
    isStreaming: state.isStreaming,
    isInitializing: state.isInitializing,
    error: state.error,
    latencyStats: state.latencyStats,
    
    // Actions
    initialize: initializeService,
    startStreaming,
    stopStreaming,
    toggleStreaming,
    disconnect,
    
    // Computed
    canStream: state.isConnected && !state.isStreaming && !state.isInitializing,
    isLowLatency: state.latencyStats.avgRtt > 0 && state.latencyStats.avgRtt < (options.latencyTarget || 50)
  };
}

// Hook for voice activity detection
export function useVoiceActivityDetection(
  stream: MediaStream | null,
  options: {
    threshold?: number;
    smoothingTime?: number;
    onSpeechStart?: () => void;
    onSpeechEnd?: () => void;
  } = {}
) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const speechTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (!stream) {
      cleanup();
      return;
    }

    const setupAudioAnalysis = async () => {
      try {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        const source = audioContextRef.current.createMediaStreamSource(stream);
        
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        analyserRef.current.smoothingTimeConstant = options.smoothingTime || 0.8;
        
        source.connect(analyserRef.current);
        
        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const checkAudioLevel = () => {
          if (!analyserRef.current) return;
          
          analyserRef.current.getByteFrequencyData(dataArray);
          
          // Calculate average volume
          const average = dataArray.reduce((acc, val) => acc + val, 0) / bufferLength;
          const normalizedLevel = average / 255;
          
          setAudioLevel(normalizedLevel);
          
          // Detect speech
          const threshold = options.threshold || 0.1;
          const wasSpeaking = isSpeaking;
          const isNowSpeaking = normalizedLevel > threshold;
          
          if (isNowSpeaking && !wasSpeaking) {
            setIsSpeaking(true);
            options.onSpeechStart?.();
            
            // Clear any existing timeout
            if (speechTimeoutRef.current) {
              clearTimeout(speechTimeoutRef.current);
              speechTimeoutRef.current = null;
            }
          } else if (!isNowSpeaking && wasSpeaking) {
            // Debounce speech end
            speechTimeoutRef.current = setTimeout(() => {
              setIsSpeaking(false);
              options.onSpeechEnd?.();
            }, 300);
          }
          
          animationFrameRef.current = requestAnimationFrame(checkAudioLevel);
        };
        
        checkAudioLevel();
      } catch (error) {
        console.error('Failed to setup audio analysis:', error);
      }
    };

    setupAudioAnalysis();

    return cleanup;
  }, [stream, options.threshold, options.smoothingTime, isSpeaking]);

  const cleanup = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (speechTimeoutRef.current) {
      clearTimeout(speechTimeoutRef.current);
      speechTimeoutRef.current = null;
    }
    
    if (analyserRef.current) {
      analyserRef.current.disconnect();
      analyserRef.current = null;
    }
    
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    
    setIsSpeaking(false);
    setAudioLevel(0);
  };

  return {
    isSpeaking,
    audioLevel
  };
}