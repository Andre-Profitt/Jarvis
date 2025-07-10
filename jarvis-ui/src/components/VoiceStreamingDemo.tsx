'use client';

import React, { useState, useEffect } from 'react';
import { useVoiceStreaming, useVoiceActivityDetection } from '@/hooks/useVoiceStreaming';

export function VoiceStreamingDemo() {
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  
  const {
    isConnected,
    isStreaming,
    isInitializing,
    error,
    latencyStats,
    initialize,
    startStreaming,
    stopStreaming,
    toggleStreaming,
    canStream,
    isLowLatency
  } = useVoiceStreaming({
    autoConnect: true,
    latencyTarget: 50,
    vadThreshold: 0.3,
    onStreamStart: () => {
      console.log('Voice streaming started');
    },
    onStreamEnd: () => {
      console.log('Voice streaming ended');
    },
    onError: (err) => {
      console.error('Voice streaming error:', err);
    },
    onLatencyUpdate: (stats) => {
      console.log('Latency stats:', stats);
    }
  });

  const { isSpeaking, audioLevel } = useVoiceActivityDetection(mediaStream, {
    threshold: 0.1,
    onSpeechStart: () => {
      console.log('Speech detected');
    },
    onSpeechEnd: () => {
      console.log('Speech ended');
    }
  });

  // Get media stream when streaming starts
  useEffect(() => {
    if (isStreaming) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(setMediaStream)
        .catch(console.error);
    } else {
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        setMediaStream(null);
      }
    }
  }, [isStreaming]);

  return (
    <div className="p-6 max-w-md mx-auto bg-gray-900 rounded-xl shadow-lg space-y-4 text-white">
      <h2 className="text-2xl font-bold text-center">Voice Streaming Demo</h2>
      
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">Connection Status:</span>
        <span className={`text-sm font-medium ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      {/* Streaming Status */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">Streaming:</span>
        <span className={`text-sm font-medium ${isStreaming ? 'text-green-400' : 'text-gray-500'}`}>
          {isStreaming ? 'Active' : 'Inactive'}
        </span>
      </div>

      {/* Voice Activity */}
      <div className="flex items-center justify-between">
        <span className="text-sm text-gray-400">Voice Activity:</span>
        <div className="flex items-center space-x-2">
          <span className={`text-sm font-medium ${isSpeaking ? 'text-green-400' : 'text-gray-500'}`}>
            {isSpeaking ? 'Speaking' : 'Silent'}
          </span>
          <div className="w-24 h-2 bg-gray-700 rounded-full overflow-hidden">
            <div 
              className="h-full bg-green-500 transition-all duration-100"
              style={{ width: `${audioLevel * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Latency Stats */}
      <div className="space-y-2">
        <span className="text-sm text-gray-400">Latency (ms):</span>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-gray-800 rounded p-2">
            <div className="text-xs text-gray-500">Avg</div>
            <div className={`text-sm font-mono ${isLowLatency ? 'text-green-400' : 'text-yellow-400'}`}>
              {latencyStats.avgRtt.toFixed(1)}
            </div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-xs text-gray-500">Min</div>
            <div className="text-sm font-mono text-blue-400">
              {latencyStats.minRtt.toFixed(1)}
            </div>
          </div>
          <div className="bg-gray-800 rounded p-2">
            <div className="text-xs text-gray-500">Max</div>
            <div className="text-sm font-mono text-orange-400">
              {latencyStats.maxRtt.toFixed(1)}
            </div>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/20 border border-red-500 rounded p-3">
          <p className="text-sm text-red-400">{error.message}</p>
        </div>
      )}

      {/* Control Buttons */}
      <div className="space-y-2">
        {!isConnected && (
          <button
            onClick={initialize}
            disabled={isInitializing}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {isInitializing ? 'Connecting...' : 'Connect'}
          </button>
        )}

        {isConnected && (
          <button
            onClick={toggleStreaming}
            disabled={!canStream && !isStreaming}
            className={`w-full py-2 px-4 rounded-lg font-medium transition-colors ${
              isStreaming 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed'
            }`}
          >
            {isStreaming ? 'Stop Streaming' : 'Start Streaming'}
          </button>
        )}
      </div>

      {/* Performance Indicators */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">WebSocket</span>
          <span className={`font-medium ${isConnected ? 'text-green-400' : 'text-gray-600'}`}>
            {isConnected ? '✓' : '✗'}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-gray-500">Opus Codec</span>
          <span className="font-medium text-green-400">✓</span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-gray-500">VAD</span>
          <span className={`font-medium ${isSpeaking !== null ? 'text-green-400' : 'text-gray-600'}`}>
            {isSpeaking !== null ? '✓' : '✗'}
          </span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-gray-500">Low Latency</span>
          <span className={`font-medium ${isLowLatency ? 'text-green-400' : 'text-yellow-400'}`}>
            {isLowLatency ? '✓' : '~'}
          </span>
        </div>
      </div>
    </div>
  );
}