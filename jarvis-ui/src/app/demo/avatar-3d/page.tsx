'use client';

import React, { useState, useEffect } from 'react';
import { JarvisAvatar3D, AvatarEmotion } from '@/components/3d/JarvisAvatar3D';
import { useVoiceStreaming } from '@/hooks/useVoiceStreaming';

export default function Avatar3DDemo() {
  const [currentEmotion, setCurrentEmotion] = useState<AvatarEmotion>('neutral');
  const [mediaStream, setMediaStream] = useState<MediaStream | null>(null);
  const { isStreaming, startStreaming, stopStreaming } = useVoiceStreaming({
    autoConnect: false
  });

  // Get user media for voice input
  useEffect(() => {
    const setupMedia = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: true 
        });
        setMediaStream(stream);
      } catch (error) {
        console.error('Failed to get user media:', error);
      }
    };

    setupMedia();

    return () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const emotions: AvatarEmotion[] = ['neutral', 'happy', 'sad', 'thinking', 'speaking', 'excited', 'concerned'];

  return (
    <div className="min-h-screen bg-black text-white">
      {/* 3D Avatar Container */}
      <div className="relative w-full h-screen">
        <JarvisAvatar3D 
          stream={mediaStream} 
          emotion={currentEmotion}
          className="absolute inset-0"
        />
        
        {/* UI Overlay */}
        <div className="absolute bottom-0 left-0 right-0 p-8">
          {/* Emotion Controls */}
          <div className="max-w-4xl mx-auto mb-8">
            <h3 className="text-sm uppercase tracking-wider text-cyan-400 mb-4">Avatar Emotion</h3>
            <div className="flex flex-wrap gap-2">
              {emotions.map((emotion) => (
                <button
                  key={emotion}
                  onClick={() => setCurrentEmotion(emotion)}
                  className={`px-4 py-2 rounded-lg border transition-all ${
                    currentEmotion === emotion
                      ? 'bg-cyan-500/20 border-cyan-400 text-cyan-400'
                      : 'border-white/20 hover:border-white/40 text-white/60 hover:text-white'
                  }`}
                >
                  {emotion.charAt(0).toUpperCase() + emotion.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Voice Controls */}
          <div className="max-w-4xl mx-auto">
            <h3 className="text-sm uppercase tracking-wider text-cyan-400 mb-4">Voice Interaction</h3>
            <div className="flex gap-4">
              <button
                onClick={() => isStreaming ? stopStreaming() : startStreaming()}
                className={`px-6 py-3 rounded-lg border transition-all ${
                  isStreaming
                    ? 'bg-red-500/20 border-red-400 text-red-400 hover:bg-red-500/30'
                    : 'bg-green-500/20 border-green-400 text-green-400 hover:bg-green-500/30'
                }`}
              >
                {isStreaming ? 'Stop Voice' : 'Start Voice'}
              </button>
              
              <div className="flex items-center px-4 py-2 rounded-lg border border-white/20">
                <div className={`w-3 h-3 rounded-full mr-2 ${
                  isStreaming ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
                }`} />
                <span className="text-sm text-white/60">
                  {isStreaming ? 'Voice Active' : 'Voice Inactive'}
                </span>
              </div>
            </div>
          </div>

          {/* Instructions */}
          <div className="max-w-4xl mx-auto mt-8 text-center">
            <p className="text-white/40 text-sm">
              Move your mouse to see the avatar's eyes track • 
              Speak to see lip-sync animation • 
              The avatar breathes and has holographic effects
            </p>
          </div>
        </div>

        {/* Title */}
        <div className="absolute top-8 left-8">
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
            Jarvis 3D Avatar
          </h1>
          <p className="text-white/60 mt-2">Holographic AI Assistant</p>
        </div>
      </div>
    </div>
  );
}