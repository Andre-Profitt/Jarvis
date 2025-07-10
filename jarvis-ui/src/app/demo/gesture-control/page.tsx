'use client';

import React, { useState, useCallback } from 'react';
import { GestureController } from '@/components/GestureController';
import { SmartHomePanel } from '@/components/SmartHomePanel';
import { JarvisAvatar3D } from '@/components/3d/JarvisAvatar3D';
import { VoiceAssistant } from '@/components/VoiceAssistant';
import { motion } from 'framer-motion';
import { Brain, Home, Mic, Hand } from 'lucide-react';

export default function GestureControlDemo() {
  const [activeMode, setActiveMode] = useState<'gesture' | 'voice' | 'multimodal'>('gesture');
  const [lastCommand, setLastCommand] = useState<any>(null);
  const [avatarAction, setAvatarAction] = useState<string>('idle');

  // Handle gesture commands
  const handleGestureCommand = useCallback((command: any) => {
    console.log('Gesture command received:', command);
    setLastCommand(command);

    // Map gestures to smart home actions
    switch (command.value) {
      case 'swipe_left':
        // Previous room or decrease
        if (command.metadata?.expression === 'happy') {
          setAvatarAction('celebrate');
        }
        break;
      
      case 'swipe_right':
        // Next room or increase
        setAvatarAction('wave');
        break;
      
      case 'swipe_up':
        // Increase brightness/temperature
        setAvatarAction('nod');
        break;
      
      case 'swipe_down':
        // Decrease brightness/temperature
        setAvatarAction('shake');
        break;
      
      case 'pinch':
        // Toggle device on/off
        setAvatarAction('think');
        break;
      
      case 'point':
        // Select device
        setAvatarAction('point');
        break;
      
      case 'thumbs_up':
        // Confirm action
        setAvatarAction('thumbsUp');
        break;
      
      case 'thumbs_down':
        // Cancel action
        setAvatarAction('thumbsDown');
        break;
      
      case 'peace':
        // Activate scene or preset
        setAvatarAction('peace');
        break;
      
      case 'open_palm':
        // Stop all actions
        setAvatarAction('stop');
        break;
    }

    // Reset avatar action after animation
    setTimeout(() => setAvatarAction('idle'), 2000);
  }, []);

  // Handle voice commands
  const handleVoiceCommand = useCallback((transcript: string) => {
    console.log('Voice command received:', transcript);
    setLastCommand({ type: 'voice', value: transcript, timestamp: Date.now() });
    setAvatarAction('speak');
    setTimeout(() => setAvatarAction('idle'), 2000);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 to-black text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Multi-Modal Interaction Demo
          </h1>
          <p className="text-gray-400">Control your smart home with gestures, voice, and facial expressions</p>
        </motion.div>

        {/* Mode Selector */}
        <div className="flex justify-center gap-4 mb-8">
          <button
            onClick={() => setActiveMode('gesture')}
            className={`px-6 py-3 rounded-lg flex items-center gap-2 transition-all ${
              activeMode === 'gesture' 
                ? 'bg-blue-600 text-white' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <Hand className="w-5 h-5" />
            Gesture Only
          </button>
          <button
            onClick={() => setActiveMode('voice')}
            className={`px-6 py-3 rounded-lg flex items-center gap-2 transition-all ${
              activeMode === 'voice' 
                ? 'bg-green-600 text-white' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <Mic className="w-5 h-5" />
            Voice Only
          </button>
          <button
            onClick={() => setActiveMode('multimodal')}
            className={`px-6 py-3 rounded-lg flex items-center gap-2 transition-all ${
              activeMode === 'multimodal' 
                ? 'bg-purple-600 text-white' 
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            <Brain className="w-5 h-5" />
            Multi-Modal
          </button>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Control Interface */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700"
            >
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <Hand className="w-6 h-6" />
                Control Interface
              </h2>
              
              {(activeMode === 'gesture' || activeMode === 'multimodal') && (
                <GestureController
                  onCommand={handleGestureCommand}
                  showVideo={true}
                  showOverlay={true}
                  multiModal={activeMode === 'multimodal'}
                  className="mb-6"
                />
              )}

              {(activeMode === 'voice' || activeMode === 'multimodal') && (
                <div className="mt-6">
                  <VoiceAssistant
                    onTranscript={handleVoiceCommand}
                    className="w-full"
                  />
                </div>
              )}

              {/* Last Command Display */}
              {lastCommand && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="mt-4 p-3 bg-gray-700/50 rounded-lg"
                >
                  <p className="text-sm text-gray-400">Last Command:</p>
                  <p className="font-mono text-green-400">
                    {lastCommand.type}: {lastCommand.value}
                  </p>
                  {lastCommand.metadata?.expression && (
                    <p className="text-sm text-purple-400">
                      Expression: {lastCommand.metadata.expression}
                    </p>
                  )}
                </motion.div>
              )}
            </motion.div>
          </div>

          {/* Middle Column - 3D Avatar */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700 h-[600px]"
            >
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <Brain className="w-6 h-6" />
                AI Assistant
              </h2>
              <JarvisAvatar3D
                action={avatarAction}
                responsive={true}
                className="h-full"
              />
            </motion.div>
          </div>

          {/* Right Column - Smart Home Panel */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-gray-700"
            >
              <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
                <Home className="w-6 h-6" />
                Smart Home Control
              </h2>
              <SmartHomePanel />
            </motion.div>
          </div>
        </div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-8 bg-gray-800/30 backdrop-blur-sm rounded-lg p-6 border border-gray-700"
        >
          <h3 className="text-xl font-semibold mb-4">How to Use</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold text-blue-400 mb-2">Gesture Controls</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Swipe left/right: Navigate rooms</li>
                <li>• Swipe up/down: Adjust values</li>
                <li>• Pinch: Toggle devices</li>
                <li>• Point: Select items</li>
                <li>• Thumbs up/down: Confirm/cancel</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-green-400 mb-2">Voice Commands</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• "Turn on the lights"</li>
                <li>• "Set temperature to 72"</li>
                <li>• "Activate movie mode"</li>
                <li>• "Show living room"</li>
                <li>• "Dim lights to 50%"</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-purple-400 mb-2">Multi-Modal</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Point + say device name</li>
                <li>• Smile + thumbs up for confirm</li>
                <li>• Gesture + expression combos</li>
                <li>• Voice override for precision</li>
                <li>• Natural interaction flow</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}