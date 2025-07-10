'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useGestureControl } from '@/hooks/useGestureControl';
import { GestureType, FacialExpression } from '@/services/gestureRecognition';
import { 
  Camera, 
  CameraOff, 
  Hand, 
  Smile, 
  Frown,
  ThumbsUp,
  ThumbsDown,
  MoveHorizontal,
  MoveVertical,
  Grab,
  Pointer
} from 'lucide-react';

interface GestureControllerProps {
  onCommand?: (command: any) => void;
  showVideo?: boolean;
  showOverlay?: boolean;
  multiModal?: boolean;
  className?: string;
}

const gestureIcons: Record<GestureType, React.ReactNode> = {
  swipe_left: <MoveHorizontal className="w-6 h-6" />,
  swipe_right: <MoveHorizontal className="w-6 h-6 scale-x-[-1]" />,
  swipe_up: <MoveVertical className="w-6 h-6" />,
  swipe_down: <MoveVertical className="w-6 h-6 scale-y-[-1]" />,
  pinch: <Grab className="w-6 h-6" />,
  point: <Pointer className="w-6 h-6" />,
  thumbs_up: <ThumbsUp className="w-6 h-6" />,
  thumbs_down: <ThumbsDown className="w-6 h-6" />,
  peace: <Hand className="w-6 h-6" />,
  fist: <Hand className="w-6 h-6 fill-current" />,
  open_palm: <Hand className="w-6 h-6" />,
  unknown: <Hand className="w-6 h-6 opacity-50" />
};

const expressionIcons: Record<FacialExpression, React.ReactNode> = {
  neutral: <Smile className="w-6 h-6 opacity-50" />,
  happy: <Smile className="w-6 h-6" />,
  sad: <Frown className="w-6 h-6" />,
  surprised: <Smile className="w-6 h-6" style={{ transform: 'rotate(180deg)' }} />,
  angry: <Frown className="w-6 h-6 text-red-500" />,
  disgusted: <Frown className="w-6 h-6 text-green-500" />,
  fearful: <Frown className="w-6 h-6 text-purple-500" />
};

export function GestureController({
  onCommand,
  showVideo = true,
  showOverlay = true,
  multiModal = true,
  className = ''
}: GestureControllerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [notification, setNotification] = useState<string | null>(null);

  const {
    videoRef,
    isInitialized,
    isActive,
    currentGesture,
    currentExpression,
    lastCommand,
    error,
    start,
    stop,
    getMultiModalCommand
  } = useGestureControl({
    onCommand: (cmd) => {
      // Show notification
      const message = cmd.type === 'gesture' 
        ? `Gesture: ${cmd.value}` 
        : `Expression: ${cmd.value}`;
      
      setNotification(message);
      setTimeout(() => setNotification(null), 2000);

      // Check for multi-modal commands
      if (multiModal) {
        const multiModalCmd = getMultiModalCommand();
        if (multiModalCmd) {
          onCommand?.(multiModalCmd);
          return;
        }
      }

      onCommand?.(cmd);
    },
    multiModal,
    gestureThreshold: 0.7,
    expressionThreshold: 0.7
  });

  useEffect(() => {
    if (isInitialized) {
      setIsLoading(false);
    }
  }, [isInitialized]);

  // Draw hand landmarks on canvas
  useEffect(() => {
    if (!showOverlay || !canvasRef.current || !videoRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    // Animation loop would go here for drawing landmarks
    // For now, we'll just clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, [showOverlay, isActive]);

  return (
    <div className={`relative ${className}`}>
      {/* Video Container */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        {showVideo && (
          <>
            <video
              ref={videoRef}
              className="w-full h-full object-cover"
              playsInline
              muted
              style={{ transform: 'scaleX(-1)' }} // Mirror the video
            />
            
            {showOverlay && (
              <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full"
                style={{ transform: 'scaleX(-1)' }} // Mirror the canvas
              />
            )}
          </>
        )}

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white mb-4" />
              <p className="text-white">Initializing camera...</p>
            </div>
          </div>
        )}

        {/* Error Overlay */}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-red-900/80">
            <div className="text-center text-white p-4">
              <p className="mb-2">Error: {error}</p>
              <button
                onClick={start}
                className="px-4 py-2 bg-white text-red-900 rounded hover:bg-gray-100"
              >
                Retry
              </button>
            </div>
          </div>
        )}

        {/* Status Indicators */}
        <div className="absolute top-4 left-4 flex flex-col gap-2">
          {/* Camera Status */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={`flex items-center gap-2 px-3 py-1 rounded-full ${
              isActive ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}
          >
            {isActive ? <Camera className="w-4 h-4" /> : <CameraOff className="w-4 h-4" />}
            <span className="text-sm">{isActive ? 'Active' : 'Inactive'}</span>
          </motion.div>

          {/* Current Gesture */}
          <AnimatePresence>
            {currentGesture && currentGesture !== 'unknown' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/20 text-blue-400"
              >
                {gestureIcons[currentGesture]}
                <span className="text-sm capitalize">{currentGesture.replace('_', ' ')}</span>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Current Expression */}
          <AnimatePresence>
            {currentExpression && currentExpression !== 'neutral' && (
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                className="flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/20 text-purple-400"
              >
                {expressionIcons[currentExpression]}
                <span className="text-sm capitalize">{currentExpression}</span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Notifications */}
        <AnimatePresence>
          {notification && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="absolute bottom-4 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-white/10 backdrop-blur-md rounded-full text-white"
            >
              {notification}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Control Buttons */}
      <div className="flex justify-center gap-4 mt-4">
        {!isActive ? (
          <button
            onClick={start}
            className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
          >
            <Camera className="w-5 h-5" />
            Start Gesture Control
          </button>
        ) : (
          <button
            onClick={stop}
            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex items-center gap-2"
          >
            <CameraOff className="w-5 h-5" />
            Stop Gesture Control
          </button>
        )}
      </div>

      {/* Gesture Guide */}
      <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-3">Available Gestures</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {Object.entries(gestureIcons).map(([gesture, icon]) => (
            gesture !== 'unknown' && (
              <div key={gesture} className="flex items-center gap-2 text-sm">
                <div className="p-2 bg-white dark:bg-gray-700 rounded">
                  {icon}
                </div>
                <span className="capitalize">{gesture.replace('_', ' ')}</span>
              </div>
            )
          ))}
        </div>
      </div>
    </div>
  );
}