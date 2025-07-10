import { useEffect, useRef, useState, useCallback } from 'react';
import { 
  gestureRecognition, 
  GestureType, 
  FacialExpression, 
  GestureResult, 
  FaceResult 
} from '@/services/gestureRecognition';

export interface GestureCommand {
  type: 'gesture' | 'expression';
  value: GestureType | FacialExpression;
  confidence: number;
  timestamp: number;
  metadata?: any;
}

export interface UseGestureControlOptions {
  onCommand?: (command: GestureCommand) => void;
  onGesture?: (gesture: GestureResult) => void;
  onExpression?: (expression: FaceResult) => void;
  multiModal?: boolean;
  gestureThreshold?: number;
  expressionThreshold?: number;
}

export function useGestureControl(options: UseGestureControlOptions = {}) {
  const {
    onCommand,
    onGesture,
    onExpression,
    multiModal = true,
    gestureThreshold = 0.7,
    expressionThreshold = 0.7
  } = options;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [isActive, setIsActive] = useState(false);
  const [currentGesture, setCurrentGesture] = useState<GestureType | null>(null);
  const [currentExpression, setCurrentExpression] = useState<FacialExpression | null>(null);
  const [lastCommand, setLastCommand] = useState<GestureCommand | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Multi-modal state for combining gestures with voice/expressions
  const multiModalStateRef = useRef({
    pendingGesture: null as GestureType | null,
    pendingExpression: null as FacialExpression | null,
    lastGestureTime: 0,
    lastExpressionTime: 0
  });

  // Handle gesture detection
  const handleGesture = useCallback((result: GestureResult) => {
    if (result.confidence >= gestureThreshold) {
      setCurrentGesture(result.gesture);
      
      // Call gesture callback
      onGesture?.(result);

      // Process as command
      const command: GestureCommand = {
        type: 'gesture',
        value: result.gesture,
        confidence: result.confidence,
        timestamp: result.timestamp,
        metadata: result.direction
      };

      setLastCommand(command);
      onCommand?.(command);

      // Update multi-modal state
      if (multiModal) {
        multiModalStateRef.current.pendingGesture = result.gesture;
        multiModalStateRef.current.lastGestureTime = result.timestamp;
      }
    }
  }, [gestureThreshold, onGesture, onCommand, multiModal]);

  // Handle facial expression detection
  const handleExpression = useCallback((result: FaceResult) => {
    if (result.confidence >= expressionThreshold) {
      setCurrentExpression(result.expression);
      
      // Call expression callback
      onExpression?.(result);

      // Process as command
      const command: GestureCommand = {
        type: 'expression',
        value: result.expression,
        confidence: result.confidence,
        timestamp: result.timestamp
      };

      setLastCommand(command);
      onCommand?.(command);

      // Update multi-modal state
      if (multiModal) {
        multiModalStateRef.current.pendingExpression = result.expression;
        multiModalStateRef.current.lastExpressionTime = result.timestamp;
      }
    }
  }, [expressionThreshold, onExpression, onCommand, multiModal]);

  // Initialize gesture recognition
  const initialize = useCallback(async () => {
    if (!videoRef.current) {
      setError('Video element not available');
      return;
    }

    try {
      // Request camera permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 1280, 
          height: 720,
          facingMode: 'user'
        } 
      });

      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      // Initialize gesture recognition service
      await gestureRecognition.initialize(videoRef.current);

      // Set up callbacks
      gestureRecognition.onGesture(handleGesture);
      gestureRecognition.onFacialExpression(handleExpression);

      setIsInitialized(true);
      setIsActive(true);
      setError(null);
    } catch (err) {
      console.error('Failed to initialize gesture recognition:', err);
      setError(err instanceof Error ? err.message : 'Failed to initialize');
    }
  }, [handleGesture, handleExpression]);

  // Start gesture recognition
  const start = useCallback(() => {
    if (!isInitialized) {
      initialize();
    } else {
      setIsActive(true);
    }
  }, [isInitialized, initialize]);

  // Stop gesture recognition
  const stop = useCallback(() => {
    setIsActive(false);
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
    }
  }, []);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stop();
      gestureRecognition.stop();
    };
  }, [stop]);

  // Utility functions for gesture-based control
  const isGesture = useCallback((gesture: GestureType): boolean => {
    return currentGesture === gesture;
  }, [currentGesture]);

  const isExpression = useCallback((expression: FacialExpression): boolean => {
    return currentExpression === expression;
  }, [currentExpression]);

  // Get gesture history for pattern recognition
  const getGestureHistory = useCallback(() => {
    return gestureRecognition.getGestureHistory();
  }, []);

  const getFaceHistory = useCallback(() => {
    return gestureRecognition.getFaceHistory();
  }, []);

  // Multi-modal command builder
  const getMultiModalCommand = useCallback((): GestureCommand | null => {
    if (!multiModal) return null;

    const state = multiModalStateRef.current;
    const now = Date.now();
    const timeDiff = Math.abs(state.lastGestureTime - state.lastExpressionTime);

    // If gesture and expression happen within 1 second, combine them
    if (timeDiff < 1000 && state.pendingGesture && state.pendingExpression) {
      const command: GestureCommand = {
        type: 'gesture',
        value: state.pendingGesture,
        confidence: 0.9,
        timestamp: now,
        metadata: {
          expression: state.pendingExpression,
          multiModal: true
        }
      };

      // Clear pending state
      state.pendingGesture = null;
      state.pendingExpression = null;

      return command;
    }

    return null;
  }, [multiModal]);

  return {
    // Refs
    videoRef,

    // State
    isInitialized,
    isActive,
    currentGesture,
    currentExpression,
    lastCommand,
    error,

    // Control functions
    start,
    stop,
    initialize,

    // Utility functions
    isGesture,
    isExpression,
    getGestureHistory,
    getFaceHistory,
    getMultiModalCommand
  };
}