import { Hands, HAND_CONNECTIONS } from '@mediapipe/hands';
import { FaceMesh, FACEMESH_TESSELATION } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

export type GestureType = 
  | 'swipe_left' 
  | 'swipe_right' 
  | 'swipe_up' 
  | 'swipe_down'
  | 'pinch'
  | 'point'
  | 'thumbs_up'
  | 'thumbs_down'
  | 'peace'
  | 'fist'
  | 'open_palm'
  | 'unknown';

export type FacialExpression = 
  | 'neutral'
  | 'happy'
  | 'sad'
  | 'surprised'
  | 'angry'
  | 'disgusted'
  | 'fearful';

export interface GestureResult {
  gesture: GestureType;
  confidence: number;
  landmarks?: any;
  direction?: { x: number; y: number };
  timestamp: number;
}

export interface FaceResult {
  expression: FacialExpression;
  confidence: number;
  landmarks?: any;
  timestamp: number;
}

export class GestureRecognitionService {
  private hands: Hands;
  private faceMesh: FaceMesh;
  private camera: Camera | null = null;
  private videoElement: HTMLVideoElement | null = null;
  private gestureHistory: GestureResult[] = [];
  private faceHistory: FaceResult[] = [];
  private previousHandPosition: { x: number; y: number } | null = null;
  private gestureCallbacks: ((result: GestureResult) => void)[] = [];
  private faceCallbacks: ((result: FaceResult) => void)[] = [];

  constructor() {
    // Initialize MediaPipe Hands
    this.hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
      }
    });

    this.hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 1,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5
    });

    // Initialize MediaPipe Face Mesh
    this.faceMesh = new FaceMesh({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
      }
    });

    this.faceMesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.7,
      minTrackingConfidence: 0.5
    });

    // Set up result handlers
    this.hands.onResults(this.handleHandResults.bind(this));
    this.faceMesh.onResults(this.handleFaceResults.bind(this));
  }

  public async initialize(videoElement: HTMLVideoElement): Promise<void> {
    this.videoElement = videoElement;

    // Set up camera
    this.camera = new Camera(videoElement, {
      onFrame: async () => {
        await this.hands.send({ image: videoElement });
        await this.faceMesh.send({ image: videoElement });
      },
      width: 1280,
      height: 720
    });

    await this.camera.start();
  }

  public onGesture(callback: (result: GestureResult) => void): void {
    this.gestureCallbacks.push(callback);
  }

  public onFacialExpression(callback: (result: FaceResult) => void): void {
    this.faceCallbacks.push(callback);
  }

  private handleHandResults(results: any): void {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      this.previousHandPosition = null;
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    const gesture = this.recognizeGesture(landmarks);
    const currentPosition = this.getHandCenter(landmarks);

    // Detect swipe gestures
    if (this.previousHandPosition) {
      const dx = currentPosition.x - this.previousHandPosition.x;
      const dy = currentPosition.y - this.previousHandPosition.y;
      const distance = Math.sqrt(dx * dx + dy * dy);

      if (distance > 0.1) { // Threshold for swipe detection
        if (Math.abs(dx) > Math.abs(dy)) {
          gesture.gesture = dx > 0 ? 'swipe_right' : 'swipe_left';
        } else {
          gesture.gesture = dy > 0 ? 'swipe_down' : 'swipe_up';
        }
        gesture.direction = { x: dx, y: dy };
      }
    }

    this.previousHandPosition = currentPosition;
    
    // Store in history and notify callbacks
    this.gestureHistory.push(gesture);
    if (this.gestureHistory.length > 30) {
      this.gestureHistory.shift();
    }

    this.gestureCallbacks.forEach(callback => callback(gesture));
  }

  private handleFaceResults(results: any): void {
    if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
      return;
    }

    const landmarks = results.multiFaceLandmarks[0];
    const expression = this.recognizeFacialExpression(landmarks);

    // Store in history and notify callbacks
    this.faceHistory.push(expression);
    if (this.faceHistory.length > 30) {
      this.faceHistory.shift();
    }

    this.faceCallbacks.forEach(callback => callback(expression));
  }

  private recognizeGesture(landmarks: any[]): GestureResult {
    const result: GestureResult = {
      gesture: 'unknown',
      confidence: 0,
      landmarks,
      timestamp: Date.now()
    };

    // Extract key points
    const thumbTip = landmarks[4];
    const thumbIP = landmarks[3];
    const indexTip = landmarks[8];
    const indexPIP = landmarks[6];
    const middleTip = landmarks[12];
    const middlePIP = landmarks[10];
    const ringTip = landmarks[16];
    const ringPIP = landmarks[14];
    const pinkyTip = landmarks[20];
    const pinkyPIP = landmarks[18];
    const palmBase = landmarks[0];

    // Helper function to check if finger is extended
    const isFingerExtended = (tip: any, pip: any): boolean => {
      return tip.y < pip.y;
    };

    // Count extended fingers
    const extendedFingers = [
      isFingerExtended(indexTip, indexPIP),
      isFingerExtended(middleTip, middlePIP),
      isFingerExtended(ringTip, ringPIP),
      isFingerExtended(pinkyTip, pinkyPIP)
    ].filter(Boolean).length;

    // Thumb position (special case)
    const thumbExtended = thumbTip.x > thumbIP.x + 0.05 || thumbTip.x < thumbIP.x - 0.05;

    // Recognize specific gestures
    if (extendedFingers === 0 && !thumbExtended) {
      result.gesture = 'fist';
      result.confidence = 0.9;
    } else if (extendedFingers === 4 && thumbExtended) {
      result.gesture = 'open_palm';
      result.confidence = 0.9;
    } else if (extendedFingers === 1 && isFingerExtended(indexTip, indexPIP)) {
      result.gesture = 'point';
      result.confidence = 0.85;
    } else if (extendedFingers === 2 && isFingerExtended(indexTip, indexPIP) && isFingerExtended(middleTip, middlePIP)) {
      result.gesture = 'peace';
      result.confidence = 0.85;
    } else if (thumbExtended && extendedFingers === 0) {
      // Check thumb direction for thumbs up/down
      if (thumbTip.y < thumbIP.y) {
        result.gesture = 'thumbs_up';
        result.confidence = 0.8;
      } else if (thumbTip.y > thumbIP.y) {
        result.gesture = 'thumbs_down';
        result.confidence = 0.8;
      }
    }

    // Detect pinch gesture
    const pinchDistance = Math.sqrt(
      Math.pow(thumbTip.x - indexTip.x, 2) +
      Math.pow(thumbTip.y - indexTip.y, 2) +
      Math.pow(thumbTip.z - indexTip.z, 2)
    );

    if (pinchDistance < 0.05) {
      result.gesture = 'pinch';
      result.confidence = 0.9;
    }

    return result;
  }

  private recognizeFacialExpression(landmarks: any[]): FaceResult {
    // Simple facial expression detection based on key landmarks
    // In a real implementation, you'd use more sophisticated analysis
    const result: FaceResult = {
      expression: 'neutral',
      confidence: 0.7,
      landmarks,
      timestamp: Date.now()
    };

    // Extract key facial points for expression analysis
    const leftEyeOuter = landmarks[33];
    const leftEyeInner = landmarks[133];
    const rightEyeOuter = landmarks[263];
    const rightEyeInner = landmarks[362];
    const mouthLeft = landmarks[61];
    const mouthRight = landmarks[291];
    const mouthTop = landmarks[13];
    const mouthBottom = landmarks[14];
    const leftBrow = landmarks[70];
    const rightBrow = landmarks[300];

    // Calculate mouth openness
    const mouthOpenness = Math.abs(mouthTop.y - mouthBottom.y);
    
    // Calculate mouth width
    const mouthWidth = Math.abs(mouthRight.x - mouthLeft.x);
    
    // Calculate eye openness
    const leftEyeOpenness = Math.abs(leftEyeOuter.y - leftEyeInner.y);
    const rightEyeOpenness = Math.abs(rightEyeOuter.y - rightEyeInner.y);
    
    // Simple expression detection logic
    if (mouthOpenness > 0.05 && leftEyeOpenness > 0.02 && rightEyeOpenness > 0.02) {
      result.expression = 'surprised';
      result.confidence = 0.8;
    } else if (mouthWidth > 0.12 && mouthTop.y > mouthBottom.y) {
      result.expression = 'happy';
      result.confidence = 0.85;
    } else if (mouthTop.y < mouthBottom.y - 0.01) {
      result.expression = 'sad';
      result.confidence = 0.75;
    }

    return result;
  }

  private getHandCenter(landmarks: any[]): { x: number; y: number } {
    const sum = landmarks.reduce((acc, landmark) => ({
      x: acc.x + landmark.x,
      y: acc.y + landmark.y
    }), { x: 0, y: 0 });

    return {
      x: sum.x / landmarks.length,
      y: sum.y / landmarks.length
    };
  }

  public getGestureHistory(): GestureResult[] {
    return [...this.gestureHistory];
  }

  public getFaceHistory(): FaceResult[] {
    return [...this.faceHistory];
  }

  public stop(): void {
    if (this.camera) {
      this.camera.stop();
    }
  }
}

// Singleton instance
export const gestureRecognition = new GestureRecognitionService();