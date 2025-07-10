import { GestureType, FacialExpression } from './gestureRecognition';
import { smartHomeService } from './smartHomeService';
import { nlpProcessor } from './nlpProcessor';

export interface MultiModalCommand {
  primary: {
    type: 'gesture' | 'voice' | 'expression';
    value: string;
    confidence: number;
  };
  secondary?: {
    type: 'gesture' | 'voice' | 'expression';
    value: string;
    confidence: number;
  };
  context?: {
    currentRoom?: string;
    selectedDevice?: string;
    lastAction?: string;
  };
  timestamp: number;
}

export interface CommandInterpretation {
  action: string;
  target?: string;
  parameters?: Record<string, any>;
  confidence: number;
}

export class MultiModalControlService {
  private commandQueue: MultiModalCommand[] = [];
  private currentContext: Record<string, any> = {
    currentRoom: 'living_room',
    selectedDevice: null,
    lastAction: null,
    gestureHistory: [],
    voiceHistory: []
  };

  constructor() {
    // Initialize service
  }

  public processMultiModalCommand(command: MultiModalCommand): CommandInterpretation {
    // Add to command queue for pattern recognition
    this.commandQueue.push(command);
    if (this.commandQueue.length > 10) {
      this.commandQueue.shift();
    }

    // Update context
    this.updateContext(command);

    // Interpret command based on modality combination
    if (command.primary.type === 'gesture' && command.secondary?.type === 'voice') {
      return this.interpretGestureVoiceCombination(command);
    } else if (command.primary.type === 'gesture' && command.secondary?.type === 'expression') {
      return this.interpretGestureExpressionCombination(command);
    } else if (command.primary.type === 'gesture') {
      return this.interpretGesture(command.primary.value as GestureType);
    } else if (command.primary.type === 'voice') {
      return this.interpretVoice(command.primary.value);
    }

    return {
      action: 'unknown',
      confidence: 0
    };
  }

  private interpretGesture(gesture: GestureType): CommandInterpretation {
    const context = this.currentContext;
    
    switch (gesture) {
      case 'swipe_left':
        if (context.selectedDevice) {
          return {
            action: 'decrease',
            target: context.selectedDevice,
            parameters: { amount: 10 },
            confidence: 0.8
          };
        } else {
          return {
            action: 'navigate',
            parameters: { direction: 'previous' },
            confidence: 0.7
          };
        }

      case 'swipe_right':
        if (context.selectedDevice) {
          return {
            action: 'increase',
            target: context.selectedDevice,
            parameters: { amount: 10 },
            confidence: 0.8
          };
        } else {
          return {
            action: 'navigate',
            parameters: { direction: 'next' },
            confidence: 0.7
          };
        }

      case 'swipe_up':
        return {
          action: 'increase',
          target: context.selectedDevice || 'brightness',
          parameters: { amount: 20 },
          confidence: 0.75
        };

      case 'swipe_down':
        return {
          action: 'decrease',
          target: context.selectedDevice || 'brightness',
          parameters: { amount: 20 },
          confidence: 0.75
        };

      case 'pinch':
        return {
          action: 'toggle',
          target: context.selectedDevice || 'main_light',
          confidence: 0.85
        };

      case 'point':
        return {
          action: 'select',
          parameters: { waitingForTarget: true },
          confidence: 0.9
        };

      case 'thumbs_up':
        return {
          action: 'confirm',
          target: context.lastAction,
          confidence: 0.95
        };

      case 'thumbs_down':
        return {
          action: 'cancel',
          target: context.lastAction,
          confidence: 0.95
        };

      case 'peace':
        return {
          action: 'activate_scene',
          parameters: { scene: 'relax' },
          confidence: 0.8
        };

      case 'open_palm':
        return {
          action: 'stop_all',
          confidence: 0.9
        };

      case 'fist':
        return {
          action: 'emergency_stop',
          confidence: 0.95
        };

      default:
        return {
          action: 'unknown',
          confidence: 0
        };
    }
  }

  private interpretVoice(transcript: string): CommandInterpretation {
    // Use NLP processor to understand voice command
    const nlpResult = nlpProcessor.processCommand(transcript);
    
    return {
      action: nlpResult.intent,
      target: nlpResult.entities.device || nlpResult.entities.room,
      parameters: nlpResult.entities,
      confidence: nlpResult.confidence
    };
  }

  private interpretGestureVoiceCombination(command: MultiModalCommand): CommandInterpretation {
    const gesture = command.primary.value as GestureType;
    const voiceText = command.secondary!.value;
    
    // Point + voice = precise selection
    if (gesture === 'point') {
      const nlpResult = nlpProcessor.processCommand(voiceText);
      return {
        action: 'select_and_control',
        target: nlpResult.entities.device,
        parameters: nlpResult.entities,
        confidence: 0.95
      };
    }

    // Gesture for action, voice for target
    const gestureAction = this.interpretGesture(gesture);
    const voiceTarget = nlpProcessor.processCommand(voiceText);
    
    return {
      action: gestureAction.action,
      target: voiceTarget.entities.device || voiceTarget.entities.room,
      parameters: { ...gestureAction.parameters, ...voiceTarget.entities },
      confidence: (gestureAction.confidence + voiceTarget.confidence) / 2
    };
  }

  private interpretGestureExpressionCombination(command: MultiModalCommand): CommandInterpretation {
    const gesture = command.primary.value as GestureType;
    const expression = command.secondary!.value as FacialExpression;
    
    // Modify gesture interpretation based on expression
    const baseInterpretation = this.interpretGesture(gesture);
    
    switch (expression) {
      case 'happy':
        // Amplify positive actions
        if (baseInterpretation.parameters?.amount) {
          baseInterpretation.parameters.amount *= 1.5;
        }
        baseInterpretation.confidence *= 1.1;
        break;
        
      case 'angry':
        // Emergency or forceful actions
        if (gesture === 'fist') {
          return {
            action: 'emergency_shutdown',
            confidence: 0.99
          };
        }
        break;
        
      case 'surprised':
        // Quick actions
        baseInterpretation.parameters = {
          ...baseInterpretation.parameters,
          speed: 'fast'
        };
        break;
    }
    
    return baseInterpretation;
  }

  private updateContext(command: MultiModalCommand): void {
    // Update gesture history
    if (command.primary.type === 'gesture') {
      this.currentContext.gestureHistory.push(command.primary.value);
      if (this.currentContext.gestureHistory.length > 5) {
        this.currentContext.gestureHistory.shift();
      }
    }

    // Update voice history
    if (command.primary.type === 'voice' || command.secondary?.type === 'voice') {
      const voiceValue = command.primary.type === 'voice' 
        ? command.primary.value 
        : command.secondary!.value;
      this.currentContext.voiceHistory.push(voiceValue);
      if (this.currentContext.voiceHistory.length > 5) {
        this.currentContext.voiceHistory.shift();
      }
    }

    // Update other context based on command
    if (command.context) {
      Object.assign(this.currentContext, command.context);
    }
  }

  public async executeCommand(interpretation: CommandInterpretation): Promise<void> {
    const { action, target, parameters } = interpretation;

    switch (action) {
      case 'toggle':
        if (target) {
          await smartHomeService.toggleDevice(target);
        }
        break;

      case 'increase':
      case 'decrease':
        if (target && parameters?.amount) {
          const currentValue = await smartHomeService.getDeviceValue(target);
          const newValue = action === 'increase' 
            ? currentValue + parameters.amount 
            : currentValue - parameters.amount;
          await smartHomeService.setDeviceValue(target, newValue);
        }
        break;

      case 'activate_scene':
        if (parameters?.scene) {
          await smartHomeService.activateScene(parameters.scene);
        }
        break;

      case 'navigate':
        if (parameters?.direction) {
          const rooms = ['living_room', 'bedroom', 'kitchen', 'bathroom'];
          const currentIndex = rooms.indexOf(this.currentContext.currentRoom);
          const newIndex = parameters.direction === 'next' 
            ? (currentIndex + 1) % rooms.length 
            : (currentIndex - 1 + rooms.length) % rooms.length;
          this.currentContext.currentRoom = rooms[newIndex];
        }
        break;

      case 'select':
        this.currentContext.waitingForSelection = true;
        break;

      case 'confirm':
        if (this.currentContext.pendingAction) {
          await this.executePendingAction();
        }
        break;

      case 'cancel':
        this.currentContext.pendingAction = null;
        break;

      case 'stop_all':
        await smartHomeService.stopAllDevices();
        break;

      case 'emergency_stop':
      case 'emergency_shutdown':
        await smartHomeService.emergencyShutdown();
        break;
    }

    // Update last action
    this.currentContext.lastAction = action;
  }

  private async executePendingAction(): Promise<void> {
    const pending = this.currentContext.pendingAction;
    if (pending) {
      await this.executeCommand(pending);
      this.currentContext.pendingAction = null;
    }
  }

  public getContext(): Record<string, any> {
    return { ...this.currentContext };
  }

  public detectPatterns(): string[] {
    const patterns: string[] = [];
    const gestureHistory = this.currentContext.gestureHistory;

    // Detect repeated gestures
    if (gestureHistory.length >= 3) {
      const last3 = gestureHistory.slice(-3);
      if (last3.every((g: string) => g === last3[0])) {
        patterns.push('repeated_gesture');
      }
    }

    // Detect swipe patterns
    if (gestureHistory.includes('swipe_left') && gestureHistory.includes('swipe_right')) {
      patterns.push('navigation_pattern');
    }

    // Detect confirmation patterns
    if (gestureHistory.includes('thumbs_up') || gestureHistory.includes('thumbs_down')) {
      patterns.push('decision_pattern');
    }

    return patterns;
  }
}

// Singleton instance
export const multiModalControl = new MultiModalControlService();