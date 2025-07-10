// Voice Activity Detection AudioWorklet Processor
class VADProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    
    // Configuration
    this.threshold = options.parameterData?.threshold || 0.5;
    this.smoothingFactor = 0.92;
    this.minSilenceDuration = 300; // ms
    this.minSpeechDuration = 100; // ms
    
    // State
    this.smoothedEnergy = 0;
    this.isSpeaking = false;
    this.speechStartTime = 0;
    this.silenceStartTime = 0;
    this.frameCount = 0;
    
    // Ring buffer for energy history
    this.energyHistory = new Float32Array(50);
    this.historyIndex = 0;
    
    // Noise floor estimation
    this.noiseFloor = 0.01;
    this.noiseFloorUpdateRate = 0.001;
  }
  
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    
    if (!input || !input[0]) {
      return true;
    }
    
    const inputChannel = input[0];
    const outputChannel = output[0];
    
    // Pass through audio
    if (outputChannel) {
      outputChannel.set(inputChannel);
    }
    
    // Calculate frame energy
    let energy = 0;
    for (let i = 0; i < inputChannel.length; i++) {
      energy += inputChannel[i] * inputChannel[i];
    }
    energy = Math.sqrt(energy / inputChannel.length);
    
    // Apply smoothing
    this.smoothedEnergy = this.smoothedEnergy * this.smoothingFactor + 
                          energy * (1 - this.smoothingFactor);
    
    // Update energy history
    this.energyHistory[this.historyIndex] = energy;
    this.historyIndex = (this.historyIndex + 1) % this.energyHistory.length;
    
    // Update noise floor estimation during silence
    if (!this.isSpeaking && energy < this.noiseFloor * 2) {
      this.noiseFloor = this.noiseFloor * (1 - this.noiseFloorUpdateRate) + 
                       energy * this.noiseFloorUpdateRate;
    }
    
    // Calculate dynamic threshold
    const dynamicThreshold = Math.max(
      this.threshold,
      this.noiseFloor * 3
    );
    
    // Voice activity detection with hysteresis
    const currentTime = currentFrame * 128 / sampleRate * 1000; // Convert to ms
    const wasActivePreviously = this.isSpeaking;
    
    if (this.smoothedEnergy > dynamicThreshold) {
      if (!this.isSpeaking) {
        // Potential speech start
        if (this.speechStartTime === 0) {
          this.speechStartTime = currentTime;
        } else if (currentTime - this.speechStartTime > this.minSpeechDuration) {
          // Confirmed speech start
          this.isSpeaking = true;
          this.silenceStartTime = 0;
          this.port.postMessage({
            type: 'vad-result',
            isSpeech: true,
            energy: this.smoothedEnergy,
            threshold: dynamicThreshold,
            timestamp: currentTime
          });
        }
      }
    } else {
      if (this.isSpeaking) {
        // Potential speech end
        if (this.silenceStartTime === 0) {
          this.silenceStartTime = currentTime;
        } else if (currentTime - this.silenceStartTime > this.minSilenceDuration) {
          // Confirmed speech end
          this.isSpeaking = false;
          this.speechStartTime = 0;
          this.port.postMessage({
            type: 'vad-result',
            isSpeech: false,
            energy: this.smoothedEnergy,
            threshold: dynamicThreshold,
            timestamp: currentTime
          });
        }
      } else {
        // Reset speech start time if energy drops
        this.speechStartTime = 0;
      }
    }
    
    // Send periodic updates
    this.frameCount++;
    if (this.frameCount % 10 === 0) {
      this.port.postMessage({
        type: 'energy-update',
        energy: this.smoothedEnergy,
        noiseFloor: this.noiseFloor,
        isSpeech: this.isSpeaking
      });
    }
    
    return true;
  }
  
  // Handle parameter updates
  static get parameterDescriptors() {
    return [
      {
        name: 'threshold',
        defaultValue: 0.5,
        minValue: 0,
        maxValue: 1,
        automationRate: 'k-rate'
      }
    ];
  }
}

registerProcessor('vad-processor', VADProcessor);