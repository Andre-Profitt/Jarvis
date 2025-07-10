export interface VoiceOptions {
  rate?: number
  pitch?: number
  volume?: number
  voice?: SpeechSynthesisVoice | null
  emotion?: 'neutral' | 'happy' | 'sad' | 'excited' | 'sarcastic'
}

export class JarvisVoice {
  private synthesis: SpeechSynthesis | null = null
  private queue: string[] = []
  private isSpeaking = false
  private currentUtterance: SpeechSynthesisUtterance | null = null
  private jarvisVoice: SpeechSynthesisVoice | null = null
  
  constructor() {
    if (typeof window !== 'undefined' && window.speechSynthesis) {
      this.synthesis = window.speechSynthesis
      this.loadVoices()
      
      // Reload voices when they change
      if (this.synthesis.onvoiceschanged !== undefined) {
        this.synthesis.onvoiceschanged = () => this.loadVoices()
      }
    }
  }
  
  private loadVoices() {
    if (!this.synthesis) return
    const voices = this.synthesis.getVoices()
    
    // Try to find a suitable JARVIS-like voice
    this.jarvisVoice = 
      voices.find(v => v.name.includes('Daniel') || v.name.includes('UK Male')) ||
      voices.find(v => v.lang.startsWith('en-GB') && !v.name.includes('Female')) ||
      voices.find(v => v.lang.startsWith('en') && !v.name.includes('Female')) ||
      voices[0]
  }
  
  private applyEmotion(utterance: SpeechSynthesisUtterance, emotion: VoiceOptions['emotion']) {
    switch (emotion) {
      case 'happy':
        utterance.rate = 1.1
        utterance.pitch = 1.2
        break
      case 'sad':
        utterance.rate = 0.9
        utterance.pitch = 0.8
        break
      case 'excited':
        utterance.rate = 1.3
        utterance.pitch = 1.3
        break
      case 'sarcastic':
        utterance.rate = 0.95
        utterance.pitch = 1.1
        // Add slight pauses for sarcastic effect
        utterance.text = utterance.text.replace(/\s+/g, '... ')
        break
      default:
        // Neutral JARVIS settings
        utterance.rate = 1.0
        utterance.pitch = 0.9
    }
  }
  
  private processJarvisPersonality(text: string): string {
    // Add JARVIS personality to responses
    const personalityMap: Record<string, string> = {
      'Hello': 'Good day, sir',
      'Hi': 'Greetings',
      'Yes': 'Indeed, sir',
      'No': 'I\'m afraid not, sir',
      'OK': 'Very well, sir',
      'Thanks': 'Always at your service, sir',
      'Error': 'I do apologize, but there seems to be an issue',
    }
    
    let processedText = text
    Object.entries(personalityMap).forEach(([key, value]) => {
      const regex = new RegExp(`\\b${key}\\b`, 'gi')
      processedText = processedText.replace(regex, value)
    })
    
    return processedText
  }
  
  speak(text: string, options: VoiceOptions = {}) {
    if (!this.synthesis) return
    const processedText = this.processJarvisPersonality(text)
    
    const utterance = new SpeechSynthesisUtterance(processedText)
    
    // Set voice
    if (options.voice) {
      utterance.voice = options.voice
    } else if (this.jarvisVoice) {
      utterance.voice = this.jarvisVoice
    }
    
    // Apply base settings
    utterance.rate = options.rate ?? 1.0
    utterance.pitch = options.pitch ?? 0.9
    utterance.volume = options.volume ?? 1.0
    
    // Apply emotion
    this.applyEmotion(utterance, options.emotion)
    
    // Handle completion
    utterance.onend = () => {
      this.isSpeaking = false
      this.currentUtterance = null
      this.processQueue()
    }
    
    utterance.onerror = (event) => {
      console.error('Speech synthesis error:', event)
      this.isSpeaking = false
      this.currentUtterance = null
      this.processQueue()
    }
    
    // Add to queue or speak immediately
    if (this.isSpeaking) {
      this.queue.push(processedText)
    } else {
      this.isSpeaking = true
      this.currentUtterance = utterance
      this.synthesis.speak(utterance)
    }
  }
  
  private processQueue() {
    if (this.queue.length > 0 && !this.isSpeaking) {
      const text = this.queue.shift()!
      this.speak(text)
    }
  }
  
  stop() {
    if (!this.synthesis) return
    this.synthesis.cancel()
    this.queue = []
    this.isSpeaking = false
    this.currentUtterance = null
  }
  
  pause() {
    if (!this.synthesis) return
    if (this.isSpeaking) {
      this.synthesis.pause()
    }
  }
  
  resume() {
    if (!this.synthesis) return
    if (this.synthesis.paused) {
      this.synthesis.resume()
    }
  }
  
  getVoices(): SpeechSynthesisVoice[] {
    if (!this.synthesis) return []
    return this.synthesis.getVoices()
  }
  
  // Special JARVIS responses
  greet() {
    const hour = new Date().getHours()
    let greeting = 'Good '
    
    if (hour < 12) greeting += 'morning'
    else if (hour < 17) greeting += 'afternoon'
    else greeting += 'evening'
    
    greeting += ', sir. All systems are operational.'
    
    this.speak(greeting, { emotion: 'neutral' })
  }
  
  acknowledge() {
    const acknowledgments = [
      'Understood, sir.',
      'Right away, sir.',
      'Consider it done.',
      'As you wish, sir.',
      'Immediately, sir.',
    ]
    
    const random = acknowledgments[Math.floor(Math.random() * acknowledgments.length)]
    this.speak(random, { emotion: 'neutral' })
  }
  
  error(message: string) {
    this.speak(`I do apologize, sir, but ${message}`, { emotion: 'sad' })
  }
  
  success(message: string) {
    this.speak(`Excellent news, sir. ${message}`, { emotion: 'happy' })
  }
  
  think() {
    const thoughts = [
      'Processing...',
      'Analyzing the request...',
      'Running calculations...',
      'One moment, sir...',
    ]
    
    const random = thoughts[Math.floor(Math.random() * thoughts.length)]
    this.speak(random, { emotion: 'neutral' })
  }
}

// Export singleton instance
export const jarvisVoice = new JarvisVoice()