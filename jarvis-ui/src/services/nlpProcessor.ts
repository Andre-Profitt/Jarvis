interface NLPResult {
  intent: string
  entities: string[]
  confidence: number
  sentiment: 'positive' | 'negative' | 'neutral'
}

// Intent patterns for natural language understanding
const intentPatterns = {
  greeting: /^(hello|hi|hey|good morning|good afternoon|good evening|greetings)/i,
  time: /what('s| is) the time|what time|current time|time is it/i,
  weather: /weather|temperature|forecast|rain|sunny|cloudy/i,
  control: /turn (on|off)|switch|control|activate|deactivate|dim|brighten|set|adjust/i,
  search: /search|find|look up|google|what is|who is|tell me about/i,
  help: /help|what can you do|capabilities|commands/i,
  status: /status|how are you|system check|diagnostics|energy|consumption/i,
  reminder: /remind|reminder|alert me|notify/i,
  calculation: /calculate|compute|what('s| is) \d+|math/i,
  music: /play|music|song|spotify|pause|stop|next|previous/i,
  scene: /scene|morning|evening|movie|sleep|good night|wake up/i,
  security: /arm|disarm|security|alarm|lock|unlock/i,
}

// Entity extraction patterns
const entityPatterns = {
  lights: /lights?|lamp|bulb|lighting/i,
  temperature: /temperature|thermostat|heating|cooling|ac|air conditioning|degrees?/i,
  device: /tv|television|computer|phone|device|media|appliance|coffee maker/i,
  room: /bedroom|living room|kitchen|bathroom|office|garage|hallway/i,
  time: /\d{1,2}:\d{2}|\d{1,2}\s*(am|pm)|morning|afternoon|evening|night/i,
  number: /\d+/g,
  action: /on|off|dim|bright|increase|decrease|up|down|activate|deactivate/i,
  scene: /morning|evening|movie|sleep|night|wake/i,
  security_mode: /home|away|night/i,
}

// Sentiment keywords
const sentimentKeywords = {
  positive: ['great', 'awesome', 'excellent', 'good', 'happy', 'love', 'thank', 'please'],
  negative: ['bad', 'terrible', 'awful', 'hate', 'angry', 'frustrated', 'annoying'],
}

export async function processNaturalLanguage(text: string): Promise<NLPResult> {
  const normalizedText = text.toLowerCase().trim()
  
  // Detect intent
  let detectedIntent = 'unknown'
  let confidence = 0
  
  for (const [intent, pattern] of Object.entries(intentPatterns)) {
    if (pattern.test(normalizedText)) {
      detectedIntent = intent
      confidence = 0.9
      break
    }
  }
  
  // Fallback intent detection using keywords
  if (detectedIntent === 'unknown') {
    const words = normalizedText.split(/\s+/)
    for (const word of words) {
      for (const [intent, pattern] of Object.entries(intentPatterns)) {
        if (pattern.test(word)) {
          detectedIntent = intent
          confidence = 0.7
          break
        }
      }
      if (detectedIntent !== 'unknown') break
    }
  }
  
  // Extract entities
  const entities: string[] = []
  
  for (const [entityType, pattern] of Object.entries(entityPatterns)) {
    const matches = normalizedText.match(pattern)
    if (matches) {
      entities.push(...matches.map(m => m.toLowerCase()))
    }
  }
  
  // Detect sentiment
  let sentiment: NLPResult['sentiment'] = 'neutral'
  let positiveScore = 0
  let negativeScore = 0
  
  for (const word of normalizedText.split(/\s+/)) {
    if (sentimentKeywords.positive.some(kw => word.includes(kw))) {
      positiveScore++
    }
    if (sentimentKeywords.negative.some(kw => word.includes(kw))) {
      negativeScore++
    }
  }
  
  if (positiveScore > negativeScore) {
    sentiment = 'positive'
  } else if (negativeScore > positiveScore) {
    sentiment = 'negative'
  }
  
  // Special case handling
  if (normalizedText.includes('calculate') || normalizedText.includes('what is')) {
    // Extract mathematical expressions
    const mathExpression = normalizedText.match(/\d+\s*[\+\-\*\/]\s*\d+/)
    if (mathExpression) {
      entities.push(mathExpression[0])
      detectedIntent = 'calculation'
      confidence = 0.95
    }
  }
  
  // Context enhancement
  if (detectedIntent === 'control' && entities.length === 0) {
    // Try to infer what to control
    if (normalizedText.includes('light')) entities.push('lights')
    if (normalizedText.includes('temp')) entities.push('temperature')
  }
  
  return {
    intent: detectedIntent,
    entities: [...new Set(entities)], // Remove duplicates
    confidence,
    sentiment,
  }
}

// Command execution mapping
export function getCommandAction(intent: string, entities: string[]): string {
  const actions: Record<string, () => string> = {
    greeting: () => 'Greet the user warmly',
    time: () => 'Show current time',
    weather: () => `Check weather ${entities.includes('tomorrow') ? 'forecast' : 'conditions'}`,
    control: () => {
      if (entities.includes('lights')) return 'Control smart lights'
      if (entities.includes('temperature')) return 'Adjust thermostat'
      if (entities.includes('tv') || entities.includes('television')) return 'Control TV'
      if (entities.includes('coffee maker')) return 'Control coffee maker'
      return 'Control smart device'
    },
    search: () => `Search for: ${entities.join(' ')}`,
    help: () => 'Show available commands',
    status: () => {
      if (entities.includes('energy') || entities.includes('consumption')) return 'Show energy usage'
      return 'Run system diagnostics'
    },
    reminder: () => 'Set a reminder',
    calculation: () => `Calculate: ${entities.find(e => /\d/.test(e))}`,
    music: () => 'Control music playback',
    scene: () => {
      const sceneMatch = entities.find(e => ['morning', 'evening', 'movie', 'sleep', 'night', 'wake'].includes(e))
      return `Activate ${sceneMatch || 'scene'} scene`
    },
    security: () => {
      if (entities.includes('arm')) return 'Arm security system'
      if (entities.includes('disarm')) return 'Disarm security system'
      return 'Control security system'
    },
  }
  
  return actions[intent]?.() || 'Process general command'
}

// Conversation memory
class ConversationMemory {
  private history: Array<{
    timestamp: Date
    input: string
    intent: string
    response: string
  }> = []
  
  private context: Map<string, any> = new Map()
  
  addInteraction(input: string, intent: string, response: string) {
    this.history.push({
      timestamp: new Date(),
      input,
      intent,
      response,
    })
    
    // Keep only last 10 interactions
    if (this.history.length > 10) {
      this.history.shift()
    }
  }
  
  getContext(key: string): any {
    return this.context.get(key)
  }
  
  setContext(key: string, value: any) {
    this.context.set(key, value)
  }
  
  getRecentIntent(): string | null {
    if (this.history.length === 0) return null
    return this.history[this.history.length - 1].intent
  }
  
  getSummary(): string {
    const intents = this.history.map(h => h.intent)
    const uniqueIntents = [...new Set(intents)]
    return `Recent interactions: ${uniqueIntents.join(', ')}`
  }
}

export const conversationMemory = new ConversationMemory()