"use client"

import { useEffect, useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Mic, MicOff, Volume2, VolumeX } from 'lucide-react'
import { useVoiceRecognition } from '@/hooks/useVoiceRecognition'
import { jarvisVoice } from '@/services/voiceSynthesis'
import { HolographicCard } from './ui/holographic-card'
import { processNaturalLanguage } from '@/services/nlpProcessor'
import { smartHomeService } from '@/services/smartHomeService'

export function VoiceAssistant() {
  const [isActive, setIsActive] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [response, setResponse] = useState('')
  const [voiceWaveform, setVoiceWaveform] = useState<number[]>(new Array(20).fill(0))
  const animationFrameRef = useRef<number>()
  const audioContextRef = useRef<AudioContext>()
  const analyserRef = useRef<AnalyserNode>()
  
  const {
    isListening,
    transcript,
    interimTranscript,
    isSupported,
    startListening,
    stopListening,
    resetTranscript,
  } = useVoiceRecognition({
    continuous: true,
    interimResults: true,
    wakeWord: 'hey jarvis',
    onResult: async (text, isFinal) => {
      if (isFinal && text.trim()) {
        await processCommand(text)
      }
    },
  })
  
  // Initialize audio visualization
  useEffect(() => {
    if (isListening && !audioContextRef.current) {
      navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
          audioContextRef.current = new AudioContext()
          analyserRef.current = audioContextRef.current.createAnalyser()
          const source = audioContextRef.current.createMediaStreamSource(stream)
          source.connect(analyserRef.current)
          analyserRef.current.fftSize = 64
          
          const updateWaveform = () => {
            if (analyserRef.current) {
              const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
              analyserRef.current.getByteFrequencyData(dataArray)
              
              const normalizedData = Array.from(dataArray.slice(0, 20)).map(
                value => value / 255
              )
              setVoiceWaveform(normalizedData)
            }
            
            animationFrameRef.current = requestAnimationFrame(updateWaveform)
          }
          
          updateWaveform()
        })
        .catch(console.error)
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [isListening])
  
  const processCommand = async (command: string) => {
    setIsProcessing(true)
    setResponse('')
    
    // Use JARVIS voice to acknowledge
    jarvisVoice.think()
    
    try {
      // Process with NLP
      const result = await processNaturalLanguage(command)
      
      // Execute command based on intent
      let responseText = ''
      
      switch (result.intent) {
        case 'greeting':
          jarvisVoice.greet()
          responseText = 'Hello! How may I assist you today?'
          break
          
        case 'time':
          const time = new Date().toLocaleTimeString()
          responseText = `The current time is ${time}`
          jarvisVoice.speak(responseText)
          break
          
        case 'weather':
          responseText = 'Checking weather conditions...'
          jarvisVoice.speak('I\'m checking the weather for you, sir.')
          // Real weather API integration would go here
          break
          
        case 'control':
          const controlResult = smartHomeService.processVoiceCommand(result.intent, result.entities)
          responseText = controlResult.message
          if (controlResult.success) {
            jarvisVoice.speak(controlResult.message + ', sir.')
          } else {
            jarvisVoice.error('unable to process that command')
          }
          break
          
        case 'scene':
          const sceneResult = smartHomeService.processVoiceCommand(result.intent, result.entities)
          responseText = sceneResult.message
          if (sceneResult.success) {
            jarvisVoice.acknowledge()
            jarvisVoice.speak(sceneResult.message + ', sir.')
          } else {
            jarvisVoice.error('scene activation failed')
          }
          break
          
        case 'security':
          const securityResult = smartHomeService.processVoiceCommand(result.intent, result.entities)
          responseText = securityResult.message
          if (securityResult.success) {
            jarvisVoice.speak(securityResult.message + ', sir.')
          } else {
            jarvisVoice.error('security command failed')
          }
          break
          
        case 'status':
          const statusResult = smartHomeService.processVoiceCommand(result.intent, result.entities)
          responseText = statusResult.message
          jarvisVoice.speak(statusResult.message)
          break
          
        case 'search':
          responseText = `Searching for: ${result.entities.join(' ')}`
          jarvisVoice.speak(`I'll search for that information right away, sir.`)
          break
          
        default:
          responseText = 'I understand your request. Let me help you with that.'
          jarvisVoice.acknowledge()
      }
      
      setResponse(responseText)
    } catch (error) {
      console.error('Command processing error:', error)
      jarvisVoice.error('encountered an unexpected issue')
      setResponse('Sorry, I encountered an error processing your request.')
    } finally {
      setIsProcessing(false)
      resetTranscript()
    }
  }
  
  const toggleVoiceAssistant = () => {
    if (isListening) {
      stopListening()
      jarvisVoice.speak('Voice assistant deactivated. Call me when you need me, sir.')
    } else {
      startListening()
      jarvisVoice.greet()
    }
    setIsActive(!isActive)
  }
  
  if (!isSupported) {
    return (
      <HolographicCard className="p-8">
        <p className="text-red-400">
          Voice recognition is not supported in your browser. 
          Please use Chrome, Edge, or Safari.
        </p>
      </HolographicCard>
    )
  }
  
  return (
    <div className="relative">
      <HolographicCard className="min-h-[400px] p-8">
        <div className="flex flex-col items-center space-y-6">
          {/* Voice Activation Button */}
          <motion.button
            onClick={toggleVoiceAssistant}
            className={`relative w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300 ${
              isActive 
                ? 'bg-cyan-500/20 border-2 border-cyan-500 shadow-[0_0_30px_rgba(0,212,255,0.5)]' 
                : 'bg-gray-800/50 border-2 border-gray-600 hover:border-cyan-500/50'
            }`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isActive ? (
              <Mic className="w-12 h-12 text-cyan-400" />
            ) : (
              <MicOff className="w-12 h-12 text-gray-400" />
            )}
            
            {/* Pulse animation when active */}
            {isActive && (
              <motion.div
                className="absolute inset-0 rounded-full border-2 border-cyan-500"
                animate={{
                  scale: [1, 1.2, 1],
                  opacity: [0.5, 0, 0.5],
                }}
                transition={{
                  duration: 2,
                  repeat: Infinity,
                  ease: "easeInOut",
                }}
              />
            )}
          </motion.button>
          
          {/* Voice Waveform Visualization */}
          <div className="flex items-center justify-center space-x-1 h-16">
            {voiceWaveform.map((value, index) => (
              <motion.div
                key={index}
                className="w-1 bg-cyan-500 rounded-full"
                animate={{
                  height: `${Math.max(4, value * 64)}px`,
                }}
                transition={{
                  duration: 0.1,
                  ease: "easeOut",
                }}
              />
            ))}
          </div>
          
          {/* Status Text */}
          <div className="text-center space-y-2">
            <p className="text-cyan-400 text-sm">
              {isActive ? 'Listening...' : 'Click to activate'}
            </p>
            
            {/* Live Transcript */}
            {(transcript || interimTranscript) && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="max-w-md"
              >
                <p className="text-gray-300 text-sm">
                  {transcript}
                  <span className="text-gray-500 italic">
                    {interimTranscript}
                  </span>
                </p>
              </motion.div>
            )}
            
            {/* Processing Indicator */}
            {isProcessing && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-center justify-center space-x-2"
              >
                <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-cyan-500 rounded-full animate-bounce delay-200" />
              </motion.div>
            )}
            
            {/* Response */}
            {response && !isProcessing && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 p-4 bg-cyan-500/10 rounded-lg border border-cyan-500/30"
              >
                <p className="text-cyan-300">{response}</p>
              </motion.div>
            )}
          </div>
          
          {/* Voice Settings */}
          <div className="flex items-center space-x-4 mt-4">
            <button
              onClick={() => jarvisVoice.stop()}
              className="p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors"
            >
              <VolumeX className="w-5 h-5 text-gray-400" />
            </button>
            <button
              onClick={() => jarvisVoice.speak('Testing voice synthesis', { emotion: 'happy' })}
              className="p-2 rounded-lg bg-gray-800/50 hover:bg-gray-700/50 transition-colors"
            >
              <Volume2 className="w-5 h-5 text-gray-400" />
            </button>
          </div>
        </div>
      </HolographicCard>
      
      {/* Instructions */}
      <div className="mt-4 text-center text-sm text-gray-500">
        <p>Say "Hey JARVIS" to activate, or click the microphone</p>
        <p>Try: "What time is it?" • "Turn on the lights" • "Search for AI news"</p>
      </div>
    </div>
  )
}