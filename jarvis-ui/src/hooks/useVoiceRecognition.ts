"use client"

import { useEffect, useState, useRef, useCallback } from 'react'

interface VoiceRecognitionOptions {
  continuous?: boolean
  interimResults?: boolean
  language?: string
  onResult?: (transcript: string, isFinal: boolean) => void
  onError?: (error: Event) => void
  wakeWord?: string
}

export function useVoiceRecognition(options: VoiceRecognitionOptions = {}) {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [interimTranscript, setInterimTranscript] = useState('')
  const [isSupported, setIsSupported] = useState(false)
  
  const recognitionRef = useRef<any>(null)
  const wakeWordTimeoutRef = useRef<NodeJS.Timeout>()
  
  useEffect(() => {
    // Check for browser support
    const SpeechRecognition = 
      (window as any).SpeechRecognition || 
      (window as any).webkitSpeechRecognition
    
    if (SpeechRecognition) {
      setIsSupported(true)
      const recognition = new SpeechRecognition()
      
      // Configure recognition
      recognition.continuous = options.continuous ?? true
      recognition.interimResults = options.interimResults ?? true
      recognition.language = options.language ?? 'en-US'
      
      // Handle results
      recognition.onresult = (event: any) => {
        let finalTranscript = ''
        let interimTranscript = ''
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript
          
          if (event.results[i].isFinal) {
            finalTranscript += transcript + ' '
          } else {
            interimTranscript += transcript
          }
        }
        
        if (finalTranscript) {
          setTranscript(prev => prev + finalTranscript)
          
          // Check for wake word
          if (options.wakeWord && 
              finalTranscript.toLowerCase().includes(options.wakeWord.toLowerCase())) {
            handleWakeWordDetected()
          }
        }
        
        setInterimTranscript(interimTranscript)
        
        // Call custom handler
        options.onResult?.(finalTranscript || interimTranscript, !!finalTranscript)
      }
      
      // Handle errors
      recognition.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error)
        options.onError?.(event)
        
        if (event.error === 'no-speech') {
          // Auto restart on no speech
          recognition.stop()
          setTimeout(() => recognition.start(), 100)
        }
      }
      
      // Handle end
      recognition.onend = () => {
        if (isListening) {
          // Auto restart if still listening
          recognition.start()
        }
      }
      
      recognitionRef.current = recognition
    }
    
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
      if (wakeWordTimeoutRef.current) {
        clearTimeout(wakeWordTimeoutRef.current)
      }
    }
  }, [options.continuous, options.interimResults, options.language, options.wakeWord])
  
  const handleWakeWordDetected = useCallback(() => {
    console.log('Wake word detected!')
    // Clear any existing timeout
    if (wakeWordTimeoutRef.current) {
      clearTimeout(wakeWordTimeoutRef.current)
    }
    
    // Set active listening mode for 30 seconds
    wakeWordTimeoutRef.current = setTimeout(() => {
      console.log('Wake word timeout - returning to passive listening')
    }, 30000)
    
    // Visual/audio feedback could go here
    const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBCuBzvLZhjYIG2m98OScTgwOUafi7qxfGgU7k9n1438hIwiA//8AAAAA//8AAAEA//0AAAAA')
    audio.play()
  }, [])
  
  const startListening = useCallback(() => {
    if (recognitionRef.current && !isListening) {
      recognitionRef.current.start()
      setIsListening(true)
      setTranscript('')
      setInterimTranscript('')
    }
  }, [isListening])
  
  const stopListening = useCallback(() => {
    if (recognitionRef.current && isListening) {
      recognitionRef.current.stop()
      setIsListening(false)
    }
  }, [isListening])
  
  const toggleListening = useCallback(() => {
    if (isListening) {
      stopListening()
    } else {
      startListening()
    }
  }, [isListening, startListening, stopListening])
  
  const resetTranscript = useCallback(() => {
    setTranscript('')
    setInterimTranscript('')
  }, [])
  
  return {
    isListening,
    transcript,
    interimTranscript,
    isSupported,
    startListening,
    stopListening,
    toggleListening,
    resetTranscript,
  }
}