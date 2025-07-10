'use client'

import { SmartHomePanel } from '@/components/SmartHomePanel'
import { VoiceAssistant } from '@/components/VoiceAssistant'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function SmartHomeDemoPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-black">
      <div className="container mx-auto p-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-cyan-400 hover:text-cyan-300 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Home
          </Link>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1 }}
        >
          <SmartHomePanel />
        </motion.div>

        {/* Floating Voice Assistant */}
        <motion.div
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="fixed bottom-8 right-8 z-50"
        >
          <div className="bg-gray-900/90 backdrop-blur-lg rounded-2xl p-4 shadow-2xl border border-cyan-500/20">
            <p className="text-sm text-gray-400 mb-2 text-center">Voice Control</p>
            <VoiceAssistant />
          </div>
        </motion.div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-8 bg-gray-800/50 backdrop-blur rounded-xl p-6 border border-gray-700"
        >
          <h2 className="text-xl font-semibold text-white mb-4">Voice Commands</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
            <div>
              <h3 className="text-cyan-400 font-medium mb-2">Device Control</h3>
              <ul className="space-y-1">
                <li>"Turn on the lights"</li>
                <li>"Turn off living room lights"</li>
                <li>"Dim the bedroom lights"</li>
                <li>"Set temperature to 72 degrees"</li>
                <li>"Turn on the TV"</li>
              </ul>
            </div>
            <div>
              <h3 className="text-cyan-400 font-medium mb-2">Scenes & Status</h3>
              <ul className="space-y-1">
                <li>"Activate morning scene"</li>
                <li>"Good night" (activates sleep scene)</li>
                <li>"Movie time"</li>
                <li>"Check energy consumption"</li>
                <li>"What's the temperature?"</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}