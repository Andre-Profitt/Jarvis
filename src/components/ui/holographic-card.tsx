"use client"

import { forwardRef } from 'react'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface HolographicCardProps extends React.HTMLAttributes<HTMLDivElement> {
  glowIntensity?: number
  scanlines?: boolean
}

export const HolographicCard = forwardRef<HTMLDivElement, HolographicCardProps>(
  ({ className, children, glowIntensity = 0.3, scanlines = true, ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        className={cn(
          'relative overflow-hidden rounded-2xl',
          'bg-gradient-to-br from-cyan-500/10 via-cyan-600/5 to-transparent',
          'backdrop-blur-xl border border-cyan-500/20',
          'before:absolute before:inset-0',
          'before:bg-gradient-to-br before:from-transparent before:to-cyan-500/10',
          'before:opacity-0 hover:before:opacity-100',
          'before:transition-opacity before:duration-500',
          'after:absolute after:inset-0 after:rounded-2xl',
          `after:shadow-[inset_0_0_30px_rgba(0,212,255,${glowIntensity})]`,
          'transform-gpu',
          className
        )}
        whileHover={{ scale: 1.02, rotateY: 2 }}
        whileTap={{ scale: 0.98 }}
        transition={{ type: "spring", stiffness: 300, damping: 20 }}
        {...props}
      >
        <div className="relative z-10 p-6">
          {children}
        </div>
        
        {/* Holographic effect */}
        <div className="absolute inset-0 opacity-30">
          <div className="absolute inset-0 bg-gradient-to-t from-transparent via-cyan-500/20 to-transparent transform translate-y-full animate-hologram" />
        </div>
        
        {/* Scanlines */}
        {scanlines && (
          <div className="absolute inset-0 opacity-20 pointer-events-none">
            <div className="h-px bg-gradient-to-r from-transparent via-cyan-500 to-transparent animate-scan" />
          </div>
        )}
      </motion.div>
    )
  }
)

HolographicCard.displayName = 'HolographicCard'
