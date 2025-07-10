#!/bin/bash
echo "🎨 Setting up World-Class JARVIS UI Project..."

# Create Next.js project with all the bells and whistles
npx create-next-app@latest jarvis-ui \
  --typescript \
  --tailwind \
  --app \
  --src-dir \
  --import-alias "@/*"

cd jarvis-ui

# Install core dependencies
echo "📦 Installing world-class dependencies..."
npm install \
  three @react-three/fiber @react-three/drei @react-three/postprocessing \
  framer-motion gsap lottie-react \
  zustand valtio jotai \
  @radix-ui/react-dialog @radix-ui/react-dropdown-menu \
  clsx tailwind-merge class-variance-authority \
  @mediapipe/hands @tensorflow/tfjs \
  socket.io-client \
  lucide-react

# Install dev dependencies
npm install -D \
  @types/three \
  million \
  prettier prettier-plugin-tailwindcss \
  eslint-config-prettier

# Create component structure
mkdir -p src/components/{ui,3d,animations,layouts}
mkdir -p src/lib/{utils,hooks,stores}
mkdir -p src/styles/shaders

# Create base components
cat > src/components/ui/holographic-card.tsx << 'COMPONENT'
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
COMPONENT

# Create tailwind config with animations
cat > tailwind.config.ts << 'TAILWIND'
import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  theme: {
    extend: {
      colors: {
        cyan: {
          50: '#E6FCFF',
          500: '#00D4FF',
          900: '#002433',
        },
        plasma: '#FF6B00',
        quantum: '#00FF88',
        neutron: '#FF0040',
      },
      animation: {
        'hologram': 'hologram 8s ease-in-out infinite',
        'scan': 'scan 4s linear infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 6s ease-in-out infinite',
      },
      keyframes: {
        hologram: {
          '0%, 100%': { transform: 'translateY(100%)' },
          '50%': { transform: 'translateY(-100%)' },
        },
        scan: {
          '0%': { transform: 'translateY(-100vh)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        glow: {
          '0%': { opacity: '0.5', filter: 'brightness(1)' },
          '100%': { opacity: '1', filter: 'brightness(1.2)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-20px)' },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Space Grotesk', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}

export default config
TAILWIND

echo "✅ World-class UI project initialized!"
echo "📁 Project created at: jarvis-ui/"
echo ""
echo "Next steps:"
echo "1. cd jarvis-ui"
echo "2. npm run dev"
echo "3. Start building incredible UI!"
