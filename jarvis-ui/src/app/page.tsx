import { HolographicCard } from "@/components/ui/holographic-card";
import { VoiceAssistant } from "@/components/VoiceAssistant";
import Link from "next/link";
import { Home as HomeIcon, Cpu, Hand } from "lucide-react";

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
      <div className="max-w-6xl w-full space-y-8">
        <h1 className="text-5xl font-bold text-center bg-gradient-to-r from-cyan-400 to-cyan-600 bg-clip-text text-transparent">
          JARVIS UI
        </h1>
        
        {/* Voice Assistant - The Star of the Show */}
        <div className="mb-12">
          <VoiceAssistant />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <HolographicCard>
            <h2 className="text-2xl font-semibold text-cyan-300 mb-4">Neural Core</h2>
            <p className="text-gray-300">
              Advanced neural processing unit with quantum entanglement capabilities.
            </p>
          </HolographicCard>
          
          <HolographicCard glowIntensity={0.5}>
            <h2 className="text-2xl font-semibold text-cyan-300 mb-4">System Status</h2>
            <p className="text-gray-300">
              All systems operational. Running at 98.7% efficiency.
            </p>
          </HolographicCard>
          
          <HolographicCard scanlines={false}>
            <h2 className="text-2xl font-semibold text-cyan-300 mb-4">Voice Engine</h2>
            <p className="text-gray-300">
              Natural language processing with emotional intelligence integration.
            </p>
          </HolographicCard>
        </div>
        
        {/* Demo Links */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
          <Link href="/demo/smart-home" className="group">
            <HolographicCard className="p-6 hover:border-cyan-400 transition-all cursor-pointer">
              <div className="flex items-center gap-4 mb-3">
                <HomeIcon className="w-8 h-8 text-cyan-400 group-hover:text-cyan-300" />
                <h3 className="text-xl font-semibold text-cyan-300 group-hover:text-cyan-200">
                  Smart Home Control
                </h3>
              </div>
              <p className="text-gray-300">
                Control lights, thermostat, security, and more with voice commands and visual interface.
              </p>
            </HolographicCard>
          </Link>
          
          <Link href="/demo/avatar-3d" className="group">
            <HolographicCard className="p-6 hover:border-cyan-400 transition-all cursor-pointer">
              <div className="flex items-center gap-4 mb-3">
                <Cpu className="w-8 h-8 text-cyan-400 group-hover:text-cyan-300" />
                <h3 className="text-xl font-semibold text-cyan-300 group-hover:text-cyan-200">
                  3D Avatar Demo
                </h3>
              </div>
              <p className="text-gray-300">
                Experience the advanced 3D holographic avatar with real-time animations.
              </p>
            </HolographicCard>
          </Link>

          <Link href="/demo/gesture-control" className="group">
            <HolographicCard className="p-6 hover:border-cyan-400 transition-all cursor-pointer">
              <div className="flex items-center gap-4 mb-3">
                <Hand className="w-8 h-8 text-cyan-400 group-hover:text-cyan-300" />
                <h3 className="text-xl font-semibold text-cyan-300 group-hover:text-cyan-200">
                  Gesture Control
                </h3>
              </div>
              <p className="text-gray-300">
                Control your smart home with hand gestures, facial expressions, and multi-modal interactions.
              </p>
            </HolographicCard>
          </Link>
        </div>
        
        <div className="text-center mt-12">
          <p className="text-cyan-500 text-lg">
            World-class UI that makes Jony Ive jealous 🚀
          </p>
        </div>
      </div>
    </div>
  );
}