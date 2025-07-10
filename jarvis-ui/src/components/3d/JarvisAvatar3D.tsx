'use client';

import React, { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  Environment, 
  Float, 
  OrbitControls,
  EffectComposer,
  Bloom,
  ChromaticAberration,
  Noise,
  Vignette
} from '@react-three/drei';
import * as THREE from 'three';
import { 
  holographicVertexShader, 
  holographicFragmentShader,
  particleVertexShader,
  particleFragmentShader
} from '@/styles/shaders/holographic.glsl';
import { useVoiceActivityDetection } from '@/hooks/useVoiceStreaming';

// Emotion types for the avatar
export type AvatarEmotion = 'neutral' | 'happy' | 'sad' | 'thinking' | 'speaking' | 'excited' | 'concerned';

// Emotion color mapping
const emotionColors: Record<AvatarEmotion, THREE.Color> = {
  neutral: new THREE.Color(0x00d4ff),
  happy: new THREE.Color(0x00ff88),
  sad: new THREE.Color(0x4488ff),
  thinking: new THREE.Color(0xff88ff),
  speaking: new THREE.Color(0x00ffff),
  excited: new THREE.Color(0xffaa00),
  concerned: new THREE.Color(0xff4444)
};

interface AvatarHeadProps {
  audioLevel: number;
  emotion: AvatarEmotion;
  mousePosition: { x: number; y: number };
}

const AvatarHead: React.FC<AvatarHeadProps> = ({ audioLevel, emotion, mousePosition }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const morphTargetsRef = useRef<{ [key: string]: number }>({
    smile: 0,
    frown: 0,
    eyebrowsUp: 0,
    eyebrowsDown: 0,
    mouthOpen: 0,
    eyesClosed: 0
  });

  // Create a procedural matcap texture
  const matcapTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d')!;
    
    // Create gradient for holographic metallic look
    const gradient = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
    gradient.addColorStop(0, '#ffffff');
    gradient.addColorStop(0.3, '#00ffff');
    gradient.addColorStop(0.6, '#0088ff');
    gradient.addColorStop(0.8, '#004488');
    gradient.addColorStop(1, '#000022');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 256, 256);
    
    // Add some noise for texture
    const imageData = ctx.getImageData(0, 0, 256, 256);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 20;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
    ctx.putImageData(imageData, 0, 0);
    
    return new THREE.CanvasTexture(canvas);
  }, []);

  // Create geometry with morph targets
  const geometry = useMemo(() => {
    const geo = new THREE.SphereGeometry(1, 64, 64);
    
    // Create morph target for smile
    const smilePositions = geo.attributes.position.array.slice();
    const frownPositions = geo.attributes.position.array.slice();
    const mouthOpenPositions = geo.attributes.position.array.slice();
    
    // Modify vertices for morphs
    for (let i = 0; i < smilePositions.length; i += 3) {
      const y = smilePositions[i + 1];
      const x = smilePositions[i];
      
      // Smile morph
      if (y < -0.3 && y > -0.7 && Math.abs(x) < 0.5) {
        smilePositions[i + 1] += Math.sin(x * Math.PI) * 0.1;
      }
      
      // Frown morph
      if (y < -0.3 && y > -0.7 && Math.abs(x) < 0.5) {
        frownPositions[i + 1] -= Math.sin(x * Math.PI) * 0.1;
      }
      
      // Mouth open morph
      if (y < -0.4 && y > -0.8 && Math.abs(x) < 0.4) {
        mouthOpenPositions[i + 1] -= Math.abs(Math.sin(x * Math.PI * 2)) * 0.2;
      }
    }
    
    geo.morphAttributes.position = [
      new THREE.Float32BufferAttribute(smilePositions, 3),
      new THREE.Float32BufferAttribute(frownPositions, 3),
      new THREE.Float32BufferAttribute(mouthOpenPositions, 3)
    ];
    
    return geo;
  }, []);

  // Shader material
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: holographicVertexShader,
      fragmentShader: holographicFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uAudioLevel: { value: 0 },
        uBreathingIntensity: { value: 1 },
        uBaseColor: { value: new THREE.Color(0x00d4ff) },
        uGlowColor: { value: new THREE.Color(0x00ffff) },
        uEmotionColor: { value: emotionColors[emotion] },
        uEmotionIntensity: { value: 0 },
        uOpacity: { value: 0.8 },
        uScanlineIntensity: { value: 0.5 },
        uGlitchIntensity: { value: 0 },
        uMatcap: { value: matcapTexture }
      },
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false
    });
  }, [matcapTexture, emotion]);

  // Update morph targets based on emotion
  useEffect(() => {
    const targetMorphs = { ...morphTargetsRef.current };
    
    // Reset all morphs
    Object.keys(targetMorphs).forEach(key => { targetMorphs[key] = 0; });
    
    // Set morphs based on emotion
    switch (emotion) {
      case 'happy':
        targetMorphs.smile = 1;
        targetMorphs.eyebrowsUp = 0.3;
        break;
      case 'sad':
        targetMorphs.frown = 1;
        targetMorphs.eyebrowsDown = 0.5;
        break;
      case 'thinking':
        targetMorphs.eyebrowsUp = 0.7;
        break;
      case 'speaking':
        targetMorphs.mouthOpen = 0.5;
        break;
      case 'excited':
        targetMorphs.smile = 0.8;
        targetMorphs.eyebrowsUp = 0.6;
        break;
      case 'concerned':
        targetMorphs.frown = 0.5;
        targetMorphs.eyebrowsDown = 0.8;
        break;
    }
    
    morphTargetsRef.current = targetMorphs;
  }, [emotion]);

  useFrame((state, delta) => {
    if (!meshRef.current || !materialRef.current) return;

    // Update time
    materialRef.current.uniforms.uTime.value += delta;
    
    // Update audio level with smoothing
    const targetAudioLevel = audioLevel;
    materialRef.current.uniforms.uAudioLevel.value = THREE.MathUtils.lerp(
      materialRef.current.uniforms.uAudioLevel.value,
      targetAudioLevel,
      0.1
    );
    
    // Lip sync - morph mouth based on audio
    if (meshRef.current.morphTargetInfluences) {
      const mouthOpenTarget = audioLevel * 1.5; // Amplify for visibility
      meshRef.current.morphTargetInfluences[2] = THREE.MathUtils.lerp(
        meshRef.current.morphTargetInfluences[2] || 0,
        Math.min(mouthOpenTarget, 1),
        0.3
      );
      
      // Apply emotion morphs with smoothing
      meshRef.current.morphTargetInfluences[0] = THREE.MathUtils.lerp(
        meshRef.current.morphTargetInfluences[0] || 0,
        morphTargetsRef.current.smile,
        0.1
      );
      
      meshRef.current.morphTargetInfluences[1] = THREE.MathUtils.lerp(
        meshRef.current.morphTargetInfluences[1] || 0,
        morphTargetsRef.current.frown,
        0.1
      );
    }
    
    // Eye tracking - follow mouse
    const targetRotationX = -mousePosition.y * 0.2;
    const targetRotationY = mousePosition.x * 0.2;
    
    meshRef.current.rotation.x = THREE.MathUtils.lerp(
      meshRef.current.rotation.x,
      targetRotationX,
      0.05
    );
    
    meshRef.current.rotation.y = THREE.MathUtils.lerp(
      meshRef.current.rotation.y,
      targetRotationY,
      0.05
    );
    
    // Breathing animation
    const breathingScale = 1 + Math.sin(state.clock.elapsedTime * 2) * 0.02;
    meshRef.current.scale.setScalar(breathingScale);
    
    // Update emotion intensity
    materialRef.current.uniforms.uEmotionIntensity.value = THREE.MathUtils.lerp(
      materialRef.current.uniforms.uEmotionIntensity.value,
      emotion !== 'neutral' ? 0.5 : 0,
      0.1
    );
    
    // Random glitch effect
    if (Math.random() < 0.002) {
      materialRef.current.uniforms.uGlitchIntensity.value = 0.5;
    } else {
      materialRef.current.uniforms.uGlitchIntensity.value *= 0.95;
    }
  });

  return (
    <mesh ref={meshRef} geometry={geometry} material={material}>
      <primitive attach="material" object={material} ref={materialRef} />
    </mesh>
  );
};

interface VoiceParticlesProps {
  audioLevel: number;
  count?: number;
}

const VoiceParticles: React.FC<VoiceParticlesProps> = ({ audioLevel, count = 1000 }) => {
  const pointsRef = useRef<THREE.Points>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);

  // Create particle texture
  const particleTexture = useMemo(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d')!;
    
    const gradient = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.5)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 32, 32);
    
    return new THREE.CanvasTexture(canvas);
  }, []);

  // Create particles geometry
  const { geometry, attributes } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const randoms = new Float32Array(count);
    const velocities = new Float32Array(count * 3);
    
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      const angle = Math.random() * Math.PI * 2;
      const radius = 2 + Math.random() * 2;
      
      positions[i3] = Math.cos(angle) * radius;
      positions[i3 + 1] = (Math.random() - 0.5) * 4;
      positions[i3 + 2] = Math.sin(angle) * radius;
      
      randoms[i] = Math.random();
      
      velocities[i3] = Math.random() * 2 - 1;
      velocities[i3 + 1] = Math.random() * 2 - 1;
      velocities[i3 + 2] = Math.random() * 2 - 1;
    }
    
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aRandom', new THREE.BufferAttribute(randoms, 1));
    geo.setAttribute('aVelocity', new THREE.BufferAttribute(velocities, 3));
    
    return { geometry: geo, attributes: { positions, randoms, velocities } };
  }, [count]);

  // Shader material for particles
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: particleVertexShader,
      fragmentShader: particleFragmentShader,
      uniforms: {
        uTime: { value: 0 },
        uAudioLevel: { value: 0 },
        uAudioFrequency: { value: 0 },
        uTexture: { value: particleTexture },
        uGlowColor: { value: new THREE.Color(0x00ffff) }
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
  }, [particleTexture]);

  useFrame((state, delta) => {
    if (!pointsRef.current || !materialRef.current) return;
    
    // Update uniforms
    materialRef.current.uniforms.uTime.value += delta;
    materialRef.current.uniforms.uAudioLevel.value = THREE.MathUtils.lerp(
      materialRef.current.uniforms.uAudioLevel.value,
      audioLevel,
      0.1
    );
    
    // Simulate audio frequency (in real app, get from audio analysis)
    materialRef.current.uniforms.uAudioFrequency.value = 
      Math.sin(state.clock.elapsedTime * 10) * audioLevel;
    
    // Rotate particle system
    pointsRef.current.rotation.y += delta * 0.1;
  });

  return (
    <points ref={pointsRef} geometry={geometry} material={material}>
      <primitive attach="material" object={material} ref={materialRef} />
    </points>
  );
};

interface JarvisAvatar3DProps {
  stream?: MediaStream | null;
  emotion?: AvatarEmotion;
  className?: string;
}

export const JarvisAvatar3D: React.FC<JarvisAvatar3DProps> = ({ 
  stream, 
  emotion = 'neutral',
  className = ''
}) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const { audioLevel } = useVoiceActivityDetection(stream);

  // Track mouse movement for eye tracking
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth) * 2 - 1;
      const y = -(e.clientY / window.innerHeight) * 2 + 1;
      setMousePosition({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className={`w-full h-full ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        gl={{ 
          antialias: true, 
          alpha: true,
          toneMapping: THREE.ACESFilmicToneMapping,
          toneMappingExposure: 1.5
        }}
      >
        <color attach="background" args={['#000000']} />
        
        {/* Lighting */}
        <ambientLight intensity={0.1} />
        <pointLight position={[10, 10, 10]} intensity={0.5} color="#00ffff" />
        <pointLight position={[-10, -10, -10]} intensity={0.3} color="#ff00ff" />
        
        {/* Environment for reflections */}
        <Environment preset="night" />
        
        {/* Main avatar */}
        <Float
          speed={2}
          rotationIntensity={0.5}
          floatIntensity={0.5}
        >
          <AvatarHead 
            audioLevel={audioLevel} 
            emotion={emotion}
            mousePosition={mousePosition}
          />
        </Float>
        
        {/* Voice-reactive particles */}
        <VoiceParticles audioLevel={audioLevel} count={1500} />
        
        {/* Camera controls */}
        <OrbitControls
          enablePan={false}
          enableZoom={false}
          minPolarAngle={Math.PI / 3}
          maxPolarAngle={Math.PI / 1.5}
          autoRotate
          autoRotateSpeed={0.5}
        />
        
        {/* Post-processing effects */}
        <EffectComposer>
          <Bloom 
            intensity={1.5}
            luminanceThreshold={0.2}
            luminanceSmoothing={0.9}
          />
          <ChromaticAberration
            offset={[0.002, 0.002]}
            radialModulation={true}
          />
          <Noise opacity={0.02} />
          <Vignette eskil={false} offset={0.1} darkness={0.5} />
        </EffectComposer>
      </Canvas>
    </div>
  );
};