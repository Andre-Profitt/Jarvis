// Holographic Avatar Shader
// Vertex Shader
export const holographicVertexShader = `
  uniform float uTime;
  uniform float uAudioLevel;
  uniform float uBreathingIntensity;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying float vDisplacement;
  
  // Noise function for organic movement
  float noise(vec3 p) {
    return sin(p.x * 10.0) * sin(p.y * 10.0) * sin(p.z * 10.0);
  }
  
  void main() {
    vUv = uv;
    vPosition = position;
    vNormal = normalize(normalMatrix * normal);
    
    // Breathing animation
    float breathing = sin(uTime * 2.0) * 0.02 * uBreathingIntensity;
    vec3 newPosition = position + normal * breathing;
    
    // Audio reactive displacement
    float audioDisplacement = uAudioLevel * 0.05;
    newPosition += normal * audioDisplacement * noise(position + uTime);
    
    // Holographic scan lines
    float scanLine = sin(position.y * 50.0 + uTime * 3.0) * 0.005;
    newPosition += normal * scanLine;
    
    vDisplacement = breathing + audioDisplacement + scanLine;
    
    gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);
  }
`;

// Fragment Shader
export const holographicFragmentShader = `
  uniform float uTime;
  uniform float uAudioLevel;
  uniform vec3 uBaseColor;
  uniform vec3 uGlowColor;
  uniform float uOpacity;
  uniform float uScanlineIntensity;
  uniform float uGlitchIntensity;
  uniform sampler2D uMatcap;
  uniform vec3 uEmotionColor;
  uniform float uEmotionIntensity;
  
  varying vec2 vUv;
  varying vec3 vPosition;
  varying vec3 vNormal;
  varying float vDisplacement;
  
  // Random function
  float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
  }
  
  // Fresnel effect
  float fresnel(vec3 normal, vec3 viewDirection, float power) {
    return pow(1.0 - dot(normal, viewDirection), power);
  }
  
  void main() {
    vec3 viewDirection = normalize(cameraPosition - vPosition);
    
    // Base holographic color
    vec3 color = uBaseColor;
    
    // Add emotion color blend
    color = mix(color, uEmotionColor, uEmotionIntensity);
    
    // Matcap for metallic appearance
    vec2 matcapUV = vNormal.xy * 0.5 + 0.5;
    vec3 matcapColor = texture2D(uMatcap, matcapUV).rgb;
    color = mix(color, matcapColor, 0.3);
    
    // Fresnel rim lighting
    float fresnelFactor = fresnel(vNormal, viewDirection, 2.0);
    vec3 rimColor = mix(uGlowColor, uEmotionColor, uEmotionIntensity);
    color += rimColor * fresnelFactor * 2.0;
    
    // Scanlines
    float scanline = sin(vPosition.y * 100.0 + uTime * 5.0) * 0.5 + 0.5;
    scanline = pow(scanline, 3.0) * uScanlineIntensity;
    color += uGlowColor * scanline;
    
    // Audio reactive glow
    float audioGlow = uAudioLevel * 0.5;
    color += uGlowColor * audioGlow;
    
    // Holographic interference patterns
    float interference = sin(vPosition.x * 50.0) * sin(vPosition.z * 50.0) * 0.1;
    color += vec3(interference) * uGlowColor;
    
    // Glitch effect
    if (uGlitchIntensity > 0.0) {
      float glitchLine = step(0.99, random(vec2(floor(vPosition.y * 20.0), uTime * 10.0)));
      vec3 glitchColor = vec3(
        random(vec2(uTime, vPosition.x)),
        random(vec2(uTime + 1.0, vPosition.y)),
        random(vec2(uTime + 2.0, vPosition.z))
      );
      color = mix(color, glitchColor, glitchLine * uGlitchIntensity);
    }
    
    // Depth fade
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float depthFade = 1.0 - smoothstep(0.0, 3.0, depth);
    
    // Final opacity with fresnel
    float finalOpacity = uOpacity * (0.6 + fresnelFactor * 0.4) * depthFade;
    
    // Add displacement glow
    color += uGlowColor * abs(vDisplacement) * 10.0;
    
    gl_FragColor = vec4(color, finalOpacity);
  }
`;

// Particle shader for voice-reactive particles
export const particleVertexShader = `
  uniform float uTime;
  uniform float uAudioLevel;
  uniform float uAudioFrequency;
  
  attribute float aRandom;
  attribute vec3 aVelocity;
  
  varying float vAlpha;
  varying vec3 vColor;
  
  void main() {
    vec3 pos = position;
    
    // Orbit around avatar
    float angle = uTime * aVelocity.x + aRandom * 6.28;
    float radius = 2.0 + sin(uTime * aVelocity.y + aRandom * 3.14) * 0.5;
    
    pos.x += cos(angle) * radius;
    pos.z += sin(angle) * radius;
    pos.y += sin(uTime * aVelocity.z + aRandom * 3.14) * 1.0;
    
    // Audio reactive movement
    float audioInfluence = uAudioLevel * 2.0;
    pos += normalize(pos) * audioInfluence * aRandom;
    
    // Frequency-based vertical movement
    pos.y += sin(uAudioFrequency * 10.0 + aRandom * 3.14) * 0.5 * uAudioLevel;
    
    // Size based on audio
    float size = 10.0 + uAudioLevel * 20.0 * aRandom;
    
    // Fade based on distance and audio
    float distance = length(pos);
    vAlpha = (1.0 - smoothstep(2.0, 4.0, distance)) * (0.5 + uAudioLevel * 0.5);
    
    // Color variation
    vColor = vec3(
      0.3 + aRandom * 0.2,
      0.6 + sin(uTime + aRandom) * 0.2,
      1.0
    );
    
    vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = size * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

export const particleFragmentShader = `
  uniform sampler2D uTexture;
  uniform vec3 uGlowColor;
  
  varying float vAlpha;
  varying vec3 vColor;
  
  void main() {
    vec2 coord = gl_PointCoord;
    vec4 tex = texture2D(uTexture, coord);
    
    // Circular particle shape
    float dist = length(coord - vec2(0.5));
    if (dist > 0.5) discard;
    
    // Soft edges
    float alpha = 1.0 - smoothstep(0.0, 0.5, dist);
    
    vec3 color = mix(vColor, uGlowColor, 0.5);
    gl_FragColor = vec4(color, alpha * vAlpha * tex.a);
  }
`;