module.exports = {
  content: ['./app/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        'quantum-cyan': '#00D4FF',
        'plasma-orange': '#FF6B00',
        'neutron-red': '#FF0040',
        'solar-gold': '#FFD700',
        'quantum-green': '#00FF88'
      },
      animation: {
        'quantum-pulse': 'quantum-pulse 2s cubic-bezier(0.5, 0, 0, 1) infinite',
        'hologram-scan': 'hologram-scan 4s linear infinite'
      }
    }
  }
}
