// Smart Home Service - Mock API that can be replaced with real IoT APIs
// Supports: Philips Hue, Nest, Ring, Sonos, etc.

export interface SmartDevice {
  id: string
  name: string
  type: 'light' | 'thermostat' | 'security' | 'media' | 'appliance'
  room: string
  status: 'online' | 'offline'
  state: any // Device-specific state
}

export interface LightState {
  on: boolean
  brightness: number // 0-100
  color?: {
    hue: number // 0-360
    saturation: number // 0-100
    temperature?: number // 2000-6500K
  }
}

export interface ThermostatState {
  mode: 'heat' | 'cool' | 'auto' | 'off'
  currentTemp: number
  targetTemp: number
  humidity: number
  schedule: boolean
}

export interface SecurityState {
  armed: boolean
  mode: 'home' | 'away' | 'night' | 'off'
  sensors: Array<{
    id: string
    name: string
    type: 'door' | 'window' | 'motion' | 'camera'
    triggered: boolean
  }>
}

export interface MediaState {
  playing: boolean
  volume: number
  source: string
  content?: {
    title: string
    artist?: string
    duration?: number
    position?: number
  }
}

export interface ApplianceState {
  on: boolean
  settings?: Record<string, any>
}

export interface Scene {
  id: string
  name: string
  description: string
  actions: Array<{
    deviceId: string
    state: any
  }>
}

export interface EnergyData {
  timestamp: Date
  consumption: number // kWh
  cost: number // USD
  breakdown: Record<string, number> // Device type -> consumption
}

class SmartHomeService {
  private devices: Map<string, SmartDevice> = new Map()
  private scenes: Map<string, Scene> = new Map()
  private energyHistory: EnergyData[] = []
  private subscribers: Set<(devices: SmartDevice[]) => void> = new Set()

  constructor() {
    this.initializeMockDevices()
    this.initializeScenes()
    this.startEnergyMonitoring()
  }

  private initializeMockDevices() {
    // Living Room
    this.addDevice({
      id: 'light-lr-1',
      name: 'Living Room Ceiling',
      type: 'light',
      room: 'Living Room',
      status: 'online',
      state: { on: true, brightness: 75, color: { hue: 45, saturation: 20 } } as LightState
    })

    this.addDevice({
      id: 'light-lr-2',
      name: 'Living Room Lamp',
      type: 'light',
      room: 'Living Room',
      status: 'online',
      state: { on: false, brightness: 50 } as LightState
    })

    this.addDevice({
      id: 'media-lr-1',
      name: 'Living Room TV',
      type: 'media',
      room: 'Living Room',
      status: 'online',
      state: { playing: false, volume: 30, source: 'Netflix' } as MediaState
    })

    // Bedroom
    this.addDevice({
      id: 'light-br-1',
      name: 'Bedroom Light',
      type: 'light',
      room: 'Bedroom',
      status: 'online',
      state: { on: false, brightness: 30, color: { temperature: 2700 } } as LightState
    })

    // Kitchen
    this.addDevice({
      id: 'light-kt-1',
      name: 'Kitchen Lights',
      type: 'light',
      room: 'Kitchen',
      status: 'online',
      state: { on: true, brightness: 100 } as LightState
    })

    this.addDevice({
      id: 'appliance-kt-1',
      name: 'Coffee Maker',
      type: 'appliance',
      room: 'Kitchen',
      status: 'online',
      state: { on: false, settings: { brew: 'medium', cups: 4 } } as ApplianceState
    })

    // Whole House
    this.addDevice({
      id: 'thermostat-1',
      name: 'Main Thermostat',
      type: 'thermostat',
      room: 'Hallway',
      status: 'online',
      state: {
        mode: 'auto',
        currentTemp: 72,
        targetTemp: 70,
        humidity: 45,
        schedule: true
      } as ThermostatState
    })

    this.addDevice({
      id: 'security-1',
      name: 'Home Security',
      type: 'security',
      room: 'System',
      status: 'online',
      state: {
        armed: false,
        mode: 'home',
        sensors: [
          { id: 's1', name: 'Front Door', type: 'door', triggered: false },
          { id: 's2', name: 'Back Door', type: 'door', triggered: false },
          { id: 's3', name: 'Living Room', type: 'motion', triggered: false }
        ]
      } as SecurityState
    })
  }

  private initializeScenes() {
    this.addScene({
      id: 'morning',
      name: 'Good Morning',
      description: 'Start your day right',
      actions: [
        { deviceId: 'light-kt-1', state: { on: true, brightness: 100 } },
        { deviceId: 'light-lr-1', state: { on: true, brightness: 60, color: { temperature: 5000 } } },
        { deviceId: 'appliance-kt-1', state: { on: true } },
        { deviceId: 'thermostat-1', state: { targetTemp: 72 } }
      ]
    })

    this.addScene({
      id: 'evening',
      name: 'Evening Relax',
      description: 'Wind down for the evening',
      actions: [
        { deviceId: 'light-lr-1', state: { on: true, brightness: 40, color: { temperature: 2700 } } },
        { deviceId: 'light-lr-2', state: { on: true, brightness: 30 } },
        { deviceId: 'light-kt-1', state: { on: false } },
        { deviceId: 'thermostat-1', state: { targetTemp: 68 } }
      ]
    })

    this.addScene({
      id: 'movie',
      name: 'Movie Time',
      description: 'Perfect for watching movies',
      actions: [
        { deviceId: 'light-lr-1', state: { on: true, brightness: 10, color: { hue: 240, saturation: 50 } } },
        { deviceId: 'light-lr-2', state: { on: false } },
        { deviceId: 'media-lr-1', state: { playing: true, volume: 40 } }
      ]
    })

    this.addScene({
      id: 'sleep',
      name: 'Good Night',
      description: 'Ready for bed',
      actions: [
        { deviceId: 'light-lr-1', state: { on: false } },
        { deviceId: 'light-lr-2', state: { on: false } },
        { deviceId: 'light-kt-1', state: { on: false } },
        { deviceId: 'light-br-1', state: { on: true, brightness: 5, color: { temperature: 2000 } } },
        { deviceId: 'thermostat-1', state: { targetTemp: 65 } },
        { deviceId: 'security-1', state: { armed: true, mode: 'night' } }
      ]
    })
  }

  private startEnergyMonitoring() {
    // Simulate energy data collection every hour
    setInterval(() => {
      const now = new Date()
      const consumption = this.calculateCurrentConsumption()
      const cost = consumption * 0.12 // $0.12 per kWh average
      
      this.energyHistory.push({
        timestamp: now,
        consumption,
        cost,
        breakdown: this.getEnergyBreakdown()
      })

      // Keep only last 24 hours
      const dayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000)
      this.energyHistory = this.energyHistory.filter(d => d.timestamp > dayAgo)
    }, 60 * 60 * 1000) // Every hour

    // Initialize with some historical data
    for (let i = 23; i >= 0; i--) {
      const timestamp = new Date(Date.now() - i * 60 * 60 * 1000)
      const consumption = 0.5 + Math.random() * 2 // 0.5-2.5 kWh
      this.energyHistory.push({
        timestamp,
        consumption,
        cost: consumption * 0.12,
        breakdown: this.getEnergyBreakdown()
      })
    }
  }

  private calculateCurrentConsumption(): number {
    let total = 0
    this.devices.forEach(device => {
      if (device.status === 'offline') return
      
      switch (device.type) {
        case 'light':
          const lightState = device.state as LightState
          if (lightState.on) {
            total += (lightState.brightness / 100) * 0.01 // 10W bulb at full brightness
          }
          break
        case 'thermostat':
          const thermoState = device.state as ThermostatState
          if (thermoState.mode !== 'off') {
            total += 2.5 // Average HVAC consumption
          }
          break
        case 'media':
          const mediaState = device.state as MediaState
          if (mediaState.playing) {
            total += 0.15 // TV consumption
          }
          break
        case 'appliance':
          const applianceState = device.state as ApplianceState
          if (applianceState.on) {
            total += 0.8 // Coffee maker
          }
          break
      }
    })
    return total
  }

  private getEnergyBreakdown(): Record<string, number> {
    const breakdown: Record<string, number> = {
      light: 0,
      thermostat: 0,
      media: 0,
      appliance: 0,
      security: 0
    }

    this.devices.forEach(device => {
      if (device.status === 'offline') return
      
      switch (device.type) {
        case 'light':
          const lightState = device.state as LightState
          if (lightState.on) {
            breakdown.light += (lightState.brightness / 100) * 0.01
          }
          break
        case 'thermostat':
          const thermoState = device.state as ThermostatState
          if (thermoState.mode !== 'off') {
            breakdown.thermostat += 2.5
          }
          break
        case 'media':
          const mediaState = device.state as MediaState
          if (mediaState.playing) {
            breakdown.media += 0.15
          }
          break
        case 'appliance':
          const applianceState = device.state as ApplianceState
          if (applianceState.on) {
            breakdown.appliance += 0.8
          }
          break
        case 'security':
          breakdown.security += 0.05 // Always on
          break
      }
    })

    return breakdown
  }

  // Device Management
  addDevice(device: SmartDevice) {
    this.devices.set(device.id, device)
    this.notifySubscribers()
  }

  getDevice(id: string): SmartDevice | undefined {
    return this.devices.get(id)
  }

  getAllDevices(): SmartDevice[] {
    return Array.from(this.devices.values())
  }

  getDevicesByRoom(room: string): SmartDevice[] {
    return this.getAllDevices().filter(d => d.room === room)
  }

  getDevicesByType(type: SmartDevice['type']): SmartDevice[] {
    return this.getAllDevices().filter(d => d.type === type)
  }

  updateDeviceState(id: string, state: Partial<any>): boolean {
    const device = this.devices.get(id)
    if (!device) return false

    device.state = { ...device.state, ...state }
    this.notifySubscribers()
    return true
  }

  // Light Controls
  setLightState(id: string, state: Partial<LightState>): boolean {
    return this.updateDeviceState(id, state)
  }

  toggleLight(id: string): boolean {
    const device = this.devices.get(id)
    if (!device || device.type !== 'light') return false

    const lightState = device.state as LightState
    return this.setLightState(id, { on: !lightState.on })
  }

  dimLight(id: string, brightness: number): boolean {
    return this.setLightState(id, { brightness: Math.max(0, Math.min(100, brightness)) })
  }

  // Thermostat Controls
  setThermostatState(id: string, state: Partial<ThermostatState>): boolean {
    return this.updateDeviceState(id, state)
  }

  setTemperature(id: string, temp: number): boolean {
    return this.setThermostatState(id, { targetTemp: temp })
  }

  // Security Controls
  setSecurityState(id: string, state: Partial<SecurityState>): boolean {
    return this.updateDeviceState(id, state)
  }

  armSecurity(id: string, mode: SecurityState['mode']): boolean {
    return this.setSecurityState(id, { armed: true, mode })
  }

  disarmSecurity(id: string): boolean {
    return this.setSecurityState(id, { armed: false, mode: 'off' })
  }

  // Media Controls
  setMediaState(id: string, state: Partial<MediaState>): boolean {
    return this.updateDeviceState(id, state)
  }

  playMedia(id: string): boolean {
    return this.setMediaState(id, { playing: true })
  }

  pauseMedia(id: string): boolean {
    return this.setMediaState(id, { playing: false })
  }

  setVolume(id: string, volume: number): boolean {
    return this.setMediaState(id, { volume: Math.max(0, Math.min(100, volume)) })
  }

  // Scene Management
  addScene(scene: Scene) {
    this.scenes.set(scene.id, scene)
  }

  getScene(id: string): Scene | undefined {
    return this.scenes.get(id)
  }

  getAllScenes(): Scene[] {
    return Array.from(this.scenes.values())
  }

  activateScene(id: string): boolean {
    const scene = this.scenes.get(id)
    if (!scene) return false

    scene.actions.forEach(action => {
      this.updateDeviceState(action.deviceId, action.state)
    })

    return true
  }

  // Energy Monitoring
  getEnergyData(hours: number = 24): EnergyData[] {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000)
    return this.energyHistory.filter(d => d.timestamp > cutoff)
  }

  getCurrentEnergyUsage(): number {
    return this.calculateCurrentConsumption()
  }

  getEnergyBreakdownCurrent(): Record<string, number> {
    return this.getEnergyBreakdown()
  }

  // Voice Command Integration
  processVoiceCommand(intent: string, entities: string[]): { success: boolean; message: string } {
    switch (intent) {
      case 'control':
        return this.handleControlCommand(entities)
      case 'status':
        return this.handleStatusCommand(entities)
      case 'scene':
        return this.handleSceneCommand(entities)
      case 'security':
        return this.handleSecurityCommand(entities)
      default:
        return { success: false, message: 'Command not recognized for smart home control' }
    }
  }

  private handleControlCommand(entities: string[]): { success: boolean; message: string } {
    const isOn = entities.some(e => e.includes('on'))
    const isOff = entities.some(e => e.includes('off'))
    const isDim = entities.some(e => e.includes('dim'))
    const isBrighten = entities.some(e => e.includes('bright'))

    // Handle lights
    if (entities.some(e => e.includes('light'))) {
      const room = this.extractRoom(entities)
      const lights = room 
        ? this.getDevicesByRoom(room).filter(d => d.type === 'light')
        : this.getDevicesByType('light')

      if (lights.length === 0) {
        return { success: false, message: 'No lights found' }
      }

      lights.forEach(light => {
        if (isOn) this.setLightState(light.id, { on: true })
        else if (isOff) this.setLightState(light.id, { on: false })
        else if (isDim) this.dimLight(light.id, 30)
        else if (isBrighten) this.dimLight(light.id, 100)
      })

      const action = isOn ? 'on' : isOff ? 'off' : isDim ? 'dimmed' : 'brightened'
      return { success: true, message: `Lights turned ${action}` }
    }

    // Handle temperature
    if (entities.some(e => e.includes('temperature') || e.includes('thermostat'))) {
      const tempMatch = entities.join(' ').match(/\d+/)
      if (tempMatch) {
        const temp = parseInt(tempMatch[0])
        this.setTemperature('thermostat-1', temp)
        return { success: true, message: `Temperature set to ${temp}°F` }
      }
    }

    // Handle scenes
    const sceneKeywords = ['morning', 'evening', 'movie', 'sleep', 'night']
    const matchedScene = sceneKeywords.find(s => entities.some(e => e.includes(s)))
    if (matchedScene) {
      const sceneId = matchedScene === 'night' ? 'sleep' : matchedScene
      this.activateScene(sceneId)
      return { success: true, message: `${this.getScene(sceneId)?.name} scene activated` }
    }

    return { success: false, message: 'Could not understand the control command' }
  }

  private handleStatusCommand(entities: string[]): { success: boolean; message: string } {
    if (entities.some(e => e.includes('energy'))) {
      const current = this.getCurrentEnergyUsage()
      return { 
        success: true, 
        message: `Current energy usage: ${current.toFixed(2)} kWh`
      }
    }

    if (entities.some(e => e.includes('temperature'))) {
      const thermostat = this.getDevice('thermostat-1')
      if (thermostat) {
        const state = thermostat.state as ThermostatState
        return {
          success: true,
          message: `Current temperature: ${state.currentTemp}°F, Target: ${state.targetTemp}°F`
        }
      }
    }

    // General status
    const onlineDevices = this.getAllDevices().filter(d => d.status === 'online').length
    const totalDevices = this.getAllDevices().length
    return {
      success: true,
      message: `Smart home status: ${onlineDevices}/${totalDevices} devices online`
    }
  }

  private extractRoom(entities: string[]): string | null {
    const rooms = ['bedroom', 'living room', 'kitchen', 'bathroom', 'office', 'garage']
    return rooms.find(room => entities.some(e => e.includes(room))) || null
  }

  private handleSceneCommand(entities: string[]): { success: boolean; message: string } {
    const sceneKeywords = ['morning', 'evening', 'movie', 'sleep', 'night', 'wake']
    const matchedScene = sceneKeywords.find(s => entities.some(e => e.includes(s)))
    
    if (matchedScene) {
      const sceneId = matchedScene === 'night' ? 'sleep' : matchedScene === 'wake' ? 'morning' : matchedScene
      const success = this.activateScene(sceneId)
      if (success) {
        const scene = this.getScene(sceneId)
        return { success: true, message: `${scene?.name} scene activated` }
      }
    }
    
    return { success: false, message: 'Scene not found' }
  }

  private handleSecurityCommand(entities: string[]): { success: boolean; message: string } {
    const isArm = entities.some(e => e.includes('arm'))
    const isDisarm = entities.some(e => e.includes('disarm'))
    const mode = entities.find(e => ['home', 'away', 'night'].includes(e)) as SecurityState['mode'] | undefined
    
    if (isArm && mode) {
      this.armSecurity('security-1', mode)
      return { success: true, message: `Security armed in ${mode} mode` }
    } else if (isDisarm) {
      this.disarmSecurity('security-1')
      return { success: true, message: 'Security disarmed' }
    }
    
    return { success: false, message: 'Please specify arm or disarm with a mode' }
  }

  // Subscription for real-time updates
  subscribe(callback: (devices: SmartDevice[]) => void): () => void {
    this.subscribers.add(callback)
    return () => this.subscribers.delete(callback)
  }

  private notifySubscribers() {
    const devices = this.getAllDevices()
    this.subscribers.forEach(callback => callback(devices))
  }
}

// Export singleton instance
export const smartHomeService = new SmartHomeService()