'use client'

import React, { useState, useEffect } from 'react'
import { smartHomeService, SmartDevice, Scene, EnergyData, LightState, ThermostatState, SecurityState, MediaState, ApplianceState } from '@/services/smartHomeService'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Lightbulb, 
  Thermometer, 
  Shield, 
  Play, 
  Pause, 
  Coffee,
  Home,
  Moon,
  Sun,
  Film,
  Battery,
  Zap,
  Volume2,
  Power,
  ChevronRight,
  Settings
} from 'lucide-react'

interface DeviceCardProps {
  device: SmartDevice
  onUpdate: (device: SmartDevice) => void
}

const DeviceCard: React.FC<DeviceCardProps> = ({ device, onUpdate }) => {
  const getIcon = () => {
    switch (device.type) {
      case 'light': return <Lightbulb className="w-6 h-6" />
      case 'thermostat': return <Thermometer className="w-6 h-6" />
      case 'security': return <Shield className="w-6 h-6" />
      case 'media': return <Volume2 className="w-6 h-6" />
      case 'appliance': return <Coffee className="w-6 h-6" />
    }
  }

  const renderControls = () => {
    switch (device.type) {
      case 'light':
        const lightState = device.state as LightState
        return (
          <div className="space-y-2">
            <button
              onClick={() => {
                smartHomeService.toggleLight(device.id)
                onUpdate({ ...device })
              }}
              className={`w-full py-2 px-4 rounded-lg transition-all ${
                lightState.on 
                  ? 'bg-yellow-500 text-black hover:bg-yellow-400' 
                  : 'bg-gray-700 text-white hover:bg-gray-600'
              }`}
            >
              <Power className="w-4 h-4 inline mr-2" />
              {lightState.on ? 'On' : 'Off'}
            </button>
            {lightState.on && (
              <div className="space-y-1">
                <label className="text-xs text-gray-400">Brightness: {lightState.brightness}%</label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={lightState.brightness}
                  onChange={(e) => {
                    smartHomeService.dimLight(device.id, parseInt(e.target.value))
                    onUpdate({ ...device })
                  }}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            )}
          </div>
        )

      case 'thermostat':
        const thermoState = device.state as ThermostatState
        return (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">{thermoState.currentTemp}°F</span>
              <div className="text-right">
                <div className="text-xs text-gray-400">Target</div>
                <div className="text-lg">{thermoState.targetTemp}°F</div>
              </div>
            </div>
            <input
              type="range"
              min="60"
              max="85"
              value={thermoState.targetTemp}
              onChange={(e) => {
                smartHomeService.setTemperature(device.id, parseInt(e.target.value))
                onUpdate({ ...device })
              }}
              className="w-full h-2 bg-gradient-to-r from-blue-500 to-red-500 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex gap-2">
              {(['heat', 'cool', 'auto', 'off'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => {
                    smartHomeService.setThermostatState(device.id, { mode })
                    onUpdate({ ...device })
                  }}
                  className={`flex-1 py-1 px-2 rounded text-xs ${
                    thermoState.mode === mode 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-700 text-gray-300'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
            </div>
          </div>
        )

      case 'security':
        const securityState = device.state as SecurityState
        return (
          <div className="space-y-2">
            <div className={`text-center py-2 px-4 rounded-lg ${
              securityState.armed 
                ? 'bg-red-500 text-white' 
                : 'bg-green-500 text-white'
            }`}>
              {securityState.armed ? 'Armed' : 'Disarmed'}
            </div>
            <div className="flex gap-2">
              {(['home', 'away', 'night'] as const).map(mode => (
                <button
                  key={mode}
                  onClick={() => {
                    smartHomeService.armSecurity(device.id, mode)
                    onUpdate({ ...device })
                  }}
                  disabled={!securityState.armed && mode !== 'home'}
                  className={`flex-1 py-1 px-2 rounded text-xs ${
                    securityState.mode === mode && securityState.armed
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-700 text-gray-300 disabled:opacity-50'
                  }`}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </button>
              ))}
              <button
                onClick={() => {
                  smartHomeService.disarmSecurity(device.id)
                  onUpdate({ ...device })
                }}
                className="flex-1 py-1 px-2 rounded text-xs bg-gray-700 text-gray-300"
              >
                Off
              </button>
            </div>
          </div>
        )

      case 'media':
        const mediaState = device.state as MediaState
        return (
          <div className="space-y-2">
            <button
              onClick={() => {
                mediaState.playing 
                  ? smartHomeService.pauseMedia(device.id)
                  : smartHomeService.playMedia(device.id)
                onUpdate({ ...device })
              }}
              className="w-full py-2 px-4 rounded-lg bg-gray-700 text-white hover:bg-gray-600 transition-all"
            >
              {mediaState.playing ? <Pause className="w-4 h-4 inline" /> : <Play className="w-4 h-4 inline" />}
              <span className="ml-2">{mediaState.playing ? 'Playing' : 'Paused'}</span>
            </button>
            <div className="space-y-1">
              <label className="text-xs text-gray-400">Volume: {mediaState.volume}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={mediaState.volume}
                onChange={(e) => {
                  smartHomeService.setVolume(device.id, parseInt(e.target.value))
                  onUpdate({ ...device })
                }}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        )

      case 'appliance':
        const applianceState = device.state as ApplianceState
        return (
          <button
            onClick={() => {
              smartHomeService.updateDeviceState(device.id, { on: !applianceState.on })
              onUpdate({ ...device })
            }}
            className={`w-full py-2 px-4 rounded-lg transition-all ${
              applianceState.on 
                ? 'bg-green-500 text-white hover:bg-green-400' 
                : 'bg-gray-700 text-white hover:bg-gray-600'
            }`}
          >
            <Power className="w-4 h-4 inline mr-2" />
            {applianceState.on ? 'On' : 'Off'}
          </button>
        )
    }
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="bg-gray-800 rounded-xl p-4 border border-gray-700 hover:border-blue-500 transition-all"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${
            device.status === 'online' ? 'bg-gray-700' : 'bg-red-900'
          }`}>
            {getIcon()}
          </div>
          <div>
            <h3 className="font-medium text-white">{device.name}</h3>
            <p className="text-xs text-gray-400">{device.room}</p>
          </div>
        </div>
        <div className={`w-2 h-2 rounded-full ${
          device.status === 'online' ? 'bg-green-400' : 'bg-red-400'
        }`} />
      </div>
      {renderControls()}
    </motion.div>
  )
}

interface SceneCardProps {
  scene: Scene
  onActivate: () => void
}

const SceneCard: React.FC<SceneCardProps> = ({ scene, onActivate }) => {
  const getIcon = () => {
    switch (scene.id) {
      case 'morning': return <Sun className="w-8 h-8" />
      case 'evening': return <Home className="w-8 h-8" />
      case 'movie': return <Film className="w-8 h-8" />
      case 'sleep': return <Moon className="w-8 h-8" />
      default: return <Settings className="w-8 h-8" />
    }
  }

  return (
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={onActivate}
      className="bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl p-6 text-white hover:from-blue-500 hover:to-purple-500 transition-all"
    >
      <div className="flex flex-col items-center gap-3">
        {getIcon()}
        <h3 className="font-medium">{scene.name}</h3>
        <p className="text-xs opacity-75">{scene.description}</p>
      </div>
    </motion.button>
  )
}

interface EnergyChartProps {
  data: EnergyData[]
}

const EnergyChart: React.FC<EnergyChartProps> = ({ data }) => {
  const maxConsumption = Math.max(...data.map(d => d.consumption))
  const totalCost = data.reduce((sum, d) => sum + d.cost, 0)
  const avgConsumption = data.reduce((sum, d) => sum + d.consumption, 0) / data.length

  return (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          Energy Monitor
        </h3>
        <div className="text-right">
          <div className="text-2xl font-bold text-white">${totalCost.toFixed(2)}</div>
          <div className="text-xs text-gray-400">Last 24 hours</div>
        </div>
      </div>
      
      <div className="mb-4">
        <div className="flex justify-between text-sm text-gray-400 mb-2">
          <span>Consumption</span>
          <span>{avgConsumption.toFixed(2)} kWh avg</span>
        </div>
        <div className="flex items-end gap-1 h-32">
          {data.slice(-24).map((d, i) => (
            <div
              key={i}
              className="flex-1 bg-gradient-to-t from-blue-500 to-cyan-400 rounded-t transition-all hover:opacity-80"
              style={{ height: `${(d.consumption / maxConsumption) * 100}%` }}
              title={`${new Date(d.timestamp).toLocaleTimeString()}: ${d.consumption.toFixed(2)} kWh`}
            />
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        {Object.entries(data[data.length - 1]?.breakdown || {}).map(([type, value]) => (
          <div key={type} className="flex items-center justify-between bg-gray-700 rounded-lg px-3 py-2">
            <span className="capitalize text-gray-300">{type}</span>
            <span className="text-white font-medium">{value.toFixed(2)} kWh</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export const SmartHomePanel: React.FC = () => {
  const [devices, setDevices] = useState<SmartDevice[]>([])
  const [scenes] = useState<Scene[]>(smartHomeService.getAllScenes())
  const [energyData, setEnergyData] = useState<EnergyData[]>([])
  const [selectedRoom, setSelectedRoom] = useState<string>('all')
  const [notification, setNotification] = useState<string | null>(null)

  useEffect(() => {
    // Load initial data
    setDevices(smartHomeService.getAllDevices())
    setEnergyData(smartHomeService.getEnergyData())

    // Subscribe to device updates
    const unsubscribe = smartHomeService.subscribe((updatedDevices) => {
      setDevices(updatedDevices)
    })

    // Update energy data periodically
    const energyInterval = setInterval(() => {
      setEnergyData(smartHomeService.getEnergyData())
    }, 60000) // Every minute

    return () => {
      unsubscribe()
      clearInterval(energyInterval)
    }
  }, [])

  const rooms = ['all', ...Array.from(new Set(devices.map(d => d.room)))]
  const filteredDevices = selectedRoom === 'all' 
    ? devices 
    : devices.filter(d => d.room === selectedRoom)

  const showNotification = (message: string) => {
    setNotification(message)
    setTimeout(() => setNotification(null), 3000)
  }

  const handleSceneActivation = (scene: Scene) => {
    smartHomeService.activateScene(scene.id)
    showNotification(`${scene.name} activated`)
  }

  const handleDeviceUpdate = (device: SmartDevice) => {
    // Force re-render by updating state
    setDevices([...smartHomeService.getAllDevices()])
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Smart Home Control</h1>
        <p className="text-gray-400">Manage your devices and monitor energy usage</p>
      </div>

      {/* Scenes */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold text-white mb-4">Scenes</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {scenes.map(scene => (
            <SceneCard
              key={scene.id}
              scene={scene}
              onActivate={() => handleSceneActivation(scene)}
            />
          ))}
        </div>
      </div>

      {/* Room Filter */}
      <div className="mb-6">
        <div className="flex gap-2 overflow-x-auto pb-2">
          {rooms.map(room => (
            <button
              key={room}
              onClick={() => setSelectedRoom(room)}
              className={`px-4 py-2 rounded-lg whitespace-nowrap transition-all ${
                selectedRoom === room
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              {room === 'all' ? 'All Rooms' : room}
            </button>
          ))}
        </div>
      </div>

      {/* Devices Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <AnimatePresence>
          {filteredDevices.map(device => (
            <DeviceCard
              key={device.id}
              device={device}
              onUpdate={handleDeviceUpdate}
            />
          ))}
        </AnimatePresence>
      </div>

      {/* Energy Monitor */}
      <EnergyChart data={energyData} />

      {/* Notification */}
      <AnimatePresence>
        {notification && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-4 right-4 bg-green-500 text-white px-6 py-3 rounded-lg shadow-lg"
          >
            {notification}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}