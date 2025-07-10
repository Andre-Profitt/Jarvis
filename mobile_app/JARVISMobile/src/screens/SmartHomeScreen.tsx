import React, { useState, useContext, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  RefreshControl,
  Switch,
  ActivityIndicator,
  Alert,
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import Slider from '@react-native-community/slider';
import { JarvisContext } from '../../App';

interface SmartDevice {
  id: string;
  name: string;
  type: string;
  state: {
    on?: boolean;
    brightness?: number;
    temperature?: number;
    color?: string;
  };
  room: string;
  capabilities: string[];
}

export default function SmartHomeScreen() {
  const { api, connected, features } = useContext(JarvisContext);
  const [devices, setDevices] = useState<SmartDevice[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedRoom, setSelectedRoom] = useState<string>('all');
  const [controllingDevice, setControllingDevice] = useState<string | null>(null);

  useEffect(() => {
    if (features.smart_home) {
      loadDevices();
    } else {
      setLoading(false);
    }
  }, [features]);

  const loadDevices = async () => {
    if (!api) return;

    try {
      setLoading(true);
      const devicesData = await api.getSmartDevices();
      setDevices(devicesData);
    } catch (error) {
      console.error('Load devices error:', error);
      Alert.alert('Error', 'Failed to load smart home devices');
    } finally {
      setLoading(false);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDevices();
    setRefreshing(false);
  };

  const getRooms = (): string[] => {
    const rooms = new Set<string>();
    rooms.add('all');
    devices.forEach(device => {
      if (device.room) {
        rooms.add(device.room);
      }
    });
    return Array.from(rooms);
  };

  const getFilteredDevices = (): SmartDevice[] => {
    if (selectedRoom === 'all') {
      return devices;
    }
    return devices.filter(device => device.room === selectedRoom);
  };

  const toggleDevice = async (device: SmartDevice) => {
    if (!api || controllingDevice) return;

    try {
      setControllingDevice(device.id);
      const newState = !device.state.on;
      
      await api.controlSmartDevice(
        device.id,
        newState ? 'turn_on' : 'turn_off'
      );

      // Update local state
      setDevices(prevDevices =>
        prevDevices.map(d =>
          d.id === device.id
            ? { ...d, state: { ...d.state, on: newState } }
            : d
        )
      );
    } catch (error) {
      console.error('Toggle device error:', error);
      Alert.alert('Error', 'Failed to control device');
    } finally {
      setControllingDevice(null);
    }
  };

  const adjustBrightness = async (device: SmartDevice, brightness: number) => {
    if (!api) return;

    try {
      await api.controlSmartDevice(device.id, 'set_brightness', {
        brightness: Math.round(brightness),
      });

      // Update local state
      setDevices(prevDevices =>
        prevDevices.map(d =>
          d.id === device.id
            ? { ...d, state: { ...d.state, brightness } }
            : d
        )
      );
    } catch (error) {
      console.error('Brightness error:', error);
    }
  };

  const getDeviceIcon = (type: string): string => {
    switch (type) {
      case 'light':
        return 'bulb';
      case 'switch':
        return 'toggle';
      case 'thermostat':
        return 'thermometer';
      case 'lock':
        return 'lock-closed';
      case 'camera':
        return 'camera';
      default:
        return 'cube';
    }
  };

  const renderDevice = (device: SmartDevice) => {
    const isOn = device.state.on;
    const isControlling = controllingDevice === device.id;

    return (
      <View key={device.id} style={styles.deviceCard}>
        <View style={styles.deviceHeader}>
          <View style={styles.deviceInfo}>
            <Icon
              name={getDeviceIcon(device.type)}
              size={24}
              color={isOn ? '#00d4ff' : '#666'}
            />
            <View style={styles.deviceText}>
              <Text style={styles.deviceName}>{device.name}</Text>
              <Text style={styles.deviceRoom}>{device.room}</Text>
            </View>
          </View>
          
          {device.capabilities.includes('on_off') && (
            <View style={styles.deviceControl}>
              {isControlling ? (
                <ActivityIndicator size="small" color="#00d4ff" />
              ) : (
                <Switch
                  value={isOn}
                  onValueChange={() => toggleDevice(device)}
                  trackColor={{ false: '#333', true: '#00d4ff' }}
                  thumbColor={isOn ? '#fff' : '#666'}
                />
              )}
            </View>
          )}
        </View>

        {/* Brightness Control */}
        {isOn && device.capabilities.includes('brightness') && (
          <View style={styles.brightnessControl}>
            <Icon name="sunny" size={16} color="#666" />
            <Slider
              style={styles.slider}
              minimumValue={0}
              maximumValue={100}
              value={device.state.brightness || 0}
              onSlidingComplete={(value) => adjustBrightness(device, value)}
              minimumTrackTintColor="#00d4ff"
              maximumTrackTintColor="#333"
              thumbTintColor="#fff"
            />
            <Text style={styles.brightnessValue}>
              {Math.round(device.state.brightness || 0)}%
            </Text>
          </View>
        )}
      </View>
    );
  };

  if (!features.smart_home) {
    return (
      <View style={styles.emptyContainer}>
        <Icon name="bulb-outline" size={64} color="#666" />
        <Text style={styles.emptyTitle}>Smart Home Not Configured</Text>
        <Text style={styles.emptyText}>
          Set up smart home integration in JARVIS to control your devices
        </Text>
      </View>
    );
  }

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#00d4ff" />
        <Text style={styles.loadingText}>Loading smart devices...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Room Filter */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={styles.roomFilter}
        contentContainerStyle={styles.roomFilterContent}
      >
        {getRooms().map((room) => (
          <TouchableOpacity
            key={room}
            style={[
              styles.roomButton,
              selectedRoom === room && styles.roomButtonActive,
            ]}
            onPress={() => setSelectedRoom(room)}
          >
            <Text
              style={[
                styles.roomButtonText,
                selectedRoom === room && styles.roomButtonTextActive,
              ]}
            >
              {room === 'all' ? 'All Rooms' : room}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Devices List */}
      <ScrollView
        style={styles.devicesList}
        contentContainerStyle={styles.devicesContent}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#00d4ff"
          />
        }
      >
        {getFilteredDevices().length === 0 ? (
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No devices found</Text>
          </View>
        ) : (
          getFilteredDevices().map(renderDevice)
        )}

        {/* Quick Actions */}
        <View style={styles.quickActions}>
          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              if (api) {
                api.sendCommand('Turn off all lights');
              }
            }}
          >
            <Icon name="power" size={24} color="#fff" />
            <Text style={styles.actionText}>All Off</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              if (api) {
                api.sendCommand('Set lights to movie mode');
              }
            }}
          >
            <Icon name="film" size={24} color="#fff" />
            <Text style={styles.actionText}>Movie Mode</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.actionButton}
            onPress={() => {
              if (api) {
                api.sendCommand('Good night');
              }
            }}
          >
            <Icon name="moon" size={24} color="#fff" />
            <Text style={styles.actionText}>Bedtime</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  loadingText: {
    color: '#fff',
    marginTop: 16,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 40,
  },
  emptyTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyText: {
    color: '#666',
    fontSize: 16,
    textAlign: 'center',
  },
  roomFilter: {
    maxHeight: 60,
    backgroundColor: '#0a0a0a',
    borderBottomWidth: 1,
    borderBottomColor: '#1a1a1a',
  },
  roomFilterContent: {
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  roomButton: {
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#1a1a1a',
    marginRight: 12,
  },
  roomButtonActive: {
    backgroundColor: '#00d4ff',
  },
  roomButtonText: {
    color: '#666',
    fontSize: 14,
    fontWeight: '500',
  },
  roomButtonTextActive: {
    color: '#000',
  },
  devicesList: {
    flex: 1,
  },
  devicesContent: {
    padding: 16,
  },
  deviceCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  deviceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  deviceInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  deviceText: {
    marginLeft: 12,
    flex: 1,
  },
  deviceName: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  deviceRoom: {
    color: '#666',
    fontSize: 14,
    marginTop: 2,
  },
  deviceControl: {
    marginLeft: 16,
  },
  brightnessControl: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  slider: {
    flex: 1,
    height: 40,
    marginHorizontal: 12,
  },
  brightnessValue: {
    color: '#fff',
    fontSize: 14,
    minWidth: 40,
    textAlign: 'right',
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: 24,
    paddingTop: 24,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  actionButton: {
    alignItems: 'center',
    padding: 16,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    minWidth: 100,
  },
  actionText: {
    color: '#fff',
    fontSize: 14,
    marginTop: 8,
  },
});