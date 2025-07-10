import React, { useState, useContext, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  RefreshControl,
  Dimensions,
} from 'react-native';
import Icon from 'react-native-vector-icons/Ionicons';
import { JarvisContext } from '../../App';

const { width } = Dimensions.get('window');

interface Shortcut {
  id: string;
  name: string;
  icon: string;
  command: string;
  description: string;
}

interface StatusSummary {
  jarvis_status: string;
  features: {
    voice: boolean;
    smart_home: boolean;
    calendar_email: boolean;
    swarm: boolean;
    anticipatory: boolean;
  };
  metrics: {
    interactions: number;
    uptime: number;
  };
  smart_home?: {
    devices_online: number;
    total_devices: number;
  };
}

export default function HomeScreen({ navigation }: any) {
  const { api, connected } = useContext(JarvisContext);
  const [shortcuts, setShortcuts] = useState<Shortcut[]>([]);
  const [status, setStatus] = useState<StatusSummary | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastCommand, setLastCommand] = useState<string>('');

  useEffect(() => {
    loadData();
    
    // Subscribe to real-time updates
    if (api) {
      api.connectWebSocket();
      api.on('status_update', handleStatusUpdate);
      api.on('command_update', handleCommandUpdate);
    }

    return () => {
      if (api) {
        api.off('status_update', handleStatusUpdate);
        api.off('command_update', handleCommandUpdate);
      }
    };
  }, [api]);

  const loadData = async () => {
    if (!api) return;

    try {
      const [shortcutsData, statusData] = await Promise.all([
        api.getShortcuts(),
        api.getStatus(),
      ]);

      if (shortcutsData.success) {
        setShortcuts(shortcutsData.shortcuts);
      }

      if (statusData) {
        setStatus(statusData);
      }
    } catch (error) {
      console.error('Load data error:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  const handleStatusUpdate = (data: any) => {
    setStatus(prev => ({
      ...prev,
      ...data,
    }));
  };

  const handleCommandUpdate = (data: any) => {
    setLastCommand(data.command);
  };

  const executeShortcut = async (shortcut: Shortcut) => {
    if (!api) return;

    try {
      const result = await api.sendCommand(shortcut.command);
      
      // Navigate to voice screen to show result
      navigation.navigate('Voice', {
        command: shortcut.command,
        response: result.response,
      });
    } catch (error) {
      console.error('Shortcut error:', error);
    }
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (hours > 24) {
      const days = Math.floor(hours / 24);
      return `${days}d ${hours % 24}h`;
    }
    
    return `${hours}h ${minutes}m`;
  };

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'online':
        return '#4ade80';
      case 'busy':
        return '#fbbf24';
      default:
        return '#ef4444';
    }
  };

  const getIconForShortcut = (iconName: string): string => {
    const iconMap: { [key: string]: string } = {
      sun: 'sunny',
      lightbulb: 'bulb',
      'lightbulb-slash': 'bulb-outline',
      calendar: 'calendar',
      envelope: 'mail',
      moon: 'moon',
      star: 'star',
    };
    
    return iconMap[iconName] || 'apps';
  };

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={onRefresh}
          tintColor="#00d4ff"
        />
      }
    >
      {/* Status Card */}
      <View style={styles.statusCard}>
        <View style={styles.statusHeader}>
          <View style={styles.statusIndicator}>
            <View
              style={[
                styles.statusDot,
                { backgroundColor: getStatusColor(status?.jarvis_status || 'offline') },
              ]}
            />
            <Text style={styles.statusText}>
              JARVIS {status?.jarvis_status || 'Offline'}
            </Text>
          </View>
          {status?.metrics && (
            <Text style={styles.uptimeText}>
              {formatUptime(status.metrics.uptime)}
            </Text>
          )}
        </View>

        {/* Feature Indicators */}
        <View style={styles.featuresGrid}>
          {status?.features && Object.entries(status.features).map(([feature, enabled]) => (
            <View key={feature} style={styles.featureItem}>
              <Icon
                name={enabled ? 'checkmark-circle' : 'close-circle'}
                size={20}
                color={enabled ? '#4ade80' : '#666'}
              />
              <Text style={[styles.featureText, { color: enabled ? '#fff' : '#666' }]}>
                {feature.replace('_', ' ')}
              </Text>
            </View>
          ))}
        </View>

        {/* Metrics */}
        {status?.metrics && (
          <View style={styles.metricsRow}>
            <View style={styles.metricItem}>
              <Text style={styles.metricValue}>{status.metrics.interactions}</Text>
              <Text style={styles.metricLabel}>Commands</Text>
            </View>
            {status.smart_home && (
              <View style={styles.metricItem}>
                <Text style={styles.metricValue}>
                  {status.smart_home.devices_online}/{status.smart_home.total_devices}
                </Text>
                <Text style={styles.metricLabel}>Devices</Text>
              </View>
            )}
          </View>
        )}
      </View>

      {/* Quick Actions */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Quick Actions</Text>
        <View style={styles.shortcutsGrid}>
          {shortcuts.slice(0, 6).map((shortcut) => (
            <TouchableOpacity
              key={shortcut.id}
              style={styles.shortcutButton}
              onPress={() => executeShortcut(shortcut)}
            >
              <Icon
                name={getIconForShortcut(shortcut.icon)}
                size={32}
                color="#00d4ff"
              />
              <Text style={styles.shortcutName} numberOfLines={2}>
                {shortcut.name}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Recent Activity */}
      {lastCommand && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Last Command</Text>
          <View style={styles.activityCard}>
            <Icon name="mic" size={20} color="#00d4ff" />
            <Text style={styles.activityText}>{lastCommand}</Text>
          </View>
        </View>
      )}

      {/* Navigation Cards */}
      <View style={styles.section}>
        <TouchableOpacity
          style={styles.navCard}
          onPress={() => navigation.navigate('Smart Home')}
        >
          <Icon name="bulb" size={32} color="#00d4ff" />
          <View style={styles.navCardContent}>
            <Text style={styles.navCardTitle}>Smart Home</Text>
            <Text style={styles.navCardSubtitle}>Control lights & devices</Text>
          </View>
          <Icon name="chevron-forward" size={24} color="#666" />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.navCard}
          onPress={() => navigation.navigate('Calendar')}
        >
          <Icon name="calendar" size={32} color="#00d4ff" />
          <View style={styles.navCardContent}>
            <Text style={styles.navCardTitle}>Calendar & Email</Text>
            <Text style={styles.navCardSubtitle}>View schedule & messages</Text>
          </View>
          <Icon name="chevron-forward" size={24} color="#666" />
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  content: {
    padding: 20,
    paddingBottom: 40,
  },
  statusCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginBottom: 24,
  },
  statusHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  statusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  statusText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  uptimeText: {
    color: '#666',
    fontSize: 14,
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 16,
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 16,
    marginBottom: 8,
  },
  featureText: {
    fontSize: 12,
    marginLeft: 4,
    textTransform: 'capitalize',
  },
  metricsRow: {
    flexDirection: 'row',
    borderTopWidth: 1,
    borderTopColor: '#333',
    paddingTop: 16,
  },
  metricItem: {
    flex: 1,
    alignItems: 'center',
  },
  metricValue: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '700',
  },
  metricLabel: {
    color: '#666',
    fontSize: 12,
    marginTop: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 16,
  },
  shortcutsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -8,
  },
  shortcutButton: {
    width: (width - 56) / 3,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    margin: 8,
    alignItems: 'center',
  },
  shortcutName: {
    color: '#fff',
    fontSize: 12,
    marginTop: 8,
    textAlign: 'center',
  },
  activityCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    alignItems: 'center',
  },
  activityText: {
    color: '#fff',
    fontSize: 14,
    marginLeft: 12,
    flex: 1,
  },
  navCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  navCardContent: {
    flex: 1,
    marginLeft: 16,
  },
  navCardTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  navCardSubtitle: {
    color: '#666',
    fontSize: 14,
    marginTop: 2,
  },
});