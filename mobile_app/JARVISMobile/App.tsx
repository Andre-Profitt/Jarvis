import React, { useEffect, useState } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  ActivityIndicator,
  Alert,
  Platform,
  Dimensions,
  StatusBar,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/Ionicons';
import DeviceInfo from 'react-native-device-info';
import Voice from '@react-native-voice/voice';

// Import screens
import HomeScreen from './src/screens/HomeScreen';
import VoiceScreen from './src/screens/VoiceScreen';
import SmartHomeScreen from './src/screens/SmartHomeScreen';
import CalendarScreen from './src/screens/CalendarScreen';
import SettingsScreen from './src/screens/SettingsScreen';

// Import API client
import JarvisAPI from './src/api/JarvisAPI';

// Import components
import ConnectionStatus from './src/components/ConnectionStatus';

const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Global context for JARVIS connection
export const JarvisContext = React.createContext<{
  api: JarvisAPI | null;
  connected: boolean;
  features: any;
}>({
  api: null,
  connected: false,
  features: {},
});

function MainTabs() {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;

          switch (route.name) {
            case 'Home':
              iconName = focused ? 'home' : 'home-outline';
              break;
            case 'Voice':
              iconName = focused ? 'mic' : 'mic-outline';
              break;
            case 'Smart Home':
              iconName = focused ? 'bulb' : 'bulb-outline';
              break;
            case 'Calendar':
              iconName = focused ? 'calendar' : 'calendar-outline';
              break;
            case 'Settings':
              iconName = focused ? 'settings' : 'settings-outline';
              break;
            default:
              iconName = 'alert-circle-outline';
          }

          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: '#00d4ff',
        tabBarInactiveTintColor: 'gray',
        tabBarStyle: {
          backgroundColor: '#1a1a1a',
          borderTopColor: '#333',
        },
        headerStyle: {
          backgroundColor: '#1a1a1a',
        },
        headerTintColor: '#fff',
      })}
    >
      <Tab.Screen name="Home" component={HomeScreen} />
      <Tab.Screen name="Voice" component={VoiceScreen} />
      <Tab.Screen name="Smart Home" component={SmartHomeScreen} />
      <Tab.Screen name="Calendar" component={CalendarScreen} />
      <Tab.Screen name="Settings" component={SettingsScreen} />
    </Tab.Navigator>
  );
}

export default function App() {
  const [api, setApi] = useState<JarvisAPI | null>(null);
  const [connected, setConnected] = useState(false);
  const [features, setFeatures] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      // Get stored auth token
      const authToken = await AsyncStorage.getItem('jarvis_auth_token');
      const apiUrl = await AsyncStorage.getItem('jarvis_api_url') || 'http://localhost:5001';

      // Initialize API client
      const jarvisApi = new JarvisAPI(apiUrl);
      setApi(jarvisApi);

      if (authToken) {
        // Try to connect with existing token
        jarvisApi.setAuthToken(authToken);
        const status = await jarvisApi.getStatus();
        
        if (status) {
          setConnected(true);
          setFeatures(status.features || {});
        }
      } else {
        // Register device
        await registerDevice(jarvisApi);
      }

      // Setup voice recognition
      await setupVoiceRecognition();

    } catch (error) {
      console.error('Initialization error:', error);
      Alert.alert(
        'Connection Error',
        'Failed to connect to JARVIS. Please check your settings.',
        [{ text: 'OK' }]
      );
    } finally {
      setLoading(false);
    }
  };

  const registerDevice = async (jarvisApi: JarvisAPI) => {
    try {
      const deviceId = await DeviceInfo.getUniqueId();
      const deviceData = {
        device_id: deviceId,
        platform: Platform.OS,
        model: DeviceInfo.getModel(),
        app_version: DeviceInfo.getVersion(),
        os_version: DeviceInfo.getSystemVersion(),
        device_name: await DeviceInfo.getDeviceName(),
        capabilities: {
          voice: true,
          push_notifications: true,
          background_mode: Platform.OS === 'ios',
        },
      };

      const response = await jarvisApi.registerDevice(deviceData);
      
      if (response.success) {
        await AsyncStorage.setItem('jarvis_auth_token', response.auth_token);
        jarvisApi.setAuthToken(response.auth_token);
        setConnected(true);
        setFeatures(response.features || {});
      }
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  };

  const setupVoiceRecognition = async () => {
    try {
      Voice.onSpeechStart = onSpeechStart;
      Voice.onSpeechEnd = onSpeechEnd;
      Voice.onSpeechError = onSpeechError;
      Voice.onSpeechResults = onSpeechResults;
      Voice.onSpeechPartialResults = onSpeechPartialResults;
    } catch (error) {
      console.error('Voice setup error:', error);
    }
  };

  const onSpeechStart = (e: any) => {
    console.log('Speech started');
  };

  const onSpeechEnd = (e: any) => {
    console.log('Speech ended');
  };

  const onSpeechError = (e: any) => {
    console.error('Speech error:', e);
  };

  const onSpeechResults = (e: any) => {
    console.log('Speech results:', e.value);
  };

  const onSpeechPartialResults = (e: any) => {
    console.log('Partial results:', e.value);
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#00d4ff" />
        <Text style={styles.loadingText}>Connecting to JARVIS...</Text>
      </View>
    );
  }

  return (
    <JarvisContext.Provider value={{ api, connected, features }}>
      <NavigationContainer>
        <StatusBar barStyle="light-content" backgroundColor="#1a1a1a" />
        <SafeAreaView style={styles.container}>
          <ConnectionStatus connected={connected} />
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Main" component={MainTabs} />
          </Stack.Navigator>
        </SafeAreaView>
      </NavigationContainer>
    </JarvisContext.Provider>
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
    marginTop: 20,
    fontSize: 16,
  },
});