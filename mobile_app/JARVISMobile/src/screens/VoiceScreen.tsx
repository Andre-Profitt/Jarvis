import React, { useState, useContext, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ScrollView,
  Animated,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import Voice from '@react-native-voice/voice';
import Icon from 'react-native-vector-icons/Ionicons';
import LottieView from 'lottie-react-native';
import { JarvisContext } from '../../App';

const { width } = Dimensions.get('window');

export default function VoiceScreen() {
  const { api, connected } = useContext(JarvisContext);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [processing, setProcessing] = useState(false);
  const [pulseAnim] = useState(new Animated.Value(1));

  useEffect(() => {
    Voice.onSpeechStart = onSpeechStart;
    Voice.onSpeechEnd = onSpeechEnd;
    Voice.onSpeechError = onSpeechError;
    Voice.onSpeechResults = onSpeechResults;
    Voice.onSpeechPartialResults = onSpeechPartialResults;

    return () => {
      Voice.destroy().then(Voice.removeAllListeners);
    };
  }, []);

  useEffect(() => {
    if (isListening) {
      startPulseAnimation();
    } else {
      stopPulseAnimation();
    }
  }, [isListening]);

  const startPulseAnimation = () => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, {
          toValue: 1.2,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(pulseAnim, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
      ]),
    ).start();
  };

  const stopPulseAnimation = () => {
    Animated.timing(pulseAnim, {
      toValue: 1,
      duration: 200,
      useNativeDriver: true,
    }).start();
  };

  const onSpeechStart = () => {
    setTranscript('');
    setResponse('');
  };

  const onSpeechEnd = async () => {
    setIsListening(false);
    
    if (transcript && api) {
      await processCommand(transcript);
    }
  };

  const onSpeechError = (error: any) => {
    console.error('Speech error:', error);
    setIsListening(false);
  };

  const onSpeechResults = (e: any) => {
    if (e.value && e.value[0]) {
      setTranscript(e.value[0]);
    }
  };

  const onSpeechPartialResults = (e: any) => {
    if (e.value && e.value[0]) {
      setTranscript(e.value[0]);
    }
  };

  const startListening = async () => {
    try {
      setTranscript('');
      setResponse('');
      setSuggestions([]);
      await Voice.start('en-US');
      setIsListening(true);
    } catch (error) {
      console.error('Start listening error:', error);
    }
  };

  const stopListening = async () => {
    try {
      await Voice.stop();
      setIsListening(false);
    } catch (error) {
      console.error('Stop listening error:', error);
    }
  };

  const processCommand = async (command: string) => {
    if (!api) return;

    setProcessing(true);
    try {
      const result = await api.sendCommand(command);
      setResponse(result.response);
      setSuggestions(result.suggestions || []);
    } catch (error) {
      console.error('Command error:', error);
      setResponse('Sorry, I encountered an error processing your request.');
    } finally {
      setProcessing(false);
    }
  };

  const handleSuggestion = async (suggestion: string) => {
    setTranscript(suggestion);
    await processCommand(suggestion);
  };

  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  };

  return (
    <View style={styles.container}>
      <ScrollView contentContainerStyle={styles.content}>
        {/* Voice Button */}
        <View style={styles.voiceSection}>
          <TouchableOpacity
            onPress={toggleListening}
            disabled={!connected}
            style={styles.voiceButtonContainer}
          >
            <Animated.View
              style={[
                styles.voiceButton,
                {
                  transform: [{ scale: pulseAnim }],
                  opacity: connected ? 1 : 0.5,
                },
              ]}
            >
              <Icon
                name={isListening ? 'mic' : 'mic-outline'}
                size={60}
                color="#fff"
              />
            </Animated.View>
          </TouchableOpacity>
          
          <Text style={styles.statusText}>
            {!connected
              ? 'Not connected to JARVIS'
              : isListening
              ? 'Listening...'
              : 'Tap to speak'}
          </Text>
        </View>

        {/* Transcript */}
        {transcript !== '' && (
          <View style={styles.transcriptSection}>
            <Text style={styles.transcriptLabel}>You said:</Text>
            <Text style={styles.transcriptText}>{transcript}</Text>
          </View>
        )}

        {/* Processing Indicator */}
        {processing && (
          <View style={styles.processingSection}>
            <ActivityIndicator size="large" color="#00d4ff" />
            <Text style={styles.processingText}>Processing...</Text>
          </View>
        )}

        {/* Response */}
        {response !== '' && !processing && (
          <View style={styles.responseSection}>
            <View style={styles.responseHeader}>
              <Icon name="logo-android" size={24} color="#00d4ff" />
              <Text style={styles.responseLabel}>JARVIS</Text>
            </View>
            <Text style={styles.responseText}>{response}</Text>
          </View>
        )}

        {/* Suggestions */}
        {suggestions.length > 0 && (
          <View style={styles.suggestionsSection}>
            <Text style={styles.suggestionsLabel}>Try saying:</Text>
            {suggestions.map((suggestion, index) => (
              <TouchableOpacity
                key={index}
                style={styles.suggestionButton}
                onPress={() => handleSuggestion(suggestion)}
              >
                <Text style={styles.suggestionText}>{suggestion}</Text>
                <Icon name="arrow-forward" size={20} color="#00d4ff" />
              </TouchableOpacity>
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  content: {
    flexGrow: 1,
    padding: 20,
  },
  voiceSection: {
    alignItems: 'center',
    marginTop: 40,
    marginBottom: 40,
  },
  voiceButtonContainer: {
    marginBottom: 20,
  },
  voiceButton: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#00d4ff',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#00d4ff',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
  },
  statusText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '500',
  },
  transcriptSection: {
    backgroundColor: '#1a1a1a',
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
  },
  transcriptLabel: {
    color: '#666',
    fontSize: 14,
    marginBottom: 8,
  },
  transcriptText: {
    color: '#fff',
    fontSize: 16,
    lineHeight: 24,
  },
  processingSection: {
    alignItems: 'center',
    padding: 40,
  },
  processingText: {
    color: '#fff',
    fontSize: 16,
    marginTop: 16,
  },
  responseSection: {
    backgroundColor: '#0a2540',
    padding: 20,
    borderRadius: 12,
    marginBottom: 20,
    borderLeftWidth: 4,
    borderLeftColor: '#00d4ff',
  },
  responseHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  responseLabel: {
    color: '#00d4ff',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  responseText: {
    color: '#fff',
    fontSize: 16,
    lineHeight: 24,
  },
  suggestionsSection: {
    marginTop: 20,
  },
  suggestionsLabel: {
    color: '#666',
    fontSize: 14,
    marginBottom: 12,
  },
  suggestionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#1a1a1a',
    padding: 16,
    borderRadius: 8,
    marginBottom: 8,
  },
  suggestionText: {
    color: '#fff',
    fontSize: 15,
    flex: 1,
  },
});