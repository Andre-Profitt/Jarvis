# JARVIS Mobile Companion App 📱

Control your JARVIS AI assistant from anywhere with this powerful mobile companion app for iOS and Android.

## Features 🚀

### Core Capabilities
- **Voice Control** - Talk to JARVIS from your phone
- **Smart Home Control** - Manage lights, devices, and scenes
- **Calendar & Email** - View schedule and email summaries
- **Real-time Updates** - WebSocket connection for instant feedback
- **Command Shortcuts** - Quick access to frequent commands
- **Offline Support** - Basic features work without connection

### Technical Features
- **React Native** - Cross-platform iOS/Android support
- **TypeScript** - Type-safe development
- **WebSocket** - Real-time bidirectional communication
- **Voice Recognition** - Native voice input
- **Secure Auth** - JWT token authentication
- **Push Notifications** - Stay updated (coming soon)

## Architecture 🏗️

### Mobile App Structure
```
JARVISMobile/
├── App.tsx                    # Main app entry point
├── src/
│   ├── api/
│   │   └── JarvisAPI.ts      # API client & WebSocket
│   ├── screens/
│   │   ├── HomeScreen.tsx    # Dashboard & shortcuts
│   │   ├── VoiceScreen.tsx   # Voice interaction
│   │   ├── SmartHomeScreen.tsx
│   │   ├── CalendarScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── components/
│   │   ├── ConnectionStatus.tsx
│   │   ├── DeviceCard.tsx
│   │   └── CommandHistory.tsx
│   └── utils/
│       ├── storage.ts        # AsyncStorage helpers
│       └── voice.ts          # Voice utilities
```

### API Server Structure
```
mobile_app/
├── jarvis_mobile_api.py      # Flask API server
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup Guide 🛠️

### Prerequisites
- Node.js 16+ and npm/yarn
- React Native development environment
- Python 3.8+ (for API server)
- JARVIS Ultimate Plus running on your computer

### 1. Start the Mobile API Server

```bash
# Install Python dependencies
cd mobile_app
pip install -r requirements.txt

# Start the API server
python jarvis_mobile_api.py
```

The API server will start on `http://localhost:5001`

### 2. Configure Network Access

For testing on physical devices, you need to make the API accessible:

**Option A: Same Network**
```bash
# Find your computer's IP address
# Mac/Linux: ifconfig | grep inet
# Windows: ipconfig

# Update the API URL in the app
# Settings > JARVIS URL > http://YOUR_IP:5001
```

**Option B: Ngrok Tunnel (Recommended for testing)**
```bash
# Install ngrok
npm install -g ngrok

# Create tunnel
ngrok http 5001

# Use the HTTPS URL provided by ngrok
```

### 3. Build & Run the Mobile App

```bash
# Navigate to mobile app directory
cd JARVISMobile

# Install dependencies
npm install
# or
yarn install

# iOS (Mac only)
npm run ios

# Android
npm run android

# Expo Go (easiest for testing)
npm start
# Scan QR code with Expo Go app
```

## Usage Guide 📱

### First Launch
1. **Launch the app** - It will auto-detect your device
2. **Enter API URL** - If not on localhost
3. **Grant permissions** - Microphone for voice commands
4. **Start talking!** - "Hey JARVIS..."

### Voice Commands
```
"Turn on the living room lights"
"What's on my calendar today?"
"Check my emails"
"Set bedroom temperature to 72"
"Turn off all lights"
"Good morning JARVIS"
```

### Smart Home Control
- **Tap devices** to toggle on/off
- **Long press** for detailed controls
- **Swipe** to access scenes
- **Pull down** to refresh device status

### Shortcuts
The app learns your most used commands and creates shortcuts on the home screen for quick access.

## API Reference 📡

### Authentication
```http
POST /api/v1/register
{
  "device_id": "unique-device-id",
  "platform": "ios|android",
  "model": "iPhone 14",
  "app_version": "1.0.0"
}

Response:
{
  "success": true,
  "auth_token": "jwt-token",
  "features": {...}
}
```

### Send Command
```http
POST /api/v1/command
Authorization: Bearer {token}
{
  "command": "Turn on the lights"
}

Response:
{
  "success": true,
  "response": "Turning on all lights",
  "suggestions": ["Dim to 50%", "Turn off bedroom"]
}
```

### WebSocket Events
```javascript
// Connect
socket.emit('authenticate', { token: authToken });

// Listen for updates
socket.on('status_update', (data) => {
  // JARVIS status changed
});

socket.on('voice_activity', (data) => {
  // JARVIS is listening/processing
});
```

## Security 🔐

### Authentication Flow
1. Device registers with unique ID
2. Server issues JWT token (30-day expiry)
3. All API calls require Bearer token
4. WebSocket authenticated on connect

### Best Practices
- **Use HTTPS** in production
- **Rotate tokens** periodically
- **Validate device IDs** on server
- **Rate limit** API endpoints
- **Encrypt sensitive data** in storage

## Customization 🎨

### Theme Customization
Edit `src/constants/theme.ts`:
```typescript
export const theme = {
  primary: '#00d4ff',
  background: '#000',
  surface: '#1a1a1a',
  text: '#fff',
  // Add your colors
};
```

### Add New Shortcuts
Edit `jarvis_mobile_api.py`:
```python
shortcuts = [
  {
    'id': 'custom',
    'name': 'Your Shortcut',
    'icon': 'star',
    'command': 'Your command',
    'description': 'What it does'
  }
]
```

### Add New Screens
1. Create screen in `src/screens/`
2. Add to navigation in `App.tsx`
3. Add tab icon and route

## Troubleshooting 🔧

### Connection Issues
- **Check IP address** - Must be reachable from device
- **Firewall** - Allow port 5001
- **JARVIS running** - Ultimate Plus must be active
- **Same network** - Device and computer on same WiFi

### Voice Recognition
- **Permissions** - Check microphone access
- **Language** - Set to en-US in settings
- **Background noise** - Find quiet environment
- **Network speed** - Voice needs good connection

### Performance
- **Reduce animations** - Settings > Performance
- **Clear cache** - Settings > Storage
- **Update app** - Check for latest version
- **Restart JARVIS** - If responses are slow

## Deployment 🚀

### iOS App Store
1. Create Apple Developer account
2. Configure bundle ID and certificates
3. Build release version
4. Submit through App Store Connect

### Google Play Store
1. Create Google Play Developer account
2. Generate signed APK/AAB
3. Create store listing
4. Submit for review

### Self-Hosting API
```bash
# Production setup with Gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:5001 jarvis_mobile_api:app

# Use reverse proxy (nginx) for HTTPS
```

## Roadmap 🗺️

### Coming Soon
- [ ] Push notifications for alerts
- [ ] Voice streaming (real-time)
- [ ] Widgets for iOS/Android
- [ ] Apple Watch / Wear OS apps
- [ ] Automation creation from mobile
- [ ] Multi-user households
- [ ] End-to-end encryption
- [ ] Biometric authentication

## Contributing 🤝

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support 💬

- **Issues**: GitHub Issues
- **Discord**: Join our community
- **Email**: support@jarvis.ai

---

Built with ❤️ for the JARVIS community. Transform your phone into a powerful AI remote control!