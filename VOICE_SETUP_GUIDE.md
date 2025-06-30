# 🎤 JARVIS Voice Recognition Setup Guide

## The Issue
Your JARVIS can **speak** (text-to-speech) but can't **hear** (speech-to-text). Let's fix that!

## 🚀 Quick Start Options

### Option 1: Web-Based Voice Recognition (Easiest!)
1. Open `jarvis_voice_web.html` in Chrome or Edge:
   ```bash
   open jarvis_voice_web.html
   ```
2. Click the microphone button
3. Say "Hey JARVIS" followed by your command
4. Copy the text and paste into JARVIS

**Pros**: Works immediately, no setup needed
**Cons**: Need to copy/paste commands

### Option 2: Install Native Voice Recognition
Run this command to set everything up:
```bash
python3 setup_voice.py
```

This will:
- Install SpeechRecognition
- Install audio libraries
- Test your microphone
- Configure everything

Then run:
```bash
python3 jarvis_v8_voice.py
```

### Option 3: Manual Installation
```bash
# Install Python packages
pip3 install SpeechRecognition pyaudio sounddevice numpy

# macOS only - install audio support
brew install portaudio

# Test it works
python3 test_voice_simple.py
```

## 🎯 How to Use Voice Mode in JARVIS v8

1. **Launch JARVIS v8**:
   ```bash
   python3 jarvis_v8_voice.py
   ```

2. **Activate Voice Mode**:
   - Type: `voice mode`
   - Or answer "yes" when asked at startup

3. **Give Commands**:
   - Say: "Hey JARVIS" (wait for beep)
   - Then say your command
   - JARVIS will respond with voice

## 🔧 Troubleshooting

### "No microphone found"
- Check System Preferences > Security & Privacy > Microphone
- Make sure Terminal has microphone access

### "PyAudio installation failed"
```bash
# macOS fix
brew install portaudio
pip3 install --global-option='build_ext' --global-option='-I/usr/local/include' --global-option='-L/usr/local/lib' pyaudio
```

### "Speech not recognized"
- Speak clearly and not too fast
- Reduce background noise
- Try the web interface instead

## 🎤 Voice Commands That Work

Once voice mode is active:

- "Hey JARVIS, what's the weather?"
- "Hey JARVIS, tell me a joke"
- "Hey JARVIS, analyze system performance"
- "Hey JARVIS, write Python code for a web server"
- "Hey JARVIS, explain quantum computing"

## 💡 Pro Tips

1. **Best Quality**: Use OpenAI Whisper
   - Make sure OPENAI_API_KEY is in your .env
   - Whisper provides the most accurate recognition

2. **Fallback**: Google Speech Recognition
   - Works without API key
   - Requires internet connection

3. **Quick Test**: Web Interface
   - Always works in Chrome/Edge
   - Good for testing if microphone works

## 🚀 Your Next Step

**Easiest Path**:
1. Open `jarvis_voice_web.html` in Chrome
2. Click microphone and speak
3. Copy text to JARVIS

**Best Path**:
1. Run `python3 setup_voice.py`
2. Run `python3 jarvis_v8_voice.py`
3. Say "voice mode"
4. Say "Hey JARVIS..."

Now your JARVIS can both speak AND listen! 🎉
