#!/usr/bin/env python3
"""
JARVIS Desktop App
Native desktop application wrapper using PyQt6
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime
import threading
import queue

# GUI imports
try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    print("Installing PyQt6...")
    os.system("pip install PyQt6 PyQt6-WebEngine")
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWebEngineWidgets import QWebEngineView

# Add JARVIS to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import JARVIS
from jarvis_seamless_v2 import IntelligentJARVIS

class JARVISThread(QThread):
    """Background thread for JARVIS processing"""
    response_ready = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.jarvis = None
        self.message_queue = queue.Queue()
        self.running = True
        
    def run(self):
        """Run JARVIS in background"""
        try:
            self.status_update.emit("Initializing JARVIS...")
            self.jarvis = IntelligentJARVIS()
            self.status_update.emit("JARVIS Online")
            
            while self.running:
                try:
                    # Check for messages
                    message = self.message_queue.get(timeout=0.5)
                    
                    # Process message
                    response = self.jarvis.process_input(message)
                    
                    # Emit response
                    self.response_ready.emit(response)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.response_ready.emit(f"Error: {str(e)}")
                    
        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
            
    def send_message(self, message):
        """Send message to JARVIS"""
        self.message_queue.put(message)
        
    def stop(self):
        """Stop the thread"""
        self.running = False

class JARVISDesktopApp(QMainWindow):
    """Main desktop application window"""
    
    def __init__(self):
        super().__init__()
        self.jarvis_thread = JARVISThread()
        self.init_ui()
        self.setup_jarvis()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("JARVIS - AI Assistant")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0A0B0D;
            }
            QTextEdit {
                background-color: #1A1B1E;
                color: #FFFFFF;
                border: 1px solid #2A2B2E;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
                font-family: 'Inter', -apple-system, sans-serif;
            }
            QLineEdit {
                background-color: #1A1B1E;
                color: #FFFFFF;
                border: 1px solid #2A2B2E;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-family: 'Inter', -apple-system, sans-serif;
            }
            QPushButton {
                background-color: #0066FF;
                color: #FFFFFF;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0052CC;
            }
            QPushButton:pressed {
                background-color: #003D99;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
            }
            QStatusBar {
                background-color: #1A1B1E;
                color: #A0A0A0;
                border-top: 1px solid #2A2B2E;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel("JARVIS")
        logo_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #00D4FF;
            padding: 10px;
        """)
        header_layout.addWidget(logo_label)
        
        header_layout.addStretch()
        
        # Status
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            background-color: rgba(0, 255, 136, 0.1);
            border: 1px solid rgba(0, 255, 136, 0.3);
            border-radius: 16px;
            padding: 8px 16px;
            color: #00FF88;
        """)
        header_layout.addWidget(self.status_label)
        
        main_layout.addLayout(header_layout)
        
        # Chat area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setMinimumHeight(400)
        main_layout.addWidget(self.chat_display)
        
        # Add initial message
        self.add_message("JARVIS", "Hello! I'm JARVIS, your AI assistant. How can I help you today?")
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message or press the mic button to speak...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)
        
        # Voice button
        self.voice_button = QPushButton("🎤")
        self.voice_button.setFixedSize(50, 50)
        self.voice_button.setStyleSheet("""
            QPushButton {
                background-color: #FF3366;
                font-size: 20px;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #FF1A4D;
            }
        """)
        self.voice_button.clicked.connect(self.toggle_voice)
        input_layout.addWidget(self.voice_button)
        
        # Send button
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        
        main_layout.addLayout(input_layout)
        
        # Quick actions
        quick_actions_layout = QHBoxLayout()
        
        actions = [
            ("📧 Email", "Open my email"),
            ("🌤️ Weather", "What's the weather?"),
            ("⏰ Reminder", "Set a reminder"),
            ("🎵 Music", "Play some music")
        ]
        
        for label, command in actions:
            btn = QPushButton(label)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2A2B2E;
                    padding: 8px 16px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #3A3B3E;
                }
            """)
            btn.clicked.connect(lambda checked, cmd=command: self.send_quick_action(cmd))
            quick_actions_layout.addWidget(btn)
            
        quick_actions_layout.addStretch()
        main_layout.addLayout(quick_actions_layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Window icon (if available)
        icon_path = Path("jarvis-icon.png")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
            
    def setup_jarvis(self):
        """Setup JARVIS thread"""
        self.jarvis_thread.response_ready.connect(self.handle_response)
        self.jarvis_thread.status_update.connect(self.update_status)
        self.jarvis_thread.start()
        
    def send_message(self):
        """Send message to JARVIS"""
        message = self.input_field.text().strip()
        if message:
            self.add_message("You", message)
            self.input_field.clear()
            self.jarvis_thread.send_message(message)
            self.statusBar().showMessage("Processing...")
            
    def send_quick_action(self, command):
        """Send quick action command"""
        self.add_message("You", command)
        self.jarvis_thread.send_message(command)
        self.statusBar().showMessage("Processing...")
        
    def handle_response(self, response):
        """Handle JARVIS response"""
        self.add_message("JARVIS", response)
        self.statusBar().showMessage("Ready")
        
        # Speak response (optional)
        # self.speak(response)
        
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
        
        if "Online" in status:
            self.status_label.setStyleSheet("""
                background-color: rgba(0, 255, 136, 0.1);
                border: 1px solid rgba(0, 255, 136, 0.3);
                border-radius: 16px;
                padding: 8px 16px;
                color: #00FF88;
            """)
        else:
            self.status_label.setStyleSheet("""
                background-color: rgba(255, 51, 102, 0.1);
                border: 1px solid rgba(255, 51, 102, 0.3);
                border-radius: 16px;
                padding: 8px 16px;
                color: #FF3366;
            """)
            
    def add_message(self, sender, message):
        """Add message to chat display"""
        timestamp = datetime.now().strftime("%I:%M %p")
        
        # Format message with HTML
        if sender == "JARVIS":
            html = f"""
            <div style="margin: 10px 0;">
                <div style="color: #00D4FF; font-weight: bold;">{sender} • {timestamp}</div>
                <div style="color: #FFFFFF; margin-top: 5px; background-color: rgba(0, 102, 255, 0.1); 
                            padding: 10px; border-radius: 8px; border-left: 3px solid #0066FF;">
                    {message}
                </div>
            </div>
            """
        else:
            html = f"""
            <div style="margin: 10px 0;">
                <div style="color: #A0A0A0; font-weight: bold;">{sender} • {timestamp}</div>
                <div style="color: #FFFFFF; margin-top: 5px; background-color: #2A2B2E; 
                            padding: 10px; border-radius: 8px;">
                    {message}
                </div>
            </div>
            """
            
        self.chat_display.append(html)
        
        # Scroll to bottom
        scrollbar = self.chat_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def toggle_voice(self):
        """Toggle voice input"""
        # This would integrate with speech recognition
        self.statusBar().showMessage("Voice input not yet implemented in desktop app")
        
    def closeEvent(self, event):
        """Handle window close"""
        self.jarvis_thread.stop()
        self.jarvis_thread.wait()
        event.accept()

def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("JARVIS")
    app.setOrganizationName("JARVIS AI")
    
    # Create and show main window
    window = JARVISDesktopApp()
    window.show()
    
    # Start event loop
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
