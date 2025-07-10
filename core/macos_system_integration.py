#!/usr/bin/env python3
"""
JARVIS macOS System Integration
Deep integration with macOS for seamless system control.
"""

import subprocess
import os
import re
from typing import List, Dict, Any, Optional, Tuple
import json
import sqlite3
from pathlib import Path
import plistlib
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger("jarvis.macos")


class MacOSIntegration:
    """Deep macOS system integration for JARVIS"""
    
    def __init__(self):
        self.home_dir = Path.home()
        self.applications_dir = Path("/Applications")
        self.user_applications_dir = self.home_dir / "Applications"
        self.spotlight_index = {}
        self._build_app_index()
        
    def _build_app_index(self):
        """Build index of installed applications"""
        # Index system applications
        for app_path in self.applications_dir.glob("*.app"):
            app_name = app_path.stem.lower()
            self.spotlight_index[app_name] = str(app_path)
            
        # Index user applications
        if self.user_applications_dir.exists():
            for app_path in self.user_applications_dir.glob("*.app"):
                app_name = app_path.stem.lower()
                self.spotlight_index[app_name] = str(app_path)
                
        # Add common name variations
        self._add_app_aliases()
        
    def _add_app_aliases(self):
        """Add common application name aliases"""
        aliases = {
            "chrome": "google chrome",
            "firefox": "firefox",
            "mail": "mail",
            "calendar": "calendar",
            "messages": "messages",
            "facetime": "facetime",
            "music": "music",
            "tv": "tv",
            "photos": "photos",
            "notes": "notes",
            "reminders": "reminders",
            "safari": "safari",
            "finder": "finder",
            "terminal": "terminal",
            "vscode": "visual studio code",
            "code": "visual studio code",
            "slack": "slack",
            "zoom": "zoom",
            "spotify": "spotify",
            "discord": "discord"
        }
        
        for alias, full_name in aliases.items():
            if full_name in self.spotlight_index:
                self.spotlight_index[alias] = self.spotlight_index[full_name]
                
    def open_application(self, app_name: str) -> Tuple[bool, str]:
        """Open an application by name"""
        app_name_lower = app_name.lower()
        
        # Check if app is in index
        if app_name_lower in self.spotlight_index:
            app_path = self.spotlight_index[app_name_lower]
            try:
                subprocess.run(["open", app_path], check=True)
                return True, f"Opening {app_name}"
            except subprocess.CalledProcessError:
                return False, f"Failed to open {app_name}"
                
        # Try direct open command
        try:
            subprocess.run(["open", "-a", app_name], check=True)
            return True, f"Opening {app_name}"
        except subprocess.CalledProcessError:
            # Try to find app using mdfind
            try:
                result = subprocess.run(
                    ["mdfind", f"kMDItemDisplayName == '*{app_name}*'cd && kMDItemKind == 'Application'"],
                    capture_output=True, text=True, check=True
                )
                apps = result.stdout.strip().split('\n')
                if apps and apps[0]:
                    subprocess.run(["open", apps[0]], check=True)
                    return True, f"Opening {app_name}"
            except:
                pass
                
        return False, f"Could not find application: {app_name}"
        
    def close_application(self, app_name: str) -> Tuple[bool, str]:
        """Close an application by name"""
        script = f'''
        tell application "{app_name}"
            quit
        end tell
        '''
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, f"Closing {app_name}"
        except subprocess.CalledProcessError:
            # Try to find and quit by process name
            try:
                subprocess.run(["pkill", "-i", app_name], check=True)
                return True, f"Closed {app_name}"
            except:
                return False, f"Could not close {app_name}"
                
    def switch_to_application(self, app_name: str) -> Tuple[bool, str]:
        """Switch to an application (bring to front)"""
        script = f'''
        tell application "{app_name}"
            activate
        end tell
        '''
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, f"Switching to {app_name}"
        except subprocess.CalledProcessError:
            return False, f"Could not switch to {app_name}"
            
    def get_active_application(self) -> str:
        """Get the currently active application"""
        script = '''
        tell application "System Events"
            get name of first application process whose frontmost is true
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except:
            return "Unknown"
            
    def set_volume(self, level: int) -> Tuple[bool, str]:
        """Set system volume (0-100)"""
        level = max(0, min(100, level))
        script = f'set volume output volume {level}'
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, f"Volume set to {level}%"
        except:
            return False, "Failed to set volume"
            
    def adjust_volume(self, change: int) -> Tuple[bool, str]:
        """Adjust volume by relative amount"""
        current = self.get_volume()
        new_level = max(0, min(100, current + change))
        return self.set_volume(new_level)
        
    def get_volume(self) -> int:
        """Get current system volume"""
        script = 'output volume of (get volume settings)'
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=True
            )
            return int(result.stdout.strip())
        except:
            return 50
            
    def mute_volume(self) -> Tuple[bool, str]:
        """Mute system volume"""
        script = 'set volume with output muted'
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, "Volume muted"
        except:
            return False, "Failed to mute"
            
    def unmute_volume(self) -> Tuple[bool, str]:
        """Unmute system volume"""
        script = 'set volume without output muted'
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, "Volume unmuted"
        except:
            return False, "Failed to unmute"
            
    def set_brightness(self, level: float) -> Tuple[bool, str]:
        """Set display brightness (0.0-1.0)"""
        level = max(0.0, min(1.0, level))
        
        try:
            # Using brightness command if available
            subprocess.run(["brightness", str(level)], check=True)
            return True, f"Brightness set to {int(level * 100)}%"
        except:
            # Fallback to AppleScript
            script = f'''
            tell application "System Preferences"
                reveal anchor "displaysDisplayTab" of pane "com.apple.preference.displays"
                tell application "System Events" to tell process "System Preferences"
                    set value of slider 1 of group 1 of tab group 1 of window 1 to {level}
                end tell
                quit
            end tell
            '''
            try:
                subprocess.run(["osascript", "-e", script], check=True)
                return True, f"Brightness set to {int(level * 100)}%"
            except:
                return False, "Failed to set brightness"
                
    def take_screenshot(self, save_to_clipboard: bool = False) -> Tuple[bool, str]:
        """Take a screenshot"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        if save_to_clipboard:
            try:
                subprocess.run(["screencapture", "-c"], check=True)
                return True, "Screenshot saved to clipboard"
            except:
                return False, "Failed to take screenshot"
        else:
            desktop = self.home_dir / "Desktop"
            filename = f"Screenshot_{timestamp}.png"
            filepath = desktop / filename
            
            try:
                subprocess.run(["screencapture", str(filepath)], check=True)
                return True, f"Screenshot saved to Desktop as {filename}"
            except:
                return False, "Failed to take screenshot"
                
    def show_notification(self, title: str, message: str, sound: bool = True) -> Tuple[bool, str]:
        """Show a system notification"""
        script = f'''
        display notification "{message}" with title "{title}"''' + (f' sound name "Default"' if sound else '')
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, "Notification shown"
        except:
            return False, "Failed to show notification"
            
    def get_running_applications(self) -> List[str]:
        """Get list of running applications"""
        script = '''
        tell application "System Events"
            get name of every application process
        end tell
        '''
        
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=True
            )
            apps = result.stdout.strip().split(", ")
            return apps
        except:
            return []
            
    def search_spotlight(self, query: str) -> List[str]:
        """Search using Spotlight"""
        try:
            result = subprocess.run(
                ["mdfind", query],
                capture_output=True, text=True, check=True
            )
            results = result.stdout.strip().split('\n')
            return [r for r in results if r][:10]  # Return top 10 results
        except:
            return []
            
    def open_url(self, url: str) -> Tuple[bool, str]:
        """Open a URL in default browser"""
        try:
            subprocess.run(["open", url], check=True)
            return True, f"Opening {url}"
        except:
            return False, f"Failed to open {url}"
            
    def system_sleep(self) -> Tuple[bool, str]:
        """Put the system to sleep"""
        script = 'tell application "System Events" to sleep'
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, "Putting system to sleep"
        except:
            return False, "Failed to sleep system"
            
    def lock_screen(self) -> Tuple[bool, str]:
        """Lock the screen"""
        try:
            subprocess.run(
                ["osascript", "-e", 'tell application "System Events" to keystroke "q" using {command down, control down}'],
                check=True
            )
            return True, "Locking screen"
        except:
            return False, "Failed to lock screen"
            
    def empty_trash(self) -> Tuple[bool, str]:
        """Empty the trash"""
        script = 'tell application "Finder" to empty trash'
        
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            return True, "Trash emptied"
        except:
            return False, "Failed to empty trash"
            
    def get_wifi_status(self) -> Dict[str, Any]:
        """Get WiFi connection status"""
        try:
            # Get current WiFi network
            result = subprocess.run(
                ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I"],
                capture_output=True, text=True, check=True
            )
            
            wifi_info = {}
            for line in result.stdout.split('\n'):
                if "SSID" in line and "BSSID" not in line:
                    wifi_info["network"] = line.split(":")[1].strip()
                elif "link auth" in line:
                    wifi_info["security"] = line.split(":")[1].strip()
                elif "lastTxRate" in line:
                    wifi_info["speed"] = line.split(":")[1].strip()
                    
            wifi_info["connected"] = "network" in wifi_info
            return wifi_info
        except:
            return {"connected": False}
            
    def toggle_wifi(self, enable: bool) -> Tuple[bool, str]:
        """Toggle WiFi on/off"""
        state = "on" if enable else "off"
        
        try:
            subprocess.run(
                ["networksetup", "-setairportpower", "en0", state],
                check=True
            )
            return True, f"WiFi turned {state}"
        except:
            return False, f"Failed to turn WiFi {state}"
            
    def get_battery_status(self) -> Dict[str, Any]:
        """Get battery status"""
        try:
            result = subprocess.run(
                ["pmset", "-g", "batt"],
                capture_output=True, text=True, check=True
            )
            
            battery_info = {}
            output = result.stdout
            
            # Parse battery percentage
            percent_match = re.search(r'(\d+)%', output)
            if percent_match:
                battery_info["percentage"] = int(percent_match.group(1))
                
            # Check if charging
            battery_info["charging"] = "AC attached" in output or "charging" in output.lower()
            
            # Parse time remaining
            time_match = re.search(r'(\d+:\d+) remaining', output)
            if time_match:
                battery_info["time_remaining"] = time_match.group(1)
                
            return battery_info
        except:
            return {}
            
    def set_do_not_disturb(self, enable: bool) -> Tuple[bool, str]:
        """Toggle Do Not Disturb mode"""
        # This is complex on modern macOS, using a workaround
        if enable:
            script = '''
            tell application "System Events"
                option key down
                click menu bar item 1 of menu bar 1 of application process "ControlCenter"
                option key up
            end tell
            '''
        else:
            script = '''
            tell application "System Events"
                click menu bar item 1 of menu bar 1 of application process "ControlCenter"
            end tell
            '''
            
        try:
            subprocess.run(["osascript", "-e", script], check=True)
            state = "enabled" if enable else "disabled"
            return True, f"Do Not Disturb {state}"
        except:
            return False, "Failed to toggle Do Not Disturb"


# Singleton instance
_macos_integration = None

def get_macos_integration() -> MacOSIntegration:
    """Get singleton MacOS integration instance"""
    global _macos_integration
    if _macos_integration is None:
        _macos_integration = MacOSIntegration()
    return _macos_integration