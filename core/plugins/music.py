#!/usr/bin/env python3
"""
Music Control Plugin for JARVIS
Controls music playback on Spotify and Apple Music
"""

from typing import Dict, Any, Tuple, Optional, List
import re
import asyncio
import subprocess
import platform
from core.plugin_system import JARVISPlugin, PluginMetadata, PluginCommand


class MusicPlugin(JARVISPlugin):
    """Music control plugin for Spotify and Apple Music"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Music",
            version="1.0.0",
            author="JARVIS Team",
            description="Control music playback on Spotify and Apple Music",
            category="entertainment",
            keywords=["music", "spotify", "apple music", "play", "pause", "skip"],
            requirements=["spotipy"],  # For Spotify Web API
            permissions=["system", "network"],
            config_schema={
                "default_service": {"type": "string", "default": "spotify", "enum": ["spotify", "apple_music"]},
                "spotify_client_id": {"type": "string", "required": False},
                "spotify_client_secret": {"type": "string", "required": False},
                "volume_step": {"type": "integer", "default": 10, "description": "Volume adjustment step"},
                "search_limit": {"type": "integer", "default": 5, "description": "Number of search results"}
            }
        )
        
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the music plugin"""
        self.config = config
        self.default_service = config.get("default_service", "spotify")
        self.volume_step = config.get("volume_step", 10)
        self.search_limit = config.get("search_limit", 5)
        
        # Detect platform
        self.platform = platform.system().lower()
        
        # Initialize Spotify if credentials provided
        self.spotify_client = None
        if config.get("spotify_client_id") and config.get("spotify_client_secret"):
            try:
                import spotipy
                from spotipy.oauth2 import SpotifyClientCredentials
                
                client_credentials_manager = SpotifyClientCredentials(
                    client_id=config["spotify_client_id"],
                    client_secret=config["spotify_client_secret"]
                )
                self.spotify_client = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
                self.logger.info("Spotify API initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Spotify API: {e}")
                
        # Check available services
        self.available_services = self._detect_available_services()
        
        # Register commands
        self._register_playback_commands()
        self._register_search_commands()
        self._register_control_commands()
        
        self.logger.info(f"Music plugin initialized. Available services: {', '.join(self.available_services)}")
        return True
        
    def _register_playback_commands(self):
        """Register playback control commands"""
        self.register_command(PluginCommand(
            name="play_pause",
            patterns=[
                r"(?:play|pause|resume)(?:\s+(?:the\s+)?music)?",
                r"music (?:play|pause)",
                r"(?:start|stop) (?:the\s+)?music"
            ],
            description="Play or pause music",
            parameters={},
            examples=["play music", "pause", "resume music"],
            handler=self.handle_play_pause
        ))
        
        self.register_command(PluginCommand(
            name="next_track",
            patterns=[
                r"(?:next|skip)(?:\s+(?:track|song))?",
                r"play (?:the\s+)?next (?:track|song)"
            ],
            description="Skip to next track",
            parameters={},
            examples=["next", "skip song", "next track"],
            handler=self.handle_next_track
        ))
        
        self.register_command(PluginCommand(
            name="previous_track",
            patterns=[
                r"(?:previous|last|back)(?:\s+(?:track|song))?",
                r"play (?:the\s+)?(?:previous|last) (?:track|song)"
            ],
            description="Go to previous track",
            parameters={},
            examples=["previous", "last song", "back"],
            handler=self.handle_previous_track
        ))
        
    def _register_search_commands(self):
        """Register music search commands"""
        self.register_command(PluginCommand(
            name="play_song",
            patterns=[
                r"play (?:the\s+)?(?:song|track)\s+(.+)",
                r"play (.+?)(?:\s+by\s+(.+))?$",
                r"(?:search|find) (?:and\s+)?play\s+(.+)"
            ],
            description="Search and play a specific song",
            parameters={
                "query": {"type": "string", "description": "Song name or artist"}
            },
            examples=[
                "play Bohemian Rhapsody",
                "play Imagine by John Lennon",
                "search and play jazz music"
            ],
            handler=self.handle_play_song
        ))
        
        self.register_command(PluginCommand(
            name="play_artist",
            patterns=[
                r"play (?:music\s+)?(?:by|from)\s+(.+)",
                r"play (.+?)(?:'s|s)?\s+music",
                r"play artist\s+(.+)"
            ],
            description="Play music by a specific artist",
            parameters={
                "artist": {"type": "string", "description": "Artist name"}
            },
            examples=[
                "play music by The Beatles",
                "play Queen's music",
                "play artist Pink Floyd"
            ],
            handler=self.handle_play_artist
        ))
        
        self.register_command(PluginCommand(
            name="play_playlist",
            patterns=[
                r"play (?:my\s+)?playlist\s+(.+)",
                r"play (.+)\s+playlist",
                r"start playlist\s+(.+)"
            ],
            description="Play a specific playlist",
            parameters={
                "playlist": {"type": "string", "description": "Playlist name"}
            },
            examples=[
                "play my workout playlist",
                "play chill playlist",
                "start playlist favorites"
            ],
            handler=self.handle_play_playlist
        ))
        
    def _register_control_commands(self):
        """Register volume and playback control commands"""
        self.register_command(PluginCommand(
            name="volume_control",
            patterns=[
                r"(?:set\s+)?volume (?:to\s+)?(\d+)(?:\s*%)?",
                r"(?:turn\s+)?volume (up|down)(?:\s+by\s+(\d+))?",
                r"(increase|decrease|raise|lower) (?:the\s+)?volume(?:\s+by\s+(\d+))?"
            ],
            description="Control music volume",
            parameters={
                "level": {"type": "integer", "description": "Volume level (0-100)", "optional": True},
                "direction": {"type": "string", "description": "up or down", "optional": True}
            },
            examples=[
                "volume 50",
                "turn volume up",
                "decrease volume by 20"
            ],
            handler=self.handle_volume_control
        ))
        
        self.register_command(PluginCommand(
            name="current_track",
            patterns=[
                r"what(?:'s| is) (?:currently\s+)?playing",
                r"(?:what|which) (?:song|track) is this",
                r"(?:current|now playing) (?:song|track|music)"
            ],
            description="Get information about current track",
            parameters={},
            examples=[
                "what's playing",
                "what song is this",
                "current track"
            ],
            handler=self.handle_current_track
        ))
        
        self.register_command(PluginCommand(
            name="shuffle_control",
            patterns=[
                r"(?:turn\s+)?(on|off|enable|disable)\s+shuffle",
                r"shuffle (?:mode\s+)?(on|off)"
            ],
            description="Control shuffle mode",
            parameters={
                "state": {"type": "string", "description": "on or off"}
            },
            examples=[
                "turn on shuffle",
                "shuffle off",
                "enable shuffle"
            ],
            handler=self.handle_shuffle_control
        ))
        
        self.register_command(PluginCommand(
            name="repeat_control",
            patterns=[
                r"(?:turn\s+)?(on|off|enable|disable)\s+repeat",
                r"repeat (?:mode\s+)?(on|off|track|all)"
            ],
            description="Control repeat mode",
            parameters={
                "state": {"type": "string", "description": "on, off, track, or all"}
            },
            examples=[
                "turn on repeat",
                "repeat track",
                "disable repeat"
            ],
            handler=self.handle_repeat_control
        ))
        
    async def shutdown(self):
        """Clean up resources"""
        self.logger.info("Music plugin shutting down")
        
    def _detect_available_services(self) -> List[str]:
        """Detect which music services are available"""
        services = []
        
        if self.platform == "darwin":  # macOS
            # Check for Apple Music
            try:
                result = subprocess.run(
                    ["osascript", "-e", "tell application \"System Events\" to (name of processes) contains \"Music\""],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip() == "true":
                    services.append("apple_music")
            except:
                pass
                
            # Check for Spotify
            try:
                result = subprocess.run(
                    ["osascript", "-e", "tell application \"System Events\" to (name of processes) contains \"Spotify\""],
                    capture_output=True, text=True
                )
                if result.returncode == 0 and result.stdout.strip() == "true":
                    services.append("spotify")
            except:
                pass
                
        elif self.platform == "linux":
            # Check for Spotify via D-Bus
            try:
                result = subprocess.run(
                    ["dbus-send", "--print-reply", "--dest=org.freedesktop.DBus", 
                     "/org/freedesktop/DBus", "org.freedesktop.DBus.ListNames"],
                    capture_output=True, text=True
                )
                if "org.mpris.MediaPlayer2.spotify" in result.stdout:
                    services.append("spotify")
            except:
                pass
                
        elif self.platform == "windows":
            # Check for Spotify process
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq Spotify.exe"],
                    capture_output=True, text=True
                )
                if "Spotify.exe" in result.stdout:
                    services.append("spotify")
            except:
                pass
                
        return services if services else ["demo"]
        
    async def handle_play_pause(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle play/pause command"""
        try:
            service = self._get_active_service()
            
            if service == "spotify" and self.platform == "darwin":
                # macOS Spotify control
                script = 'tell application "Spotify" to playpause'
                subprocess.run(["osascript", "-e", script])
                return True, "Music playback toggled"
                
            elif service == "apple_music" and self.platform == "darwin":
                # macOS Apple Music control
                script = 'tell application "Music" to playpause'
                subprocess.run(["osascript", "-e", script])
                return True, "Music playback toggled"
                
            elif service == "spotify" and self.platform == "linux":
                # Linux Spotify control via D-Bus
                subprocess.run([
                    "dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify",
                    "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.PlayPause"
                ])
                return True, "Music playback toggled"
                
            else:
                return True, self._get_demo_response("play_pause")
                
        except Exception as e:
            self.logger.error(f"Error controlling playback: {e}")
            return False, f"Failed to control playback: {str(e)}"
            
    async def handle_next_track(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle next track command"""
        try:
            service = self._get_active_service()
            
            if service == "spotify" and self.platform == "darwin":
                script = 'tell application "Spotify" to next track'
                subprocess.run(["osascript", "-e", script])
                return True, "Skipped to next track"
                
            elif service == "apple_music" and self.platform == "darwin":
                script = 'tell application "Music" to next track'
                subprocess.run(["osascript", "-e", script])
                return True, "Skipped to next track"
                
            elif service == "spotify" and self.platform == "linux":
                subprocess.run([
                    "dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify",
                    "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Next"
                ])
                return True, "Skipped to next track"
                
            else:
                return True, self._get_demo_response("next")
                
        except Exception as e:
            self.logger.error(f"Error skipping track: {e}")
            return False, f"Failed to skip track: {str(e)}"
            
    async def handle_previous_track(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle previous track command"""
        try:
            service = self._get_active_service()
            
            if service == "spotify" and self.platform == "darwin":
                script = 'tell application "Spotify" to previous track'
                subprocess.run(["osascript", "-e", script])
                return True, "Went back to previous track"
                
            elif service == "apple_music" and self.platform == "darwin":
                script = 'tell application "Music" to previous track'
                subprocess.run(["osascript", "-e", script])
                return True, "Went back to previous track"
                
            elif service == "spotify" and self.platform == "linux":
                subprocess.run([
                    "dbus-send", "--print-reply", "--dest=org.mpris.MediaPlayer2.spotify",
                    "/org/mpris/MediaPlayer2", "org.mpris.MediaPlayer2.Player.Previous"
                ])
                return True, "Went back to previous track"
                
            else:
                return True, self._get_demo_response("previous")
                
        except Exception as e:
            self.logger.error(f"Error going to previous track: {e}")
            return False, f"Failed to go to previous track: {str(e)}"
            
    async def handle_play_song(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle play specific song command"""
        try:
            # Extract query and artist if provided
            groups = match.groups()
            query = groups[0].strip()
            artist = groups[1].strip() if len(groups) > 1 and groups[1] else None
            
            if artist:
                search_query = f"{query} {artist}"
            else:
                search_query = query
                
            # Search using Spotify API if available
            if self.spotify_client:
                try:
                    results = self.spotify_client.search(q=search_query, type='track', limit=1)
                    if results['tracks']['items']:
                        track = results['tracks']['items'][0]
                        track_uri = track['uri']
                        track_name = track['name']
                        artist_name = track['artists'][0]['name']
                        
                        # Attempt to play via URI
                        if self.platform == "darwin" and "spotify" in self.available_services:
                            script = f'tell application "Spotify" to play track "{track_uri}"'
                            subprocess.run(["osascript", "-e", script])
                            return True, f"🎵 Now playing: {track_name} by {artist_name}"
                            
                except Exception as e:
                    self.logger.error(f"Spotify search error: {e}")
                    
            # Fallback to basic search
            service = self._get_active_service()
            if service == "spotify" and self.platform == "darwin":
                # Try to search in Spotify app
                script = f'tell application "Spotify" to play track "{search_query}"'
                subprocess.run(["osascript", "-e", script])
                return True, f"🎵 Searching for: {search_query}"
                
            else:
                return True, self._get_demo_response("search", search_query)
                
        except Exception as e:
            self.logger.error(f"Error playing song: {e}")
            return False, f"Failed to play song: {str(e)}"
            
    async def handle_play_artist(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle play artist command"""
        try:
            artist = match.group(1).strip()
            
            # Search using Spotify API if available
            if self.spotify_client:
                try:
                    results = self.spotify_client.search(q=artist, type='artist', limit=1)
                    if results['artists']['items']:
                        artist_obj = results['artists']['items'][0]
                        artist_uri = artist_obj['uri']
                        artist_name = artist_obj['name']
                        
                        if self.platform == "darwin" and "spotify" in self.available_services:
                            # Play artist's top tracks
                            script = f'tell application "Spotify" to play track "{artist_uri}"'
                            subprocess.run(["osascript", "-e", script])
                            return True, f"🎵 Playing music by {artist_name}"
                            
                except Exception as e:
                    self.logger.error(f"Spotify artist search error: {e}")
                    
            return True, self._get_demo_response("artist", artist)
            
        except Exception as e:
            self.logger.error(f"Error playing artist: {e}")
            return False, f"Failed to play artist: {str(e)}"
            
    async def handle_play_playlist(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle play playlist command"""
        try:
            playlist = match.group(1).strip()
            
            service = self._get_active_service()
            if service == "spotify" and self.platform == "darwin":
                # Note: This would require OAuth for user playlists
                return True, f"🎵 Looking for playlist: {playlist}\n(Note: Full playlist support requires Spotify authentication)"
                
            else:
                return True, self._get_demo_response("playlist", playlist)
                
        except Exception as e:
            self.logger.error(f"Error playing playlist: {e}")
            return False, f"Failed to play playlist: {str(e)}"
            
    async def handle_volume_control(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle volume control command"""
        try:
            groups = match.groups()
            
            # Determine volume action
            if groups[0] and groups[0].isdigit():
                # Set specific volume
                volume = int(groups[0])
                volume = max(0, min(100, volume))  # Clamp to 0-100
                
                if self.platform == "darwin":
                    script = f'set volume output volume {volume}'
                    subprocess.run(["osascript", "-e", script])
                    return True, f"🔊 Volume set to {volume}%"
                    
            else:
                # Adjust volume up/down
                direction = groups[0] if groups[0] else groups[2]
                amount = int(groups[1] or groups[3]) if len(groups) > 1 and (groups[1] or groups[3]) else self.volume_step
                
                if direction in ["up", "increase", "raise"]:
                    change = amount
                else:
                    change = -amount
                    
                if self.platform == "darwin":
                    # Get current volume
                    result = subprocess.run(
                        ["osascript", "-e", "output volume of (get volume settings)"],
                        capture_output=True, text=True
                    )
                    current = int(result.stdout.strip()) if result.returncode == 0 else 50
                    new_volume = max(0, min(100, current + change))
                    
                    script = f'set volume output volume {new_volume}'
                    subprocess.run(["osascript", "-e", script])
                    return True, f"🔊 Volume {'increased' if change > 0 else 'decreased'} to {new_volume}%"
                    
            return True, self._get_demo_response("volume")
            
        except Exception as e:
            self.logger.error(f"Error controlling volume: {e}")
            return False, f"Failed to control volume: {str(e)}"
            
    async def handle_current_track(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle current track info command"""
        try:
            service = self._get_active_service()
            
            if service == "spotify" and self.platform == "darwin":
                # Get current track info
                script = '''
                tell application "Spotify"
                    if player state is playing then
                        set trackName to name of current track
                        set artistName to artist of current track
                        set albumName to album of current track
                        return trackName & "|" & artistName & "|" & albumName
                    else
                        return "not_playing"
                    end if
                end tell
                '''
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output == "not_playing":
                        return True, "No music is currently playing"
                    else:
                        parts = output.split("|")
                        if len(parts) >= 3:
                            return True, f"🎵 Now playing:\n**{parts[0]}**\nby {parts[1]}\nfrom {parts[2]}"
                            
            elif service == "apple_music" and self.platform == "darwin":
                script = '''
                tell application "Music"
                    if player state is playing then
                        set trackName to name of current track
                        set artistName to artist of current track
                        set albumName to album of current track
                        return trackName & "|" & artistName & "|" & albumName
                    else
                        return "not_playing"
                    end if
                end tell
                '''
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output == "not_playing":
                        return True, "No music is currently playing"
                    else:
                        parts = output.split("|")
                        if len(parts) >= 3:
                            return True, f"🎵 Now playing:\n**{parts[0]}**\nby {parts[1]}\nfrom {parts[2]}"
                            
            return True, self._get_demo_response("current")
            
        except Exception as e:
            self.logger.error(f"Error getting current track: {e}")
            return False, f"Failed to get current track: {str(e)}"
            
    async def handle_shuffle_control(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle shuffle control command"""
        try:
            state = match.group(1).lower()
            enabled = state in ["on", "enable"]
            
            service = self._get_active_service()
            if service == "spotify" and self.platform == "darwin":
                script = f'tell application "Spotify" to set shuffling to {str(enabled).lower()}'
                subprocess.run(["osascript", "-e", script])
                return True, f"🔀 Shuffle {'enabled' if enabled else 'disabled'}"
                
            return True, self._get_demo_response("shuffle", enabled)
            
        except Exception as e:
            self.logger.error(f"Error controlling shuffle: {e}")
            return False, f"Failed to control shuffle: {str(e)}"
            
    async def handle_repeat_control(self, command: str, match: re.Match) -> Tuple[bool, str]:
        """Handle repeat control command"""
        try:
            state = match.group(1).lower()
            
            service = self._get_active_service()
            if service == "spotify" and self.platform == "darwin":
                if state in ["on", "enable", "all"]:
                    script = 'tell application "Spotify" to set repeating to true'
                else:
                    script = 'tell application "Spotify" to set repeating to false'
                    
                subprocess.run(["osascript", "-e", script])
                return True, f"🔁 Repeat {'enabled' if state in ['on', 'enable', 'all'] else 'disabled'}"
                
            return True, self._get_demo_response("repeat", state)
            
        except Exception as e:
            self.logger.error(f"Error controlling repeat: {e}")
            return False, f"Failed to control repeat: {str(e)}"
            
    def _get_active_service(self) -> str:
        """Get the currently active music service"""
        # Prefer the service that's currently playing
        for service in self.available_services:
            if self._is_service_playing(service):
                return service
                
        # Fallback to default or first available
        if self.default_service in self.available_services:
            return self.default_service
            
        return self.available_services[0] if self.available_services else "demo"
        
    def _is_service_playing(self, service: str) -> bool:
        """Check if a service is currently playing"""
        try:
            if service == "spotify" and self.platform == "darwin":
                script = 'tell application "Spotify" to player state as string'
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                return result.returncode == 0 and "playing" in result.stdout.lower()
                
            elif service == "apple_music" and self.platform == "darwin":
                script = 'tell application "Music" to player state as string'
                result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                return result.returncode == 0 and "playing" in result.stdout.lower()
                
        except:
            pass
            
        return False
        
    def _get_demo_response(self, action: str, query: str = "") -> str:
        """Get demo response when no service is available"""
        responses = {
            "play_pause": "🎵 Music playback toggled (demo mode)",
            "next": "⏭️ Skipped to next track (demo mode)",
            "previous": "⏮️ Went back to previous track (demo mode)",
            "search": f"🔍 Searching for: {query} (demo mode)\nNote: Install Spotify or Apple Music for full functionality",
            "artist": f"🎤 Playing music by {query} (demo mode)",
            "playlist": f"📃 Playing playlist: {query} (demo mode)",
            "volume": "🔊 Volume adjusted (demo mode)",
            "current": "🎵 Currently playing: Demo Track by Demo Artist (demo mode)",
            "shuffle": f"🔀 Shuffle {'enabled' if query else 'toggled'} (demo mode)",
            "repeat": f"🔁 Repeat {query} (demo mode)"
        }
        return responses.get(action, "Music command executed (demo mode)")