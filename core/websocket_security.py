#!/usr/bin/env python3
"""
WebSocket Security Implementation
Secure authentication and encryption for device communication
"""

import asyncio
import websockets
import jwt
import json
import ssl
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Set
from pathlib import Path
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketSecurity:
    """Secure WebSocket communication for JARVIS ecosystem"""
    
    def __init__(self):
        self.secret_key = self._load_or_generate_secret()
        self.fernet = self._create_fernet()
        self.authenticated_clients: Set[str] = set()
        self.client_tokens: Dict[str, Dict[str, Any]] = {}
        self.ssl_context = self._create_ssl_context()
        
    def _load_or_generate_secret(self) -> str:
        """Load or generate secret key for JWT tokens"""
        
        secret_file = Path.home() / ".jarvis" / "secret.key"
        secret_file.parent.mkdir(exist_ok=True)
        
        if secret_file.exists():
            return secret_file.read_text().strip()
        else:
            # Generate new secret
            secret = secrets.token_urlsafe(64)
            secret_file.write_text(secret)
            secret_file.chmod(0o600)  # Only owner can read
            return secret
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet encryption instance"""
        
        # Derive key from secret
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'jarvis-salt-2025',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(self.secret_key.encode())
        )
        return Fernet(key)
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for secure WebSocket"""
        
        cert_dir = Path.home() / ".jarvis" / "certs"
        cert_file = cert_dir / "jarvis.crt"
        key_file = cert_dir / "jarvis.key"
        
        if cert_file.exists() and key_file.exists():
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(cert_file, key_file)
            return ssl_context
        else:
            # Generate self-signed certificate for development
            self._generate_self_signed_cert(cert_dir)
            return None
    
    def _generate_self_signed_cert(self, cert_dir: Path):
        """Generate self-signed certificate for development"""
        
        cert_dir.mkdir(parents=True, exist_ok=True)
        
        # This would use OpenSSL or cryptography to generate certs
        # For now, we'll use unencrypted WebSocket in development
        logger.warning("No SSL certificates found. Using unencrypted WebSocket.")
    
    def generate_device_token(self, device_info: Dict[str, Any]) -> str:
        """Generate JWT token for device authentication"""
        
        payload = {
            "device_id": device_info.get("device_id"),
            "device_type": device_info.get("device_type"),
            "device_name": device_info.get("device_name"),
            "issued_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=30)).isoformat()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        # Store token info
        self.client_tokens[device_info["device_id"]] = {
            "token": token,
            "payload": payload,
            "last_seen": datetime.utcnow()
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            # Check expiration
            expires_at = datetime.fromisoformat(payload["expires_at"])
            if datetime.utcnow() > expires_at:
                logger.warning(f"Token expired for device {payload.get('device_id')}")
                return None
            
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            return None
    
    async def authenticate_websocket(self, websocket, path) -> bool:
        """Authenticate incoming WebSocket connection"""
        
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(
                websocket.recv(), 
                timeout=5.0
            )
            
            auth_data = json.loads(auth_message)
            
            if auth_data.get("type") != "auth":
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "First message must be authentication"
                }))
                return False
            
            # Verify token
            token = auth_data.get("token")
            if not token:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "No token provided"
                }))
                return False
            
            payload = self.verify_token(token)
            if not payload:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid or expired token"
                }))
                return False
            
            # Store authenticated client
            device_id = payload["device_id"]
            self.authenticated_clients.add(device_id)
            
            # Send success response
            await websocket.send(json.dumps({
                "type": "auth_success",
                "device_id": device_id,
                "message": "Authentication successful"
            }))
            
            # Update last seen
            if device_id in self.client_tokens:
                self.client_tokens[device_id]["last_seen"] = datetime.utcnow()
            
            return True
            
        except asyncio.TimeoutError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Authentication timeout"
            }))
            return False
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Authentication failed"
            }))
            return False
    
    def encrypt_message(self, message: Dict[str, Any]) -> str:
        """Encrypt message for transmission"""
        
        json_message = json.dumps(message)
        encrypted = self.fernet.encrypt(json_message.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_message(self, encrypted_message: str) -> Dict[str, Any]:
        """Decrypt received message"""
        
        try:
            encrypted = base64.b64decode(encrypted_message.encode())
            decrypted = self.fernet.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def remove_client(self, device_id: str):
        """Remove client from authenticated set"""
        
        self.authenticated_clients.discard(device_id)
    
    def is_authenticated(self, device_id: str) -> bool:
        """Check if device is authenticated"""
        
        return device_id in self.authenticated_clients
    
    def get_connected_devices(self) -> List[Dict[str, Any]]:
        """Get list of connected devices"""
        
        devices = []
        for device_id in self.authenticated_clients:
            if device_id in self.client_tokens:
                device_info = self.client_tokens[device_id]["payload"]
                device_info["last_seen"] = self.client_tokens[device_id]["last_seen"].isoformat()
                devices.append(device_info)
        
        return devices
    
    async def create_secure_server(self, handler, host: str, port: int):
        """Create secure WebSocket server"""
        
        async def secure_handler(websocket, path):
            # Authenticate first
            if await self.authenticate_websocket(websocket, path):
                # Get device ID from authentication
                device_id = None
                for did in self.authenticated_clients:
                    # In real implementation, we'd track websocket->device_id mapping
                    device_id = did
                    break
                
                try:
                    await handler(websocket, path, device_id)
                finally:
                    # Clean up on disconnect
                    if device_id:
                        self.remove_client(device_id)
            else:
                await websocket.close()
        
        # Start server with or without SSL
        if self.ssl_context:
            return await websockets.serve(
                secure_handler, 
                host, 
                port,
                ssl=self.ssl_context
            )
        else:
            return await websockets.serve(
                secure_handler, 
                host, 
                port
            )


# Enhanced secure WebSocket handler
class SecureWebSocketHandler:
    """Handler for secure WebSocket connections"""
    
    def __init__(self, security: WebSocketSecurity):
        self.security = security
        self.connections: Dict[str, Any] = {}
    
    async def handle_connection(self, websocket, path, device_id: str):
        """Handle authenticated WebSocket connection"""
        
        # Store connection
        self.connections[device_id] = {
            "websocket": websocket,
            "connected_at": datetime.utcnow()
        }
        
        try:
            async for message in websocket:
                # Decrypt message
                try:
                    if message.startswith("{"):
                        # Unencrypted message (for compatibility)
                        data = json.loads(message)
                    else:
                        # Encrypted message
                        data = self.security.decrypt_message(message)
                    
                    # Process message
                    response = await self.process_message(data, device_id)
                    
                    # Encrypt and send response
                    if response:
                        encrypted_response = self.security.encrypt_message(response)
                        await websocket.send(encrypted_response)
                        
                except Exception as e:
                    logger.error(f"Message processing error: {e}")
                    error_response = {
                        "type": "error",
                        "message": str(e)
                    }
                    await websocket.send(json.dumps(error_response))
                    
        finally:
            # Remove connection
            del self.connections[device_id]
    
    async def process_message(self, data: Dict[str, Any], device_id: str) -> Optional[Dict[str, Any]]:
        """Process incoming message"""
        
        message_type = data.get("type")
        
        if message_type == "ping":
            return {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
        
        elif message_type == "sync":
            # Handle state synchronization
            return {
                "type": "sync_ack",
                "device_id": device_id,
                "state": "synchronized"
            }
        
        # Add more message handlers as needed
        
        return None
    
    async def broadcast_to_devices(self, message: Dict[str, Any], exclude_device: Optional[str] = None):
        """Broadcast message to all connected devices"""
        
        for device_id, connection in self.connections.items():
            if device_id != exclude_device:
                try:
                    encrypted = self.security.encrypt_message(message)
                    await connection["websocket"].send(encrypted)
                except Exception as e:
                    logger.error(f"Broadcast error to {device_id}: {e}")


# Create singleton instances
websocket_security = WebSocketSecurity()