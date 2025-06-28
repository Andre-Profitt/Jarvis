"""
Google Cloud Storage Integration Module for JARVIS
Provides unified cloud storage interface for all components
"""

import os
import json
import pickle
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib

try:
    from google.cloud import storage
    from google.cloud.exceptions import NotFound
    from google.api_core import retry
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None
    retry = None

logger = logging.getLogger(__name__)

# Create a dummy retry decorator if google.cloud is not available
if retry is None:
    class DummyRetry:
        def Retry(self):
            def decorator(func):
                return func
            return decorator
    retry = DummyRetry()


class GCSStorageManager:
    """Unified Google Cloud Storage manager for JARVIS ecosystem"""
    
    def __init__(
        self,
        bucket_name: str = None,
        credentials_path: str = None,
        project_id: str = None,
        enable_versioning: bool = True,
        enable_lifecycle: bool = True
    ):
        """
        Initialize GCS Storage Manager
        
        Args:
            bucket_name: GCS bucket name (defaults to env var GCS_BUCKET)
            credentials_path: Path to service account JSON (defaults to env var GCS_CREDENTIALS_PATH)
            project_id: GCP project ID (defaults to env var GCP_PROJECT_ID)
            enable_versioning: Enable object versioning
            enable_lifecycle: Enable lifecycle management
        """
        self.bucket_name = bucket_name or os.getenv('GCS_BUCKET', 'jarvis-storage')
        self.credentials_path = credentials_path or os.getenv('GCS_CREDENTIALS_PATH')
        self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
        self.enable_versioning = enable_versioning
        self.enable_lifecycle = enable_lifecycle
        
        self._client = None
        self._bucket = None
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        # Initialize if credentials are available
        if self.credentials_path and os.path.exists(self.credentials_path):
            self._initialize_client()
    
    def _initialize_client(self):
        """Initialize GCS client and bucket"""
        if not GCS_AVAILABLE:
            logger.warning("Google Cloud Storage library not installed. Using local fallback.")
            return
        
        try:
            # Set credentials
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
            
            # Initialize client
            self._client = storage.Client(project=self.project_id)
            
            # Get or create bucket
            try:
                self._bucket = self._client.get_bucket(self.bucket_name)
                logger.info(f"Connected to GCS bucket: {self.bucket_name}")
            except NotFound:
                self._bucket = self._client.create_bucket(
                    self.bucket_name,
                    location="US"
                )
                logger.info(f"Created new GCS bucket: {self.bucket_name}")
            
            # Configure bucket
            self._configure_bucket()
            
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            self._client = None
            self._bucket = None
    
    def _configure_bucket(self):
        """Configure bucket with versioning and lifecycle rules"""
        if not self._bucket:
            return
        
        # Enable versioning
        if self.enable_versioning:
            self._bucket.versioning_enabled = True
            self._bucket.patch()
            logger.info("Enabled versioning on bucket")
        
        # Set lifecycle rules
        if self.enable_lifecycle:
            rules = [
                # Delete old versions after 30 days
                {
                    "action": {"type": "Delete"},
                    "condition": {
                        "age": 30,
                        "isLive": False
                    }
                },
                # Move to nearline after 7 days
                {
                    "action": {
                        "type": "SetStorageClass",
                        "storageClass": "NEARLINE"
                    },
                    "condition": {
                        "age": 7,
                        "matchesStorageClass": ["STANDARD"]
                    }
                },
                # Move to coldline after 30 days
                {
                    "action": {
                        "type": "SetStorageClass",
                        "storageClass": "COLDLINE"
                    },
                    "condition": {
                        "age": 30,
                        "matchesStorageClass": ["NEARLINE"]
                    }
                }
            ]
            self._bucket.lifecycle_rules = rules
            self._bucket.patch()
            logger.info("Configured lifecycle rules on bucket")
    
    @property
    def is_available(self) -> bool:
        """Check if GCS is available and configured"""
        return self._client is not None and self._bucket is not None
    
    def _get_local_fallback_path(self, blob_name: str) -> Path:
        """Get local fallback path when GCS is not available"""
        base_path = Path("./local_storage") / self.bucket_name
        file_path = base_path / blob_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        return file_path
    
    @retry.Retry()
    def upload(
        self,
        source: Union[str, bytes, BinaryIO],
        destination: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload file or data to GCS
        
        Args:
            source: File path, bytes, or file-like object
            destination: Destination path in GCS
            metadata: Optional metadata dict
            content_type: Optional content type
            
        Returns:
            GCS URI (gs://bucket/path)
        """
        if not self.is_available:
            # Fallback to local storage
            local_path = self._get_local_fallback_path(destination)
            
            if isinstance(source, str):
                # Copy file
                import shutil
                shutil.copy2(source, local_path)
            elif isinstance(source, bytes):
                # Write bytes
                local_path.write_bytes(source)
            else:
                # Write from file-like object
                local_path.write_bytes(source.read())
            
            logger.warning(f"GCS not available. Saved to local: {local_path}")
            return f"file://{local_path.absolute()}"
        
        # Upload to GCS
        blob = self._bucket.blob(destination)
        
        # Set metadata
        if metadata:
            blob.metadata = metadata
        if content_type:
            blob.content_type = content_type
        
        # Upload based on source type
        if isinstance(source, str):
            blob.upload_from_filename(source)
        elif isinstance(source, bytes):
            blob.upload_from_string(source)
        else:
            blob.upload_from_file(source)
        
        logger.info(f"Uploaded to GCS: gs://{self.bucket_name}/{destination}")
        return f"gs://{self.bucket_name}/{destination}"
    
    @retry.Retry()
    def download(
        self,
        source: str,
        destination: Optional[Union[str, BinaryIO]] = None
    ) -> Union[bytes, str]:
        """
        Download file from GCS
        
        Args:
            source: Source path in GCS
            destination: Optional local file path or file-like object
            
        Returns:
            Bytes if no destination specified, else destination path
        """
        if not self.is_available:
            # Fallback to local storage
            local_path = self._get_local_fallback_path(source)
            if not local_path.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            if destination is None:
                return local_path.read_bytes()
            elif isinstance(destination, str):
                import shutil
                shutil.copy2(local_path, destination)
                return destination
            else:
                destination.write(local_path.read_bytes())
                return destination
        
        # Download from GCS
        blob = self._bucket.blob(source)
        
        if not blob.exists():
            raise FileNotFoundError(f"Blob not found: gs://{self.bucket_name}/{source}")
        
        if destination is None:
            return blob.download_as_bytes()
        elif isinstance(destination, str):
            blob.download_to_filename(destination)
            return destination
        else:
            blob.download_to_file(destination)
            return destination
    
    def exists(self, path: str) -> bool:
        """Check if object exists in GCS"""
        if not self.is_available:
            local_path = self._get_local_fallback_path(path)
            return local_path.exists()
        
        blob = self._bucket.blob(path)
        return blob.exists()
    
    def delete(self, path: str) -> bool:
        """Delete object from GCS"""
        if not self.is_available:
            local_path = self._get_local_fallback_path(path)
            if local_path.exists():
                local_path.unlink()
                return True
            return False
        
        blob = self._bucket.blob(path)
        if blob.exists():
            blob.delete()
            return True
        return False
    
    def list(self, prefix: str = "", delimiter: Optional[str] = None) -> List[str]:
        """List objects in GCS with optional prefix"""
        if not self.is_available:
            base_path = Path("./local_storage") / self.bucket_name
            if not base_path.exists():
                return []
            
            pattern = f"{prefix}*" if prefix else "*"
            files = list(base_path.glob(pattern))
            return [str(f.relative_to(base_path)) for f in files if f.is_file()]
        
        blobs = self._bucket.list_blobs(prefix=prefix, delimiter=delimiter)
        return [blob.name for blob in blobs]
    
    # Convenience methods for JARVIS components
    
    def save_model(
        self,
        model_data: Any,
        model_name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save ML model to GCS with versioning"""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        path = f"models/{model_name}/v_{version}/model.pkl"
        
        # Serialize model
        model_bytes = pickle.dumps(model_data)
        
        # Create metadata
        meta = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.now().isoformat(),
            "size_bytes": str(len(model_bytes)),
            "checksum": hashlib.md5(model_bytes).hexdigest()
        }
        if metadata:
            meta.update({k: str(v) for k, v in metadata.items()})
        
        # Upload model
        uri = self.upload(model_bytes, path, metadata=meta, content_type="application/octet-stream")
        
        # Save metadata separately
        meta_path = f"models/{model_name}/v_{version}/metadata.json"
        self.upload(
            json.dumps(meta, indent=2).encode(),
            meta_path,
            content_type="application/json"
        )
        
        return uri
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load ML model from GCS"""
        if version is None:
            # Get latest version
            versions = self.list_model_versions(model_name)
            if not versions:
                raise FileNotFoundError(f"No versions found for model: {model_name}")
            version = versions[-1]  # Latest version
        
        path = f"models/{model_name}/v_{version}/model.pkl"
        model_bytes = self.download(path)
        return pickle.loads(model_bytes)
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """List all versions of a model"""
        prefix = f"models/{model_name}/"
        blobs = self.list(prefix=prefix, delimiter="/")
        
        versions = []
        for blob in blobs:
            parts = blob.split("/")
            if len(parts) >= 3 and parts[2].startswith("v_"):
                version = parts[2][2:]  # Remove 'v_' prefix
                versions.append(version)
        
        return sorted(set(versions))
    
    def save_memory(self, memory_data: Any, memory_type: str, agent_id: str) -> str:
        """Save agent memory to GCS"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"memories/{agent_id}/{memory_type}/{timestamp}.pkl"
        
        memory_bytes = pickle.dumps(memory_data)
        metadata = {
            "agent_id": agent_id,
            "memory_type": memory_type,
            "timestamp": timestamp,
            "size_bytes": str(len(memory_bytes))
        }
        
        return self.upload(memory_bytes, path, metadata=metadata)
    
    def load_latest_memory(self, memory_type: str, agent_id: str) -> Optional[Any]:
        """Load latest memory for an agent"""
        prefix = f"memories/{agent_id}/{memory_type}/"
        memories = self.list(prefix=prefix)
        
        if not memories:
            return None
        
        # Get latest by timestamp in filename
        latest = sorted(memories)[-1]
        memory_bytes = self.download(latest)
        return pickle.loads(memory_bytes)
    
    async def upload_async(self, source: Union[str, bytes], destination: str, **kwargs) -> str:
        """Async wrapper for upload"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.upload(source, destination, **kwargs)
        )
    
    async def download_async(self, source: str, destination: Optional[str] = None) -> Union[bytes, str]:
        """Async wrapper for download"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.download(source, destination)
        )


# Global instance
_gcs_manager = None


def get_gcs_manager() -> GCSStorageManager:
    """Get or create global GCS manager instance"""
    global _gcs_manager
    if _gcs_manager is None:
        _gcs_manager = GCSStorageManager()
    return _gcs_manager


# Convenience functions
def upload_to_gcs(source: Union[str, bytes], destination: str, **kwargs) -> str:
    """Upload to GCS using global manager"""
    return get_gcs_manager().upload(source, destination, **kwargs)


def download_from_gcs(source: str, destination: Optional[str] = None) -> Union[bytes, str]:
    """Download from GCS using global manager"""
    return get_gcs_manager().download(source, destination)


def save_model_to_gcs(model_data: Any, model_name: str, **kwargs) -> str:
    """Save model to GCS using global manager"""
    return get_gcs_manager().save_model(model_data, model_name, **kwargs)


def load_model_from_gcs(model_name: str, version: Optional[str] = None) -> Any:
    """Load model from GCS using global manager"""
    return get_gcs_manager().load_model(model_name, version)