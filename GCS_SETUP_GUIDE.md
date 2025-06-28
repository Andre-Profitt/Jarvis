# Google Cloud Storage Setup Guide for JARVIS

## Overview

JARVIS now includes comprehensive Google Cloud Storage (GCS) integration for:
- ML model storage and versioning
- Episodic memory persistence across sessions
- Distributed training support
- Automatic backups
- 30TB storage capability

## What's Been Implemented

### 1. **Core GCS Module** (`core/gcs_storage.py`)
- Unified storage interface for all JARVIS components
- Automatic fallback to local storage if GCS unavailable
- Async support for non-blocking operations
- Built-in versioning and lifecycle management
- Specialized methods for models and memories

### 2. **ML Training Integration** (`core/real_ml_training.py`)
- Automatic model checkpoint uploads to GCS
- Model versioning with metadata
- Training history preservation
- Load models from GCS or local storage
- Seamless failover between cloud and local

### 3. **Episodic Memory Integration** (`core/enhanced_episodic_memory.py`)
- Persistent memory storage in GCS
- Agent-specific memory organization
- Automatic local caching for performance
- Memory type categorization
- Distributed access across agents

### 4. **Configuration Updates**
- `.env`: Added GCS configuration variables
- `config.yaml`: Enabled cloud storage with feature flags

## Setup Instructions

### 1. Create GCS Service Account

```bash
# Using gcloud CLI
gcloud iam service-accounts create jarvis-storage \
    --display-name="JARVIS Storage Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:jarvis-storage@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create \
    ./credentials/gcs-service-account.json \
    --iam-account=jarvis-storage@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 2. Create Storage Bucket

```bash
# Create bucket with versioning
gsutil mb -p YOUR_PROJECT_ID -c STANDARD -l US gs://jarvis-storage/
gsutil versioning set on gs://jarvis-storage/
```

### 3. Update Environment Variables

Already configured in `.env`:
```bash
GCS_CREDENTIALS_PATH=./credentials/gcs-service-account.json
GCS_BUCKET=jarvis-storage
GCP_PROJECT_ID=jarvis-ai-project  # Update with your project ID
```

### 4. Install GCS Dependencies

```bash
pip install google-cloud-storage
```

## Usage Examples

### Training with GCS

```python
from core.real_ml_training import JARVISTrainer
from pathlib import Path

# Training automatically uses GCS
trainer = JARVISTrainer(
    model_dir=Path("./models"), 
    use_cloud_storage=True  # Default
)

# Train model - checkpoints auto-upload to GCS
trainer.train(epochs=10)

# Load model from GCS
trainer.load_model(version="final")  # or specific version
```

### Memory Persistence

```python
from core.enhanced_episodic_memory import EpisodicMemorySystem

# Memory system automatically uses GCS
memory_system = EpisodicMemorySystem(
    storage_backend="persistent",  # Uses GCS-enabled storage
    use_cloud_storage=True
)

# Memories auto-sync to GCS
await memory_system.store_memory(...)
```

### Direct GCS Operations

```python
from core.gcs_storage import get_gcs_manager

gcs = get_gcs_manager()

# Upload file
gcs.upload("local_file.txt", "remote/path/file.txt")

# Download file
data = gcs.download("remote/path/file.txt")

# List model versions
versions = gcs.list_model_versions("jarvis_brain")
```

## Storage Organization

```
gs://jarvis-storage/
├── models/
│   └── jarvis_brain/
│       ├── v_final/
│       │   ├── model.pkl
│       │   └── metadata.json
│       └── v_checkpoint_epoch_10/
│           ├── model.pkl
│           └── metadata.json
├── memories/
│   └── [agent_id]/
│       └── [memory_type]/
│           └── [timestamp].pkl
└── training_data/
    └── datasets/
```

## Lifecycle Policies

Automatic cost optimization:
- **7 days**: Move to Nearline storage
- **30 days**: Move to Coldline storage
- **Old versions**: Delete after 30 days

## Monitoring

Check GCS usage:
```bash
# Storage usage
gsutil du -sh gs://jarvis-storage/

# List recent uploads
gsutil ls -la gs://jarvis-storage/models/

# Check bucket metadata
gsutil stat gs://jarvis-storage/
```

## Troubleshooting

### GCS Not Available
- Check credentials file exists: `./credentials/gcs-service-account.json`
- Verify project ID in `.env`
- Ensure `google-cloud-storage` is installed
- Check service account permissions

### Fallback to Local Storage
The system automatically falls back to local storage if GCS is unavailable. Check logs for:
```
WARNING: GCS not available, falling back to local storage
```

### Performance Tips
1. Local caching reduces GCS calls
2. Async operations prevent blocking
3. Batch uploads for multiple files
4. Use versioning for model experiments

## Security Notes

- **Never commit** `gcs-service-account.json` to git
- Keep credentials in `./credentials/` (gitignored)
- Use IAM roles with minimal permissions
- Enable audit logging in GCP Console

## Cost Optimization

- Lifecycle rules automatically move old data to cheaper storage
- Local caching minimizes API calls
- Compression for large models (automatic with pickle)
- Monitor usage in GCP Console

---

Your JARVIS ecosystem is now fully integrated with Google Cloud Storage, providing unlimited scalability for models and memories! 🚀