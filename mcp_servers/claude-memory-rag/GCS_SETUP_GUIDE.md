# Google Cloud Storage Setup Guide

## Current Status
- ✅ Memory server is working with local storage
- ⚠️ Google Cloud Storage requires billing to be enabled
- 📁 Local storage: ~/.claude_simple_memory/

## To Enable Google Cloud Storage (30TB)

### Option 1: Enable Billing on Current Project
1. Visit: https://console.cloud.google.com/billing
2. Select project: gen-lang-client-0385977686
3. Add a billing account
4. Restart the memory server

### Option 2: Create New Project
1. Create new GCP project with billing
2. Enable Cloud Storage API
3. Create service account key
4. Replace ~/.gcs/jarvis-credentials.json
5. Update server configuration

### Option 3: Continue with Local Storage
No action needed - already working!

## Cost Considerations
- 30TB storage: ~$600/month (standard pricing)
- Consider using Nearline ($10/TB/month) for older data
- Or use lifecycle policies to auto-archive

## Testing GCS Connection
```bash
python3 test_gcs_connection.py
```
