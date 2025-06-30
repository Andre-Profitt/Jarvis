#!/usr/bin/env python3
"""
Test JARVIS minimal deployment
"""
import redis
import json
from datetime import datetime

def test_jarvis():
    """Test if JARVIS is running"""
    print("🔍 Testing JARVIS deployment...\n")
    
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        
        # Check Redis connection
        r.ping()
        print("✅ Redis connection successful")
        
        # Check JARVIS status
        status = r.get("jarvis:status")
        if status:
            print(f"✅ JARVIS status: {status}")
        else:
            print("⚠️  JARVIS status not found in Redis")
        
        # Check start time
        start_time = r.get("jarvis:start_time")
        if start_time:
            print(f"✅ JARVIS started at: {start_time}")
        else:
            print("⚠️  JARVIS start time not found")
        
        # Check version
        version = r.get("jarvis:version")
        if version:
            print(f"✅ JARVIS version: {version}")
        else:
            print("⚠️  JARVIS version not found")
        
        # Check heartbeat
        heartbeat = r.get("jarvis:heartbeat")
        if heartbeat:
            hb_time = datetime.fromisoformat(heartbeat)
            time_diff = (datetime.now() - hb_time).total_seconds()
            if time_diff < 30:
                print(f"✅ JARVIS heartbeat: Active ({time_diff:.1f}s ago)")
            else:
                print(f"⚠️  JARVIS heartbeat: Stale ({time_diff:.1f}s ago)")
        else:
            print("⚠️  JARVIS heartbeat not found")
        
        print("\n✨ JARVIS deployment test complete!")
        
        if status == "online" and heartbeat:
            print("🎉 JARVIS is RUNNING SUCCESSFULLY!")
            return True
        else:
            print("⚠️  JARVIS may not be fully operational")
            return False
            
    except redis.ConnectionError:
        print("❌ Could not connect to Redis")
        print("   Make sure Redis is running: redis-server")
        return False
    except Exception as e:
        print(f"❌ Error testing JARVIS: {e}")
        return False

if __name__ == "__main__":
    test_jarvis()
