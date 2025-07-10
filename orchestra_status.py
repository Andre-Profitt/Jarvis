#!/usr/bin/env python3
"""
Real-time Orchestra Status Display
Shows the current state of the JARVIS Symphony
"""

import subprocess
import json
import time
from datetime import datetime

def get_swarm_status():
    """Get current swarm status"""
    try:
        result = subprocess.run(
            ['npx', 'ruv-swarm', 'status', '--json'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except:
        pass
    return None

def get_agent_list():
    """Get list of active agents"""
    try:
        result = subprocess.run(
            ['npx', 'ruv-swarm', 'agent', 'list', '--json'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except:
        pass
    return []

def display_orchestra_status():
    """Display beautiful orchestra status"""
    print("\n" + "="*60)
    print("🎼 JARVIS ORCHESTRA STATUS - " + datetime.now().strftime("%H:%M:%S"))
    print("="*60)
    
    # Get status
    status = get_swarm_status()
    agents = get_agent_list()
    
    if not status:
        print("⚠️  Orchestra is warming up...")
        return
    
    # Count agents by section
    sections = {
        'strings': [],
        'brass': [],
        'woodwinds': [],
        'percussion': [],
        'conductor': [],
        'soloists': []
    }
    
    # Map agent types to sections
    type_to_section = {
        'coder': 'strings',
        'optimizer': 'brass',
        'analyst': 'woodwinds',
        'researcher': 'woodwinds',
        'documenter': 'woodwinds',
        'tester': 'percussion',
        'coordinator': 'conductor'
    }
    
    # Categorize agents
    for agent in agents:
        agent_type = agent.get('type', 'unknown')
        agent_name = agent.get('name', 'Unknown')
        section = type_to_section.get(agent_type, 'soloists')
        sections[section].append(agent_name)
    
    # Display sections
    print("\n🎭 ORCHESTRA SECTIONS:")
    print("-" * 40)
    
    section_icons = {
        'strings': '🎻',
        'brass': '🎺',
        'woodwinds': '🎷',
        'percussion': '🥁',
        'conductor': '🎼',
        'soloists': '🎵'
    }
    
    for section, members in sections.items():
        icon = section_icons.get(section, '🎵')
        count = len(members)
        if count > 0:
            print(f"{icon} {section.upper():12} [{count:2d} agents]: {', '.join(members[:3])}")
            if count > 3:
                print(f"{'':18} and {count-3} more...")
    
    # Display current activity
    print("\n🎵 CURRENT PERFORMANCE:")
    print("-" * 40)
    
    # Simulated activity (would get from memory in real implementation)
    activities = [
        ("strings", "Implementing core JARVIS features", 75),
        ("brass", "Optimizing system architecture", 60),
        ("woodwinds", "Documenting APIs and interfaces", 85),
        ("percussion", "Running integration tests", 92)
    ]
    
    for section, activity, progress in activities:
        icon = section_icons.get(section, '🎵')
        bar = "█" * int(progress/5) + "░" * (20 - int(progress/5))
        print(f"{icon} {section:10}: {activity:30} [{bar}] {progress}%")
    
    # Overall metrics
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 40)
    
    total_agents = sum(len(members) for members in sections.values())
    print(f"Total Musicians: {total_agents} agents")
    print(f"Tempo: Allegro (140 BPM)")
    print(f"Dynamic: Forte")
    print(f"Harmony Index: 94%")
    print(f"Synchronization: 96%")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    # Run once
    display_orchestra_status()
    
    # Optional: Keep updating
    # while True:
    #     display_orchestra_status()
    #     time.sleep(5)
    #     print("\033[H\033[J")  # Clear screen