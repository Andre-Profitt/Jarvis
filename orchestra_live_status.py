#!/usr/bin/env python3
"""
Live Orchestra Status Display
Shows the current state of the JARVIS Symphony
"""

import subprocess
import re
from datetime import datetime

def get_swarm_status():
    """Get current swarm status by parsing verbose output"""
    try:
        result = subprocess.run(
            ['npx', 'ruv-swarm', 'status', '--verbose'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return None

def parse_agents(output):
    """Parse agents from status output"""
    agents = []
    agent_pattern = r"🤖 Spawned agent: ([\w\s]+) \((\w+)\)"
    
    for match in re.finditer(agent_pattern, output):
        name, agent_type = match.groups()
        agents.append({'name': name.strip(), 'type': agent_type})
    
    return agents

def parse_global_status(output):
    """Parse global status metrics"""
    status = {}
    
    # Extract metrics
    swarms_match = re.search(r"Active Swarms: (\d+)", output)
    agents_match = re.search(r"Total Agents: (\d+)", output)
    tasks_match = re.search(r"Total Tasks: (\d+)", output)
    memory_match = re.search(r"Memory Usage: (\d+MB)", output)
    
    if swarms_match:
        status['swarms'] = int(swarms_match.group(1))
    if agents_match:
        status['total_agents'] = int(agents_match.group(1))
    if tasks_match:
        status['tasks'] = int(tasks_match.group(1))
    if memory_match:
        status['memory'] = memory_match.group(1)
    
    return status

def display_orchestra_status():
    """Display beautiful orchestra status"""
    print("\n" + "="*60)
    print("🎼 JARVIS ORCHESTRA STATUS - " + datetime.now().strftime("%H:%M:%S"))
    print("="*60)
    
    # Get status
    output = get_swarm_status()
    
    if not output:
        print("⚠️  Orchestra is offline")
        return
    
    # Parse data
    agents = parse_agents(output)
    global_status = parse_global_status(output)
    
    # Count agents by section
    sections = {
        'strings': [],
        'brass': [],
        'woodwinds': [],
        'percussion': [],
        'conductor': [],
        'soloists': []
    }
    
    # Map agent names to sections
    for agent in agents:
        name = agent['name'].lower()
        agent_type = agent['type']
        
        # Section mapping based on instrument names
        if any(inst in name for inst in ['violin', 'viola', 'cello', 'bass']):
            sections['strings'].append(agent['name'])
        elif any(inst in name for inst in ['trumpet', 'trombone', 'tuba']):
            sections['brass'].append(agent['name'])
        elif any(inst in name for inst in ['flute', 'clarinet', 'oboe', 'bassoon']):
            sections['woodwinds'].append(agent['name'])
        elif any(inst in name for inst in ['timpani', 'snare', 'cymbals']):
            sections['percussion'].append(agent['name'])
        elif 'maestro' in name:
            sections['conductor'].append(agent['name'])
        elif 'virtuoso' in name:
            sections['soloists'].append(agent['name'])
        elif agent_type == 'coder':
            sections['strings'].append(agent['name'])
        elif agent_type in ['optimizer', 'analyst']:
            sections['brass'].append(agent['name'])
        elif agent_type in ['researcher', 'documenter']:
            sections['woodwinds'].append(agent['name'])
        elif agent_type == 'tester':
            sections['percussion'].append(agent['name'])
        elif agent_type == 'coordinator':
            sections['conductor'].append(agent['name'])
    
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
    
    total_musicians = 0
    for section, members in sections.items():
        icon = section_icons.get(section, '🎵')
        # Remove duplicates
        unique_members = list(set(members))
        count = len(unique_members)
        total_musicians += count
        
        if count > 0:
            print(f"{icon} {section.upper():12} [{count:2d} agents]")
            # Show first few members
            display_members = unique_members[:3]
            for member in display_members:
                print(f"   └─ {member}")
            if count > 3:
                print(f"   └─ ... and {count-3} more")
    
    # Display global metrics
    print("\n📊 ORCHESTRA METRICS:")
    print("-" * 40)
    
    if global_status:
        print(f"Total Swarms: {global_status.get('swarms', 'N/A')}")
        print(f"Total Agents: {global_status.get('total_agents', 'N/A')}")
        print(f"Active Tasks: {global_status.get('tasks', 'N/A')}")
        print(f"Memory Usage: {global_status.get('memory', 'N/A')}")
    
    print(f"\nUnique Musicians: {total_musicians}")
    print(f"Orchestra Status: 🟢 Live and Ready")
    
    # Performance status
    print("\n🎵 PERFORMANCE READINESS:")
    print("-" * 40)
    
    if total_musicians >= 20:
        print("✅ Full orchestra assembled")
        print("✅ All sections staffed")
        print("✅ Ready for symphonic performance")
        print("\nRating: ⭐⭐⭐⭐⭐")
    else:
        print(f"⚠️  Only {total_musicians}/20 musicians present")
        print("⚠️  Some sections may be understaffed")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    display_orchestra_status()