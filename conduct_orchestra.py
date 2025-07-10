#!/usr/bin/env python3
"""
Orchestra Conductor - Assign tasks to your JARVIS Symphony
"""

import subprocess
import json
import sys
import time

def assign_task(agent_name, task_description):
    """Assign a task to a specific agent"""
    cmd = f"npx ruv-swarm task assign '{agent_name}' '{task_description}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

def broadcast_task(task_description):
    """Broadcast a task to all agents"""
    cmd = f"npx ruv-swarm task broadcast '{task_description}'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

def orchestrate_movement(movement_name, tasks):
    """Orchestrate a complete movement with section-specific tasks"""
    print(f"\n🎼 Starting Movement: {movement_name}")
    print("="*50)
    
    for section, task in tasks.items():
        print(f"\n🎵 {section} Section:")
        print(f"   Task: {task}")
        
        # Assign tasks based on section
        if section == "strings":
            agents = ["Violin1", "Violin2", "Viola", "Cello", "Bass1", "Bass2"]
        elif section == "brass":
            agents = ["Trumpet1", "Trumpet2", "Trombone", "Tuba"]
        elif section == "woodwinds":
            agents = ["Flute", "Clarinet", "Oboe", "Bassoon"]
        elif section == "percussion":
            agents = ["Timpani", "Snare", "Cymbals"]
        elif section == "conductor":
            agents = ["Maestro"]
        elif section == "soloists":
            agents = ["AIVirtuoso", "SecurityVirtuoso"]
        else:
            agents = []
        
        # Assign task to each agent in section
        for agent in agents:
            if assign_task(agent, task):
                print(f"   ✅ {agent} - Task assigned")
            else:
                print(f"   ⚠️  {agent} - Failed to assign")
        
        time.sleep(1)

def main():
    print("🎭 JARVIS Orchestra Conductor")
    print("="*50)
    print("\nOptions:")
    print("1. Quick Task - Assign a single task")
    print("2. Full Symphony - Run complete JARVIS build")
    print("3. Custom Movement - Design your own movement")
    print("4. Section Practice - Task for specific section")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == "1":
        # Quick task
        task = input("\nEnter task description: ").strip()
        agent = input("Enter agent name (or 'all' for broadcast): ").strip()
        
        if agent.lower() == 'all':
            if broadcast_task(task):
                print("✅ Task broadcast to all agents!")
            else:
                print("❌ Failed to broadcast task")
        else:
            if assign_task(agent, task):
                print(f"✅ Task assigned to {agent}!")
            else:
                print(f"❌ Failed to assign task to {agent}")
    
    elif choice == "2":
        # Full symphony
        print("\n🎼 Starting JARVIS Symphony...")
        
        movements = {
            "Movement I - Foundation": {
                "brass": "Design JARVIS core architecture and system components",
                "strings": "Set up project structure and base infrastructure",
                "percussion": "Create testing framework and CI/CD pipeline"
            },
            "Movement II - Features": {
                "strings": "Implement voice recognition, NLP, and core features",
                "woodwinds": "Create API documentation and user guides",
                "brass": "Optimize performance and memory usage",
                "percussion": "Test all features thoroughly"
            },
            "Movement III - Intelligence": {
                "soloists": "Integrate advanced AI capabilities and learning systems",
                "strings": "Support AI implementation with helper functions",
                "woodwinds": "Document AI features and capabilities"
            },
            "Movement IV - Finale": {
                "conductor": "Coordinate final integration and deployment",
                "strings": "Final bug fixes and polish",
                "brass": "Performance optimization sweep",
                "woodwinds": "Complete all documentation",
                "percussion": "Full system testing and validation"
            }
        }
        
        for movement, tasks in movements.items():
            orchestrate_movement(movement, tasks)
            input("\nPress Enter to continue to next movement...")
    
    elif choice == "3":
        # Custom movement
        movement_name = input("\nEnter movement name: ").strip()
        
        tasks = {}
        print("\nEnter tasks for each section (leave blank to skip):")
        
        for section in ["strings", "brass", "woodwinds", "percussion", "conductor", "soloists"]:
            task = input(f"{section.capitalize()} task: ").strip()
            if task:
                tasks[section] = task
        
        if tasks:
            orchestrate_movement(movement_name, tasks)
        else:
            print("No tasks defined!")
    
    elif choice == "4":
        # Section practice
        print("\nSections:")
        print("1. Strings (Coders)")
        print("2. Brass (Optimizers/Analysts)")
        print("3. Woodwinds (Researchers/Documenters)")
        print("4. Percussion (Testers)")
        print("5. Conductor (Coordinator)")
        print("6. Soloists (Specialists)")
        
        section_choice = input("\nSelect section (1-6): ").strip()
        task = input("Enter task for section: ").strip()
        
        section_map = {
            "1": ("strings", ["Violin1", "Violin2", "Viola", "Cello", "Bass1", "Bass2"]),
            "2": ("brass", ["Trumpet1", "Trumpet2", "Trombone", "Tuba"]),
            "3": ("woodwinds", ["Flute", "Clarinet", "Oboe", "Bassoon"]),
            "4": ("percussion", ["Timpani", "Snare", "Cymbals"]),
            "5": ("conductor", ["Maestro"]),
            "6": ("soloists", ["AIVirtuoso", "SecurityVirtuoso"])
        }
        
        if section_choice in section_map:
            section_name, agents = section_map[section_choice]
            print(f"\n🎵 Assigning to {section_name} section...")
            
            for agent in agents:
                if assign_task(agent, task):
                    print(f"✅ {agent} - Task assigned")
                else:
                    print(f"⚠️  {agent} - Not found")
    
    print("\n🎭 Conductor session complete!")

if __name__ == "__main__":
    main()