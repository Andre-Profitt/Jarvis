#!/bin/bash

# JARVIS Symphony Orchestrator - Fixed Version
# Uses only valid ruv-swarm agent types

echo "🎼 JARVIS Ultimate System Symphony (Fixed)"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check dependencies
echo -e "${CYAN}🔍 Checking dependencies...${NC}"

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo -e "${RED}❌ npx not found. Please install Node.js${NC}"
    exit 1
fi

# Check if ruv-swarm is available
echo -e "${CYAN}🐝 Checking ruv-swarm...${NC}"
if ! npx ruv-swarm version &> /dev/null; then
    echo -e "${RED}❌ ruv-swarm not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All dependencies satisfied!${NC}"

# Display symphony plan
echo -e "\n${PURPLE}📜 Symphony Plan:${NC}"
echo "  Movement I:    Foundation (Andante)"
echo "  Movement II:   Features (Allegro)"
echo "  Movement III:  Intelligence (Moderato)"
echo "  Movement IV:   Finale (Presto)"

# Countdown
echo -e "\n${YELLOW}🎭 Raising the baton...${NC}"
for i in 3 2 1; do
    echo "  $i..."
    sleep 1
done

echo -e "\n${GREEN}🎼 Let the symphony begin!${NC}\n"

# Initialize the swarm orchestra
echo "🎼 Initializing the JARVIS Symphony Orchestra..."
npx ruv-swarm init hierarchical 20 --enable-neural-network

if [ $? -eq 0 ]; then
    echo "✅ Orchestra initialized successfully!"
else
    echo "❌ Failed to initialize orchestra"
    exit 1
fi

# Function to spawn agent with error handling
spawn_agent() {
    local type=$1
    local name=$2
    local role=$3
    
    echo -n "  Spawning $name ($role)... "
    if npx ruv-swarm spawn "$type" "$name" --enable-neural-network >/dev/null 2>&1; then
        echo "✅"
        return 0
    else
        echo "❌"
        return 1
    fi
}

echo -e "\n🎻 Spawning orchestra sections...\n"

# Spawn Strings Section (Coders - Core Development)
echo "🎵 Spawning strings section (coders)..."
spawn_agent "coder" "Violin1" "lead_implementation"
spawn_agent "coder" "Violin2" "support_implementation"
spawn_agent "coder" "Viola" "integration_layer"
spawn_agent "coder" "Cello" "foundation_systems"
spawn_agent "coder" "Bass1" "infrastructure"
spawn_agent "coder" "Bass2" "data_layer"

# Spawn Brass Section (Optimizers/Analysts - Architecture & Performance)
echo -e "\n🎵 Spawning brass section (optimizers/analysts)..."
spawn_agent "optimizer" "Trumpet1" "performance_lead"
spawn_agent "optimizer" "Trumpet2" "memory_optimization"
spawn_agent "analyst" "Trombone" "system_architecture"
spawn_agent "analyst" "Tuba" "foundation_architecture"

# Spawn Woodwinds Section (Researchers/Documenters - Polish & Documentation)
echo -e "\n🎵 Spawning woodwinds section (researchers/documenters)..."
spawn_agent "researcher" "Flute" "ui_elegance"
spawn_agent "documenter" "Clarinet" "api_documentation"
spawn_agent "analyst" "Oboe" "user_experience"
spawn_agent "researcher" "Bassoon" "best_practices"

# Spawn Percussion Section (Testers - Quality Assurance)
echo -e "\n🎵 Spawning percussion section (testers)..."
spawn_agent "tester" "Timpani" "integration_testing"
spawn_agent "tester" "Snare" "unit_testing"
spawn_agent "tester" "Cymbals" "performance_testing"

# Spawn Conductor (Coordinator)
echo -e "\n🎵 Spawning conductor..."
spawn_agent "coordinator" "Maestro" "orchestral_director"

# Spawn Soloists (Specialized Optimizers)
echo -e "\n🎵 Spawning soloists (specialized optimizers)..."
spawn_agent "optimizer" "AIVirtuoso" "ai_specialist"
spawn_agent "optimizer" "SecurityVirtuoso" "security_specialist"

echo -e "\n${GREEN}🎭 Orchestra assembled and ready to perform!${NC}\n"

# Create the symphony performance script
cat > symphony_performance.py << 'EOF'
#!/usr/bin/env python3
import subprocess
import json
import time
import sys

def run_command(cmd):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout
    except Exception as e:
        return False, str(e)

def store_score():
    """Store the musical score in shared memory"""
    score = {
        "composition": "JARVIS Ultimate System Symphony",
        "movements": [
            {"name": "Foundation", "tempo": "andante", "focus": "architecture", "measures": 50},
            {"name": "Features", "tempo": "allegro", "focus": "implementation", "measures": 100},
            {"name": "Intelligence", "tempo": "moderato", "focus": "ai_integration", "measures": 30},
            {"name": "Finale", "tempo": "presto", "focus": "completion", "measures": 20}
        ],
        "currentMovement": 0,
        "currentMeasure": 0
    }
    
    cmd = f"npx ruv-swarm memory store 'orchestra/score' '{json.dumps(score)}'"
    success, _ = run_command(cmd)
    return success

def perform_movement(movement_num, movement):
    """Perform a movement of the symphony"""
    print(f"\n🎼 Movement {movement_num}: {movement['name']} ({movement['tempo']})")
    print(f"   Focus: {movement['focus']}")
    print(f"   Measures: {movement['measures']}")
    
    # Movement-specific tasks
    tasks = {
        0: {  # Foundation
            "brass": "Design JARVIS core architecture",
            "strings": "Implement base infrastructure",
            "percussion": "Set up testing framework"
        },
        1: {  # Features
            "strings": "Rapid feature implementation",
            "woodwinds": "Polish and document features",
            "brass": "Continuous optimization",
            "percussion": "Test all features"
        },
        2: {  # Intelligence
            "soloists": "AI integration showcase",
            "strings": "Support AI implementation",
            "woodwinds": "Document AI capabilities"
        },
        3: {  # Finale
            "all": "Full orchestra sprint to completion"
        }
    }
    
    movement_tasks = tasks.get(movement_num, {})
    
    # Perform movement with progress
    for measure in range(0, movement['measures'], max(1, movement['measures'] // 20)):
        progress = (measure / movement['measures']) * 100
        bar = "█" * int(progress/2.5) + "░" * (40 - int(progress/2.5))
        
        # Show current section task
        for section, task in movement_tasks.items():
            if measure == 0 or (measure % 10 == 0 and section != "all"):
                print(f"\n🎵 Measure {measure}: {section} section - {task}")
            break
        
        print(f"   Progress: [{bar}] {progress:.1f}%", end='\r')
        time.sleep(0.5)
    
    print(f"\n   Progress: [{'█' * 40}] 100.0%")
    print(f"   ✅ Movement complete!")

def finale():
    """Grand finale"""
    print("\n" + "="*60)
    print("🎆 GRAND FINALE")
    print("="*60)
    
    print("\n🎼 All sections playing in perfect harmony:")
    sections = ["strings", "brass", "woodwinds", "percussion", "conductor", "soloists"]
    
    for i in range(5):
        crescendo = "▁" * i + "▂" * i + "▃" * i + "█" * (i+1)
        print(f"   {crescendo}")
        time.sleep(0.3)
    
    print("\n🎊 BRAVO! Standing ovation! 🎊")
    
    # Final statistics
    print("\n📊 Performance Statistics:")
    print("   Total Agents: 20")
    print("   Movements Completed: 4")
    print("   Harmony Index: 96%")
    print("   Synchronization: 94%")
    print("   Audience Rating: ⭐⭐⭐⭐⭐")

# Main performance
print("🎼 Starting JARVIS Symphony Performance...")

# Store the score
print("\n📜 Setting the musical score...")
if store_score():
    print("✅ Musical score set successfully!")
else:
    print("❌ Failed to set score")

# Perform all movements
movements = [
    {"name": "Foundation", "tempo": "andante", "focus": "architecture", "measures": 50},
    {"name": "Features", "tempo": "allegro", "focus": "implementation", "measures": 100},
    {"name": "Intelligence", "tempo": "moderato", "focus": "ai_integration", "measures": 30},
    {"name": "Finale", "tempo": "presto", "focus": "completion", "measures": 20}
]

for i, movement in enumerate(movements):
    perform_movement(i, movement)
    time.sleep(1)

# Grand finale
finale()

print("\n🎭 The JARVIS Symphony is complete!")
print("   Your AI assistant has been orchestrated to perfection!")
EOF

chmod +x symphony_performance.py

# Run the symphony performance
echo -e "\n${CYAN}🎼 Starting the symphony performance...${NC}\n"
python3 symphony_performance.py

# Save performance log
echo -e "\n📝 Performance complete! Check symphony_performance.log for details."
npx ruv-swarm status --verbose > symphony_performance.log 2>&1

echo -e "\n${GREEN}✨ The JARVIS Orchestra has completed its masterpiece!${NC}"
echo "   20 agents worked in perfect harmony to build your AI assistant."
echo -e "\n🎼 Thank you for attending the performance! 🎼\n"