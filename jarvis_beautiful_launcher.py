#!/usr/bin/env python3
"""
JARVIS Beautiful Launcher
An elegant interface for the ultimate AI assistant
"""

import sys
import os
import time
import subprocess
from pathlib import Path
import random

# Colors
class Colors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def clear_screen():
    """Clear the terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')


def print_banner():
    """Print the JARVIS banner"""
    banner = f"""
{Colors.CYAN}     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝{Colors.END}
 
{Colors.BOLD}{Colors.WHITE}The Ultimate 10/10 Seamless AI Assistant{Colors.END}
{Colors.BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}
"""
    print(banner)


def typewriter_effect(text, delay=0.03):
    """Print text with typewriter effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def show_features():
    """Display JARVIS features"""
    features = [
        ("🎙️", "Voice-First Interface", "Always listening, natural conversation"),
        ("🧠", "Anticipatory AI", "Predicts your needs before you ask"),
        ("🐝", "Swarm Intelligence", "Multiple AI agents working together"),
        ("⚡", "System Control", "Deep macOS integration"),
        ("📚", "Continuous Learning", "Gets smarter with every interaction"),
        ("🔄", "Zero Friction", "No modes, no commands, just talk")
    ]
    
    print(f"\n{Colors.BOLD}Key Features:{Colors.END}")
    for icon, title, desc in features:
        print(f"  {icon} {Colors.GREEN}{title}{Colors.END}")
        print(f"     {Colors.WHITE}{desc}{Colors.END}")
        time.sleep(0.2)


def show_examples():
    """Show usage examples"""
    examples = [
        "Hey JARVIS, open Safari",
        "What's the weather like?",
        "Remind me to call mom at 3pm",
        "Calculate 15% tip on $84.50",
        "Take a screenshot",
        "Turn up the volume",
        "Search for the best Italian restaurants nearby"
    ]
    
    print(f"\n{Colors.BOLD}Example Commands:{Colors.END}")
    for example in random.sample(examples, 4):
        print(f"  {Colors.YELLOW}•{Colors.END} \"{example}\"")
        time.sleep(0.1)


def check_setup():
    """Check if JARVIS is properly set up"""
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")
        
    # Check virtual environment
    if not Path("venv").exists():
        issues.append("Virtual environment not created")
        
    # Check main script
    if not Path("jarvis_10_seamless.py").exists():
        issues.append("Main JARVIS script not found")
        
    return issues


def main_menu():
    """Display main menu"""
    while True:
        clear_screen()
        print_banner()
        
        print(f"\n{Colors.BOLD}Welcome to JARVIS!{Colors.END}")
        print(f"{Colors.WHITE}Your personal AI assistant is ready to help.{Colors.END}\n")
        
        options = [
            ("1", "Start JARVIS", Colors.GREEN),
            ("2", "Run as Background Service", Colors.BLUE),
            ("3", "View Features", Colors.CYAN),
            ("4", "Run Setup", Colors.YELLOW),
            ("5", "Exit", Colors.RED)
        ]
        
        for key, desc, color in options:
            print(f"  {color}[{key}]{Colors.END} {desc}")
            
        print(f"\n{Colors.BOLD}Choose an option:{Colors.END} ", end='')
        
        choice = input().strip()
        
        if choice == '1':
            start_jarvis()
        elif choice == '2':
            start_service()
        elif choice == '3':
            show_features()
            show_examples()
            input(f"\n{Colors.WHITE}Press Enter to continue...{Colors.END}")
        elif choice == '4':
            run_setup()
        elif choice == '5':
            print(f"\n{Colors.GREEN}Goodbye! JARVIS will be here when you need it.{Colors.END}")
            sys.exit(0)


def start_jarvis():
    """Start JARVIS in interactive mode"""
    clear_screen()
    print_banner()
    
    print(f"\n{Colors.GREEN}Starting JARVIS...{Colors.END}")
    show_examples()
    
    print(f"\n{Colors.BOLD}Tips:{Colors.END}")
    print(f"  • Say '{Colors.YELLOW}Hey JARVIS{Colors.END}' to start")
    print(f"  • Keep talking naturally - no need to repeat the wake word")
    print(f"  • Press {Colors.RED}Ctrl+C{Colors.END} to stop\n")
    
    time.sleep(2)
    
    # Check for setup
    issues = check_setup()
    if issues:
        print(f"{Colors.RED}Setup required!{Colors.END}")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\nRun setup? (y/n): ", end='')
        if input().lower() == 'y':
            run_setup()
            
    # Start JARVIS
    try:
        if Path("venv/bin/python3").exists():
            subprocess.run(["venv/bin/python3", "jarvis_10_seamless.py"])
        else:
            subprocess.run([sys.executable, "jarvis_10_seamless.py"])
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}JARVIS stopped. Until next time!{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        input("Press Enter to return to menu...")


def start_service():
    """Start JARVIS as background service"""
    clear_screen()
    print_banner()
    
    print(f"\n{Colors.BLUE}Background Service Management{Colors.END}\n")
    
    options = [
        ("1", "Install & Start Service", "Set up JARVIS to run automatically"),
        ("2", "Check Service Status", "See if JARVIS is running"),
        ("3", "Stop Service", "Stop the background service"),
        ("4", "Back to Main Menu", "")
    ]
    
    for key, title, desc in options:
        print(f"  [{key}] {Colors.BOLD}{title}{Colors.END}")
        if desc:
            print(f"      {Colors.WHITE}{desc}{Colors.END}")
            
    choice = input(f"\n{Colors.BOLD}Choose an option:{Colors.END} ").strip()
    
    if choice == '1':
        print(f"\n{Colors.GREEN}Installing JARVIS service...{Colors.END}")
        subprocess.run(["python3", "jarvis_background_service.py", "install"])
        subprocess.run(["python3", "jarvis_background_service.py", "start"])
        print(f"\n{Colors.GREEN}✅ JARVIS is now running in the background!{Colors.END}")
        print("It will start automatically when you log in.")
        
    elif choice == '2':
        subprocess.run(["python3", "jarvis_background_service.py", "status"])
        
    elif choice == '3':
        subprocess.run(["python3", "jarvis_background_service.py", "stop"])
        print(f"{Colors.YELLOW}Service stopped.{Colors.END}")
        
    if choice != '4':
        input(f"\n{Colors.WHITE}Press Enter to continue...{Colors.END}")


def run_setup():
    """Run the setup script"""
    clear_screen()
    print_banner()
    
    print(f"\n{Colors.YELLOW}Running JARVIS Setup...{Colors.END}")
    print("This will only take about 2 minutes.\n")
    
    time.sleep(1)
    
    try:
        subprocess.run(["./setup_10_seamless.sh"], check=True)
        print(f"\n{Colors.GREEN}✅ Setup complete!{Colors.END}")
        input("Press Enter to continue...")
    except Exception as e:
        print(f"\n{Colors.RED}Setup failed: {e}{Colors.END}")
        input("Press Enter to continue...")


def loading_animation(message="Loading", duration=2):
    """Show a loading animation"""
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    end_time = time.time() + duration
    
    while time.time() < end_time:
        for frame in frames:
            print(f"\r{Colors.CYAN}{frame} {message}...{Colors.END}", end='', flush=True)
            time.sleep(0.1)
            
    print(f"\r{Colors.GREEN}✓ {message} complete!{Colors.END}")


if __name__ == "__main__":
    try:
        # Check if running with arguments
        if len(sys.argv) > 1:
            # Direct command execution
            if sys.argv[1] == "start":
                start_jarvis()
            elif sys.argv[1] == "service":
                start_service()
            elif sys.argv[1] == "setup":
                run_setup()
        else:
            # Interactive menu
            main_menu()
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Goodbye!{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}")
        sys.exit(1)