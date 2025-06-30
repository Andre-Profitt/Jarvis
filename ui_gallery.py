#!/usr/bin/env python3
"""
JARVIS UI Gallery
See all the different interfaces and choose your favorite
"""

import os
import webbrowser
import time

print("""
╔══════════════════════════════════════════════════════════════════╗
║                      🎨 JARVIS UI GALLERY 🎨                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Choose which interface to view:                                 ║
║                                                                  ║
║  1. 🏆 Premium 2D UI (NEW!)                                     ║
║     Clean, professional interface inspired by Perplexity/Claude  ║
║                                                                  ║
║  2. 🔮 3D Holographic UI                                        ║
║     The original 3D sphere with particles                        ║
║                                                                  ║
║  3. 🏢 Enterprise UI                                            ║
║     The Stripe-inspired business interface                       ║
║                                                                  ║
║  4. 📊 UI Comparison                                            ║
║     See the evolution from 3D to Premium                         ║
║                                                                  ║
║  5. 🎯 Interactive Demo                                         ║
║     The working demo with quick actions                          ║
║                                                                  ║
║  6. 🚀 Launch Premium with Backend                              ║
║     Full system with AI integration                              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")

choice = input("\nEnter your choice (1-6): ")

base_dir = "/Users/andreprofitt/CloudAI/JARVIS-ECOSYSTEM"

if choice == "1":
    print("\n🏆 Opening Premium 2D UI...")
    print("This is the latest and best interface!")
    webbrowser.open(f"file://{base_dir}/jarvis-premium-ui.html")

elif choice == "2":
    print("\n🔮 Opening 3D Holographic UI...")
    webbrowser.open(f"file://{base_dir}/jarvis-3d-avatar.html")

elif choice == "3":
    print("\n🏢 Opening Enterprise UI...")
    webbrowser.open(f"file://{base_dir}/jarvis-enterprise-ui.html")

elif choice == "4":
    print("\n📊 Opening UI Comparison...")
    print("See how we evolved from 3D to Premium!")
    webbrowser.open(f"file://{base_dir}/ui-comparison.html")

elif choice == "5":
    print("\n🎯 Opening Interactive Demo...")
    webbrowser.open(f"file://{base_dir}/jarvis-demo-interactive.html")

elif choice == "6":
    print("\n🚀 Launching Premium UI with Backend...")
    print("Starting server...")
    import subprocess
    subprocess.run(["python3", f"{base_dir}/jarvis_premium_backend.py"])

else:
    print("\n❌ Invalid choice")

print("\n✨ Thanks for exploring JARVIS interfaces!")
print("\n💡 Tip: The Premium 2D UI (option 1) is the most advanced and professional!")
