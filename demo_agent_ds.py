#!/usr/bin/env python3
"""
Agent DS v2.0 - Demo Script
Demonstrates the complete one-click experience
"""

import subprocess
import sys
import time

def run_demo():
    """Run the Agent DS demo"""
    
    print("""
🚀 AGENT DS v2.0 - DEMONSTRATION SCRIPT 🚀
=========================================

This script demonstrates the complete one-click penetration testing experience.

Available demo scenarios:
1. Status Check
2. Help System  
3. One-Click Penetration Test (Safe Target)
4. View Methodology Chain

""")
    
    while True:
        try:
            choice = input("Select demo (1-4) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Thanks for trying Agent DS v2.0!")
                break
                
            elif choice == '1':
                print("\n🔍 Running: python main_agent_ds.py --status")
                print("=" * 50)
                subprocess.run([sys.executable, "main_agent_ds.py", "--status"])
                
            elif choice == '2':
                print("\n📚 Running: python main_agent_ds.py --help")
                print("=" * 50)
                subprocess.run([sys.executable, "main_agent_ds.py", "--help"])
                
            elif choice == '3':
                print("\n⚡ Running: python main_agent_ds.py httpbin.org")
                print("=" * 50)
                print("This will demonstrate the complete methodology execution...")
                confirm = input("Continue with penetration test demo? (y/n): ")
                if confirm.lower() == 'y':
                    # Create a simple input file to auto-confirm authorization
                    with open('demo_input.txt', 'w') as f:
                        f.write('yes\n')
                    
                    # Run with input redirection
                    with open('demo_input.txt', 'r') as f:
                        subprocess.run([sys.executable, "main_agent_ds.py", "httpbin.org"], stdin=f)
                    
                    # Clean up
                    import os
                    try:
                        os.remove('demo_input.txt')
                    except:
                        pass
                
            elif choice == '4':
                print("\n🎯 AGENT DS v2.0 METHODOLOGY CHAIN:")
                print("=" * 50)
                methodology = [
                    "01. 🔧 INITIALIZATION - System startup and target validation",
                    "02. 🔍 RECONNAISSANCE - Intelligence gathering and enumeration", 
                    "03. 🧠 VULNERABILITY_ANALYSIS - AI-powered vulnerability discovery",
                    "04. 🌐 WEB_EXPLOITATION - Web application attack vectors",
                    "05. 🗄️ DATABASE_EXPLOITATION - Database penetration and extraction",
                    "06. 👑 ADMIN_ACCESS_TESTING - Administrative interface compromise",
                    "07. 🤖 AI_ADAPTIVE_ATTACKS - Machine learning enhanced exploitation",
                    "08. ⬆️ PRIVILEGE_ESCALATION - System-level access attempts", 
                    "09. 🔒 PERSISTENCE_TESTING - Backdoor and persistence mechanisms",
                    "10. 📊 REPORTING - Comprehensive security assessment report"
                ]
                
                for phase in methodology:
                    print(f"  {phase}")
                    time.sleep(0.3)
                
                print("\n🚀 This complete methodology runs automatically with:")
                print("   python main_agent_ds.py <target>")
                
            else:
                print("❌ Invalid choice. Please select 1-4 or 'q'")
            
            print("\n" + "=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Demo interrupted. Thanks for trying Agent DS v2.0!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_demo()