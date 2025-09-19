#!/usr/bin/env python3
"""
Agent DS v2.0 - Main Entry Point
===============================

Simple entry point for Agent DS v2.0 with multiple interfaces:
- Interactive CLI
- Quick command execution  
- AI system demonstration
- System status checking

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import sys
import os
import asyncio

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def print_agent_ds_logo():
    """Print Agent DS logo and version info"""
    logo = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗              ║
    ║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝              ║
    ║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║                 ║
    ║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║                 ║
    ║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║                 ║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝                 ║
    ║                                                              ║
    ║              ██████╗ ███████╗    ██╗   ██╗██████╗             ║
    ║              ██╔══██╗██╔════╝    ██║   ██║╚════██╗            ║
    ║              ██║  ██║███████╗    ██║   ██║ █████╔╝            ║
    ║              ██║  ██║╚════██║    ╚██╗ ██╔╝██╔═══╝             ║
    ║              ██████╔╝███████║     ╚████╔╝ ███████╗            ║
    ║              ╚═════╝ ╚══════╝      ╚═══╝  ╚══════╝            ║
    ║                                                              ║
    ║           🤖 AI-Powered Autonomous Penetration Testing        ║
    ║                                                              ║
    ║  🧠 Intelligent Decision Making  🎯 Smart Payload Generation  ║
    ║  🔄 Adaptive Attack Sequencing   🛡️  Real-time Coordination  ║
    ║  📊 Live Monitoring & Analytics  💡 Interactive CLI Interface ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    🚀 Agent DS v2.0 - The Future of Autonomous Security Testing
    
    """
    print(logo)

def show_usage():
    """Show usage information"""
    usage = """
🔧 Usage Options:

1. 🎮 Interactive CLI Mode (Recommended):
   python agent_ds.py
   
2. 🎯 Quick Penetration Test:
   python agent_ds.py --pentest <target_url>
   
3. 🎭 AI Capabilities Demo:
   python agent_ds.py --demo
   
4. 📊 System Status Check:
   python agent_ds.py --status
   
5. 💡 Help & Information:
   python agent_ds.py --help

📚 Examples:
   python agent_ds.py                                    # Start interactive mode
   python agent_ds.py --pentest https://example.com      # Quick pentest
   python agent_ds.py --demo                             # Show AI capabilities
   python agent_ds.py --status                           # Check system status

🔗 For detailed command help, use the interactive mode and type 'help'
    """
    print(usage)

async def quick_status_check():
    """Quick status check of all systems"""
    print("🔍 Agent DS v2.0 - Quick System Status Check")
    print("=" * 50)
    
    try:
        # Check AI integration
        try:
            from core.attacks.complete_ai_integration import get_ai_integration_system
            ai_system = get_ai_integration_system()
            status = ai_system.get_system_status()
            print("🤖 AI Integration System: 🟢 AVAILABLE")
            
            systems = status.get('systems', {})
            for system_name, system_status in systems.items():
                if isinstance(system_status, dict):
                    if system_status.get('status') == 'not_available':
                        print(f"   {system_name.replace('_', ' ').title()}: 🔴 NOT AVAILABLE")
                    else:
                        print(f"   {system_name.replace('_', ' ').title()}: 🟢 AVAILABLE")
                        
        except ImportError as e:
            print("🤖 AI Integration System: 🔴 NOT AVAILABLE")
            print(f"   Reason: {e}")
        
        # Check terminal interface
        try:
            from core.attacks.terminal_theme import get_live_terminal_interface
            terminal = get_live_terminal_interface()
            print("📺 Terminal Interface: 🟢 AVAILABLE")
        except ImportError:
            print("📺 Terminal Interface: 🔴 NOT AVAILABLE")
        
        # Check core modules
        try:
            from core.attacks.ai_coordination_manager import get_ai_coordinator
            coordinator = get_ai_coordinator()
            print("🧠 AI Coordinator: 🟢 AVAILABLE")
        except ImportError:
            print("🧠 AI Coordinator: 🔴 NOT AVAILABLE")
        
        print("\n✅ System check complete!")
        
    except Exception as e:
        print(f"❌ System check failed: {e}")

async def quick_demo():
    """Quick demonstration of AI capabilities"""
    print("🎭 Agent DS v2.0 - AI Capabilities Demonstration")
    print("=" * 50)
    
    try:
        from core.attacks.complete_ai_integration import demo_ai_capabilities
        
        print("🚀 Running AI demonstration...")
        result = await demo_ai_capabilities("https://example.com")
        
        if result.get('success'):
            print("✅ AI demonstration completed successfully!")
            
            capabilities = result.get('capabilities_tested', [])
            print(f"\n🧪 Tested Capabilities ({len(capabilities)}):")
            for cap in capabilities:
                print(f"   ✓ {cap.replace('_', ' ').title()}")
                
            results = result.get('results', {})
            if 'ai_recommendations' in results:
                recommendations = results['ai_recommendations']
                print(f"\n💡 Sample AI Recommendations ({len(recommendations)}):")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec}")
                    
        else:
            print(f"❌ AI demonstration failed: {result.get('error', 'Unknown error')}")
            
    except ImportError:
        print("❌ AI system not available for demonstration")
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")

async def quick_pentest(target_url: str):
    """Quick penetration test execution"""
    print(f"🎯 Agent DS v2.0 - Quick Penetration Test")
    print(f"Target: {target_url}")
    print("=" * 50)
    
    try:
        from core.attacks.complete_ai_integration import execute_ai_pentest
        
        print("🚀 Initializing AI-powered penetration test...")
        result = await execute_ai_pentest(target_url)
        
        if result.get('success'):
            print("✅ Penetration test completed!")
            
            print(f"\n📋 Mission Summary:")
            print(f"   Mission ID: {result.get('mission_id', 'Unknown')}")
            print(f"   Execution Time: {result.get('execution_time', 0):.2f} seconds")
            print(f"   Phases Executed: {len(result.get('phases_executed', []))}")
            
            findings = result.get('findings', [])
            if findings:
                print(f"\n🔍 Key Findings ({len(findings)}):")
                for i, finding in enumerate(findings[:5], 1):
                    print(f"   {i}. {finding}")
                if len(findings) > 5:
                    print(f"   ... and {len(findings) - 5} more findings")
            
            recommendations = result.get('recommendations', [])
            if recommendations:
                print(f"\n💡 AI Recommendations ({len(recommendations)}):")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec}")
                if len(recommendations) > 3:
                    print(f"   ... and {len(recommendations) - 3} more recommendations")
                    
        else:
            print(f"❌ Penetration test failed: {result.get('error', 'Unknown error')}")
            if result.get('fallback_mode'):
                print("⚠️  AI features were limited - check system dependencies")
                
    except ImportError:
        print("❌ AI penetration testing system not available")
        print("💡 Please ensure all dependencies are installed")
    except Exception as e:
        print(f"❌ Penetration test failed: {e}")

def main():
    """Main entry point"""
    
    # Check command line arguments
    if len(sys.argv) == 1:
        # No arguments - show logo and start interactive mode
        print_agent_ds_logo()
        print("🎮 Starting Interactive CLI Mode...")
        print("💡 Use 'help' for available commands\n")
        
        try:
            from interactive_cli import InteractiveCLI
            cli = InteractiveCLI()
            asyncio.run(cli.run_interactive())
        except ImportError:
            print("❌ Interactive CLI not available")
            print("💡 Try: python agent_ds.py --status")
        except Exception as e:
            print(f"❌ Failed to start interactive mode: {e}")
    
    elif '--help' in sys.argv or '-h' in sys.argv:
        print_agent_ds_logo()
        show_usage()
    
    elif '--status' in sys.argv:
        print_agent_ds_logo()
        asyncio.run(quick_status_check())
    
    elif '--demo' in sys.argv:
        print_agent_ds_logo()
        asyncio.run(quick_demo())
    
    elif '--pentest' in sys.argv:
        print_agent_ds_logo()
        try:
            # Find target URL
            pentest_index = sys.argv.index('--pentest')
            if pentest_index + 1 < len(sys.argv):
                target_url = sys.argv[pentest_index + 1]
                asyncio.run(quick_pentest(target_url))
            else:
                print("❌ Error: --pentest requires a target URL")
                print("💡 Usage: python agent_ds.py --pentest <target_url>")
        except ValueError:
            print("❌ Error: Invalid --pentest usage")
    
    else:
        print_agent_ds_logo()
        print("❌ Unknown command line options")
        show_usage()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for using Agent DS v2.0!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)