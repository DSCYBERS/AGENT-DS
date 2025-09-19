#!/usr/bin/env python3
"""
Agent DS v2.0 - Interactive CLI System
=====================================

Comprehensive command-line interface with:
- Interactive commands for all AI systems
- Real-time status updates and monitoring
- Mission management and execution
- Integration with complete AI suite
- User-friendly operation and feedback

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import argparse
import logging
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.attacks.complete_ai_integration import (
        get_ai_integration_system, execute_ai_pentest, 
        get_ai_recommendations, get_ai_system_status, demo_ai_capabilities
    )
    from core.attacks.ai_coordination_manager import MissionContext, MissionPhase
    from core.attacks.terminal_theme import get_live_terminal_interface
    AI_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI integration not fully available: {e}")
    AI_INTEGRATION_AVAILABLE = False

class InteractiveCLI:
    """
    Interactive command-line interface for Agent DS v2.0
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize systems
        self.ai_system = None
        self.terminal_interface = None
        self.current_mission = None
        self.mission_history = []
        
        # Command registry
        self.commands = {
            'help': self.cmd_help,
            'status': self.cmd_status,
            'demo': self.cmd_demo,
            'pentest': self.cmd_pentest,
            'recommend': self.cmd_recommend,
            'mission': self.cmd_mission,
            'history': self.cmd_history,
            'ai': self.cmd_ai,
            'modules': self.cmd_modules,
            'config': self.cmd_config,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit
        }
        
        self.initialize_systems()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('agent_ds_cli.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def initialize_systems(self):
        """Initialize AI and terminal systems"""
        try:
            if AI_INTEGRATION_AVAILABLE:
                self.ai_system = get_ai_integration_system()
                self.logger.info("AI integration system initialized")
            
            try:
                self.terminal_interface = get_live_terminal_interface()
                self.logger.info("Terminal interface initialized") 
            except:
                self.logger.warning("Terminal interface not available")
                
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
    
    def print_banner(self):
        """Print Agent DS banner"""
        banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🤖 Agent DS v2.0 - Interactive CLI              ║
    ║                                                              ║
    ║           AI-Powered Autonomous Penetration Testing         ║
    ║                                                              ║
    ║  🧠 AI Thinking Model     🎯 Smart Payload Generation        ║
    ║  🔄 Adaptive Sequencing   🛡️  Intelligence Coordination      ║
    ║  📊 Real-time Monitoring  💡 Interactive Decision Making     ║
    ║                                                              ║
    ║                  Type 'help' for commands                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
        
        # Show system status
        if AI_INTEGRATION_AVAILABLE and self.ai_system:
            print("🟢 AI Integration System: ACTIVE")
        else:
            print("🔴 AI Integration System: NOT AVAILABLE")
        
        if self.terminal_interface:
            print("🟢 Terminal Interface: ACTIVE")
        else:
            print("🔴 Terminal Interface: NOT AVAILABLE")
        
        print("\n" + "="*60 + "\n")
    
    async def run_interactive(self):
        """Run interactive CLI mode"""
        self.print_banner()
        
        while True:
            try:
                # Get user input
                user_input = input("\n[Agent DS] > ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split()
                command = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Execute command
                if command in self.commands:
                    await self.commands[command](args)
                else:
                    print(f"❌ Unknown command: {command}")
                    print("💡 Type 'help' to see available commands")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Exiting Agent DS CLI...")
                break
            except EOFError:
                print("\n\n👋 Goodbye! Exiting Agent DS CLI...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                self.logger.error(f"CLI error: {e}")
    
    async def cmd_help(self, args: List[str]):
        """Show help information"""
        if args:
            # Show help for specific command
            command = args[0].lower()
            if command in self.get_detailed_help():
                help_info = self.get_detailed_help()[command]
                print(f"\n📖 Help for '{command}':")
                print(f"   Description: {help_info['description']}")
                print(f"   Usage: {help_info['usage']}")
                if 'examples' in help_info:
                    print("   Examples:")
                    for example in help_info['examples']:
                        print(f"     {example}")
            else:
                print(f"❌ No help available for command: {command}")
        else:
            # Show general help
            print("\n📖 Agent DS v2.0 - Available Commands:")
            print("\n🤖 AI Operations:")
            print("   status     - Show AI system status")
            print("   demo       - Demonstrate AI capabilities")
            print("   ai         - AI system information and controls")
            
            print("\n🎯 Penetration Testing:")
            print("   pentest    - Execute AI-powered penetration test")
            print("   recommend  - Get AI recommendations for target")
            print("   mission    - Mission management commands")
            
            print("\n📊 Information & History:")
            print("   history    - Show mission history")
            print("   modules    - Show enhanced module status")
            print("   config     - Configuration management")
            
            print("\n💬 General:")
            print("   help       - Show this help (use 'help <command>' for details)")
            print("   exit/quit  - Exit the CLI")
            
            print(f"\n💡 Use 'help <command>' for detailed information")
    
    async def cmd_status(self, args: List[str]):
        """Show system status"""
        print("\n📊 Agent DS v2.0 System Status")
        print("=" * 40)
        
        if AI_INTEGRATION_AVAILABLE and self.ai_system:
            try:
                status = self.ai_system.get_system_status()
                
                print(f"🤖 AI Integration: {'🟢 ACTIVE' if status.get('ai_integration_active') else '🔴 INACTIVE'}")
                
                # Show individual system status
                systems = status.get('systems', {})
                
                if 'coordination_manager' in systems:
                    cm_status = systems['coordination_manager']
                    if isinstance(cm_status, dict) and 'enhanced_modules' in cm_status:
                        print(f"🧠 Coordination Manager: 🟢 ACTIVE ({len(cm_status['enhanced_modules'])} modules)")
                    else:
                        print("🧠 Coordination Manager: 🔴 NOT AVAILABLE")
                
                if 'module_enhancer' in systems:
                    me_status = systems['module_enhancer']
                    if isinstance(me_status, dict) and 'enhanced_modules' in me_status:
                        print(f"⚡ Module Enhancer: 🟢 ACTIVE ({me_status['total_enhanced']} enhanced)")
                    else:
                        print("⚡ Module Enhancer: 🔴 NOT AVAILABLE")
                
                if 'ai_components' in systems:
                    ac_status = systems['ai_components']
                    if isinstance(ac_status, dict) and ac_status.get('status') != 'not_available':
                        print("🧮 AI Components: 🟢 ACTIVE")
                        if 'payload_generator' in ac_status:
                            print(f"   🎯 Payload Generator: {'🟢' if ac_status['payload_generator'] else '🔴'}")
                        if 'attack_sequencer' in ac_status:
                            print(f"   🔄 Attack Sequencer: {'🟢' if ac_status['attack_sequencer'] else '🔴'}")
                    else:
                        print("🧮 AI Components: 🔴 NOT AVAILABLE")
                
            except Exception as e:
                print(f"❌ Error getting status: {e}")
        else:
            print("🔴 AI Integration System: NOT AVAILABLE")
        
        # Show mission status
        if self.current_mission:
            print(f"\n📋 Current Mission: {self.current_mission.get('mission_id', 'Unknown')}")
            print(f"   Target: {self.current_mission.get('target_url', 'Unknown')}")
            print(f"   Status: {self.current_mission.get('status', 'Unknown')}")
        else:
            print("\n📋 Current Mission: None")
        
        print(f"\n📚 Mission History: {len(self.mission_history)} completed missions")
    
    async def cmd_demo(self, args: List[str]):
        """Demonstrate AI capabilities"""
        print("\n🎭 Demonstrating Agent DS v2.0 AI Capabilities...")
        
        if not AI_INTEGRATION_AVAILABLE or not self.ai_system:
            print("❌ AI integration system not available")
            return
        
        target_url = args[0] if args else "https://example.com"
        
        try:
            demo_results = await self.ai_system.demonstrate_ai_capabilities(target_url)
            
            if demo_results.get('success'):
                print(f"✅ Demo completed successfully for {demo_results['demo_target']}")
                
                print(f"\n🧪 Capabilities Tested:")
                for capability in demo_results['capabilities_tested']:
                    print(f"   ✓ {capability.replace('_', ' ').title()}")
                
                results = demo_results.get('results', {})
                
                if 'ai_recommendations' in results:
                    recommendations = results['ai_recommendations']
                    print(f"\n💡 AI Recommendations ({len(recommendations)}):")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec}")
                
                if 'system_status' in results:
                    status = results['system_status']
                    print(f"\n📊 System Status: {'🟢 Active' if status.get('ai_integration_active') else '🔴 Inactive'}")
                
                if 'module_enhancement' in results:
                    enhancement = results['module_enhancement']
                    enhanced_count = enhancement.get('total_enhanced', 0)
                    print(f"\n⚡ Enhanced Modules: {enhanced_count} modules with AI capabilities")
                
            else:
                print(f"❌ Demo failed: {demo_results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Demo execution failed: {e}")
    
    async def cmd_pentest(self, args: List[str]):
        """Execute AI-powered penetration test"""
        if not args:
            print("❌ Usage: pentest <target_url> [options]")
            print("💡 Example: pentest https://example.com")
            return
        
        target_url = args[0]
        
        print(f"\n🎯 Starting AI-Powered Penetration Test")
        print(f"Target: {target_url}")
        print("=" * 50)
        
        if not AI_INTEGRATION_AVAILABLE or not self.ai_system:
            print("❌ AI integration system not available")
            return
        
        try:
            # Parse additional options
            kwargs = {}
            if len(args) > 1:
                for arg in args[1:]:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        kwargs[key] = value
            
            # Execute penetration test
            print("🚀 Launching AI coordination systems...")
            mission_result = await self.ai_system.execute_intelligent_pentest(target_url, **kwargs)
            
            if mission_result.get('success'):
                print(f"✅ Mission completed successfully!")
                
                # Show results summary
                print(f"\n📋 Mission Summary:")
                print(f"   Mission ID: {mission_result.get('mission_id', 'Unknown')}")
                print(f"   Target: {mission_result.get('target_url', target_url)}")
                print(f"   Phases Executed: {len(mission_result.get('phases_executed', []))}")
                print(f"   Execution Time: {mission_result.get('execution_time', 0):.2f} seconds")
                
                # Show findings
                findings = mission_result.get('findings', [])
                if findings:
                    print(f"\n🔍 Findings ({len(findings)}):")
                    for i, finding in enumerate(findings, 1):
                        print(f"   {i}. {finding}")
                
                # Show recommendations
                recommendations = mission_result.get('recommendations', [])
                if recommendations:
                    print(f"\n💡 AI Recommendations ({len(recommendations)}):")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"   {i}. {rec}")
                
                # Store mission
                self.current_mission = mission_result
                self.mission_history.append(mission_result)
                
            else:
                print(f"❌ Mission failed: {mission_result.get('error', 'Unknown error')}")
                if mission_result.get('fallback_mode'):
                    print("⚠️  Running in fallback mode - AI features limited")
                
        except Exception as e:
            print(f"❌ Penetration test failed: {e}")
            self.logger.error(f"Pentest error: {e}")
    
    async def cmd_recommend(self, args: List[str]):
        """Get AI recommendations for target"""
        if not args:
            print("❌ Usage: recommend <target_url> [findings_json]")
            print("💡 Example: recommend https://example.com")
            return
        
        target_url = args[0]
        
        # Parse findings if provided
        findings = {}
        if len(args) > 1:
            try:
                findings = json.loads(args[1])
            except json.JSONDecodeError:
                print("⚠️  Invalid JSON for findings, using default")
                findings = {
                    'vulnerabilities_found': True,
                    'admin_panels': ['/admin'],
                    'database_detected': True
                }
        else:
            # Use example findings
            findings = {
                'vulnerabilities_found': True,
                'admin_panels': ['/admin', '/wp-admin'],
                'database_detected': True,
                'waf_detected': False
            }
        
        print(f"\n💡 Getting AI Recommendations for {target_url}")
        print("=" * 50)
        
        if not AI_INTEGRATION_AVAILABLE or not self.ai_system:
            print("❌ AI integration system not available")
            return
        
        try:
            recommendations = await self.ai_system.get_ai_recommendations(target_url, findings)
            
            if recommendations:
                print(f"🤖 AI Analysis Complete - {len(recommendations)} recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            else:
                print("ℹ️  No specific recommendations at this time")
                
        except Exception as e:
            print(f"❌ Failed to get recommendations: {e}")
    
    async def cmd_mission(self, args: List[str]):
        """Mission management commands"""
        if not args:
            print("❌ Usage: mission <subcommand>")
            print("Available subcommands: info, save, load, clear")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == 'info':
            if self.current_mission:
                print(f"\n📋 Current Mission Information:")
                print(f"   Mission ID: {self.current_mission.get('mission_id', 'Unknown')}")
                print(f"   Target: {self.current_mission.get('target_url', 'Unknown')}")
                print(f"   Success: {self.current_mission.get('overall_success', False)}")
                print(f"   Phases: {', '.join(self.current_mission.get('phases_executed', []))}")
                print(f"   Findings: {len(self.current_mission.get('findings', []))}")
                print(f"   Recommendations: {len(self.current_mission.get('recommendations', []))}")
                print(f"   Execution Time: {self.current_mission.get('execution_time', 0):.2f}s")
            else:
                print("ℹ️  No current mission")
        
        elif subcommand == 'save':
            if self.current_mission:
                filename = args[1] if len(args) > 1 else f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                try:
                    with open(filename, 'w') as f:
                        json.dump(self.current_mission, f, indent=2)
                    print(f"✅ Mission saved to {filename}")
                except Exception as e:
                    print(f"❌ Failed to save mission: {e}")
            else:
                print("ℹ️  No current mission to save")
        
        elif subcommand == 'load':
            if len(args) < 2:
                print("❌ Usage: mission load <filename>")
                return
            
            filename = args[1]
            try:
                with open(filename, 'r') as f:
                    self.current_mission = json.load(f)
                print(f"✅ Mission loaded from {filename}")
            except Exception as e:
                print(f"❌ Failed to load mission: {e}")
        
        elif subcommand == 'clear':
            self.current_mission = None
            print("✅ Current mission cleared")
        
        else:
            print(f"❌ Unknown mission subcommand: {subcommand}")
    
    async def cmd_history(self, args: List[str]):
        """Show mission history"""
        if not self.mission_history:
            print("ℹ️  No mission history")
            return
        
        print(f"\n📚 Mission History ({len(self.mission_history)} missions):")
        print("=" * 60)
        
        for i, mission in enumerate(self.mission_history, 1):
            success_icon = "✅" if mission.get('overall_success') else "❌"
            print(f"{i:2d}. {success_icon} {mission.get('mission_id', 'Unknown')}")
            print(f"     Target: {mission.get('target_url', 'Unknown')}")
            print(f"     Time: {mission.get('execution_time', 0):.2f}s")
            print(f"     Findings: {len(mission.get('findings', []))}")
            print()
    
    async def cmd_ai(self, args: List[str]):
        """AI system information and controls"""
        if not AI_INTEGRATION_AVAILABLE:
            print("❌ AI integration system not available")
            return
        
        if not args:
            print("Available AI commands: status, components, demo")
            return
        
        subcommand = args[0].lower()
        
        if subcommand == 'status':
            await self.cmd_status([])
        elif subcommand == 'components':
            print("\n🧮 AI Components Status:")
            try:
                status = self.ai_system.get_system_status()
                components = status.get('systems', {}).get('ai_components', {})
                if components.get('status') != 'not_available':
                    for component, active in components.items():
                        if isinstance(active, bool):
                            icon = "🟢" if active else "🔴"
                            print(f"   {icon} {component.replace('_', ' ').title()}")
                else:
                    print("   🔴 AI components not available")
            except Exception as e:
                print(f"❌ Error getting component status: {e}")
        elif subcommand == 'demo':
            await self.cmd_demo(args[1:])
        else:
            print(f"❌ Unknown AI subcommand: {subcommand}")
    
    async def cmd_modules(self, args: List[str]):
        """Show enhanced module status"""
        print("\n⚡ Enhanced Module Status:")
        
        if not AI_INTEGRATION_AVAILABLE or not self.ai_system:
            print("❌ AI integration system not available")
            return
        
        try:
            status = self.ai_system.get_system_status()
            enhancer_status = status.get('systems', {}).get('module_enhancer', {})
            
            if 'enhanced_modules' in enhancer_status:
                modules = enhancer_status['enhanced_modules']
                print(f"🔧 Total Enhanced: {len(modules)} modules")
                
                for module in modules:
                    print(f"   🟢 {module.replace('_', ' ').title()}")
                
                print(f"\n📊 AI Available: {'🟢 Yes' if enhancer_status.get('ai_available') else '🔴 No'}")
                
            else:
                print("🔴 Module enhancer not available")
                
        except Exception as e:
            print(f"❌ Error getting module status: {e}")
    
    async def cmd_config(self, args: List[str]):
        """Configuration management"""
        print("\n⚙️  Configuration Management:")
        print("   Log Level: INFO")
        print("   AI Integration: " + ("🟢 Enabled" if AI_INTEGRATION_AVAILABLE else "🔴 Disabled"))
        print("   Terminal Interface: " + ("🟢 Available" if self.terminal_interface else "🔴 Not Available"))
        print("   Mission Auto-save: 🟢 Enabled")
        print("   Debug Mode: 🔴 Disabled")
    
    async def cmd_exit(self, args: List[str]):
        """Exit the CLI"""
        print("\n👋 Thank you for using Agent DS v2.0!")
        print("💾 Saving session data...")
        
        # Save mission history if any
        if self.mission_history:
            try:
                filename = f"session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(self.mission_history, f, indent=2)
                print(f"✅ Session history saved to {filename}")
            except Exception as e:
                print(f"⚠️  Could not save session history: {e}")
        
        print("🚀 Happy hacking!")
        sys.exit(0)
    
    def get_detailed_help(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed help information for commands"""
        return {
            'pentest': {
                'description': 'Execute AI-powered penetration test on target',
                'usage': 'pentest <target_url> [key=value options]',
                'examples': [
                    'pentest https://example.com',
                    'pentest https://test.com waf_detected=true',
                    'pentest https://app.com database_type=mysql'
                ]
            },
            'recommend': {
                'description': 'Get AI recommendations based on findings',
                'usage': 'recommend <target_url> [findings_json]',
                'examples': [
                    'recommend https://example.com',
                    'recommend https://test.com \'{"admin_panels": ["/admin"]}\''
                ]
            },
            'mission': {
                'description': 'Mission management operations',
                'usage': 'mission <info|save|load|clear> [args]',
                'examples': [
                    'mission info',
                    'mission save my_mission.json',
                    'mission load my_mission.json',
                    'mission clear'
                ]
            },
            'ai': {
                'description': 'AI system information and controls',
                'usage': 'ai <status|components|demo> [args]',
                'examples': [
                    'ai status',
                    'ai components',
                    'ai demo https://example.com'
                ]
            }
        }

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Agent DS v2.0 - AI-Powered Penetration Testing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py                          # Interactive mode
  python cli.py --demo                   # Run demonstration
  python cli.py --pentest https://example.com  # Quick pentest
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run AI capabilities demonstration')
    parser.add_argument('--pentest', type=str, metavar='URL',
                       help='Execute penetration test on target URL')
    parser.add_argument('--status', action='store_true',
                       help='Show system status and exit')
    parser.add_argument('--interactive', action='store_true', default=True,
                       help='Run in interactive mode (default)')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = InteractiveCLI()
    
    async def run_cli():
        if args.demo:
            await cli.cmd_demo([])
        elif args.pentest:
            await cli.cmd_pentest([args.pentest])
        elif args.status:
            await cli.cmd_status([])
        else:
            # Interactive mode
            await cli.run_interactive()
    
    # Run the CLI
    try:
        asyncio.run(run_cli())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()