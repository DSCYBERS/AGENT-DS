
"""
Agent DS v2.0 - AI Enhanced Autonomous Hacker Agent
===================================================

Enhanced launcher with AI Thinking Model integration and Live Terminal Interface
for next-generation autonomous penetration testing with real-time intelligence.

New Features in v2.0:
- AI Thinking Model: Central intelligence controlling all attack modules
- Live Metasploit-Themed Terminal: Real-time updates with interactive commands
- Autonomous Mission Control: Fully automated attack execution with AI adaptation
- Real-Time Feedback: Live monitoring and intelligence updates
- Advanced Payload Generation: AI-powered context-aware exploit creation
- Adaptive Attack Sequencing: ML-based optimal attack ordering

Usage:
    python launcher_v2.py --target <URL> --mode <mode> [options]

Modes:
    --one-click         : Original one-click attack (compatibility mode)
    --ai-autonomous     : AI autonomous mode with live terminal
    --interactive       : Interactive terminal mode with manual control
    --live-demo         : Live demonstration mode
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.orchestrator.one_click import OneClickAttack
from core.interface.ai_terminal_integration import AITerminalIntegration
from core.interface.live_terminal import LiveTerminalInterface
from core.attacks.ai_core import AIThinkingModel


class AgentDSLauncher:
    """
    Enhanced Agent DS launcher with AI capabilities
    """
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger('AgentDS.Launcher')
        
        # Component instances
        self.ai_integration = None
        self.one_click_engine = None
        self.terminal_interface = None
        
        self.logger.info("Agent DS v2.0 Launcher initialized")
    
    def setup_logging(self):
        """Setup enhanced logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agent_ds_v2.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    async def launch_ai_autonomous_mode(self, target_url: str, options: Dict[str, Any]):
        """
        Launch AI autonomous mode with live terminal interface
        """
        print("🚀 Starting Agent DS v2.0 - AI Autonomous Mode")
        print("=" * 80)
        print()
        
        try:
            # Initialize AI Terminal Integration
            self.ai_integration = AITerminalIntegration()
            await self.ai_integration.initialize()
            
            print("✅ AI Thinking Model: LOADED")
            print("✅ Live Terminal Interface: ACTIVE")
            print("✅ Autonomous Attack Engine: READY")
            print()
            
            # Start terminal interface
            terminal_task = asyncio.create_task(
                self.ai_integration.terminal.start_interface()
            )
            
            # Wait for interface to initialize
            await asyncio.sleep(2)
            
            # Start autonomous mission
            print(f"🎯 Launching autonomous mission against: {target_url}")
            print("🧠 AI will control all attack phases automatically")
            print("📊 Monitor progress in live terminal interface")
            print()
            
            mission_task = asyncio.create_task(
                self.ai_integration.start_autonomous_mission(target_url, options)
            )
            
            # Run both tasks concurrently
            await asyncio.gather(terminal_task, mission_task)
            
        except Exception as e:
            self.logger.error(f"Error in AI autonomous mode: {e}")
            print(f"❌ Error: {e}")
            return False
        
        return True
    
    async def launch_interactive_mode(self, target_url: Optional[str] = None):
        """
        Launch interactive terminal mode
        """
        print("💻 Starting Agent DS v2.0 - Interactive Terminal Mode")
        print("=" * 80)
        print()
        
        try:
            # Initialize terminal interface
            self.terminal_interface = LiveTerminalInterface()
            
            print("✅ Live Terminal Interface: ACTIVE")
            print("✅ Interactive Commands: READY")
            print("✅ AI Thinking Model: STANDBY")
            print()
            print("📱 Use interactive commands to control Agent DS")
            print("💡 Type 'help' for available commands")
            print("🎯 Use 'ai-mission start <target>' for autonomous attacks")
            print()
            
            # Start terminal interface
            await self.terminal_interface.start_interface()
            
        except Exception as e:
            self.logger.error(f"Error in interactive mode: {e}")
            print(f"❌ Error: {e}")
            return False
        
        return True
    
    async def launch_one_click_mode(self, target_url: str, options: Dict[str, Any]):
        """
        Launch original one-click attack mode (compatibility)
        """
        print("⚡ Starting Agent DS v2.0 - One-Click Attack Mode (Compatibility)")
        print("=" * 80)
        print()
        
        try:
            # Initialize one-click engine
            self.one_click_engine = OneClickAttack()
            
            print("✅ One-Click Attack Engine: LOADED")
            print("✅ All Attack Modules: READY")
            print()
            print(f"🎯 Target: {target_url}")
            print("⚡ Launching sequential attack phases...")
            print()
            
            # Execute one-click attack
            results = await self.one_click_engine.execute_full_attack(target_url)
            
            # Display results summary
            self._display_one_click_results(results)
            
        except Exception as e:
            self.logger.error(f"Error in one-click mode: {e}")
            print(f"❌ Error: {e}")
            return False
        
        return True
    
    async def launch_live_demo_mode(self):
        """
        Launch live demonstration mode
        """
        print("🎬 Starting Agent DS v2.0 - Live Demonstration Mode")
        print("=" * 80)
        print()
        
        try:
            # Initialize AI Terminal Integration
            self.ai_integration = AITerminalIntegration()
            await self.ai_integration.initialize()
            
            print("✅ AI Thinking Model: DEMO MODE")
            print("✅ Live Terminal Interface: ACTIVE")
            print("✅ Simulation Engine: READY")
            print()
            print("🎭 Demonstration will show AI autonomous capabilities")
            print("📊 All attacks are simulated - no real penetration testing")
            print()
            
            # Start terminal interface
            terminal_task = asyncio.create_task(
                self.ai_integration.terminal.start_interface()
            )
            
            # Wait for interface to start
            await asyncio.sleep(3)
            
            # Start demo mission
            demo_targets = [
                "https://demo.testfire.net",
                "http://testphp.vulnweb.com",
                "https://dvwa.local"
            ]
            
            for target in demo_targets:
                print(f"🎯 Demo Mission: {target}")
                
                mission_task = asyncio.create_task(
                    self.ai_integration.start_autonomous_mission(
                        target, 
                        {'mode': 'demonstration', 'simulate': True}
                    )
                )
                
                # Run demo mission
                await mission_task
                
                # Brief pause between demos
                await asyncio.sleep(5)
            
            # Keep terminal running
            await terminal_task
            
        except Exception as e:
            self.logger.error(f"Error in demo mode: {e}")
            print(f"❌ Error: {e}")
            return False
        
        return True
    
    def _display_one_click_results(self, results: Dict[str, Any]):
        """
        Display one-click attack results
        """
        print("📊 ONE-CLICK ATTACK RESULTS")
        print("=" * 50)
        
        total_phases = len(results)
        successful_phases = sum(1 for result in results.values() 
                              if isinstance(result, dict) and result.get('vulnerable', False))
        
        print(f"🎯 Total Attack Phases: {total_phases}")
        print(f"✅ Successful Phases: {successful_phases}")
        print(f"📈 Success Rate: {successful_phases/total_phases*100:.1f}%")
        print()
        
        for phase_name, phase_result in results.items():
            if isinstance(phase_result, dict):
                vulnerable = phase_result.get('vulnerable', False)
                vuln_count = len(phase_result.get('vulnerabilities', []))
                
                status = "✅ VULNERABLE" if vulnerable else "❌ SECURE"
                print(f"{phase_name.upper()}: {status}")
                
                if vulnerable and vuln_count > 0:
                    print(f"   └── {vuln_count} vulnerabilities found")
        
        print()
        print("📝 Detailed reports saved to reports/ directory")
        print("🔍 Check logs for complete attack details")
    
    def _display_banner(self):
        """
        Display Agent DS v2.0 banner
        """
        banner = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                                  ║
║     █████╗  ██████╗ ███████╗███╗   ██╗████████╗    ██████╗ ███████╗    ██╗   ██╗██████╗    ██████╗             ║
║    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝    ██╔══██╗██╔════╝    ██║   ██║╚════██╗  ██╔═████╗             ║
║    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║       ██║  ██║███████╗    ██║   ██║ █████╔╝  ██║██╔██║             ║
║    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║       ██║  ██║╚════██║    ╚██╗ ██╔╝██╔═══╝   ████╔╝██║             ║
║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║       ██████╔╝███████║     ╚████╔╝ ███████╗██╗╚██████╔╝██╗          ║
║    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝       ╚═════╝ ╚══════╝      ╚═══╝  ╚══════╝╚═╝ ╚═════╝ ╚═╝          ║
║                                                                                                                  ║
║                           🧠 AI THINKING MODEL • 🎯 AUTONOMOUS ATTACKS • 📊 LIVE MONITORING                     ║
║                                                                                                                  ║
║                                    Next-Generation Autonomous Hacker Agent                                      ║
║                                             with AI Intelligence                                                 ║
║                                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)


def create_arg_parser():
    """
    Create command line argument parser
    """
    parser = argparse.ArgumentParser(
        description="Agent DS v2.0 - AI Enhanced Autonomous Hacker Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # AI Autonomous Mode (Full automation with live terminal)
  python launcher_v2.py --target https://demo.testfire.net --mode ai-autonomous
  
  # Interactive Terminal Mode
  python launcher_v2.py --mode interactive
  
  # Original One-Click Attack (Compatibility)
  python launcher_v2.py --target https://example.com --mode one-click
  
  # Live Demonstration Mode
  python launcher_v2.py --mode live-demo
  
  # AI Autonomous with custom parameters
  python launcher_v2.py --target https://target.com --mode ai-autonomous --depth comprehensive --timeout 3600
        """
    )
    
    # Main arguments
    parser.add_argument(
        '--target', '-t',
        type=str,
        help='Target URL for penetration testing'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['ai-autonomous', 'interactive', 'one-click', 'live-demo'],
        required=True,
        help='Execution mode'
    )
    
    # AI and attack options
    parser.add_argument(
        '--depth',
        choices=['quick', 'standard', 'comprehensive'],
        default='standard',
        help='Attack depth level'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Mission timeout in seconds (default: 1800)'
    )
    
    parser.add_argument(
        '--ai-learning',
        action='store_true',
        help='Enable AI learning mode'
    )
    
    parser.add_argument(
        '--real-time-adaptation',
        action='store_true',
        default=True,
        help='Enable real-time AI adaptation'
    )
    
    parser.add_argument(
        '--parallel-attacks',
        type=int,
        default=3,
        help='Number of parallel attack threads'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'html', 'pdf', 'all'],
        default='all',
        help='Report format'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - minimal output'
    )
    
    return parser


async def main():
    """
    Main launcher function
    """
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = AgentDSLauncher()
    
    # Display banner (unless quiet mode)
    if not args.quiet:
        launcher._display_banner()
    
    # Validate arguments
    if args.mode in ['ai-autonomous', 'one-click'] and not args.target:
        print("❌ Error: Target URL required for attack modes")
        print("Use --target <URL> or try --mode interactive or --mode live-demo")
        sys.exit(1)
    
    # Prepare options
    options = {
        'depth': args.depth,
        'timeout': args.timeout,
        'ai_learning': args.ai_learning,
        'real_time_adaptation': args.real_time_adaptation,
        'parallel_attacks': args.parallel_attacks,
        'output_dir': args.output,
        'report_format': args.format,
        'verbose': args.verbose
    }
    
    # Launch based on mode
    try:
        if args.mode == 'ai-autonomous':
            success = await launcher.launch_ai_autonomous_mode(args.target, options)
        elif args.mode == 'interactive':
            success = await launcher.launch_interactive_mode(args.target)
        elif args.mode == 'one-click':
            success = await launcher.launch_one_click_mode(args.target, options)
        elif args.mode == 'live-demo':
            success = await launcher.launch_live_demo_mode()
        else:
            print(f"❌ Unknown mode: {args.mode}")
            success = False
        
        if success:
            print("\n✅ Agent DS v2.0 execution completed successfully")
        else:
            print("\n❌ Agent DS v2.0 execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Agent DS v2.0 interrupted by user")
        print("👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.error(f"Unexpected error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure event loop compatibility
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run main launcher
    asyncio.run(main())