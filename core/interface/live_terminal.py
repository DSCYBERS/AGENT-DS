"""
Agent DS - Live Terminal Interface
==================================

Metasploit-themed terminal interface with dark theme, neon colors, real-time updates,
interactive commands, and live feedback system for autonomous attack monitoring.

Features:
- Metasploit-style dark theme with neon colors
- AgentDS> prompt with interactive commands
- Real-time attack progress visualization
- Live status updates and intelligence feedback
- Interactive command processing
- Attack monitoring dashboard
- Mission control interface
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Callable, Optional
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich.columns import Columns
from rich import box
import threading
import queue


class LiveTerminalInterface:
    """
    Metasploit-themed live terminal interface for Agent DS
    """
    
    def __init__(self):
        # Console setup with dark theme
        self.console = Console(
            theme={
                'info': 'bright_cyan',
                'warning': 'bright_yellow', 
                'error': 'bright_red',
                'success': 'bright_green',
                'highlight': 'bright_magenta',
                'dim': 'dim white'
            },
            width=120,
            height=40
        )
        
        # Terminal state
        self.is_running = False
        self.current_mission = None
        self.attack_status = {}
        self.ai_thinking_log = []
        self.mission_stats = {
            'attacks_launched': 0,
            'vulnerabilities_found': 0,
            'success_rate': 0.0,
            'time_elapsed': 0
        }
        
        # Message queues for real-time updates
        self.message_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.thinking_queue = queue.Queue()
        
        # Layout components
        self.layout = Layout()
        self.live_display = None
        
        # Command handlers
        self.command_handlers = {
            'help': self._cmd_help,
            'status': self._cmd_status,
            'mission': self._cmd_mission,
            'attacks': self._cmd_attacks,
            'ai': self._cmd_ai,
            'stats': self._cmd_stats,
            'logs': self._cmd_logs,
            'clear': self._cmd_clear,
            'exit': self._cmd_exit,
            'quit': self._cmd_exit
        }
        
        # Initialize layout
        self._setup_layout()
    
    def _setup_layout(self):
        """
        Setup the Metasploit-style terminal layout
        """
        # Main layout with header, body, and footer
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=6)
        )
        
        # Split body into main content and sidebar
        self.layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="sidebar", ratio=1)
        )
        
        # Split sidebar into status and AI thinking
        self.layout["sidebar"].split_column(
            Layout(name="status", ratio=1),
            Layout(name="ai_thinking", ratio=1)
        )
        
        # Update header
        self._update_header()
    
    def _update_header(self):
        """
        Update the terminal header with Agent DS branding
        """
        title = Text()
        title.append("Agent DS", style="bold bright_red")
        title.append(" - ", style="dim white")
        title.append("Next-Gen Autonomous Hacker Agent", style="bright_cyan")
        title.append(" v2.0", style="bright_green")
        
        header_panel = Panel(
            Align.center(title),
            style="bright_white on black",
            box=box.DOUBLE
        )
        
        self.layout["header"].update(header_panel)
    
    def _update_main_content(self) -> Panel:
        """
        Update main content area with mission information
        """
        if not self.current_mission:
            content = Text()
            content.append("💻 Welcome to Agent DS - Autonomous Hacker Agent\n\n", style="bright_cyan")
            content.append("🎯 Ready for autonomous penetration testing\n", style="bright_green")
            content.append("🧠 AI Thinking Model loaded and operational\n", style="bright_magenta")
            content.append("⚡ Live terminal interface active\n\n", style="bright_yellow")
            content.append("Type 'help' for available commands\n", style="dim white")
            content.append("Type 'mission start <target>' to begin\n", style="bright_white")
            
            return Panel(
                content,
                title="[bright_green]Mission Control[/bright_green]",
                border_style="bright_green",
                box=box.ROUNDED
            )
        else:
            # Show active mission details
            mission_table = Table(
                title="🎯 Active Mission",
                show_header=True,
                header_style="bright_cyan",
                border_style="bright_green",
                box=box.ROUNDED
            )
            
            mission_table.add_column("Parameter", style="bright_white")
            mission_table.add_column("Value", style="bright_green")
            
            mission_table.add_row("Target", self.current_mission.get('target', 'Unknown'))
            mission_table.add_row("Status", self.current_mission.get('status', 'Unknown'))
            mission_table.add_row("Phase", self.current_mission.get('current_phase', 'Initialization'))
            mission_table.add_row("Progress", f"{self.current_mission.get('progress', 0):.1f}%")
            mission_table.add_row("Attacks Launched", str(self.mission_stats['attacks_launched']))
            mission_table.add_row("Vulnerabilities", str(self.mission_stats['vulnerabilities_found']))
            mission_table.add_row("Success Rate", f"{self.mission_stats['success_rate']:.1f}%")
            
            return Panel(
                mission_table,
                title="[bright_green]Mission Control[/bright_green]",
                border_style="bright_green",
                box=box.ROUNDED
            )
    
    def _update_status_panel(self) -> Panel:
        """
        Update status panel with real-time attack information
        """
        if not self.attack_status:
            status_text = Text()
            status_text.append("🔍 No active attacks\n", style="dim white")
            status_text.append("⏳ Waiting for mission start", style="bright_yellow")
            
            return Panel(
                status_text,
                title="[bright_yellow]Attack Status[/bright_yellow]",
                border_style="bright_yellow",
                box=box.ROUNDED
            )
        
        # Create attack status table
        status_table = Table(
            show_header=True,
            header_style="bright_yellow",
            border_style="bright_yellow",
            box=box.SIMPLE
        )
        
        status_table.add_column("Attack", width=12)
        status_table.add_column("Status", width=8)
        status_table.add_column("Progress", width=8)
        
        for attack_name, status_info in self.attack_status.items():
            status_style = "bright_green" if status_info.get('success') else "bright_red" if status_info.get('failed') else "bright_yellow"
            progress = status_info.get('progress', 0)
            
            status_table.add_row(
                attack_name[:12],
                status_info.get('status', 'Running')[:8],
                f"{progress:.0f}%",
                style=status_style
            )
        
        return Panel(
            status_table,
            title="[bright_yellow]Attack Status[/bright_yellow]",
            border_style="bright_yellow",
            box=box.ROUNDED
        )
    
    def _update_ai_thinking_panel(self) -> Panel:
        """
        Update AI thinking panel with latest AI insights
        """
        if not self.ai_thinking_log:
            thinking_text = Text()
            thinking_text.append("🧠 AI Thinking Model ready\n", style="bright_magenta")
            thinking_text.append("💭 Waiting for analysis tasks", style="dim white")
            
            return Panel(
                thinking_text,
                title="[bright_magenta]AI Thinking[/bright_magenta]",
                border_style="bright_magenta",
                box=box.ROUNDED
            )
        
        # Show latest AI thinking logs
        thinking_content = Text()
        
        # Show last 5 thinking entries
        for entry in self.ai_thinking_log[-5:]:
            timestamp = entry.get('timestamp', '')
            message = entry.get('message', '')
            thinking_type = entry.get('type', 'info')
            
            if thinking_type == 'analyzing':
                style = "bright_cyan"
                icon = "🔍"
            elif thinking_type == 'adapting':
                style = "bright_yellow"
                icon = "🔄"
            elif thinking_type == 'recommending':
                style = "bright_green"
                icon = "💡"
            elif thinking_type == 'orchestrating':
                style = "bright_magenta"
                icon = "🎼"
            else:
                style = "bright_white"
                icon = "🧠"
            
            thinking_content.append(f"{icon} ", style=style)
            thinking_content.append(f"{message}\n", style="bright_white")
        
        return Panel(
            thinking_content,
            title="[bright_magenta]AI Thinking[/bright_magenta]",
            border_style="bright_magenta",
            box=box.ROUNDED
        )
    
    def _update_footer(self) -> Panel:
        """
        Update footer with command prompt and system info
        """
        footer_content = Text()
        
        # System info line
        current_time = datetime.now().strftime("%H:%M:%S")
        footer_content.append(f"[{current_time}] ", style="dim white")
        footer_content.append("Agent DS Terminal", style="bright_cyan")
        footer_content.append(" | ", style="dim white")
        footer_content.append("AI Status: ", style="dim white")
        footer_content.append("ACTIVE", style="bright_green")
        footer_content.append(" | ", style="dim white")
        footer_content.append("Connection: ", style="dim white")
        footer_content.append("SECURE", style="bright_green")
        footer_content.append("\n\n", style="dim white")
        
        # Command prompt
        footer_content.append("AgentDS", style="bright_red")
        footer_content.append("> ", style="bright_white")
        footer_content.append("█", style="bright_green")  # Cursor
        
        return Panel(
            footer_content,
            title="[bright_cyan]Command Interface[/bright_cyan]",
            border_style="bright_cyan",
            box=box.ROUNDED
        )
    
    async def start_interface(self):
        """
        Start the live terminal interface
        """
        self.is_running = True
        
        # Welcome message
        self.console.print()
        self.console.print("╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗", style="bright_red")
        self.console.print("║                                           AGENT DS v2.0                                                         ║", style="bright_red")
        self.console.print("║                                    Next-Gen Autonomous Hacker Agent                                             ║", style="bright_cyan")
        self.console.print("║                                     AI Thinking Model: OPERATIONAL                                               ║", style="bright_green")
        self.console.print("╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝", style="bright_red")
        self.console.print()
        
        # Start live display
        with Live(self._generate_layout(), refresh_per_second=2, console=self.console) as live:
            self.live_display = live
            
            # Start message processing thread
            message_thread = threading.Thread(target=self._process_messages, daemon=True)
            message_thread.start()
            
            # Main command loop
            await self._command_loop()
    
    def _generate_layout(self):
        """
        Generate the complete terminal layout
        """
        # Update all panels
        self.layout["main"].update(self._update_main_content())
        self.layout["status"].update(self._update_status_panel())
        self.layout["ai_thinking"].update(self._update_ai_thinking_panel())
        self.layout["footer"].update(self._update_footer())
        
        return self.layout
    
    async def _command_loop(self):
        """
        Main command processing loop
        """
        while self.is_running:
            try:
                # In a real implementation, this would handle user input
                # For demo purposes, we'll simulate some activity
                await asyncio.sleep(1)
                
                # Update layout
                if self.live_display:
                    self.live_display.update(self._generate_layout())
                
            except KeyboardInterrupt:
                break
    
    def _process_messages(self):
        """
        Process incoming messages from attack modules
        """
        while self.is_running:
            try:
                # Process AI thinking messages
                while not self.thinking_queue.empty():
                    thinking_msg = self.thinking_queue.get_nowait()
                    self.ai_thinking_log.append({
                        'timestamp': datetime.now().isoformat(),
                        'message': thinking_msg.get('message', ''),
                        'type': thinking_msg.get('type', 'info')
                    })
                    
                    # Keep only last 20 entries
                    if len(self.ai_thinking_log) > 20:
                        self.ai_thinking_log = self.ai_thinking_log[-20:]
                
                # Process status updates
                while not self.status_queue.empty():
                    status_msg = self.status_queue.get_nowait()
                    attack_name = status_msg.get('attack_name', 'unknown')
                    self.attack_status[attack_name] = status_msg
                
                # Process general messages
                while not self.message_queue.empty():
                    msg = self.message_queue.get_nowait()
                    # Handle general messages
                    pass
                
                time.sleep(0.1)
                
            except Exception as e:
                # Error handling
                pass
    
    # Terminal callback methods for AI core integration
    async def ai_thinking_callback(self, thinking_type: str, message: str):
        """
        Callback for AI thinking updates
        """
        self.thinking_queue.put({
            'type': thinking_type,
            'message': message
        })
    
    async def attack_status_callback(self, attack_name: str, status_info: Dict[str, Any]):
        """
        Callback for attack status updates
        """
        self.status_queue.put({
            'attack_name': attack_name,
            **status_info
        })
    
    async def mission_update_callback(self, mission_info: Dict[str, Any]):
        """
        Callback for mission updates
        """
        self.current_mission = mission_info
        
        # Update mission stats
        if 'attacks_launched' in mission_info:
            self.mission_stats['attacks_launched'] = mission_info['attacks_launched']
        if 'vulnerabilities_found' in mission_info:
            self.mission_stats['vulnerabilities_found'] = mission_info['vulnerabilities_found']
        if 'success_rate' in mission_info:
            self.mission_stats['success_rate'] = mission_info['success_rate'] * 100
    
    # Command handlers
    def _cmd_help(self, args: List[str]) -> str:
        """Display help information"""
        help_text = """
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                              AGENT DS COMMANDS                                                  ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║ mission start <target>  - Start autonomous penetration testing mission                                          ║
║ mission stop            - Stop current mission                                                                  ║
║ mission status          - Show detailed mission status                                                          ║
║ attacks list            - List all active attack vectors                                                        ║
║ attacks stop <name>     - Stop specific attack vector                                                           ║
║ ai status               - Show AI thinking model status                                                         ║
║ ai insights             - Display latest AI insights and recommendations                                        ║
║ stats                   - Show mission statistics and performance metrics                                       ║
║ logs view               - View attack logs and detailed results                                                 ║
║ logs export             - Export logs and reports                                                               ║
║ status                  - Show overall system status                                                            ║
║ clear                   - Clear terminal screen                                                                 ║
║ help                    - Show this help message                                                                ║
║ exit/quit               - Exit Agent DS terminal                                                                ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        return help_text
    
    def _cmd_status(self, args: List[str]) -> str:
        """Show system status"""
        status_info = f"""
🔥 AGENT DS SYSTEM STATUS 🔥

🧠 AI Thinking Model: OPERATIONAL
⚡ Live Terminal Interface: ACTIVE
🎯 Mission Status: {'ACTIVE' if self.current_mission else 'READY'}
📊 Attacks Monitored: {len(self.attack_status)}
🔍 Vulnerabilities Found: {self.mission_stats['vulnerabilities_found']}
📈 Success Rate: {self.mission_stats['success_rate']:.1f}%

System ready for autonomous operations.
        """
        return status_info
    
    def _cmd_mission(self, args: List[str]) -> str:
        """Handle mission commands"""
        if not args:
            return "Usage: mission [start|stop|status] [target]"
        
        subcmd = args[0].lower()
        
        if subcmd == "start":
            if len(args) < 2:
                return "Usage: mission start <target_url>"
            
            target = args[1]
            self.current_mission = {
                'target': target,
                'status': 'Starting',
                'current_phase': 'Initialization',
                'progress': 0.0,
                'start_time': datetime.now().isoformat()
            }
            
            return f"🎯 Mission started against target: {target}\n🚀 Initializing AI thinking model and attack vectors..."
        
        elif subcmd == "stop":
            if self.current_mission:
                target = self.current_mission.get('target', 'Unknown')
                self.current_mission = None
                self.attack_status.clear()
                return f"🛑 Mission against {target} stopped."
            else:
                return "⚠️ No active mission to stop."
        
        elif subcmd == "status":
            if self.current_mission:
                return f"""
🎯 MISSION STATUS:
   Target: {self.current_mission.get('target', 'Unknown')}
   Status: {self.current_mission.get('status', 'Unknown')}
   Phase: {self.current_mission.get('current_phase', 'Unknown')}
   Progress: {self.current_mission.get('progress', 0):.1f}%
   Duration: {self._calculate_mission_duration()}
                """
            else:
                return "ℹ️ No active mission."
        
        return "Unknown mission command."
    
    def _cmd_attacks(self, args: List[str]) -> str:
        """Handle attack commands"""
        if not args:
            return "Usage: attacks [list|stop] [attack_name]"
        
        subcmd = args[0].lower()
        
        if subcmd == "list":
            if not self.attack_status:
                return "📋 No active attacks."
            
            attack_list = "🔥 ACTIVE ATTACKS:\n\n"
            for attack_name, status in self.attack_status.items():
                attack_list += f"   {attack_name}: {status.get('status', 'Unknown')} ({status.get('progress', 0):.0f}%)\n"
            
            return attack_list
        
        elif subcmd == "stop":
            if len(args) < 2:
                return "Usage: attacks stop <attack_name>"
            
            attack_name = args[1]
            if attack_name in self.attack_status:
                del self.attack_status[attack_name]
                return f"🛑 Attack '{attack_name}' stopped."
            else:
                return f"⚠️ Attack '{attack_name}' not found."
        
        return "Unknown attack command."
    
    def _cmd_ai(self, args: List[str]) -> str:
        """Handle AI commands"""
        if not args:
            return "Usage: ai [status|insights]"
        
        subcmd = args[0].lower()
        
        if subcmd == "status":
            return f"""
🧠 AI THINKING MODEL STATUS:

   Model State: OPERATIONAL
   Thinking Processes: {len(self.ai_thinking_log)} logged
   Learning Mode: ACTIVE
   Adaptation Count: {len(self.attack_status)}
   
   Latest AI Activity:
   {self.ai_thinking_log[-1].get('message', 'No recent activity') if self.ai_thinking_log else 'No activity logged'}
            """
        
        elif subcmd == "insights":
            insights = "🧠 AI INSIGHTS & RECOMMENDATIONS:\n\n"
            
            if self.ai_thinking_log:
                insights += "Recent AI Analysis:\n"
                for entry in self.ai_thinking_log[-3:]:
                    insights += f"   • {entry.get('message', '')}\n"
            else:
                insights += "   No insights available yet.\n   Start a mission to see AI analysis."
            
            return insights
        
        return "Unknown AI command."
    
    def _cmd_stats(self, args: List[str]) -> str:
        """Show mission statistics"""
        return f"""
📊 MISSION STATISTICS:

   🎯 Attacks Launched: {self.mission_stats['attacks_launched']}
   🔍 Vulnerabilities Found: {self.mission_stats['vulnerabilities_found']}
   📈 Success Rate: {self.mission_stats['success_rate']:.1f}%
   ⏱️ Time Elapsed: {self._calculate_mission_duration()}
   
   🤖 AI Performance:
   • Thinking Processes: {len(self.ai_thinking_log)}
   • Active Attacks: {len(self.attack_status)}
   • System Status: OPERATIONAL
        """
    
    def _cmd_logs(self, args: List[str]) -> str:
        """Handle log commands"""
        return "📝 Log viewing and export features will be implemented in the full version."
    
    def _cmd_clear(self, args: List[str]) -> str:
        """Clear terminal"""
        self.console.clear()
        return ""
    
    def _cmd_exit(self, args: List[str]) -> str:
        """Exit terminal"""
        self.is_running = False
        return "👋 Agent DS terminal shutting down..."
    
    def _calculate_mission_duration(self) -> str:
        """Calculate mission duration"""
        if not self.current_mission or 'start_time' not in self.current_mission:
            return "00:00:00"
        
        start_time = datetime.fromisoformat(self.current_mission['start_time'])
        duration = datetime.now() - start_time
        
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    # Simulation methods for demo
    async def simulate_mission_activity(self):
        """
        Simulate mission activity for demonstration
        """
        if not self.current_mission:
            return
        
        # Simulate AI thinking
        ai_messages = [
            {"type": "analyzing", "message": "Analyzing target attack surface..."},
            {"type": "orchestrating", "message": "Optimizing attack sequence based on target profile"},
            {"type": "adapting", "message": "Adapting payload strategies for detected defenses"},
            {"type": "recommending", "message": "Recommending high-probability attack vectors"}
        ]
        
        for msg in ai_messages:
            await self.ai_thinking_callback(msg["type"], msg["message"])
            await asyncio.sleep(2)
        
        # Simulate attack progress
        attacks = ["reconnaissance", "web_scanning", "sql_injection", "xss_testing", "admin_bruteforce"]
        
        for i, attack in enumerate(attacks):
            for progress in range(0, 101, 20):
                await self.attack_status_callback(attack, {
                    'status': 'Running',
                    'progress': progress,
                    'success': progress == 100
                })
                await asyncio.sleep(1)
            
            # Update mission stats
            self.mission_stats['attacks_launched'] += 1
            if i % 2 == 0:  # Simulate 50% success rate
                self.mission_stats['vulnerabilities_found'] += 1
            
            self.mission_stats['success_rate'] = self.mission_stats['vulnerabilities_found'] / self.mission_stats['attacks_launched']
            
            await self.mission_update_callback({
                'progress': ((i + 1) / len(attacks)) * 100,
                'current_phase': f"Attack Phase {i + 1}/5",
                'attacks_launched': self.mission_stats['attacks_launched'],
                'vulnerabilities_found': self.mission_stats['vulnerabilities_found'],
                'success_rate': self.mission_stats['success_rate']
            })


# Demo function
async def main():
    """
    Demo the live terminal interface
    """
    terminal = LiveTerminalInterface()
    
    # Start the interface
    interface_task = asyncio.create_task(terminal.start_interface())
    
    # Start simulation after a delay
    await asyncio.sleep(3)
    
    # Simulate a mission
    await terminal.mission_update_callback({
        'target': 'https://demo.testfire.net',
        'status': 'Active',
        'current_phase': 'AI Analysis',
        'progress': 15.0
    })
    
    simulation_task = asyncio.create_task(terminal.simulate_mission_activity())
    
    # Wait for both tasks
    await asyncio.gather(interface_task, simulation_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Agent DS Terminal shutting down...")