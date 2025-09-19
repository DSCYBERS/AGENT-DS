"""
Terminal Interface for Agent DS v2.0
Provides interactive terminal-based attack execution
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich.align import Align
    from rich.layout import Layout
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .terminal_theme import HackerTerminalTheme

class TerminalInterface:
    """
    Interactive terminal interface for Agent DS attacks
    """
    
    def __init__(self):
        self.theme = HackerTerminalTheme()
        self.console = Console() if RICH_AVAILABLE else None
        self.current_target = None
        self.attack_phase = "IDLE"
        self.is_active = False
        
        # Attack results storage
        self.attack_results = {
            'reconnaissance': {},
            'web_attacks': {},
            'database': {},
            'admin_login': {},
            'file_upload': {},
            'privilege_escalation': {}
        }
        
        # Statistics
        self.vulnerabilities_found = 0
        self.exploits_successful = 0
        self.attack_duration = 0
        self.start_time = None
    
    async def initialize_interface(self):
        """Initialize the terminal interface"""
        self.theme.clear_screen()
        self.theme.print_banner('agent_ds', 'bright_green', 'glitch')
        
        if RICH_AVAILABLE:
            welcome_panel = Panel(
                """[bold bright_cyan]Welcome to Agent DS v2.0 Terminal Interface[/bold bright_cyan]

[bright_white]An advanced autonomous penetration testing framework[/bright_white]
[bright_yellow]⚠ Use only on systems you own or have explicit permission to test ⚠[/bright_yellow]

[bright_green]Commands available:[/bright_green]
• [bold]quick_scan[/bold] - Quick vulnerability assessment
• [bold]full_attack[/bold] - Comprehensive penetration test
• [bold]recon[/bold] - Reconnaissance only
• [bold]status[/bold] - View current status
• [bold]help[/bold] - Show available commands
• [bold]exit[/bold] - Exit Agent DS
""",
                title="[bold bright_cyan]🚀 AGENT DS TERMINAL 🚀[/bold bright_cyan]",
                border_style="bright_green",
                padding=(1, 2)
            )
            self.console.print(welcome_panel)
        else:
            print("""
Agent DS v2.0 Terminal Interface
Advanced Autonomous Penetration Testing Framework

Commands:
- quick_scan: Quick vulnerability assessment
- full_attack: Comprehensive penetration test
- recon: Reconnaissance only
- status: View current status
- help: Show available commands
- exit: Exit Agent DS
""")
    
    async def show_target_selection(self):
        """Show target selection interface"""
        if RICH_AVAILABLE:
            target_panel = Panel(
                """[bold bright_yellow]🎯 TARGET SELECTION 🎯[/bold bright_yellow]

[bright_cyan]Enter target information:[/bright_cyan]
• [bold]URL[/bold]: Website URL (e.g., https://example.com)
• [bold]IP[/bold]: IP address (e.g., 192.168.1.1)
• [bold]Domain[/bold]: Domain name (e.g., example.com)

[bright_red]⚠ WARNING: Only test targets you own or have permission to test![/bright_red]
""",
                border_style="bright_yellow",
                padding=(1, 2)
            )
            self.console.print(target_panel)
        
        target = input("\n🎯 Enter target: ").strip()
        
        if target:
            self.current_target = target
            self.theme.show_success_message(f"Target set: {target}")
            return target
        else:
            self.theme.show_error_message("No target specified")
            return None
    
    async def execute_quick_scan(self, target: str):
        """Execute quick vulnerability scan"""
        self.attack_phase = "QUICK_SCAN"
        self.start_time = time.time()
        
        self.theme.print_banner('recon', 'bright_cyan', 'typewriter')
        
        # Simulated quick scan phases
        scan_phases = [
            ("Port Scanning", "Scanning common ports", 3),
            ("Service Detection", "Identifying running services", 2),
            ("Web Technologies", "Detecting web technologies", 2),
            ("Quick Vuln Check", "Basic vulnerability assessment", 4)
        ]
        
        for phase_name, phase_desc, duration in scan_phases:
            await self.theme.animated_progress(
                f"{phase_name}: {phase_desc}",
                duration,
                style="hacker"
            )
            
            # Simulate findings
            await self._simulate_scan_results(phase_name.lower().replace(' ', '_'))
        
        # Show results summary
        await self._show_scan_summary()
    
    async def execute_full_attack(self, target: str):
        """Execute comprehensive penetration test"""
        self.attack_phase = "FULL_ATTACK"
        self.start_time = time.time()
        
        self.theme.print_banner('one_click', 'bright_red', 'glitch')
        
        # Comprehensive attack phases
        attack_phases = [
            ("Reconnaissance", "Gathering target intelligence", 5),
            ("Port Scanning", "Comprehensive port scan", 4),
            ("Service Enumeration", "Detailed service analysis", 3),
            ("Web Attack Testing", "Testing web vulnerabilities", 6),
            ("Database Testing", "Database vulnerability assessment", 4),
            ("Admin Portal Discovery", "Finding admin interfaces", 3),
            ("File Upload Testing", "Testing file upload vulnerabilities", 4),
            ("Privilege Escalation", "Testing privilege escalation", 5)
        ]
        
        for phase_name, phase_desc, duration in attack_phases:
            await self.theme.animated_progress(
                f"{phase_name}: {phase_desc}",
                duration,
                style="matrix"
            )
            
            # Simulate attack results
            await self._simulate_attack_results(phase_name.lower().replace(' ', '_'))
            
            # Show live status
            await self._update_live_status()
        
        # Show final results
        await self._show_attack_summary()
    
    async def execute_reconnaissance(self, target: str):
        """Execute reconnaissance only"""
        self.attack_phase = "RECONNAISSANCE"
        self.start_time = time.time()
        
        self.theme.print_banner('recon', 'bright_green', 'matrix')
        
        # Reconnaissance phases
        recon_phases = [
            ("Subdomain Discovery", "Finding subdomains", 4),
            ("DNS Enumeration", "DNS record analysis", 3),
            ("Technology Detection", "Web technology fingerprinting", 2),
            ("Social Media OSINT", "Social media reconnaissance", 3),
            ("Email Harvesting", "Finding email addresses", 2)
        ]
        
        for phase_name, phase_desc, duration in recon_phases:
            await self.theme.animated_progress(
                f"{phase_name}: {phase_desc}",
                duration,
                style="neon"
            )
            
            # Simulate recon results
            await self._simulate_recon_results(phase_name.lower().replace(' ', '_'))
        
        # Show recon summary
        await self._show_recon_summary()
    
    async def _simulate_scan_results(self, phase: str):
        """Simulate scan results for different phases"""
        if phase == "port_scanning":
            self.attack_results['reconnaissance']['ports'] = [
                {'port': 22, 'service': 'ssh', 'status': 'open'},
                {'port': 80, 'service': 'http', 'status': 'open'},
                {'port': 443, 'service': 'https', 'status': 'open'},
                {'port': 3306, 'service': 'mysql', 'status': 'open'}
            ]
            
        elif phase == "service_detection":
            self.attack_results['reconnaissance']['services'] = [
                {'service': 'Apache 2.4.41', 'version': '2.4.41', 'vulnerabilities': ['CVE-2021-41773']},
                {'service': 'MySQL', 'version': '8.0.25', 'vulnerabilities': []},
                {'service': 'OpenSSH', 'version': '7.4', 'vulnerabilities': ['CVE-2018-15473']}
            ]
            
        elif phase == "web_technologies":
            self.attack_results['reconnaissance']['technologies'] = [
                'PHP 7.4.21', 'WordPress 5.8', 'jQuery 3.6.0', 'Bootstrap 4.6'
            ]
            
        elif phase == "quick_vuln_check":
            vulnerabilities = [
                {
                    'type': 'SQL Injection',
                    'severity': 'HIGH',
                    'description': 'Potential SQL injection in login form',
                    'confidence': 'medium'
                },
                {
                    'type': 'XSS',
                    'severity': 'MEDIUM',
                    'description': 'Reflected XSS in search parameter',
                    'confidence': 'high'
                }
            ]
            self.attack_results['web_attacks']['vulnerabilities'] = vulnerabilities
            self.vulnerabilities_found += len(vulnerabilities)
    
    async def _simulate_attack_results(self, phase: str):
        """Simulate attack results for comprehensive testing"""
        success_rate = random.uniform(0.3, 0.8)  # Simulate varying success rates
        
        if phase == "reconnaissance":
            self.attack_results['reconnaissance'] = {
                'subdomains': ['www', 'mail', 'ftp', 'admin'],
                'emails': ['admin@target.com', 'info@target.com'],
                'technologies': ['Apache', 'PHP', 'MySQL', 'WordPress'],
                'dns_records': ['A', 'MX', 'TXT', 'CNAME']
            }
            
        elif phase == "web_attack_testing":
            vulnerabilities = []
            if random.random() < success_rate:
                vulnerabilities.append({
                    'type': 'SQL Injection',
                    'severity': 'CRITICAL',
                    'description': 'Time-based SQL injection in user parameter',
                    'confidence': 'high'
                })
                self.exploits_successful += 1
            
            if random.random() < success_rate:
                vulnerabilities.append({
                    'type': 'XSS',
                    'severity': 'HIGH',
                    'description': 'Stored XSS in comment field',
                    'confidence': 'high'
                })
                self.exploits_successful += 1
            
            self.attack_results['web_attacks'] = {
                'vulnerabilities': vulnerabilities,
                'vulnerable': len(vulnerabilities) > 0
            }
            self.vulnerabilities_found += len(vulnerabilities)
            
        elif phase == "database_testing":
            if random.random() < success_rate:
                self.attack_results['database'] = {
                    'fingerprint': {'database_type': 'MySQL', 'version': '8.0.25'},
                    'tables': ['users', 'admin', 'products', 'orders'],
                    'vulnerable': True
                }
                self.exploits_successful += 1
            else:
                self.attack_results['database'] = {'vulnerable': False}
        
        elif phase == "admin_portal_discovery":
            if random.random() < success_rate:
                self.attack_results['admin_login'] = {
                    'portals': ['/admin', '/administrator', '/wp-admin'],
                    'credentials': {'admin': 'admin123'},
                    'vulnerable': True
                }
                self.exploits_successful += 1
    
    async def _simulate_recon_results(self, phase: str):
        """Simulate reconnaissance results"""
        if phase == "subdomain_discovery":
            self.attack_results['reconnaissance']['subdomains'] = [
                'www.target.com', 'mail.target.com', 'ftp.target.com',
                'admin.target.com', 'dev.target.com'
            ]
            
        elif phase == "dns_enumeration":
            self.attack_results['reconnaissance']['dns'] = {
                'A_records': ['192.168.1.100'],
                'MX_records': ['mail.target.com'],
                'TXT_records': ['v=spf1 include:_spf.google.com ~all']
            }
            
        elif phase == "technology_detection":
            self.attack_results['reconnaissance']['technologies'] = [
                'Apache/2.4.41', 'PHP/7.4.21', 'WordPress/5.8',
                'jQuery/3.6.0', 'Bootstrap/4.6'
            ]
    
    async def _update_live_status(self):
        """Update live status during attacks"""
        if RICH_AVAILABLE:
            status_panel = self.theme.create_status_panel(
                target=self.current_target,
                phase=self.attack_phase,
                vulnerabilities=self.vulnerabilities_found,
                exploited=self.exploits_successful
            )
            self.console.print(status_panel)
    
    async def _show_scan_summary(self):
        """Show quick scan summary"""
        self.attack_duration = time.time() - self.start_time
        
        if RICH_AVAILABLE:
            # Create vulnerability table
            vulnerabilities = self.attack_results.get('web_attacks', {}).get('vulnerabilities', [])
            if vulnerabilities:
                vuln_table = self.theme.create_vulnerability_table(vulnerabilities)
                self.console.print(vuln_table)
            
            # Summary panel
            summary_text = f"""
[bold bright_cyan]QUICK SCAN COMPLETED[/bold bright_cyan]

[bright_white]Target:[/bright_white] [bright_yellow]{self.current_target}[/bright_yellow]
[bright_white]Duration:[/bright_white] [bright_green]{self.attack_duration:.1f} seconds[/bright_green]
[bright_white]Vulnerabilities Found:[/bright_white] [bright_red]{self.vulnerabilities_found}[/bright_red]

[bright_cyan]Recommendations:[/bright_cyan]
• Run full attack for comprehensive testing
• Review identified vulnerabilities
• Implement security patches
"""
            
            summary_panel = Panel(
                summary_text,
                title="[bold bright_green]📊 SCAN SUMMARY 📊[/bold bright_green]",
                border_style="bright_green",
                padding=(1, 2)
            )
            self.console.print(summary_panel)
        
        if self.vulnerabilities_found > 0:
            self.theme.print_banner('warning', 'bright_yellow')
        else:
            self.theme.show_success_message("No critical vulnerabilities found in quick scan")
    
    async def _show_attack_summary(self):
        """Show comprehensive attack summary"""
        self.attack_duration = time.time() - self.start_time
        
        # Show attack tree
        if RICH_AVAILABLE:
            attack_tree = self.theme.create_attack_tree(self.attack_results)
            self.console.print(attack_tree)
        
        # Show completion summary
        self.theme.show_completion_summary(self.attack_results)
        
        # Final banner based on results
        if self.exploits_successful > 0:
            self.theme.print_banner('success', 'bright_red', 'glitch')
        else:
            self.theme.print_banner('warning', 'bright_yellow')
    
    async def _show_recon_summary(self):
        """Show reconnaissance summary"""
        self.attack_duration = time.time() - self.start_time
        
        if RICH_AVAILABLE:
            recon_data = self.attack_results.get('reconnaissance', {})
            
            # Create recon table
            recon_table = Table(
                title="[bold bright_cyan]🔍 RECONNAISSANCE RESULTS 🔍[/bold bright_cyan]",
                show_header=True,
                header_style="bold bright_yellow",
                border_style="bright_green"
            )
            
            recon_table.add_column("Category", style="bright_cyan", width=20)
            recon_table.add_column("Found", style="bright_white", width=10)
            recon_table.add_column("Details", style="bright_green", width=50)
            
            # Add data to table
            for category, data in recon_data.items():
                if isinstance(data, list):
                    count = len(data)
                    details = ", ".join(data[:3])  # Show first 3 items
                    if len(data) > 3:
                        details += "..."
                elif isinstance(data, dict):
                    count = len(data)
                    details = ", ".join(list(data.keys())[:3])
                    if len(data) > 3:
                        details += "..."
                else:
                    count = 1
                    details = str(data)[:50]
                
                recon_table.add_row(
                    category.replace('_', ' ').title(),
                    str(count),
                    details
                )
            
            self.console.print(recon_table)
        
        self.theme.show_success_message(
            f"Reconnaissance completed in {self.attack_duration:.1f} seconds",
            [f"Gathered intelligence on {self.current_target}"]
        )
    
    async def show_status(self):
        """Show current status"""
        if RICH_AVAILABLE:
            status_text = f"""
[bold bright_cyan]AGENT DS STATUS[/bold bright_cyan]

[bright_white]Current Target:[/bright_white] [bright_yellow]{self.current_target or 'None'}[/bright_yellow]
[bright_white]Attack Phase:[/bright_white] [bright_green]{self.attack_phase}[/bright_green]
[bright_white]Active:[/bright_white] [bright_red]{'Yes' if self.is_active else 'No'}[/bright_red]
[bright_white]Vulnerabilities Found:[/bright_white] [bright_red]{self.vulnerabilities_found}[/bright_red]
[bright_white]Successful Exploits:[/bright_white] [bright_magenta]{self.exploits_successful}[/bright_magenta]

[bright_cyan]System Time:[/bright_cyan] [bright_white]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/bright_white]
"""
            
            if self.start_time:
                elapsed = time.time() - self.start_time
                status_text += f"[bright_white]Elapsed Time:[/bright_white] [bright_green]{elapsed:.1f} seconds[/bright_green]\n"
            
            status_panel = Panel(
                status_text,
                title="[bold bright_green]📊 SYSTEM STATUS 📊[/bold bright_green]",
                border_style="bright_green",
                padding=(1, 2)
            )
            self.console.print(status_panel)
        else:
            print(f"""
Agent DS Status:
- Target: {self.current_target or 'None'}
- Phase: {self.attack_phase}
- Vulnerabilities: {self.vulnerabilities_found}
- Exploits: {self.exploits_successful}
""")
    
    async def show_help(self):
        """Show help information"""
        if RICH_AVAILABLE:
            help_text = """
[bold bright_cyan]AGENT DS v2.0 COMMANDS[/bold bright_cyan]

[bold bright_yellow]Attack Commands:[/bold bright_yellow]
• [bold]quick_scan <target>[/bold] - Quick vulnerability assessment
• [bold]full_attack <target>[/bold] - Comprehensive penetration test
• [bold]recon <target>[/bold] - Reconnaissance only

[bold bright_yellow]Information Commands:[/bold bright_yellow]
• [bold]status[/bold] - Show current system status
• [bold]results[/bold] - Show last attack results
• [bold]target[/bold] - Show current target information

[bold bright_yellow]System Commands:[/bold bright_yellow]
• [bold]clear[/bold] - Clear the terminal screen
• [bold]help[/bold] - Show this help message
• [bold]exit[/bold] - Exit Agent DS

[bold bright_red]⚠ LEGAL WARNING ⚠[/bold bright_red]
[bright_yellow]Only use Agent DS on systems you own or have explicit written permission to test.
Unauthorized access to computer systems is illegal and punishable by law.[/bright_yellow]

[bold bright_green]Examples:[/bold bright_green]
• quick_scan https://example.com
• full_attack 192.168.1.100
• recon example.com
"""
            
            help_panel = Panel(
                help_text,
                title="[bold bright_cyan]🚀 AGENT DS HELP 🚀[/bold bright_cyan]",
                border_style="bright_cyan",
                padding=(1, 2)
            )
            self.console.print(help_panel)
        else:
            print("""
Agent DS v2.0 Commands:

Attack Commands:
- quick_scan <target>: Quick vulnerability assessment
- full_attack <target>: Comprehensive penetration test  
- recon <target>: Reconnaissance only

Information Commands:
- status: Show current system status
- help: Show this help message
- exit: Exit Agent DS

⚠ LEGAL WARNING ⚠
Only use on systems you own or have permission to test!
""")
    
    def reset_session(self):
        """Reset the current session"""
        self.current_target = None
        self.attack_phase = "IDLE"
        self.is_active = False
        self.vulnerabilities_found = 0
        self.exploits_successful = 0
        self.attack_duration = 0
        self.start_time = None
        
        # Clear results
        for key in self.attack_results:
            self.attack_results[key] = {}
        
        self.theme.show_success_message("Session reset successfully")

# Main interface function for integration
async def run_terminal_interface():
    """Main function to run the terminal interface"""
    interface = TerminalInterface()
    await interface.initialize_interface()
    
    while True:
        try:
            if RICH_AVAILABLE:
                command = input("\n[Agent DS] >>> ").strip().lower()
            else:
                command = input("\n[Agent DS] >>> ").strip().lower()
            
            if not command:
                continue
            
            parts = command.split()
            cmd = parts[0]
            
            if cmd == "exit" or cmd == "quit":
                interface.theme.show_success_message("Goodbye! Stay safe and hack ethically!")
                break
            
            elif cmd == "clear":
                interface.theme.clear_screen()
                await interface.initialize_interface()
            
            elif cmd == "help":
                await interface.show_help()
            
            elif cmd == "status":
                await interface.show_status()
            
            elif cmd == "reset":
                interface.reset_session()
            
            elif cmd == "quick_scan":
                if len(parts) > 1:
                    target = parts[1]
                else:
                    target = await interface.show_target_selection()
                
                if target:
                    interface.is_active = True
                    await interface.execute_quick_scan(target)
                    interface.is_active = False
            
            elif cmd == "full_attack":
                if len(parts) > 1:
                    target = parts[1]
                else:
                    target = await interface.show_target_selection()
                
                if target:
                    interface.is_active = True
                    await interface.execute_full_attack(target)
                    interface.is_active = False
            
            elif cmd == "recon":
                if len(parts) > 1:
                    target = parts[1]
                else:
                    target = await interface.show_target_selection()
                
                if target:
                    interface.is_active = True
                    await interface.execute_reconnaissance(target)
                    interface.is_active = False
            
            else:
                interface.theme.show_error_message(
                    f"Unknown command: {cmd}",
                    ["Type 'help' for available commands"]
                )
        
        except KeyboardInterrupt:
            interface.theme.show_warning_message("Operation interrupted by user")
            interface.is_active = False
        
        except Exception as e:
            interface.theme.show_error_message(
                f"An error occurred: {str(e)}",
                ["Please try again or contact support"]
            )
            interface.is_active = False

if __name__ == "__main__":
    asyncio.run(run_terminal_interface())