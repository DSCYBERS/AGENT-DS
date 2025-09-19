#!/usr/bin/env python3
"""
Agent DS - CLI Integration for One-Click Attack
Command-line interface integration for the autonomous hacker agent

This module integrates the One-Click Attack system into the Agent DS CLI including:
- Command: agent-ds attack --target <URL> --one-click
- Support for all command-line flags and options
- Integration with existing Agent DS architecture
- Comprehensive help and usage information

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import argparse
import asyncio
import sys
import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

# Import our attack modules
try:
    from .one_click import OneClickAttackOrchestrator
    from .terminal_theme import HackerTerminalTheme
    from .ai_core import AIAdaptiveCore
except ImportError:
    # Fallback for direct execution
    from one_click import OneClickAttackOrchestrator
    from terminal_theme import HackerTerminalTheme
    from ai_core import AIAdaptiveCore

class AgentDSCLI:
    """
    Agent DS Command Line Interface with One-Click Attack integration
    """
    
    def __init__(self):
        self.theme = HackerTerminalTheme()
        self.version = "2.0"
        self.banner_shown = False
        
        # CLI configuration
        self.commands = {
            'attack': self.attack_command,
            'scan': self.scan_command,
            'exploit': self.exploit_command,
            'report': self.report_command,
            'config': self.config_command,
            'version': self.version_command,
            'help': self.help_command
        }
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser"""
        
        parser = argparse.ArgumentParser(
            prog='agent-ds',
            description='Agent DS - Next-Generation Autonomous Hacker Agent',
            epilog='For more information, visit: https://github.com/agent-ds/core',
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Version
        parser.add_argument(
            '--version', '-v',
            action='version',
            version=f'Agent DS v{self.version}'
        )
        
        # Global options
        parser.add_argument(
            '--verbose', '-V',
            action='store_true',
            help='Enable verbose output'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='Suppress output (conflicts with --verbose)'
        )
        
        parser.add_argument(
            '--config-file', '-c',
            type=str,
            default='./config/agent_ds.json',
            help='Path to configuration file'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Attack command
        attack_parser = subparsers.add_parser(
            'attack',
            help='Launch attacks against targets',
            description='Autonomous attack capabilities with various modes'
        )
        
        self._create_attack_parser(attack_parser)
        
        # Scan command
        scan_parser = subparsers.add_parser(
            'scan',
            help='Reconnaissance and scanning operations',
            description='Advanced reconnaissance capabilities'
        )
        
        self._create_scan_parser(scan_parser)
        
        # Exploit command
        exploit_parser = subparsers.add_parser(
            'exploit',
            help='Exploit specific vulnerabilities',
            description='Targeted exploitation capabilities'
        )
        
        self._create_exploit_parser(exploit_parser)
        
        # Report command
        report_parser = subparsers.add_parser(
            'report',
            help='Generate and manage reports',
            description='Comprehensive reporting system'
        )
        
        self._create_report_parser(report_parser)
        
        # Config command
        config_parser = subparsers.add_parser(
            'config',
            help='Configuration management',
            description='Manage Agent DS configuration'
        )
        
        self._create_config_parser(config_parser)
        
        return parser
    
    def _create_attack_parser(self, parser: argparse.ArgumentParser):
        """Create attack command parser"""
        
        # Target specification
        parser.add_argument(
            '--target', '-t',
            type=str,
            required=True,
            help='Target URL or IP address'
        )
        
        # Attack modes
        attack_mode = parser.add_mutually_exclusive_group(required=True)
        
        attack_mode.add_argument(
            '--one-click',
            action='store_true',
            help='Execute comprehensive one-click attack sequence'
        )
        
        attack_mode.add_argument(
            '--web-only',
            action='store_true',
            help='Focus on web application attacks only'
        )
        
        attack_mode.add_argument(
            '--network-only',
            action='store_true',
            help='Focus on network-level attacks only'
        )
        
        attack_mode.add_argument(
            '--custom',
            type=str,
            help='Custom attack sequence (comma-separated: recon,web,db,admin)'
        )
        
        # Attack options
        parser.add_argument(
            '--safe-mode',
            action='store_true',
            help='Enable safe mode (less aggressive attacks)'
        )
        
        parser.add_argument(
            '--stealth',
            action='store_true',
            help='Enable stealth mode (slower but less detectable)'
        )
        
        parser.add_argument(
            '--aggressive',
            action='store_true',
            help='Enable aggressive mode (faster but more detectable)'
        )
        
        # Output and reporting
        parser.add_argument(
            '--export-report',
            type=str,
            choices=['json', 'pdf', 'html', 'txt'],
            help='Export report format'
        )
        
        parser.add_argument(
            '--output-dir', '-o',
            type=str,
            default='./reports',
            help='Output directory for reports'
        )
        
        parser.add_argument(
            '--session-name',
            type=str,
            help='Custom session name for this attack'
        )
        
        # Authentication
        parser.add_argument(
            '--auth-user',
            type=str,
            help='Username for authenticated attacks'
        )
        
        parser.add_argument(
            '--auth-pass',
            type=str,
            help='Password for authenticated attacks'
        )
        
        parser.add_argument(
            '--auth-token',
            type=str,
            help='Authentication token (JWT, API key, etc.)'
        )
        
        # Advanced options
        parser.add_argument(
            '--threads',
            type=int,
            default=10,
            help='Number of concurrent threads'
        )
        
        parser.add_argument(
            '--timeout',
            type=int,
            default=30,
            help='Request timeout in seconds'
        )
        
        parser.add_argument(
            '--delay',
            type=float,
            default=0.5,
            help='Delay between requests in seconds'
        )
        
        parser.add_argument(
            '--user-agent',
            type=str,
            help='Custom User-Agent string'
        )
        
        parser.add_argument(
            '--proxy',
            type=str,
            help='Proxy server (http://host:port)'
        )
        
        # AI options
        parser.add_argument(
            '--ai-mode',
            action='store_true',
            help='Enable AI-powered adaptive attacks'
        )
        
        parser.add_argument(
            '--learning-mode',
            action='store_true',
            help='Enable learning from attack results'
        )
    
    def _create_scan_parser(self, parser: argparse.ArgumentParser):
        """Create scan command parser"""
        
        parser.add_argument(
            '--target', '-t',
            type=str,
            required=True,
            help='Target URL or IP address'
        )
        
        parser.add_argument(
            '--type',
            choices=['port', 'subdomain', 'tech', 'all'],
            default='all',
            help='Type of scan to perform'
        )
        
        parser.add_argument(
            '--ports',
            type=str,
            help='Port range (e.g., 1-1000, 80,443,8080)'
        )
        
        parser.add_argument(
            '--wordlist',
            type=str,
            help='Wordlist file for subdomain enumeration'
        )
    
    def _create_exploit_parser(self, parser: argparse.ArgumentParser):
        """Create exploit command parser"""
        
        parser.add_argument(
            '--target', '-t',
            type=str,
            required=True,
            help='Target URL or IP address'
        )
        
        parser.add_argument(
            '--vulnerability',
            choices=['sql', 'xss', 'ssrf', 'ssti', 'rce'],
            required=True,
            help='Vulnerability type to exploit'
        )
        
        parser.add_argument(
            '--payload',
            type=str,
            help='Custom payload to use'
        )
    
    def _create_report_parser(self, parser: argparse.ArgumentParser):
        """Create report command parser"""
        
        parser.add_argument(
            '--session',
            type=str,
            help='Session ID or name to generate report for'
        )
        
        parser.add_argument(
            '--format',
            choices=['json', 'pdf', 'html', 'txt'],
            default='html',
            help='Report format'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output file path'
        )
    
    def _create_config_parser(self, parser: argparse.ArgumentParser):
        """Create config command parser"""
        
        config_action = parser.add_mutually_exclusive_group(required=True)
        
        config_action.add_argument(
            '--show',
            action='store_true',
            help='Show current configuration'
        )
        
        config_action.add_argument(
            '--set',
            nargs=2,
            metavar=('KEY', 'VALUE'),
            help='Set configuration value'
        )
        
        config_action.add_argument(
            '--reset',
            action='store_true',
            help='Reset to default configuration'
        )
    
    def show_banner(self):
        """Show Agent DS banner"""
        if not self.banner_shown:
            self.theme.clear_screen()
            self.theme.print_banner('agent_ds', 'bright_green', 'matrix')
            self.banner_shown = True
    
    async def attack_command(self, args) -> int:
        """Execute attack command"""
        
        self.show_banner()
        
        # Validate arguments
        if args.quiet and args.verbose:
            self.theme.show_error_message("Cannot use --quiet and --verbose together")
            return 1
        
        # Show attack initiation
        if args.one_click:
            self.theme.print_banner('one_click', 'bright_cyan', 'glitch')
            attack_type = "One-Click Autonomous Attack"
        elif args.web_only:
            attack_type = "Web Application Attack"
        elif args.network_only:
            attack_type = "Network-Level Attack"
        else:
            attack_type = "Custom Attack Sequence"
        
        # Show target and mode information
        if not args.quiet:
            self.theme.show_warning_message(
                f"Initiating {attack_type}",
                [
                    f"Target: {args.target}",
                    f"Mode: {'Safe' if args.safe_mode else 'Aggressive' if args.aggressive else 'Normal'}",
                    f"AI Mode: {'Enabled' if args.ai_mode else 'Disabled'}",
                    f"Stealth: {'Enabled' if args.stealth else 'Disabled'}"
                ]
            )
        
        # Show countdown
        if not args.quiet:
            await self.theme.show_countdown(5, "Attack commencing in")
        
        try:
            # Initialize the one-click attack orchestrator
            orchestrator = OneClickAttackOrchestrator()
            
            # Configure attack options
            attack_config = {
                'target_url': args.target,
                'safe_mode': args.safe_mode,
                'stealth_mode': args.stealth,
                'aggressive_mode': args.aggressive,
                'ai_mode': args.ai_mode,
                'learning_mode': args.learning_mode,
                'threads': args.threads,
                'timeout': args.timeout,
                'delay': args.delay,
                'user_agent': args.user_agent,
                'proxy': args.proxy,
                'session_name': args.session_name,
                'verbose': args.verbose and not args.quiet,
                'quiet': args.quiet
            }
            
            # Add authentication if provided
            if args.auth_user and args.auth_pass:
                attack_config['auth'] = {
                    'username': args.auth_user,
                    'password': args.auth_pass
                }
            elif args.auth_token:
                attack_config['auth'] = {
                    'token': args.auth_token
                }
            
            # Execute the appropriate attack mode
            if args.one_click:
                results = await orchestrator.execute_one_click_attack(attack_config)
            elif args.web_only:
                results = await orchestrator.execute_web_only_attack(attack_config)
            elif args.network_only:
                results = await orchestrator.execute_network_only_attack(attack_config)
            elif args.custom:
                phases = [phase.strip() for phase in args.custom.split(',')]
                results = await orchestrator.execute_custom_attack(attack_config, phases)
            else:
                self.theme.show_error_message("Invalid attack mode specified")
                return 1
            
            # Show completion summary
            if not args.quiet:
                self.theme.show_completion_summary(results)
            
            # Export report if requested
            if args.export_report:
                await self._export_report(results, args)
            
            # Determine exit code based on results
            if results.get('success', False):
                if not args.quiet:
                    self.theme.show_success_message("Attack completed successfully")
                return 0
            else:
                if not args.quiet:
                    self.theme.show_error_message("Attack completed with errors")
                return 1
                
        except KeyboardInterrupt:
            if not args.quiet:
                self.theme.show_warning_message("Attack interrupted by user")
            return 2
        except Exception as e:
            if not args.quiet:
                self.theme.show_error_message(f"Attack failed: {str(e)}")
            if args.verbose:
                import traceback
                print(traceback.format_exc())
            return 1
    
    async def scan_command(self, args) -> int:
        """Execute scan command"""
        
        self.show_banner()
        self.theme.print_banner('recon', 'bright_yellow')
        
        # Implementation would call reconnaissance module
        self.theme.show_warning_message("Scan command not yet implemented")
        return 0
    
    async def exploit_command(self, args) -> int:
        """Execute exploit command"""
        
        self.show_banner()
        self.theme.print_banner('exploit', 'bright_red')
        
        # Implementation would call specific exploit modules
        self.theme.show_warning_message("Exploit command not yet implemented")
        return 0
    
    async def report_command(self, args) -> int:
        """Execute report command"""
        
        # Implementation would call reporting module
        self.theme.show_warning_message("Report command not yet implemented")
        return 0
    
    async def config_command(self, args) -> int:
        """Execute config command"""
        
        if args.show:
            # Show current configuration
            config = self._load_config(args.config_file)
            print(json.dumps(config, indent=2))
        elif args.set:
            # Set configuration value
            key, value = args.set
            config = self._load_config(args.config_file)
            self._set_config_value(config, key, value)
            self._save_config(config, args.config_file)
            print(f"Configuration updated: {key} = {value}")
        elif args.reset:
            # Reset configuration
            self._create_default_config(args.config_file)
            print("Configuration reset to defaults")
        
        return 0
    
    async def version_command(self, args) -> int:
        """Show version information"""
        
        self.show_banner()
        
        version_info = f"""
Agent DS v{self.version}
Next-Generation Autonomous Hacker Agent

Components:
  - One-Click Attack Engine: v2.0
  - AI Adaptive Core: v2.0
  - Terminal Theme Engine: v2.0
  - Reconnaissance Module: v2.0
  - Web Attack Engine: v2.0
  - Database Exploitation: v2.0
  - Admin Login Testing: v2.0

Python Version: {sys.version}
Platform: {sys.platform}
"""
        
        print(version_info)
        return 0
    
    async def help_command(self, args) -> int:
        """Show help information"""
        
        self.show_banner()
        
        help_text = """
Agent DS - Next-Generation Autonomous Hacker Agent

USAGE:
    agent-ds <command> [options]

COMMANDS:
    attack      Launch attacks against targets
    scan        Reconnaissance and scanning operations  
    exploit     Exploit specific vulnerabilities
    report      Generate and manage reports
    config      Configuration management
    version     Show version information
    help        Show this help message

EXAMPLES:
    # One-click autonomous attack
    agent-ds attack --target https://example.com --one-click
    
    # Safe mode with report export
    agent-ds attack --target https://example.com --one-click --safe-mode --export-report pdf
    
    # Stealth mode with custom session
    agent-ds attack --target https://example.com --one-click --stealth --session-name "target-recon"
    
    # Web-only attack with authentication
    agent-ds attack --target https://example.com --web-only --auth-user admin --auth-pass password
    
    # Custom attack sequence
    agent-ds attack --target https://example.com --custom "recon,web,admin" --ai-mode
    
    # Port scanning only
    agent-ds scan --target example.com --type port --ports "1-1000"

For more information on a specific command, use:
    agent-ds <command> --help
"""
        
        print(help_text)
        return 0
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from file"""
        
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Return default configuration
        return self._get_default_config()
    
    def _save_config(self, config: Dict[str, Any], config_file: str):
        """Save configuration to file"""
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _create_default_config(self, config_file: str):
        """Create default configuration file"""
        
        default_config = self._get_default_config()
        self._save_config(default_config, config_file)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        
        return {
            "version": "2.0",
            "global": {
                "timeout": 30,
                "threads": 10,
                "delay": 0.5,
                "user_agent": "Agent DS v2.0 - Autonomous Hacker Agent",
                "output_dir": "./reports",
                "verbose": False,
                "safe_mode": False
            },
            "attack": {
                "ai_mode": True,
                "learning_mode": True,
                "stealth_mode": False,
                "aggressive_mode": False
            },
            "reconnaissance": {
                "subdomain_wordlist": "./wordlists/subdomains.txt",
                "port_range": "1-1000",
                "technology_detection": True
            },
            "web_attacks": {
                "sql_injection": True,
                "xss": True,
                "ssrf": True,
                "ssti": True,
                "command_injection": True
            },
            "database": {
                "sqlmap_path": "/usr/bin/sqlmap",
                "extract_data": False,
                "max_tables": 10
            },
            "admin_login": {
                "brute_force": True,
                "default_credentials": True,
                "session_analysis": True
            },
            "reporting": {
                "default_format": "html",
                "include_payloads": True,
                "include_screenshots": False
            }
        }
    
    def _set_config_value(self, config: Dict[str, Any], key: str, value: str):
        """Set a configuration value using dot notation"""
        
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value, attempting to preserve type
        final_key = keys[-1]
        if final_key in current:
            # Try to preserve the original type
            original_type = type(current[final_key])
            if original_type == bool:
                current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
            elif original_type == int:
                current[final_key] = int(value)
            elif original_type == float:
                current[final_key] = float(value)
            else:
                current[final_key] = value
        else:
            # New key, try to infer type
            if value.lower() in ('true', 'false'):
                current[final_key] = value.lower() == 'true'
            elif value.isdigit():
                current[final_key] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                current[final_key] = float(value)
            else:
                current[final_key] = value
    
    async def _export_report(self, results: Dict[str, Any], args):
        """Export attack results to specified format"""
        
        try:
            from .reporting import ReportGenerator
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            session_name = args.session_name or f"attack_{args.target.replace('://', '_').replace('/', '_')}"
            timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
            
            filename = f"{session_name}_{timestamp}.{args.export_report}"
            output_path = output_dir / filename
            
            generator = ReportGenerator()
            
            if args.export_report == 'json':
                await generator.generate_json_report(results, output_path)
            elif args.export_report == 'pdf':
                await generator.generate_pdf_report(results, output_path)
            elif args.export_report == 'html':
                await generator.generate_html_report(results, output_path)
            elif args.export_report == 'txt':
                await generator.generate_text_report(results, output_path)
            
            if not args.quiet:
                self.theme.show_success_message(f"Report exported to: {output_path}")
                
        except ImportError:
            if not args.quiet:
                self.theme.show_warning_message("Reporting module not available")
        except Exception as e:
            if not args.quiet:
                self.theme.show_error_message(f"Failed to export report: {str(e)}")
    
    async def run(self, argv: Optional[List[str]] = None) -> int:
        """Main CLI entry point"""
        
        parser = self.create_parser()
        
        # Parse arguments
        if argv is None:
            argv = sys.argv[1:]
        
        if not argv:
            self.show_banner()
            parser.print_help()
            return 0
        
        try:
            args = parser.parse_args(argv)
        except SystemExit as e:
            return e.code
        
        # Execute command
        if args.command in self.commands:
            return await self.commands[args.command](args)
        else:
            self.show_banner()
            parser.print_help()
            return 1


async def main():
    """Main entry point"""
    cli = AgentDSCLI()
    return await cli.run()


def cli_main():
    """Synchronous CLI entry point"""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 2
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())