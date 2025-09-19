"""
Agent DS - Autonomous Red-Team CLI AI Framework
Main entry point for the CLI application
"""

import sys
import os
import click
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.auth.authentication import AuthManager
from core.ai_orchestrator.planner import AIOrchestrator
from core.recon.scanner import ReconModule
from core.vulnerability_intel.analyzer import VulnIntel
from core.attack_engine.executor import AttackEngine
from core.reporting.generator import ReportGenerator
from core.config.settings import Config
from core.database.manager import DatabaseManager
from core.utils.logger import setup_logger
from core.ai_learning.cli_commands import ai_learning_commands

console = Console()
logger = setup_logger()

class AgentDS:
    """Main Agent DS application class"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        self.auth_manager = AuthManager()
        self.ai_orchestrator = AIOrchestrator()
        self.recon_module = ReconModule()
        self.vuln_intel = VulnIntel()
        self.attack_engine = AttackEngine()
        self.report_generator = ReportGenerator()
        self.session_token = None
        self.current_mission = None
        
    def display_banner(self):
        """Display the Agent DS banner"""
        banner = Text("""
╔═══════════════════════════════════════════════════════════════╗
║                    AGENT DS v1.0                              ║
║           Autonomous Red-Team CLI AI Framework               ║
║                                                               ║
║     ⚠️  AUTHORIZED GOVERNMENT USE ONLY ⚠️                     ║
║     NSA-Approved | Classified Operations                     ║
╚═══════════════════════════════════════════════════════════════╝
        """, style="bold red")
        
        console.print(Panel(banner, border_style="red", padding=(1, 2)))

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Agent DS - Autonomous Red-Team CLI AI Framework"""
    if ctx.invoked_subcommand is None:
        agent = AgentDS()
        agent.display_banner()
        console.print("\n[bold yellow]Available Commands:[/bold yellow]")
        console.print("  • [green]agent-ds login[/green] - Authenticate with admin credentials")
        console.print("  • [green]agent-ds mission start[/green] - Initialize a new mission")
        console.print("  • [green]agent-ds recon[/green] - Perform reconnaissance")
        console.print("  • [green]agent-ds analyze[/green] - Vulnerability analysis")
        console.print("  • [green]agent-ds attack[/green] - Execute attacks")
        console.print("  • [green]agent-ds report[/green] - Generate reports")
        console.print("\n[dim]Use --help with any command for more information[/dim]")

@cli.command()
@click.option('--username', prompt='Admin Username', hide_input=False)
@click.option('--password', prompt='Admin Password', hide_input=True)
@click.option('--mfa', help='Multi-factor authentication code (optional)')
def login(username, password, mfa):
    """Authenticate with admin credentials"""
    agent = AgentDS()
    
    try:
        with console.status("[bold green]Authenticating..."):
            success, token, message = agent.auth_manager.authenticate(username, password, mfa)
            
        if success:
            agent.session_token = token
            console.print(f"[bold green]✓ Authentication successful[/bold green]")
            console.print(f"[dim]Session token: {token[:20]}...[/dim]")
            
            # Store session for subsequent commands
            agent.config.set('session_token', token)
            agent.config.save()
            
        else:
            console.print(f"[bold red]✗ Authentication failed: {message}[/bold red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[bold red]✗ Authentication error: {str(e)}[/bold red]")
        sys.exit(1)

@cli.group()
def mission():
    """Mission management commands"""
    pass

@mission.command()
@click.option('--target', required=True, help='Target URL or IP address')
@click.option('--scope', help='Additional scope definition')
@click.option('--authorized', is_flag=True, help='Confirm target is authorized')
def start(target, scope, authorized):
    """Start a new authorized mission"""
    agent = AgentDS()
    
    # Verify authentication
    if not agent.auth_manager.verify_session():
        console.print("[bold red]✗ Authentication required. Run 'agent-ds login' first.[/bold red]")
        sys.exit(1)
    
    if not authorized:
        console.print("[bold red]✗ You must confirm the target is authorized with --authorized flag[/bold red]")
        sys.exit(1)
    
    try:
        with console.status(f"[bold green]Initializing mission for {target}..."):
            mission_id = agent.ai_orchestrator.create_mission(target, scope)
            
        console.print(f"[bold green]✓ Mission {mission_id} started for target: {target}[/bold green]")
        console.print(f"[dim]Mission logged and tracked for compliance[/dim]")
        
        # Store current mission
        agent.config.set('current_mission', mission_id)
        agent.config.save()
        
    except Exception as e:
        console.print(f"[bold red]✗ Mission start failed: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
@click.option('--auto', is_flag=True, help='Fully automated reconnaissance')
@click.option('--target', help='Specific target (overrides mission target)')
@click.option('--modules', help='Specific modules to run (nmap,amass,gobuster)')
def recon(auto, target, modules):
    """Perform reconnaissance on target"""
    agent = AgentDS()
    
    # Verify authentication and active mission
    if not agent.auth_manager.verify_session():
        console.print("[bold red]✗ Authentication required[/bold red]")
        sys.exit(1)
    
    try:
        if auto:
            console.print("[bold cyan]🔍 Starting automated reconnaissance...[/bold cyan]")
            results = agent.recon_module.run_full_recon(target)
        else:
            console.print("[bold cyan]🔍 Starting targeted reconnaissance...[/bold cyan]")
            results = agent.recon_module.run_custom_recon(target, modules)
            
        console.print(f"[bold green]✓ Reconnaissance completed[/bold green]")
        console.print(f"[dim]Found {len(results.get('hosts', []))} hosts, {len(results.get('services', []))} services[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Reconnaissance failed: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
@click.option('--cve', is_flag=True, help='Check CVE database')
@click.option('--alienvault', is_flag=True, help='Check AlienVault OTX')
@click.option('--exploit-db', is_flag=True, help='Check Exploit Database')
@click.option('--ai-fuzzing', is_flag=True, help='Enable AI-driven fuzzing')
def analyze(cve, alienvault, exploit_db, ai_fuzzing):
    """Analyze vulnerabilities and plan attacks"""
    agent = AgentDS()
    
    if not agent.auth_manager.verify_session():
        console.print("[bold red]✗ Authentication required[/bold red]")
        sys.exit(1)
    
    try:
        console.print("[bold cyan]🧠 Starting AI vulnerability analysis...[/bold cyan]")
        
        analysis_results = agent.vuln_intel.analyze_target({
            'cve_check': cve,
            'alienvault': alienvault,
            'exploit_db': exploit_db,
            'ai_fuzzing': ai_fuzzing
        })
        
        # Generate attack plan
        attack_plan = agent.ai_orchestrator.generate_attack_plan(analysis_results)
        
        console.print(f"[bold green]✓ Analysis completed[/bold green]")
        console.print(f"[dim]Found {len(analysis_results.get('vulnerabilities', []))} vulnerabilities[/dim]")
        console.print(f"[dim]Generated {len(attack_plan.get('attack_vectors', []))} attack vectors[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Analysis failed: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
@click.option('--auto', is_flag=True, help='Fully automated attack execution')
@click.option('--dry-run', is_flag=True, help='Simulation mode only')
@click.option('--vector', help='Specific attack vector to execute')
def attack(auto, dry_run, vector):
    """Execute AI-planned attacks"""
    agent = AgentDS()
    
    if not agent.auth_manager.verify_session():
        console.print("[bold red]✗ Authentication required[/bold red]")
        sys.exit(1)
    
    if dry_run:
        console.print("[bold yellow]⚠️  Running in simulation mode[/bold yellow]")
    
    try:
        console.print("[bold red]⚔️  Starting attack execution...[/bold red]")
        
        attack_results = agent.attack_engine.execute_attacks({
            'auto_mode': auto,
            'dry_run': dry_run,
            'specific_vector': vector
        })
        
        console.print(f"[bold green]✓ Attack execution completed[/bold green]")
        console.print(f"[dim]Executed {len(attack_results.get('attacks', []))} attack attempts[/dim]")
        console.print(f"[dim]Success rate: {attack_results.get('success_rate', 0)}%[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Attack execution failed: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
@click.option('--format', default='pdf', help='Report format (pdf, html, json)')
@click.option('--output', help='Output file path')
@click.option('--classification', default='CONFIDENTIAL', help='Report classification level')
def report(format, output, classification):
    """Generate mission report"""
    agent = AgentDS()
    
    if not agent.auth_manager.verify_session():
        console.print("[bold red]✗ Authentication required[/bold red]")
        sys.exit(1)
    
    try:
        console.print("[bold cyan]📊 Generating mission report...[/bold cyan]")
        
        report_path = agent.report_generator.generate_report({
            'format': format,
            'output_path': output,
            'classification': classification
        })
        
        console.print(f"[bold green]✓ Report generated: {report_path}[/bold green]")
        console.print(f"[dim]Classification: {classification}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Report generation failed: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
@click.option('--data', help='Path to custom training data')
@click.option('--model', help='Specific model to train')
def train(data, model):
    """Train AI models with custom data"""
    agent = AgentDS()
    
    if not agent.auth_manager.verify_session():
        console.print("[bold red]✗ Authentication required[/bold red]")
        sys.exit(1)
    
    try:
        console.print("[bold cyan]🤖 Starting AI model training...[/bold cyan]")
        
        training_results = agent.ai_orchestrator.train_models({
            'data_path': data,
            'model_type': model
        })
        
        console.print(f"[bold green]✓ Training completed[/bold green]")
        console.print(f"[dim]Model accuracy: {training_results.get('accuracy', 0)}%[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]✗ Training failed: {str(e)}[/bold red]")
        sys.exit(1)

# Add AI learning commands to main CLI
cli.add_command(ai_learning_commands)

if __name__ == '__main__':
    cli()