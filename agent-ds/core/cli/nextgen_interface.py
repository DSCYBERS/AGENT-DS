"""
Agent DS - Enhanced CLI Interface
Advanced CLI interface integrating all Next-Gen features with AI-powered recommendations
"""

import asyncio
import click
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Rich console imports for beautiful CLI
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from rich.prompt import Prompt, Confirm
    from rich.tree import Tree
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from core.config.settings import Config
from core.utils.logger import get_logger
from core.ai_learning.reinforcement_engine import reinforcement_engine
from core.ai_learning.payload_mutation import PayloadMutationEngine
from core.ai_learning.underrated_attacks import UnderratedAttackOrchestrator
from core.ai_learning.chained_exploit_engine import chained_exploit_engine, ChainType, ExploitNode
from core.ai_learning.training_pipeline import training_pipeline
from core.ai_learning.sandbox_environment import sandbox_environment

logger = get_logger('enhanced_cli')

if RICH_AVAILABLE:
    console = Console()
else:
    # Fallback console
    class FallbackConsole:
        def print(self, *args, **kwargs):
            print(*args)
        
        def input(self, prompt=""):
            return input(prompt)
    
    console = FallbackConsole()

class AIRecommendationEngine:
    """AI-powered recommendation engine for CLI suggestions"""
    
    def __init__(self):
        self.logger = get_logger('ai_recommendations')
        self.usage_history = []
        self.success_patterns = {}
    
    async def get_attack_recommendations(self, target_url: str, 
                                       context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get AI-powered attack recommendations"""
        recommendations = []
        
        try:
            # Analyze target characteristics
            target_analysis = await self._analyze_target(target_url, context)
            
            # Get recommendations based on analysis
            if target_analysis.get('technology_stack'):
                tech_stack = target_analysis['technology_stack']
                
                if 'python' in tech_stack:
                    recommendations.append({
                        'attack_type': 'ssti',
                        'confidence': 0.85,
                        'reason': 'Python framework detected - SSTI likely effective',
                        'payload_suggestion': '{{config.__class__}}',
                        'priority': 'high'
                    })
                
                if 'database' in tech_stack:
                    recommendations.append({
                        'attack_type': 'sql_injection',
                        'confidence': 0.75,
                        'reason': 'Database detected - SQL injection may be possible',
                        'payload_suggestion': "' UNION SELECT 1,2,3-- ",
                        'priority': 'medium'
                    })
                
                if 'java' in tech_stack:
                    recommendations.append({
                        'attack_type': 'xxe',
                        'confidence': 0.70,
                        'reason': 'Java application - XXE vulnerabilities common',
                        'payload_suggestion': '<!DOCTYPE root [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>',
                        'priority': 'medium'
                    })
            
            # Add chain recommendations
            chain_recommendations = await self._get_chain_recommendations(target_analysis)
            recommendations.extend(chain_recommendations)
            
            # Sort by confidence and priority
            recommendations.sort(key=lambda x: (x['confidence'], x['priority'] == 'high'), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def _analyze_target(self, target_url: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze target for AI recommendations"""
        # Mock analysis - in real implementation would use reconnaissance tools
        analysis = {
            'technology_stack': ['python', 'database'],
            'security_measures': ['waf'],
            'port_scan_results': [80, 443, 22],
            'framework_detected': 'flask',
            'confidence': 0.8
        }
        
        if context:
            analysis.update(context)
        
        return analysis
    
    async def _get_chain_recommendations(self, target_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get exploit chain recommendations"""
        chain_recs = []
        
        tech_stack = target_analysis.get('technology_stack', [])
        
        if 'python' in tech_stack:
            chain_recs.append({
                'attack_type': 'ssti_to_rce_chain',
                'confidence': 0.80,
                'reason': 'Python detected - SSTI→RCE chain highly effective',
                'payload_suggestion': 'Multi-stage SSTI exploitation',
                'priority': 'high'
            })
        
        if 'database' in tech_stack:
            chain_recs.append({
                'attack_type': 'sqli_to_lateral_chain',
                'confidence': 0.75,
                'reason': 'Database access may enable lateral movement',
                'payload_suggestion': 'SQL injection → credential extraction → lateral movement',
                'priority': 'medium'
            })
        
        return chain_recs
    
    def record_usage(self, command: str, success: bool, execution_time: float):
        """Record command usage for learning"""
        self.usage_history.append({
            'command': command,
            'success': success,
            'execution_time': execution_time,
            'timestamp': datetime.now()
        })
        
        # Update success patterns
        if command not in self.success_patterns:
            self.success_patterns[command] = {'total': 0, 'successful': 0}
        
        self.success_patterns[command]['total'] += 1
        if success:
            self.success_patterns[command]['successful'] += 1

class InteractiveTrainingMode:
    """Interactive AI training mode for the CLI"""
    
    def __init__(self):
        self.logger = get_logger('interactive_training')
        self.training_session_active = False
        self.session_data = []
    
    async def start_training_session(self):
        """Start interactive training session"""
        if RICH_AVAILABLE:
            console.print(Panel("[bold green]🤖 AI Training Mode Activated[/bold green]", 
                              title="Interactive Training"))
        else:
            print("🤖 AI Training Mode Activated")
        
        self.training_session_active = True
        self.session_data = []
        
        await self._training_menu()
    
    async def _training_menu(self):
        """Interactive training menu"""
        while self.training_session_active:
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]Training Options:[/bold cyan]")
                console.print("1. Train Attack Effectiveness Model")
                console.print("2. Train Payload Optimization Model") 
                console.print("3. Train Reinforcement Learning Model")
                console.print("4. Continuous Learning Status")
                console.print("5. Manual Feedback Input")
                console.print("6. Exit Training Mode")
                
                choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5", "6"])
            else:
                print("\nTraining Options:")
                print("1. Train Attack Effectiveness Model")
                print("2. Train Payload Optimization Model")
                print("3. Train Reinforcement Learning Model") 
                print("4. Continuous Learning Status")
                print("5. Manual Feedback Input")
                print("6. Exit Training Mode")
                
                choice = input("Select option (1-6): ")
            
            await self._handle_training_choice(choice)
    
    async def _handle_training_choice(self, choice: str):
        """Handle training menu choice"""
        try:
            if choice == "1":
                await self._train_attack_effectiveness()
            elif choice == "2":
                await self._train_payload_optimization()
            elif choice == "3":
                await self._train_reinforcement_learning()
            elif choice == "4":
                await self._show_learning_status()
            elif choice == "5":
                await self._manual_feedback_input()
            elif choice == "6":
                self.training_session_active = False
                if RICH_AVAILABLE:
                    console.print("[green]Exiting training mode[/green]")
                else:
                    print("Exiting training mode")
        
        except Exception as e:
            self.logger.error(f"Training choice handling failed: {str(e)}")
            if RICH_AVAILABLE:
                console.print(f"[red]Error: {str(e)}[/red]")
            else:
                print(f"Error: {str(e)}")
    
    async def _train_attack_effectiveness(self):
        """Train attack effectiveness model"""
        if RICH_AVAILABLE:
            with console.status("[bold green]Training attack effectiveness model..."):
                success = await training_pipeline.trigger_immediate_training("attack_effectiveness")
        else:
            print("Training attack effectiveness model...")
            success = await training_pipeline.trigger_immediate_training("attack_effectiveness")
        
        if success:
            if RICH_AVAILABLE:
                console.print("[green]✅ Attack effectiveness model training initiated[/green]")
            else:
                print("✅ Attack effectiveness model training initiated")
        else:
            if RICH_AVAILABLE:
                console.print("[red]❌ Training failed - insufficient data[/red]")
            else:
                print("❌ Training failed - insufficient data")
    
    async def _train_payload_optimization(self):
        """Train payload optimization model"""
        if RICH_AVAILABLE:
            with console.status("[bold green]Training payload optimization model..."):
                success = await training_pipeline.trigger_immediate_training("payload_optimization")
        else:
            print("Training payload optimization model...")
            success = await training_pipeline.trigger_immediate_training("payload_optimization")
        
        if success:
            if RICH_AVAILABLE:
                console.print("[green]✅ Payload optimization model training initiated[/green]")
            else:
                print("✅ Payload optimization model training initiated")
        else:
            if RICH_AVAILABLE:
                console.print("[red]❌ Training failed - insufficient data[/red]")
            else:
                print("❌ Training failed - insufficient data")
    
    async def _train_reinforcement_learning(self):
        """Train reinforcement learning model"""
        if RICH_AVAILABLE:
            with console.status("[bold green]Training reinforcement learning model..."):
                success = await training_pipeline.trigger_immediate_training("reinforcement_learning")
        else:
            print("Training reinforcement learning model...")
            success = await training_pipeline.trigger_immediate_training("reinforcement_learning")
        
        if success:
            if RICH_AVAILABLE:
                console.print("[green]✅ Reinforcement learning model training initiated[/green]")
            else:
                print("✅ Reinforcement learning model training initiated")
        else:
            if RICH_AVAILABLE:
                console.print("[red]❌ Training failed - insufficient data[/red]")
            else:
                print("❌ Training failed - insufficient data")
    
    async def _show_learning_status(self):
        """Show continuous learning status"""
        stats = training_pipeline.get_learning_statistics()
        
        if RICH_AVAILABLE:
            table = Table(title="Continuous Learning Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Learning Active", str(stats.get('learning_active', False)))
            table.add_row("Total Training Sessions", str(stats.get('total_training_sessions', 0)))
            table.add_row("Active Models", str(stats.get('active_models', 0)))
            table.add_row("Last Training", str(stats.get('last_training_time', 'Never')))
            table.add_row("Queue Size", str(stats.get('training_queue_size', 0)))
            
            console.print(table)
        else:
            print("\nContinuous Learning Status:")
            print(f"Learning Active: {stats.get('learning_active', False)}")
            print(f"Total Training Sessions: {stats.get('total_training_sessions', 0)}")
            print(f"Active Models: {stats.get('active_models', 0)}")
            print(f"Last Training: {stats.get('last_training_time', 'Never')}")
            print(f"Queue Size: {stats.get('training_queue_size', 0)}")
    
    async def _manual_feedback_input(self):
        """Manual feedback input for training"""
        if RICH_AVAILABLE:
            console.print("[bold yellow]Manual Feedback Input[/bold yellow]")
            
            attack_type = Prompt.ask("Attack type", default="sql_injection")
            payload = Prompt.ask("Payload used")
            success = Confirm.ask("Was the attack successful?")
            impact = Prompt.ask("Impact level", choices=["low", "medium", "high", "critical"], default="medium")
        else:
            print("\nManual Feedback Input")
            attack_type = input("Attack type (default: sql_injection): ") or "sql_injection"
            payload = input("Payload used: ")
            success = input("Was the attack successful? (y/n): ").lower().startswith('y')
            impact = input("Impact level (low/medium/high/critical): ") or "medium"
        
        # Store feedback for training
        feedback_data = {
            'attack_type': attack_type,
            'payload': payload,
            'success': success,
            'impact_level': impact,
            'timestamp': datetime.now().isoformat(),
            'source': 'manual_feedback'
        }
        
        self.session_data.append(feedback_data)
        
        if RICH_AVAILABLE:
            console.print("[green]✅ Feedback recorded[/green]")
        else:
            print("✅ Feedback recorded")

# Enhanced CLI Group
@click.group()
@click.version_option(version="2.0.0")
def nextgen_cli():
    """Agent DS - Next-Generation AI-Powered Penetration Testing Framework"""
    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold red]Agent DS[/bold red] - [bold cyan]Next-Gen AI Framework[/bold cyan]\n"
            "[yellow]Government Authorized Penetration Testing Tool[/yellow]",
            title="🤖 Agent DS v2.0",
            border_style="blue"
        ))

@nextgen_cli.command()
@click.option('--target', '-t', required=True, help='Target URL for autonomous attack')
@click.option('--objective', '-o', default='privilege_escalation', 
              type=click.Choice(['reconnaissance', 'privilege_escalation', 'lateral_movement', 'data_exfiltration']),
              help='Attack objective')
@click.option('--ai-recommendations', '--ai', is_flag=True, help='Get AI-powered attack recommendations')
@click.option('--sandbox-test', '--sandbox', is_flag=True, help='Test in sandbox environment first')
async def autonomous_attack(target, objective, ai_recommendations, sandbox_test):
    """Execute autonomous AI-powered attack with chained exploits"""
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]🎯 Autonomous Attack Mode[/bold green]")
        console.print(f"Target: [yellow]{target}[/yellow]")
        console.print(f"Objective: [cyan]{objective}[/cyan]")
    else:
        print(f"\n🎯 Autonomous Attack Mode")
        print(f"Target: {target}")
        print(f"Objective: {objective}")
    
    try:
        # Get AI recommendations if requested
        if ai_recommendations:
            rec_engine = AIRecommendationEngine()
            recommendations = await rec_engine.get_attack_recommendations(target)
            
            if RICH_AVAILABLE:
                console.print("\n[bold cyan]🤖 AI Recommendations:[/bold cyan]")
                for i, rec in enumerate(recommendations[:3], 1):
                    console.print(f"{i}. [yellow]{rec['attack_type']}[/yellow] "
                                f"(Confidence: {rec['confidence']:.1%})")
                    console.print(f"   Reason: {rec['reason']}")
                    console.print(f"   Payload: [green]{rec['payload_suggestion']}[/green]\n")
            else:
                print("\n🤖 AI Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"{i}. {rec['attack_type']} (Confidence: {rec['confidence']:.1%})")
                    print(f"   Reason: {rec['reason']}")
                    print(f"   Payload: {rec['payload_suggestion']}\n")
        
        # Convert objective to enum
        chain_type = ChainType(objective)
        
        # Execute autonomous attack
        if RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("Executing autonomous attack...", total=100)
                
                # Simulate progress updates
                progress.update(task, advance=20, description="Planning optimal exploit chain...")
                await asyncio.sleep(1)
                
                progress.update(task, advance=30, description="Executing exploit chain...")
                result = await chained_exploit_engine.execute_autonomous_attack(target, chain_type)
                
                progress.update(task, advance=50, description="Analyzing results...")
                await asyncio.sleep(0.5)
        else:
            print("Executing autonomous attack...")
            result = await chained_exploit_engine.execute_autonomous_attack(target, chain_type)
        
        # Display results
        if RICH_AVAILABLE:
            if result['success']:
                console.print(f"\n[bold green]✅ Autonomous Attack Successful![/bold green]")
                console.print(f"Chain Used: [yellow]{result['chain_id']}[/yellow]")
                console.print(f"Impact Achieved: [red]{result['impact_achieved']:.1f}/10[/red]")
                
                # Create results table
                table = Table(title="Attack Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Overall Success", "✅ True")
                table.add_row("Chain ID", result['chain_id'])
                table.add_row("Execution Time", f"{result['execution_results'].get('execution_time', 0):.2f}s")
                table.add_row("Nodes Executed", str(len(result['execution_results'].get('nodes_executed', []))))
                table.add_row("Impact Score", f"{result['impact_achieved']:.1f}/10")
                
                console.print(table)
            else:
                console.print(f"\n[bold red]❌ Autonomous Attack Failed[/bold red]")
                console.print(f"Error: [yellow]{result.get('error', 'Unknown error')}[/yellow]")
        else:
            if result['success']:
                print(f"\n✅ Autonomous Attack Successful!")
                print(f"Chain Used: {result['chain_id']}")
                print(f"Impact Achieved: {result['impact_achieved']:.1f}/10")
                print(f"Execution Time: {result['execution_results'].get('execution_time', 0):.2f}s")
            else:
                print(f"\n❌ Autonomous Attack Failed")
                print(f"Error: {result.get('error', 'Unknown error')}")
        
        # Sandbox testing if requested
        if sandbox_test and result['success']:
            if RICH_AVAILABLE:
                console.print(f"\n[bold yellow]🧪 Testing in Sandbox Environment[/bold yellow]")
            else:
                print(f"\n🧪 Testing in Sandbox Environment")
            
            # Test successful payloads in sandbox
            for node in result['execution_results'].get('nodes_executed', []):
                if node.get('success'):
                    sandbox_result = await sandbox_environment.test_payload_safely(
                        node.get('payload_used', ''),
                        node.get('exploit_type', 'generic')
                    )
                    
                    if RICH_AVAILABLE:
                        status = "✅" if sandbox_result.success else "❌"
                        console.print(f"{status} {node.get('exploit_type', 'unknown')}: "
                                    f"{sandbox_result.execution_time:.2f}s")
                    else:
                        status = "✅" if sandbox_result.success else "❌"
                        print(f"{status} {node.get('exploit_type', 'unknown')}: "
                              f"{sandbox_result.execution_time:.2f}s")
    
    except Exception as e:
        logger.error(f"Autonomous attack failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        else:
            print(f"Error: {str(e)}")

@nextgen_cli.command()
@click.option('--target', '-t', required=True, help='Target URL')
@click.option('--attack-types', '-a', multiple=True, 
              type=click.Choice(['ssti', 'xxe', 'deserialization', 'business_logic', 'cache_poisoning', 'request_smuggling']),
              help='Specific underrated attack types to execute')
@click.option('--ai-mutation', '--mutation', is_flag=True, help='Use AI-powered payload mutation')
async def underrated_attacks(target, attack_types, ai_mutation):
    """Execute underrated attack techniques with AI enhancement"""
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold red]🔥 Underrated Attacks Mode[/bold red]")
        console.print(f"Target: [yellow]{target}[/yellow]")
    else:
        print(f"\n🔥 Underrated Attacks Mode")
        print(f"Target: {target}")
    
    try:
        # Use all attack types if none specified
        if not attack_types:
            attack_types = ['ssti', 'xxe', 'deserialization', 'business_logic', 'cache_poisoning', 'request_smuggling']
        
        if RICH_AVAILABLE:
            console.print(f"Attack Types: [cyan]{', '.join(attack_types)}[/cyan]")
        else:
            print(f"Attack Types: {', '.join(attack_types)}")
        
        # Execute attacks
        orchestrator = UnderratedAttackOrchestrator()
        
        if RICH_AVAILABLE:
            with Progress(console=console) as progress:
                task = progress.add_task("Executing underrated attacks...", total=len(attack_types))
                
                results = await orchestrator.execute_comprehensive_scan(target, list(attack_types))
                
                progress.update(task, completed=len(attack_types))
        else:
            print("Executing underrated attacks...")
            results = await orchestrator.execute_comprehensive_scan(target, list(attack_types))
        
        # Display results
        if RICH_AVAILABLE:
            table = Table(title="Underrated Attack Results")
            table.add_column("Attack Type", style="cyan")
            table.add_column("Success", style="green")
            table.add_column("Impact", style="red")
            table.add_column("Payload Used", style="yellow")
            
            for attack_type, result in results.items():
                success_icon = "✅" if result.success else "❌"
                impact_color = "red" if result.impact_level == "critical" else "yellow" if result.impact_level == "high" else "green"
                
                table.add_row(
                    attack_type,
                    success_icon,
                    f"[{impact_color}]{result.impact_level}[/{impact_color}]",
                    result.payload_used[:50] + "..." if len(result.payload_used) > 50 else result.payload_used
                )
            
            console.print(table)
            
            # Statistics
            stats = orchestrator.get_attack_statistics(results)
            console.print(f"\n[bold cyan]📊 Statistics:[/bold cyan]")
            console.print(f"Success Rate: [green]{stats['success_rate']:.1%}[/green]")
            console.print(f"Critical Impact: [red]{stats['impact_distribution']['critical']}[/red]")
            console.print(f"High Impact: [yellow]{stats['impact_distribution']['high']}[/yellow]")
        else:
            print("\nUnderrated Attack Results:")
            for attack_type, result in results.items():
                success_icon = "✅" if result.success else "❌"
                print(f"{success_icon} {attack_type}: {result.impact_level} impact")
                print(f"   Payload: {result.payload_used[:100]}...")
            
            # Statistics
            stats = orchestrator.get_attack_statistics(results)
            print(f"\n📊 Statistics:")
            print(f"Success Rate: {stats['success_rate']:.1%}")
            print(f"Critical Impact: {stats['impact_distribution']['critical']}")
            print(f"High Impact: {stats['impact_distribution']['high']}")
    
    except Exception as e:
        logger.error(f"Underrated attacks failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        else:
            print(f"Error: {str(e)}")

@nextgen_cli.command()
@click.option('--payload', '-p', required=True, help='Payload to test')
@click.option('--attack-type', '-a', required=True, help='Attack type')
@click.option('--environment', '-e', default='web_app', 
              type=click.Choice(['web_app', 'database', 'api_server']),
              help='Sandbox environment type')
@click.option('--batch-file', '-f', type=click.Path(exists=True), help='File containing payloads to test')
async def sandbox_test(payload, attack_type, environment, batch_file):
    """Test payloads in isolated sandbox environment"""
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold yellow]🧪 Sandbox Testing Mode[/bold yellow]")
    else:
        print(f"\n🧪 Sandbox Testing Mode")
    
    try:
        if batch_file:
            # Batch testing
            payloads = []
            with open(batch_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            payloads.append((parts[0], parts[1]))
                        else:
                            payloads.append((line, attack_type))
            
            if RICH_AVAILABLE:
                console.print(f"Testing [cyan]{len(payloads)}[/cyan] payloads from file")
                
                with Progress(console=console) as progress:
                    task = progress.add_task("Batch testing payloads...", total=len(payloads))
                    
                    results = await sandbox_environment.batch_test_payloads(payloads, environment)
                    progress.update(task, completed=len(payloads))
            else:
                print(f"Testing {len(payloads)} payloads from file")
                results = await sandbox_environment.batch_test_payloads(payloads, environment)
            
            # Display batch results
            successful = sum(1 for r in results if r.success)
            
            if RICH_AVAILABLE:
                console.print(f"\n[bold green]✅ Batch Test Complete[/bold green]")
                console.print(f"Success Rate: [green]{successful}/{len(results)} ({successful/len(results):.1%})[/green]")
                
                # Show top results
                table = Table(title="Top Sandbox Results")
                table.add_column("Payload", style="yellow")
                table.add_column("Attack Type", style="cyan")
                table.add_column("Success", style="green")
                table.add_column("Time", style="blue")
                
                for result in results[:10]:  # Top 10
                    success_icon = "✅" if result.success else "❌"
                    table.add_row(
                        result.payload[:50] + "..." if len(result.payload) > 50 else result.payload,
                        result.attack_type,
                        success_icon,
                        f"{result.execution_time:.2f}s"
                    )
                
                console.print(table)
            else:
                print(f"\n✅ Batch Test Complete")
                print(f"Success Rate: {successful}/{len(results)} ({successful/len(results):.1%})")
                
                for result in results[:5]:  # Top 5
                    success_icon = "✅" if result.success else "❌"
                    print(f"{success_icon} {result.attack_type}: {result.execution_time:.2f}s")
        
        else:
            # Single payload testing
            if RICH_AVAILABLE:
                console.print(f"Payload: [yellow]{payload}[/yellow]")
                console.print(f"Attack Type: [cyan]{attack_type}[/cyan]")
                console.print(f"Environment: [blue]{environment}[/blue]")
                
                with console.status("[bold green]Testing payload in sandbox..."):
                    result = await sandbox_environment.test_payload_safely(payload, attack_type, environment)
            else:
                print(f"Payload: {payload}")
                print(f"Attack Type: {attack_type}")
                print(f"Environment: {environment}")
                print("Testing payload in sandbox...")
                result = await sandbox_environment.test_payload_safely(payload, attack_type, environment)
            
            # Display single result
            if RICH_AVAILABLE:
                if result.success:
                    console.print(f"\n[bold green]✅ Sandbox Test Successful![/bold green]")
                else:
                    console.print(f"\n[bold red]❌ Sandbox Test Failed[/bold red]")
                
                table = Table(title="Sandbox Test Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Success", "✅ True" if result.success else "❌ False")
                table.add_row("Execution Time", f"{result.execution_time:.3f}s")
                table.add_row("Security Violations", str(len(result.security_violations)))
                table.add_row("Output Length", str(len(result.output)))
                
                if result.security_violations:
                    table.add_row("Violations", ", ".join(result.security_violations))
                
                console.print(table)
                
                if result.output:
                    console.print(f"\n[bold cyan]Output:[/bold cyan]")
                    console.print(Panel(result.output, border_style="blue"))
            else:
                if result.success:
                    print(f"\n✅ Sandbox Test Successful!")
                else:
                    print(f"\n❌ Sandbox Test Failed")
                
                print(f"Execution Time: {result.execution_time:.3f}s")
                print(f"Security Violations: {len(result.security_violations)}")
                
                if result.security_violations:
                    print(f"Violations: {', '.join(result.security_violations)}")
                
                if result.output:
                    print(f"\nOutput:\n{result.output}")
    
    except Exception as e:
        logger.error(f"Sandbox testing failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        else:
            print(f"Error: {str(e)}")

@nextgen_cli.command()
async def interactive_training():
    """Enter interactive AI training mode"""
    try:
        trainer = InteractiveTrainingMode()
        await trainer.start_training_session()
    
    except Exception as e:
        logger.error(f"Interactive training failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        else:
            print(f"Error: {str(e)}")

@nextgen_cli.command()
async def learning_status():
    """Show AI learning and training status"""
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold cyan]🤖 AI Learning Status[/bold cyan]")
    else:
        print(f"\n🤖 AI Learning Status")
    
    try:
        # Get training pipeline status
        training_stats = training_pipeline.get_learning_statistics()
        
        # Get chained exploit engine stats
        chain_stats = chained_exploit_engine.get_execution_statistics()
        
        # Get sandbox metrics
        sandbox_metrics = sandbox_environment.get_sandbox_metrics()
        
        if RICH_AVAILABLE:
            # Create layout
            layout = Layout()
            layout.split_column(
                Layout(name="training", ratio=1),
                Layout(name="chains", ratio=1),
                Layout(name="sandbox", ratio=1)
            )
            
            # Training status table
            training_table = Table(title="Training Pipeline Status")
            training_table.add_column("Metric", style="cyan")
            training_table.add_column("Value", style="green")
            
            training_table.add_row("Learning Active", str(training_stats.get('learning_active', False)))
            training_table.add_row("Total Sessions", str(training_stats.get('total_training_sessions', 0)))
            training_table.add_row("Active Models", str(training_stats.get('active_models', 0)))
            training_table.add_row("Queue Size", str(training_stats.get('training_queue_size', 0)))
            
            # Chain execution table
            chain_table = Table(title="Exploit Chain Statistics")
            chain_table.add_column("Metric", style="cyan")
            chain_table.add_column("Value", style="green")
            
            chain_table.add_row("Total Executions", str(chain_stats.get('total_executions', 0)))
            chain_table.add_row("Success Rate", f"{chain_stats.get('success_rate', 0):.1%}")
            chain_table.add_row("Avg Execution Time", f"{chain_stats.get('average_execution_time', 0):.2f}s")
            chain_table.add_row("Known Chains", str(chain_stats.get('known_chains', 0)))
            
            # Sandbox metrics table
            sandbox_table = Table(title="Sandbox Environment")
            sandbox_table.add_column("Metric", style="cyan")
            sandbox_table.add_column("Value", style="green")
            
            sandbox_table.add_row("Total Tests", str(sandbox_metrics.get('total_executions', 0)))
            sandbox_table.add_row("Success Rate", f"{sandbox_metrics.get('success_rate', 0):.1%}")
            sandbox_table.add_row("Active Containers", str(sandbox_metrics.get('active_containers', 0)))
            sandbox_table.add_row("Docker Available", str(sandbox_metrics.get('docker_available', False)))
            
            layout["training"].update(Panel(training_table, title="Training", border_style="blue"))
            layout["chains"].update(Panel(chain_table, title="Chains", border_style="green"))
            layout["sandbox"].update(Panel(sandbox_table, title="Sandbox", border_style="yellow"))
            
            console.print(layout)
        else:
            print("\nTraining Pipeline Status:")
            print(f"Learning Active: {training_stats.get('learning_active', False)}")
            print(f"Total Sessions: {training_stats.get('total_training_sessions', 0)}")
            print(f"Active Models: {training_stats.get('active_models', 0)}")
            
            print("\nExploit Chain Statistics:")
            print(f"Total Executions: {chain_stats.get('total_executions', 0)}")
            print(f"Success Rate: {chain_stats.get('success_rate', 0):.1%}")
            print(f"Known Chains: {chain_stats.get('known_chains', 0)}")
            
            print("\nSandbox Environment:")
            print(f"Total Tests: {sandbox_metrics.get('total_executions', 0)}")
            print(f"Success Rate: {sandbox_metrics.get('success_rate', 0):.1%}")
            print(f"Docker Available: {sandbox_metrics.get('docker_available', False)}")
    
    except Exception as e:
        logger.error(f"Status retrieval failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        else:
            print(f"Error: {str(e)}")

@nextgen_cli.command()
@click.option('--enable', is_flag=True, help='Enable continuous learning')
@click.option('--disable', is_flag=True, help='Disable continuous learning')
async def continuous_learning(enable, disable):
    """Manage continuous AI learning"""
    
    try:
        if enable:
            await training_pipeline.start_continuous_learning()
            if RICH_AVAILABLE:
                console.print("[bold green]✅ Continuous learning enabled[/bold green]")
            else:
                print("✅ Continuous learning enabled")
        
        elif disable:
            await training_pipeline.stop_continuous_learning()
            if RICH_AVAILABLE:
                console.print("[bold yellow]⏸️ Continuous learning disabled[/bold yellow]")
            else:
                print("⏸️ Continuous learning disabled")
        
        else:
            # Show current status
            stats = training_pipeline.get_learning_statistics()
            status = "enabled" if stats.get('learning_active', False) else "disabled"
            
            if RICH_AVAILABLE:
                console.print(f"Continuous learning is [bold cyan]{status}[/bold cyan]")
            else:
                print(f"Continuous learning is {status}")
    
    except Exception as e:
        logger.error(f"Continuous learning management failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")
        else:
            print(f"Error: {str(e)}")

def main():
    """Main CLI entry point"""
    try:
        # Run async CLI
        import sys
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        
        asyncio.run(nextgen_cli())
    
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("\n[bold red]🛑 Operation cancelled by user[/bold red]")
        else:
            print("\n🛑 Operation cancelled by user")
    
    except Exception as e:
        logger.error(f"CLI execution failed: {str(e)}")
        if RICH_AVAILABLE:
            console.print(f"[bold red]Fatal Error: {str(e)}[/bold red]")
        else:
            print(f"Fatal Error: {str(e)}")

if __name__ == "__main__":
    main()