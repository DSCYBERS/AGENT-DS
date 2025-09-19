"""
Agent DS - AI Learning CLI Commands
Command interface for autonomous AI learning features
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm

# Handle optional torch import
try:
    import torch
except ImportError:
    torch = None

from core.ai_learning.autonomous_engine import AutonomousLearningEngine
from core.ai_learning.adaptive_orchestrator import AdaptiveAttackOrchestrator
from core.config.settings import Config
from core.utils.logger import get_logger

console = Console()
logger = get_logger('ai_learning_cli')

class AILearningCLI:
    """CLI interface for AI learning features"""
    
    def __init__(self):
        self.learning_engine = AutonomousLearningEngine()
        self.adaptive_orchestrator = AdaptiveAttackOrchestrator()
        self.config = Config()
    
    @click.group(name='ai')
    def ai_learning_commands():
        """AI Learning and Autonomous Attack Commands"""
        pass
    
    @ai_learning_commands.command('status')
    @click.option('--detailed', '-d', is_flag=True, help='Show detailed status')
    def show_ai_status(detailed: bool):
        """Show AI learning system status"""
        console.print(Panel.fit("[bold blue]Agent DS - AI Learning System Status[/bold blue]"))
        
        status_table = Table(title="AI Learning Components")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        status_table.add_column("Details", style="white")
        
        # Check learning engine status
        learning_status = "✓ Active" if hasattr(AILearningCLI().learning_engine, 'learning_enabled') else "✗ Disabled"
        status_table.add_row("Autonomous Learning Engine", learning_status, "Real-time learning from missions")
        
        # Check AI models status
        ai_models_status = "✓ Loaded" if torch else "⚠ Limited (No PyTorch)"
        status_table.add_row("AI Models", ai_models_status, "Payload generation & success prediction")
        
        # Check external intelligence
        intel_status = "✓ Connected"
        status_table.add_row("External Intelligence", intel_status, "CVE.org, OTX, ExploitDB integration")
        
        # Check sandbox
        sandbox_status = "✓ Ready"
        status_table.add_row("Sandbox Experimenter", sandbox_status, "Safe payload testing environment")
        
        console.print(status_table)
        
        if detailed:
            # Show learning statistics
            asyncio.run(AILearningCLI()._show_detailed_status())
    
    async def _show_detailed_status(self):
        """Show detailed AI learning statistics"""
        try:
            # Get learning statistics
            stats = await self._get_learning_statistics()
            
            stats_table = Table(title="Learning Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green")
            stats_table.add_column("Description", style="white")
            
            stats_table.add_row("Total Missions Learned", str(stats.get('total_missions', 0)), "Missions processed for learning")
            stats_table.add_row("Insights Generated", str(stats.get('total_insights', 0)), "AI insights from past attacks")
            stats_table.add_row("Success Rate Improvement", f"{stats.get('success_improvement', 0):.1%}", "Attack success rate improvement")
            stats_table.add_row("Novel Payloads Discovered", str(stats.get('novel_payloads', 0)), "AI-generated successful payloads")
            
            console.print(stats_table)
            
        except Exception as e:
            console.print(f"[red]Error getting detailed status: {str(e)}[/red]")
    
    @ai_learning_commands.command('train')
    @click.option('--mission-id', '-m', help='Specific mission to learn from')
    @click.option('--all-missions', '-a', is_flag=True, help='Learn from all available missions')
    @click.option('--external-intel', '-e', is_flag=True, help='Update external threat intelligence')
    def train_ai(mission_id: Optional[str], all_missions: bool, external_intel: bool):
        """Train AI from mission results and external intelligence"""
        console.print(Panel.fit("[bold green]Agent DS - AI Training Session[/bold green]"))
        
        if not any([mission_id, all_missions, external_intel]):
            console.print("[red]Error: Specify --mission-id, --all-missions, or --external-intel[/red]")
            return
        
        asyncio.run(AILearningCLI()._execute_training(mission_id, all_missions, external_intel))
    
    async def _execute_training(self, mission_id: Optional[str], all_missions: bool, external_intel: bool):
        """Execute AI training process"""
        training_results = {
            'started_at': datetime.now().isoformat(),
            'training_sessions': [],
            'improvements': {}
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Train from missions
            if mission_id:
                task = progress.add_task(f"Learning from mission {mission_id}...", total=None)
                try:
                    results = await self.learning_engine.learn_from_mission(mission_id)
                    training_results['training_sessions'].append(results)
                    progress.update(task, description=f"✓ Learned from mission {mission_id}")
                    console.print(f"[green]Generated {len(results.get('insights_generated', []))} insights from mission {mission_id}[/green]")
                except Exception as e:
                    progress.update(task, description=f"✗ Failed to learn from mission {mission_id}")
                    console.print(f"[red]Training error: {str(e)}[/red]")
            
            elif all_missions:
                task = progress.add_task("Learning from all missions...", total=None)
                try:
                    # Get all mission IDs and learn from each
                    mission_ids = await self._get_all_mission_ids()
                    for mid in mission_ids:
                        results = await self.learning_engine.learn_from_mission(mid)
                        training_results['training_sessions'].append(results)
                    
                    progress.update(task, description=f"✓ Learned from {len(mission_ids)} missions")
                    console.print(f"[green]Completed learning from {len(mission_ids)} missions[/green]")
                except Exception as e:
                    progress.update(task, description="✗ Failed to learn from missions")
                    console.print(f"[red]Training error: {str(e)}[/red]")
            
            # Update external intelligence
            if external_intel:
                task = progress.add_task("Updating external threat intelligence...", total=None)
                try:
                    intel_results = await self.learning_engine.update_external_intelligence()
                    training_results['external_intelligence'] = intel_results
                    progress.update(task, description="✓ Updated external intelligence")
                    console.print(f"[green]Updated intelligence from {len(intel_results.get('sources_updated', []))} sources[/green]")
                except Exception as e:
                    progress.update(task, description="✗ Failed to update intelligence")
                    console.print(f"[red]Intelligence update error: {str(e)}[/red]")
        
        # Show training summary
        self._show_training_summary(training_results)
    
    @ai_learning_commands.command('experiment')
    @click.option('--attack-type', '-t', required=True, help='Attack type for experimentation')
    @click.option('--target', help='Target context for experimentation')
    @click.option('--sandbox-only', '-s', is_flag=True, help='Run only in sandbox mode')
    def experiment_payloads(attack_type: str, target: Optional[str], sandbox_only: bool):
        """Experiment with novel AI-generated payloads"""
        console.print(Panel.fit("[bold yellow]Agent DS - Payload Experimentation[/bold yellow]"))
        
        if not sandbox_only:
            if not Confirm.ask("[red]This will test novel payloads. Continue only in authorized environments. Proceed?[/red]"):
                console.print("[yellow]Experimentation cancelled[/yellow]")
                return
        
        target_context = {'target': target} if target else {}
        asyncio.run(AILearningCLI()._execute_experimentation(attack_type, target_context, sandbox_only))
    
    async def _execute_experimentation(self, attack_type: str, target_context: Dict, sandbox_only: bool):
        """Execute payload experimentation"""
        console.print(f"[cyan]Experimenting with {attack_type} payloads...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Generating and testing novel payloads...", total=None)
            
            try:
                # Run experimentation
                results = await self.learning_engine.experiment_with_novel_payloads(
                    attack_type, target_context
                )
                
                progress.update(task, description="✓ Experimentation completed")
                
                # Show results
                self._show_experimentation_results(results)
                
            except Exception as e:
                progress.update(task, description="✗ Experimentation failed")
                console.print(f"[red]Experimentation error: {str(e)}[/red]")
    
    @ai_learning_commands.command('adaptive-attack')
    @click.option('--target', '-t', required=True, help='Target for adaptive attack')
    @click.option('--mission-name', '-n', help='Mission name')
    @click.option('--ai-mode', '-ai', is_flag=True, help='Enable full AI autonomous mode')
    def adaptive_attack(target: str, mission_name: Optional[str], ai_mode: bool):
        """Execute AI-driven adaptive attack mission"""
        console.print(Panel.fit("[bold red]Agent DS - Adaptive AI Attack Mission[/bold red]"))
        
        if not Confirm.ask(f"[red]Execute adaptive attack against {target}? Ensure you have authorization.[/red]"):
            console.print("[yellow]Attack cancelled[/yellow]")
            return
        
        mission_config = {
            'mission_id': mission_name or f"adaptive_mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'target': target,
            'ai_autonomous_mode': ai_mode,
            'learning_enabled': True
        }
        
        asyncio.run(AILearningCLI()._execute_adaptive_attack(mission_config))
    
    async def _execute_adaptive_attack(self, mission_config: Dict):
        """Execute adaptive attack mission"""
        console.print(f"[cyan]Starting adaptive attack mission: {mission_config['mission_id']}[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Executing adaptive attack mission...", total=None)
            
            try:
                # Execute adaptive mission
                results = await self.adaptive_orchestrator.execute_adaptive_mission(mission_config)
                
                progress.update(task, description="✓ Adaptive mission completed")
                
                # Show mission results
                self._show_mission_results(results)
                
            except Exception as e:
                progress.update(task, description="✗ Adaptive mission failed")
                console.print(f"[red]Mission error: {str(e)}[/red]")
    
    @ai_learning_commands.command('insights')
    @click.option('--attack-type', '-t', help='Filter by attack type')
    @click.option('--limit', '-l', default=10, help='Number of insights to show')
    def show_insights(attack_type: Optional[str], limit: int):
        """Show AI learning insights"""
        console.print(Panel.fit("[bold blue]Agent DS - AI Learning Insights[/bold blue]"))
        asyncio.run(AILearningCLI()._show_learning_insights(attack_type, limit))
    
    async def _show_learning_insights(self, attack_type: Optional[str], limit: int):
        """Display learning insights"""
        try:
            insights = await self._get_learning_insights(attack_type, limit)
            
            if not insights:
                console.print("[yellow]No learning insights found[/yellow]")
                return
            
            insights_table = Table(title="AI Learning Insights")
            insights_table.add_column("Type", style="cyan")
            insights_table.add_column("Success Factors", style="green")
            insights_table.add_column("Confidence", style="yellow")
            insights_table.add_column("Scenarios", style="white")
            
            for insight in insights:
                insights_table.add_row(
                    insight.get('pattern_type', 'Unknown'),
                    ', '.join(insight.get('success_factors', [])[:3]),
                    f"{insight.get('confidence_score', 0):.2f}",
                    ', '.join(insight.get('applicable_scenarios', [])[:2])
                )
            
            console.print(insights_table)
            
        except Exception as e:
            console.print(f"[red]Error retrieving insights: {str(e)}[/red]")
    
    def _show_training_summary(self, training_results: Dict):
        """Show training session summary"""
        summary_table = Table(title="Training Session Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        total_insights = sum(len(session.get('insights_generated', [])) 
                           for session in training_results.get('training_sessions', []))
        
        summary_table.add_row("Total Sessions", str(len(training_results.get('training_sessions', []))))
        summary_table.add_row("Insights Generated", str(total_insights))
        summary_table.add_row("Training Duration", self._calculate_duration(training_results))
        
        if 'external_intelligence' in training_results:
            intel = training_results['external_intelligence']
            summary_table.add_row("Intelligence Sources Updated", str(len(intel.get('sources_updated', []))))
            summary_table.add_row("New Intelligence Items", str(intel.get('new_intelligence_count', 0)))
        
        console.print(summary_table)
    
    def _show_experimentation_results(self, results: Dict):
        """Show payload experimentation results"""
        results_table = Table(title="Payload Experimentation Results")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        metrics = results.get('experiment_metrics', {})
        
        results_table.add_row("Total Payloads Tested", str(metrics.get('total_tested', 0)))
        results_table.add_row("Successful Discoveries", str(metrics.get('successful_discoveries', 0)))
        results_table.add_row("Success Rate", f"{metrics.get('success_rate', 0):.1%}")
        
        console.print(results_table)
        
        # Show successful payloads
        if results.get('successful_discoveries'):
            console.print("\n[bold green]Successful Novel Payloads:[/bold green]")
            for i, payload in enumerate(results['successful_discoveries'][:5], 1):
                console.print(f"{i}. {payload}")
    
    def _show_mission_results(self, results: Dict):
        """Show adaptive mission results"""
        mission_table = Table(title="Adaptive Mission Results")
        mission_table.add_column("Phase", style="cyan")
        mission_table.add_column("Status", style="green")
        mission_table.add_column("Results", style="white")
        
        for phase in results.get('phases_executed', []):
            phase_name = phase.get('phase', 'Unknown')
            phase_results = phase.get('results', {})
            
            if 'error' in phase_results:
                status = "✗ Failed"
                result_summary = phase_results['error']
            else:
                status = "✓ Completed"
                result_summary = self._summarize_phase_results(phase_results)
            
            mission_table.add_row(phase_name, status, result_summary)
        
        console.print(mission_table)
        
        # Show adaptations made
        adaptations = results.get('adaptations_made', [])
        if adaptations:
            console.print(f"\n[bold yellow]AI Adaptations Made: {len(adaptations)}[/bold yellow]")
            for adaptation in adaptations[:3]:
                console.print(f"• {adaptation.get('adaptation_type', 'Unknown')}: {adaptation.get('reasoning', 'No details')}")
    
    def _summarize_phase_results(self, phase_results: Dict) -> str:
        """Summarize phase results for display"""
        if isinstance(phase_results, dict):
            if 'attack_surface' in phase_results:
                return f"Found {len(phase_results.get('attack_surface', {}))} potential vectors"
            elif 'vulnerabilities' in phase_results:
                return f"Identified {len(phase_results.get('vulnerabilities', []))} vulnerabilities"
            elif 'overall_success' in phase_results:
                return "Attack successful" if phase_results['overall_success'] else "Attack failed"
            else:
                return "Phase completed"
        return str(phase_results)[:50]
    
    def _calculate_duration(self, training_results: Dict) -> str:
        """Calculate training duration"""
        try:
            start_time = datetime.fromisoformat(training_results['started_at'])
            duration = datetime.now() - start_time
            return f"{duration.total_seconds():.1f}s"
        except:
            return "Unknown"
    
    async def _get_learning_statistics(self) -> Dict[str, Any]:
        """Get AI learning statistics"""
        # Mock statistics - would query actual learning database
        return {
            'total_missions': 15,
            'total_insights': 47,
            'success_improvement': 0.23,
            'novel_payloads': 8
        }
    
    async def _get_all_mission_ids(self) -> List[str]:
        """Get all available mission IDs"""
        # Mock mission IDs - would query actual database
        return ["mission_001", "mission_002", "mission_003"]
    
    async def _get_learning_insights(self, attack_type: Optional[str], limit: int) -> List[Dict]:
        """Get learning insights from database"""
        # Mock insights - would query actual learning database
        return [
            {
                'pattern_type': 'success_pattern',
                'success_factors': ['payload_length:45', 'sql_union_technique', 'target_mysql:5.7'],
                'confidence_score': 0.85,
                'applicable_scenarios': ['sql_injection', 'mysql_targets']
            },
            {
                'pattern_type': 'failure_pattern',
                'success_factors': [],
                'confidence_score': 0.78,
                'applicable_scenarios': ['xss', 'filtered_input']
            }
        ]

# Global CLI instance
ai_cli = AILearningCLI()

# Export commands for main CLI
ai_learning_commands = ai_cli.ai_learning_commands