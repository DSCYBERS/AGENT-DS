"""
Agent DS - AI Terminal Integration
==================================

Integration module connecting AI Thinking Model with Live Terminal Interface
for real-time autonomous attack monitoring and control.

Features:
- Real-time AI thinking visualization
- Attack progress monitoring
- Mission control integration
- Interactive command processing
- Autonomous feedback loops
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import logging

from core.interface.live_terminal import LiveTerminalInterface
from core.attacks.ai_core import AIThinkingModel


class AITerminalIntegration:
    """
    Integration layer between AI Thinking Model and Live Terminal Interface
    """
    
    def __init__(self):
        # Initialize components
        self.terminal = LiveTerminalInterface()
        self.ai_core = None  # Will be initialized with terminal callback
        
        # Mission state
        self.active_mission = None
        self.mission_context = {}
        self.attack_results = {}
        self.real_time_stats = {
            'mission_start_time': None,
            'attacks_completed': 0,
            'vulnerabilities_discovered': 0,
            'ai_adaptations': 0,
            'success_rate': 0.0
        }
        
        # Message routing
        self.message_handlers = {
            'ai_thinking': self._handle_ai_thinking,
            'attack_progress': self._handle_attack_progress,
            'mission_update': self._handle_mission_update,
            'vulnerability_found': self._handle_vulnerability_found,
            'adaptation_made': self._handle_adaptation_made
        }
        
        # Setup logging
        self.logger = logging.getLogger('AgentDS.AITerminal')
    
    async def initialize(self):
        """
        Initialize the AI Terminal Integration system
        """
        # Initialize AI core with terminal callback
        self.ai_core = AIThinkingModel(terminal_callback=self._ai_terminal_callback)
        
        # Setup terminal message routing
        await self._setup_terminal_integration()
        
        self.logger.info("AI Terminal Integration initialized")
    
    async def _setup_terminal_integration(self):
        """
        Setup integration between AI core and terminal interface
        """
        # Connect AI callbacks to terminal
        self.ai_core.terminal_callback = self._ai_terminal_callback
        
        # Enhanced terminal with AI command handlers
        self.terminal.command_handlers.update({
            'ai-mission': self._cmd_ai_mission,
            'ai-adapt': self._cmd_ai_adapt,
            'ai-analyze': self._cmd_ai_analyze,
            'ai-recommend': self._cmd_ai_recommend,
            'mission-auto': self._cmd_mission_auto,
            'attack-auto': self._cmd_attack_auto,
            'feedback': self._cmd_feedback,
            'insights': self._cmd_insights
        })
    
    async def start_autonomous_mission(self, target_url: str, mission_params: Optional[Dict] = None):
        """
        Start fully autonomous penetration testing mission
        """
        try:
            # Initialize mission
            self.active_mission = {
                'target_url': target_url,
                'mission_id': f"mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'start_time': datetime.now().isoformat(),
                'status': 'initializing',
                'parameters': mission_params or {}
            }
            
            self.real_time_stats['mission_start_time'] = datetime.now()
            
            # Update terminal with mission start
            await self.terminal.mission_update_callback({
                'target': target_url,
                'status': 'Starting AI Analysis',
                'current_phase': 'AI Planning',
                'progress': 5.0
            })
            
            # AI Thinking Phase 1: Target Analysis and Planning
            await self._ai_terminal_callback('analyzing', f"🎯 Starting autonomous mission against {target_url}")
            
            target_data = {
                'target_url': target_url,
                'technologies': [],  # Will be populated by reconnaissance
                'ports': [],
                'mission_params': mission_params or {}
            }
            
            # AI generates comprehensive attack plan
            mission_context = await self.ai_core.think_and_plan(target_data)
            self.mission_context = mission_context
            
            await self.terminal.mission_update_callback({
                'status': 'AI Planning Complete',
                'current_phase': 'Autonomous Execution',
                'progress': 15.0
            })
            
            # AI Thinking Phase 2: Workflow Orchestration
            orchestration_plan = await self.ai_core.workflow_orchestration(mission_context)
            
            await self._ai_terminal_callback('orchestrating', 
                f"📋 AI orchestrated {len(orchestration_plan['next_phases'])} attack phases")
            
            # Start autonomous execution
            await self._execute_autonomous_attacks(orchestration_plan)
            
        except Exception as e:
            self.logger.error(f"Error in autonomous mission: {e}")
            await self._ai_terminal_callback('error', f"❌ Mission error: {str(e)}")
    
    async def _execute_autonomous_attacks(self, orchestration_plan: Dict[str, Any]):
        """
        Execute autonomous attacks based on AI orchestration plan
        """
        phases = orchestration_plan.get('next_phases', [])
        total_phases = len(phases)
        
        for i, phase in enumerate(phases):
            phase_name = phase if isinstance(phase, str) else phase.get('phase', f'Phase {i+1}')
            
            # Update progress
            progress = ((i + 1) / total_phases) * 85 + 15  # Start from 15%
            
            await self.terminal.mission_update_callback({
                'current_phase': phase_name.replace('_', ' ').title(),
                'progress': progress
            })
            
            # AI monitors and adapts in real-time
            await self._ai_terminal_callback('analyzing', f"🔍 Executing {phase_name}")
            
            # Simulate attack execution with real-time monitoring
            attack_result = await self._execute_attack_phase(phase_name, phase)
            
            # AI real-time adaptation based on results
            if not attack_result.get('success', False):
                await self._ai_terminal_callback('adapting', 
                    f"🔄 Adapting strategy for {phase_name}")
                
                adaptation = await self.ai_core.real_time_adaptation(
                    attack_result, 
                    attack_result.get('payload', ''),
                    attack_result.get('response', '')
                )
                
                if adaptation.get('should_retry', False):
                    await self._ai_terminal_callback('adapting', 
                        f"🧬 Retrying {phase_name} with adapted payload")
                    
                    # Retry with adapted strategy
                    retry_result = await self._execute_attack_phase(
                        phase_name, 
                        phase, 
                        adapted_payload=adaptation.get('new_payload')
                    )
                    
                    if retry_result.get('success', False):
                        self.real_time_stats['ai_adaptations'] += 1
                        await self._ai_terminal_callback('success', 
                            f"✅ AI adaptation successful for {phase_name}")
            
            # Update attack results
            self.attack_results[phase_name] = attack_result
            self._update_real_time_stats(attack_result)
            
            # AI generates next attack recommendation
            if i < total_phases - 1:  # Not the last phase
                recommendation = await self.ai_core.get_next_attack_recommendation(self.attack_results)
                
                await self._ai_terminal_callback('recommending', 
                    f"💡 AI recommends: {recommendation.get('recommended_phase', 'Continue sequence')}")
        
        # Mission completion
        await self._complete_autonomous_mission()
    
    async def _execute_attack_phase(self, phase_name: str, phase_info: Any, 
                                  adapted_payload: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute individual attack phase with real-time monitoring
        """
        # Update terminal attack status
        await self.terminal.attack_status_callback(phase_name, {
            'status': 'Starting',
            'progress': 0,
            'success': False
        })
        
        # Simulate attack execution with progress updates
        for progress in range(0, 101, 25):
            await asyncio.sleep(1)  # Simulate attack time
            
            await self.terminal.attack_status_callback(phase_name, {
                'status': 'Running',
                'progress': progress,
                'success': False
            })
            
            # AI continuous monitoring
            if progress == 50:
                await self._ai_terminal_callback('analyzing', 
                    f"📊 Monitoring {phase_name} progress: {progress}%")
        
        # Simulate attack result
        import random
        success = random.random() > 0.4  # 60% success rate
        
        # Final status update
        await self.terminal.attack_status_callback(phase_name, {
            'status': 'Completed' if success else 'Failed',
            'progress': 100,
            'success': success
        })
        
        # Generate realistic attack result
        attack_result = {
            'phase': phase_name,
            'success': success,
            'attack_type': phase_name,
            'vulnerabilities_found': random.randint(0, 3) if success else 0,
            'payload': adapted_payload or f"auto_payload_{phase_name}",
            'response': f"Response for {phase_name}",
            'confidence': 'high' if success else 'low',
            'duration': random.uniform(30, 120)
        }
        
        if success:
            await self._ai_terminal_callback('success', 
                f"🎯 {phase_name} successful - {attack_result['vulnerabilities_found']} vulnerabilities found")
        else:
            await self._ai_terminal_callback('info', 
                f"⚠️ {phase_name} completed - no vulnerabilities found")
        
        return attack_result
    
    async def _complete_autonomous_mission(self):
        """
        Complete autonomous mission and generate AI report
        """
        await self.terminal.mission_update_callback({
            'status': 'Generating AI Report',
            'current_phase': 'Mission Analysis',
            'progress': 95.0
        })
        
        # AI generates comprehensive mission report
        await self._ai_terminal_callback('analyzing', "📊 AI generating comprehensive mission report...")
        
        mission_results = {
            'target_url': self.active_mission['target_url'],
            'attack_results': self.attack_results,
            'total_duration': (datetime.now() - self.real_time_stats['mission_start_time']).total_seconds(),
            'overall_success_rate': self.real_time_stats['success_rate'],
            'adaptation_count': self.real_time_stats['ai_adaptations'],
            'all_vulnerabilities': self._extract_all_vulnerabilities()
        }
        
        ai_report = await self.ai_core.generate_mission_report(mission_results)
        
        await self.terminal.mission_update_callback({
            'status': 'Mission Complete',
            'current_phase': 'Report Generated',
            'progress': 100.0
        })
        
        # AI provides final insights
        await self._ai_terminal_callback('success', 
            f"✅ Autonomous mission complete - {self.real_time_stats['vulnerabilities_discovered']} vulnerabilities found")
        
        await self._ai_terminal_callback('recommending', 
            f"📋 AI generated {len(ai_report)} section comprehensive report")
        
        # Store mission data for future learning
        await self.ai_core.update_learning_model([
            {
                'attack_type': phase,
                'success': result.get('success', False),
                'target_characteristics': self.mission_context.get('target_analysis', {}),
                'success_indicators': result.get('vulnerabilities_found', []),
                'failure_indicators': [] if result.get('success') else ['no_vulnerabilities']
            }
            for phase, result in self.attack_results.items()
        ])
        
        self.logger.info(f"Autonomous mission completed: {len(self.attack_results)} phases executed")
    
    def _update_real_time_stats(self, attack_result: Dict[str, Any]):
        """
        Update real-time mission statistics
        """
        self.real_time_stats['attacks_completed'] += 1
        
        vulns_found = attack_result.get('vulnerabilities_found', 0)
        self.real_time_stats['vulnerabilities_discovered'] += vulns_found
        
        # Update success rate
        successful_attacks = sum(1 for result in self.attack_results.values() 
                               if result.get('success', False))
        total_attacks = len(self.attack_results)
        
        if total_attacks > 0:
            self.real_time_stats['success_rate'] = successful_attacks / total_attacks
        
        # Update terminal mission stats
        asyncio.create_task(self.terminal.mission_update_callback({
            'attacks_launched': self.real_time_stats['attacks_completed'],
            'vulnerabilities_found': self.real_time_stats['vulnerabilities_discovered'],
            'success_rate': self.real_time_stats['success_rate']
        }))
    
    def _extract_all_vulnerabilities(self) -> List[Dict[str, Any]]:
        """
        Extract all vulnerabilities found during mission
        """
        all_vulns = []
        
        for phase, result in self.attack_results.items():
            if result.get('success', False):
                vuln_count = result.get('vulnerabilities_found', 0)
                for i in range(vuln_count):
                    all_vulns.append({
                        'type': phase,
                        'severity': 'high' if result.get('confidence') == 'high' else 'medium',
                        'exploitable': True,
                        'phase': phase,
                        'payload': result.get('payload', ''),
                        'found_at': datetime.now().isoformat()
                    })
        
        return all_vulns
    
    # AI Terminal Callback Implementation
    async def _ai_terminal_callback(self, thinking_type: str, message: str):
        """
        Callback for AI thinking updates to terminal interface
        """
        await self.terminal.ai_thinking_callback(thinking_type, message)
        
        # Log AI activity
        self.logger.info(f"AI {thinking_type}: {message}")
    
    # Message Handlers
    async def _handle_ai_thinking(self, data: Dict[str, Any]):
        """Handle AI thinking messages"""
        await self._ai_terminal_callback(data.get('type', 'info'), data.get('message', ''))
    
    async def _handle_attack_progress(self, data: Dict[str, Any]):
        """Handle attack progress updates"""
        await self.terminal.attack_status_callback(
            data.get('attack_name', 'unknown'),
            data
        )
    
    async def _handle_mission_update(self, data: Dict[str, Any]):
        """Handle mission updates"""
        await self.terminal.mission_update_callback(data)
    
    async def _handle_vulnerability_found(self, data: Dict[str, Any]):
        """Handle vulnerability discovery"""
        self.real_time_stats['vulnerabilities_discovered'] += 1
        await self._ai_terminal_callback('success', 
            f"🔍 Vulnerability found: {data.get('type', 'Unknown')} - {data.get('severity', 'medium')} severity")
    
    async def _handle_adaptation_made(self, data: Dict[str, Any]):
        """Handle AI adaptations"""
        self.real_time_stats['ai_adaptations'] += 1
        await self._ai_terminal_callback('adapting', 
            f"🧬 AI adapted strategy: {data.get('reason', 'Strategy optimization')}")
    
    # Enhanced Command Handlers
    async def _cmd_ai_mission(self, args: List[str]) -> str:
        """AI mission control commands"""
        if not args:
            return "Usage: ai-mission [start|status|report] [target]"
        
        subcmd = args[0].lower()
        
        if subcmd == "start":
            if len(args) < 2:
                return "Usage: ai-mission start <target_url>"
            
            target = args[1]
            await self.start_autonomous_mission(target)
            return f"🚀 AI autonomous mission started against {target}"
        
        elif subcmd == "status":
            if self.active_mission:
                return f"""
🤖 AI MISSION STATUS:
   Target: {self.active_mission['target_url']}
   Duration: {(datetime.now() - self.real_time_stats['mission_start_time']).total_seconds():.0f}s
   Attacks: {self.real_time_stats['attacks_completed']}
   Vulnerabilities: {self.real_time_stats['vulnerabilities_discovered']}
   AI Adaptations: {self.real_time_stats['ai_adaptations']}
   Success Rate: {self.real_time_stats['success_rate']:.1%}
                """
            else:
                return "ℹ️ No active AI mission."
        
        elif subcmd == "report":
            if self.attack_results:
                return f"📊 AI mission report: {len(self.attack_results)} phases completed"
            else:
                return "⚠️ No mission data for report generation."
        
        return "Unknown AI mission command."
    
    async def _cmd_ai_adapt(self, args: List[str]) -> str:
        """Trigger AI adaptation"""
        if self.ai_core and self.attack_results:
            await self._ai_terminal_callback('adapting', "🔄 Manual AI adaptation triggered")
            return "🧠 AI adaptation process initiated"
        else:
            return "⚠️ No active mission for AI adaptation"
    
    async def _cmd_ai_analyze(self, args: List[str]) -> str:
        """Trigger AI analysis"""
        if self.ai_core:
            await self._ai_terminal_callback('analyzing', "🔍 Manual AI analysis triggered")
            return "🧠 AI analysis process initiated"
        else:
            return "⚠️ AI core not initialized"
    
    async def _cmd_ai_recommend(self, args: List[str]) -> str:
        """Get AI recommendations"""
        if self.ai_core and self.attack_results:
            recommendation = await self.ai_core.get_next_attack_recommendation(self.attack_results)
            
            await self._ai_terminal_callback('recommending', 
                f"💡 {recommendation.get('reasoning', 'No specific recommendations')}")
            
            return f"🎯 AI Recommendation: {recommendation.get('recommended_phase', 'Continue current strategy')}"
        else:
            return "⚠️ No mission data for AI recommendations"
    
    async def _cmd_mission_auto(self, args: List[str]) -> str:
        """Start fully autonomous mission"""
        if len(args) < 1:
            return "Usage: mission-auto <target_url>"
        
        target = args[0]
        await self.start_autonomous_mission(target, {'mode': 'fully_autonomous'})
        return f"🤖 Fully autonomous mission started against {target}"
    
    async def _cmd_attack_auto(self, args: List[str]) -> str:
        """Enable automatic attack execution"""
        return "🎯 Automatic attack execution enabled - AI will control all attack phases"
    
    async def _cmd_feedback(self, args: List[str]) -> str:
        """Show real-time feedback"""
        return f"""
📡 REAL-TIME FEEDBACK:
   
   🎯 Mission Progress: {self.real_time_stats['attacks_completed']} attacks completed
   🔍 Discoveries: {self.real_time_stats['vulnerabilities_discovered']} vulnerabilities
   🧠 AI Adaptations: {self.real_time_stats['ai_adaptations']} strategy changes
   📈 Success Rate: {self.real_time_stats['success_rate']:.1%}
   
   🤖 AI Status: {'ACTIVE' if self.ai_core else 'INACTIVE'}
   📊 Terminal: LIVE MONITORING
        """
    
    async def _cmd_insights(self, args: List[str]) -> str:
        """Get AI insights"""
        insights = "🧠 LATEST AI INSIGHTS:\n\n"
        
        if self.attack_results:
            successful_phases = [name for name, result in self.attack_results.items() 
                               if result.get('success', False)]
            failed_phases = [name for name, result in self.attack_results.items() 
                           if not result.get('success', False)]
            
            insights += f"✅ Successful: {', '.join(successful_phases) if successful_phases else 'None'}\n"
            insights += f"❌ Failed: {', '.join(failed_phases) if failed_phases else 'None'}\n"
            insights += f"🧬 Adaptations: {self.real_time_stats['ai_adaptations']} AI strategy changes\n"
        else:
            insights += "No mission data available for insights."
        
        return insights


# Demo function for testing
async def demo_ai_terminal_integration():
    """
    Demonstrate AI Terminal Integration
    """
    integration = AITerminalIntegration()
    await integration.initialize()
    
    # Start terminal interface
    terminal_task = asyncio.create_task(integration.terminal.start_interface())
    
    # Wait for interface to start
    await asyncio.sleep(3)
    
    # Start autonomous mission
    mission_task = asyncio.create_task(
        integration.start_autonomous_mission("https://demo.testfire.net")
    )
    
    # Run both tasks
    await asyncio.gather(terminal_task, mission_task)


if __name__ == "__main__":
    try:
        asyncio.run(demo_ai_terminal_integration())
    except KeyboardInterrupt:
        print("\n👋 AI Terminal Integration shutting down...")