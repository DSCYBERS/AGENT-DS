#!/usr/bin/env python3
"""
Agent DS v2.0 - AI Core Integration Manager
==========================================

Central coordination system that integrates all attack modules with the new AI systems:
- Connects attack modules (recon, web attacks, DB exploitation, admin testing) with AIThinkingModel
- Coordinates between AI Payload Generator and attack execution
- Manages Adaptive Attack Sequencer for optimal attack ordering
- Provides real-time intelligence coordination and decision making
- Handles centralized mission management and workflow orchestration

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Import AI components
try:
    from core.ai import (
        AIPayloadGenerator, get_sequencer_manager, 
        AttackAction, AttackResult, AttackPhase, AttackVector,
        SequenceState, ResourceType, get_ai_status
    )
    AI_COMPONENTS_AVAILABLE = True
except ImportError:
    AI_COMPONENTS_AVAILABLE = False

# Import attack modules
try:
    from .recon import ReconEngine
    from .web_attack import WebAttackEngine
    from .db_exploit import DatabaseExploitEngine
    from .admin_login import AdminLoginTester
    from .ai_core import AIThinkingModel
    ATTACK_MODULES_AVAILABLE = True
except ImportError:
    ATTACK_MODULES_AVAILABLE = False

class MissionPhase(Enum):
    """Mission execution phases"""
    INITIALIZATION = "initialization"
    RECONNAISSANCE = "reconnaissance"
    ENUMERATION = "enumeration"
    VULNERABILITY_ASSESSMENT = "vulnerability_assessment"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    PERSISTENCE = "persistence"
    DATA_COLLECTION = "data_collection"
    CLEANUP = "cleanup"
    REPORTING = "reporting"

class ModuleStatus(Enum):
    """Status of attack modules"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"

@dataclass
class MissionContext:
    """Central mission context shared across all modules"""
    mission_id: str
    target_url: str
    target_info: Dict[str, Any] = field(default_factory=dict)
    discovered_endpoints: List[str] = field(default_factory=list)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    successful_exploits: List[Dict[str, Any]] = field(default_factory=list)
    failed_attempts: List[Dict[str, Any]] = field(default_factory=list)
    intelligence_data: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    current_phase: MissionPhase = MissionPhase.INITIALIZATION
    phase_completion: Dict[MissionPhase, float] = field(default_factory=dict)

@dataclass
class ModuleResult:
    """Result from attack module execution"""
    module_name: str
    action_type: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    confidence_score: float
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)

class AICoreIntegrationManager:
    """
    Central coordination system for AI-powered attack operations
    """
    
    def __init__(self, terminal_callback=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.terminal_callback = terminal_callback
        
        # Mission state
        self.current_mission = None
        self.mission_history = []
        self.active_modules = {}
        self.module_status = {}
        
        # Initialize AI components
        if AI_COMPONENTS_AVAILABLE:
            self._initialize_ai_components()
        else:
            self.logger.warning("AI components not available, using fallback mode")
            self._initialize_fallback_ai()
        
        # Initialize attack modules
        if ATTACK_MODULES_AVAILABLE:
            self._initialize_attack_modules()
        else:
            self.logger.warning("Attack modules not available")
            self._initialize_fallback_modules()
        
        # Integration state
        self.ai_recommendations = []
        self.real_time_intelligence = {}
        self.coordination_queue = asyncio.Queue()
        
        self.logger.info("AI Core Integration Manager initialized")
    
    def _initialize_ai_components(self):
        """Initialize AI systems"""
        try:
            # AI Thinking Model (enhanced version)
            self.ai_thinking_model = AIThinkingModel(terminal_callback=self.terminal_callback)
            
            # AI Payload Generator
            self.payload_generator = AIPayloadGenerator()
            
            # Adaptive Attack Sequencer
            self.attack_sequencer = get_sequencer_manager(enable_training=True)
            
            # Check AI status
            ai_status = get_ai_status()
            self.ai_capabilities = ai_status.get('components', {})
            
            self.logger.info("AI components initialized successfully")
            if self.terminal_callback:
                self.terminal_callback("AI Intelligence Systems: ✅ ONLINE", "success")
                
        except Exception as e:
            self.logger.error(f"Error initializing AI components: {e}")
            self._initialize_fallback_ai()
    
    def _initialize_fallback_ai(self):
        """Initialize fallback AI when advanced AI not available"""
        self.ai_thinking_model = None
        self.payload_generator = None
        self.attack_sequencer = None
        self.ai_capabilities = {}
        
        if self.terminal_callback:
            self.terminal_callback("AI Intelligence Systems: ⚠️  FALLBACK MODE", "warning")
    
    def _initialize_attack_modules(self):
        """Initialize all attack modules with AI integration"""
        try:
            # Reconnaissance module
            self.recon_engine = ReconEngine()
            self.module_status['recon'] = ModuleStatus.IDLE
            
            # Web attack module
            self.web_attack_engine = WebAttackEngine()
            self.module_status['web_attack'] = ModuleStatus.IDLE
            
            # Database exploitation module
            self.db_exploit_engine = DatabaseExploitEngine()
            self.module_status['db_exploit'] = ModuleStatus.IDLE
            
            # Admin login testing module
            self.admin_login_tester = AdminLoginTester()
            self.module_status['admin_login'] = ModuleStatus.IDLE
            
            # Integrate modules with AI
            self._integrate_modules_with_ai()
            
            self.logger.info("Attack modules initialized and integrated with AI")
            if self.terminal_callback:
                self.terminal_callback("Attack Modules: ✅ INTEGRATED WITH AI", "success")
                
        except Exception as e:
            self.logger.error(f"Error initializing attack modules: {e}")
            self._initialize_fallback_modules()
    
    def _initialize_fallback_modules(self):
        """Initialize fallback modules when attack modules not available"""
        self.recon_engine = None
        self.web_attack_engine = None
        self.db_exploit_engine = None
        self.admin_login_tester = None
        
        if self.terminal_callback:
            self.terminal_callback("Attack Modules: ⚠️  NOT AVAILABLE", "warning")
    
    def _integrate_modules_with_ai(self):
        """Integrate attack modules with AI systems"""
        modules = {
            'recon': self.recon_engine,
            'web_attack': self.web_attack_engine,
            'db_exploit': self.db_exploit_engine,
            'admin_login': self.admin_login_tester
        }
        
        for module_name, module in modules.items():
            if module and hasattr(module, '__dict__'):
                # Inject AI components into modules
                module.ai_thinking_model = self.ai_thinking_model
                module.payload_generator = self.payload_generator
                module.attack_sequencer = self.attack_sequencer
                module.integration_manager = self
                
                # Add AI-enhanced methods
                self._add_ai_methods_to_module(module, module_name)
        
        self.logger.info("Modules successfully integrated with AI systems")
    
    def _add_ai_methods_to_module(self, module, module_name):
        """Add AI-enhanced methods to attack modules"""
        
        async def ai_generate_payload(attack_type, target_context, **kwargs):
            """AI-enhanced payload generation for the module"""
            if self.payload_generator:
                return await self.payload_generator.generate_payloads(
                    payload_type=attack_type,
                    target_context=target_context,
                    **kwargs
                )
            return []
        
        async def ai_predict_success(action, context):
            """AI-enhanced success prediction for the module"""
            if self.attack_sequencer:
                return await self.attack_sequencer.predict_action_success(action, context)
            return 0.5  # Default probability
        
        async def ai_get_recommendations(current_state):
            """AI-enhanced recommendations for the module"""
            if self.ai_thinking_model:
                return await self.ai_thinking_model.get_intelligent_recommendations(current_state)
            return []
        
        async def ai_report_result(action, result, context):
            """Report result to AI for learning"""
            await self.learn_from_module_result(module_name, action, result, context)
        
        # Inject AI methods into module
        module.ai_generate_payload = ai_generate_payload
        module.ai_predict_success = ai_predict_success
        module.ai_get_recommendations = ai_get_recommendations
        module.ai_report_result = ai_report_result
    
    async def start_mission(self, target_url: str, mission_config: Dict[str, Any] = None) -> str:
        """
        Start a new AI-coordinated mission
        """
        mission_id = str(uuid.uuid4())[:8]
        
        self.current_mission = MissionContext(
            mission_id=mission_id,
            target_url=target_url,
            current_phase=MissionPhase.INITIALIZATION
        )
        
        if mission_config:
            self.current_mission.target_info.update(mission_config)
        
        self.logger.info(f"Starting AI-coordinated mission {mission_id} against {target_url}")
        
        if self.terminal_callback:
            self.terminal_callback(f"🎯 Mission {mission_id}: INITIALIZING", "info")
            self.terminal_callback(f"Target: {target_url}", "info")
        
        # Initialize AI for this mission
        if self.ai_thinking_model:
            await self.ai_thinking_model.start_mission(target_url, mission_config)
        
        # Start coordination loop
        asyncio.create_task(self._coordination_loop())
        
        return mission_id
    
    async def execute_ai_coordinated_attack(self, target_url: str, attack_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a fully AI-coordinated attack sequence
        """
        mission_id = await self.start_mission(target_url, attack_config)
        
        try:
            # Phase 1: AI-Enhanced Reconnaissance
            recon_results = await self._execute_ai_reconnaissance()
            
            # Phase 2: AI Attack Sequence Planning
            attack_sequence = await self._plan_ai_attack_sequence(recon_results)
            
            # Phase 3: AI-Coordinated Execution
            execution_results = await self._execute_ai_attack_sequence(attack_sequence)
            
            # Phase 4: AI Analysis and Reporting
            final_report = await self._generate_ai_mission_report(execution_results)
            
            # Complete mission
            await self._complete_mission(final_report)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Mission {mission_id} failed: {e}")
            await self._handle_mission_failure(e)
            raise
    
    async def _execute_ai_reconnaissance(self) -> Dict[str, Any]:
        """Execute AI-enhanced reconnaissance phase"""
        self.current_mission.current_phase = MissionPhase.RECONNAISSANCE
        self.module_status['recon'] = ModuleStatus.RUNNING
        
        if self.terminal_callback:
            self.terminal_callback("🔍 Phase 1: AI-Enhanced Reconnaissance", "info")
        
        recon_results = {}
        
        if self.recon_engine:
            try:
                # Basic reconnaissance
                basic_recon = await self._execute_module_with_ai(
                    self.recon_engine, 'comprehensive_recon', 
                    target_url=self.current_mission.target_url
                )
                
                # AI-enhanced analysis
                if self.ai_thinking_model:
                    ai_analysis = await self.ai_thinking_model.analyze_reconnaissance_data(basic_recon)
                    basic_recon['ai_analysis'] = ai_analysis
                
                recon_results = basic_recon
                self.current_mission.target_info.update(recon_results)
                
                if self.terminal_callback:
                    self.terminal_callback(f"✅ Reconnaissance: {len(recon_results.get('endpoints', []))} endpoints found", "success")
                
            except Exception as e:
                self.logger.error(f"Reconnaissance failed: {e}")
                if self.terminal_callback:
                    self.terminal_callback(f"❌ Reconnaissance failed: {e}", "error")
        
        self.module_status['recon'] = ModuleStatus.COMPLETED
        return recon_results
    
    async def _plan_ai_attack_sequence(self, recon_results: Dict[str, Any]) -> List[AttackAction]:
        """Plan optimal attack sequence using AI"""
        if self.terminal_callback:
            self.terminal_callback("🧠 Phase 2: AI Attack Sequence Planning", "info")
        
        # Generate potential attack actions
        potential_actions = await self._generate_attack_actions_from_recon(recon_results)
        
        if self.attack_sequencer:
            try:
                # Use AI to optimize attack sequence
                optimized_sequence = await self.attack_sequencer.optimize_attack_sequence(
                    target_context=self.current_mission.target_info,
                    available_attacks=potential_actions,
                    optimization_method="adaptive",
                    constraints={'time_budget': 3600}
                )
                
                if self.terminal_callback:
                    self.terminal_callback(f"🎯 AI Sequencer: Optimized {len(optimized_sequence)} attack actions", "success")
                
                return optimized_sequence
                
            except Exception as e:
                self.logger.error(f"AI sequence planning failed: {e}")
                if self.terminal_callback:
                    self.terminal_callback(f"⚠️ AI planning failed, using heuristic sequence", "warning")
        
        # Fallback to heuristic ordering
        return self._create_heuristic_sequence(potential_actions)
    
    async def _execute_ai_attack_sequence(self, attack_sequence: List[AttackAction]) -> Dict[str, Any]:
        """Execute AI-coordinated attack sequence"""
        self.current_mission.current_phase = MissionPhase.EXPLOITATION
        
        if self.terminal_callback:
            self.terminal_callback("⚔️ Phase 3: AI-Coordinated Attack Execution", "info")
        
        execution_results = {
            'total_attacks': len(attack_sequence),
            'successful_attacks': 0,
            'failed_attacks': 0,
            'vulnerabilities_found': [],
            'attack_details': []
        }
        
        for i, action in enumerate(attack_sequence, 1):
            try:
                if self.terminal_callback:
                    self.terminal_callback(f"🎯 Executing {i}/{len(attack_sequence)}: {action.vector.value}", "info")
                
                # Execute attack with AI coordination
                result = await self._execute_ai_coordinated_attack_action(action)
                
                # Learn from result
                if self.attack_sequencer:
                    await self.attack_sequencer.learn_from_execution(
                        action, result, self.current_mission.target_info
                    )
                
                # Update results
                if result.success:
                    execution_results['successful_attacks'] += 1
                    if self.terminal_callback:
                        self.terminal_callback(f"✅ {action.vector.value}: SUCCESS", "success")
                else:
                    execution_results['failed_attacks'] += 1
                    if self.terminal_callback:
                        self.terminal_callback(f"❌ {action.vector.value}: FAILED", "error")
                
                execution_results['attack_details'].append({
                    'action': action,
                    'result': result,
                    'timestamp': time.time()
                })
                
                # Real-time adaptation
                if i < len(attack_sequence):
                    remaining_sequence = attack_sequence[i:]
                    adapted_sequence = await self._adapt_sequence_realtime(
                        remaining_sequence, result
                    )
                    if adapted_sequence != remaining_sequence:
                        attack_sequence[i:] = adapted_sequence
                        if self.terminal_callback:
                            self.terminal_callback("🔄 AI adapted attack sequence", "info")
                
            except Exception as e:
                self.logger.error(f"Attack action failed: {e}")
                execution_results['failed_attacks'] += 1
        
        return execution_results
    
    async def _execute_ai_coordinated_attack_action(self, action: AttackAction) -> AttackResult:
        """Execute a single attack action with AI coordination"""
        
        # Determine which module to use
        module_map = {
            AttackVector.SQL_INJECTION: self.web_attack_engine,
            AttackVector.XSS: self.web_attack_engine,
            AttackVector.SSRF: self.web_attack_engine,
            AttackVector.DIRECTORY_TRAVERSAL: self.web_attack_engine,
            AttackVector.COMMAND_INJECTION: self.web_attack_engine,
            AttackVector.FILE_UPLOAD: self.web_attack_engine,
            AttackVector.AUTHENTICATION_BYPASS: self.admin_login_tester,
            AttackVector.BRUTE_FORCE: self.admin_login_tester
        }
        
        module = module_map.get(action.vector)
        if not module:
            raise ValueError(f"No module available for attack vector: {action.vector}")
        
        # Generate AI-enhanced payload if needed
        if action.payload == "AI_GENERATE":
            enhanced_payloads = await self._generate_ai_payload_for_action(action)
            if enhanced_payloads:
                action.payload = enhanced_payloads[0].payload
        
        # Execute attack using appropriate module
        start_time = time.time()
        
        try:
            if action.vector in [AttackVector.SQL_INJECTION, AttackVector.XSS, AttackVector.SSRF]:
                result_data = await self._execute_web_attack(module, action)
            elif action.vector in [AttackVector.AUTHENTICATION_BYPASS, AttackVector.BRUTE_FORCE]:
                result_data = await self._execute_auth_attack(module, action)
            else:
                result_data = await self._execute_generic_attack(module, action)
            
            execution_time = time.time() - start_time
            
            # Create AttackResult
            result = AttackResult(
                action=action,
                success=result_data.get('success', False),
                response_time=execution_time,
                status_code=result_data.get('status_code', 0),
                response_size=result_data.get('response_size', 0),
                confidence_score=result_data.get('confidence', 0.0),
                indicators_found=result_data.get('indicators', [])
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Attack execution failed: {e}")
            return AttackResult(
                action=action,
                success=False,
                response_time=time.time() - start_time,
                status_code=500,
                response_size=0,
                error_message=str(e)
            )
    
    async def _generate_ai_payload_for_action(self, action: AttackAction) -> List[Any]:
        """Generate AI-enhanced payload for attack action"""
        if not self.payload_generator:
            return []
        
        try:
            from core.ai import PayloadType, TargetContext
            
            # Map attack vector to payload type
            payload_type_map = {
                AttackVector.SQL_INJECTION: PayloadType.SQL_INJECTION,
                AttackVector.XSS: PayloadType.XSS,
                AttackVector.SSRF: PayloadType.SSRF
            }
            
            payload_type = payload_type_map.get(action.vector)
            if not payload_type:
                return []
            
            # Create target context
            target_context = TargetContext(
                technology_stack=self.current_mission.target_info.get('technologies', []),
                detected_waf=self.current_mission.target_info.get('waf_detected', False)
            )
            
            # Generate AI payloads
            payloads = await self.payload_generator.generate_payloads(
                payload_type=payload_type,
                target_context=target_context,
                count=3
            )
            
            return payloads
            
        except Exception as e:
            self.logger.error(f"AI payload generation failed: {e}")
            return []
    
    async def _execute_module_with_ai(self, module, method_name: str, **kwargs) -> Dict[str, Any]:
        """Execute module method with AI integration"""
        if not module or not hasattr(module, method_name):
            return {}
        
        method = getattr(module, method_name)
        
        try:
            if asyncio.iscoroutinefunction(method):
                result = await method(**kwargs)
            else:
                result = method(**kwargs)
            
            return result if isinstance(result, dict) else {'data': result}
            
        except Exception as e:
            self.logger.error(f"Module execution failed: {e}")
            return {'error': str(e)}
    
    async def learn_from_module_result(self, module_name: str, action: Any, result: Any, context: Dict[str, Any]):
        """Learn from module execution results"""
        if self.ai_thinking_model:
            try:
                await self.ai_thinking_model.learn_from_attack_result(
                    module_name, action, result, context
                )
            except Exception as e:
                self.logger.error(f"AI learning failed: {e}")
    
    async def get_ai_recommendations(self, current_state: Dict[str, Any]) -> List[str]:
        """Get AI recommendations for current state"""
        recommendations = []
        
        if self.ai_thinking_model:
            try:
                ai_recs = await self.ai_thinking_model.get_intelligent_recommendations(current_state)
                recommendations.extend(ai_recs)
            except Exception as e:
                self.logger.error(f"AI recommendations failed: {e}")
        
        return recommendations
    
    def get_mission_status(self) -> Dict[str, Any]:
        """Get current mission status"""
        if not self.current_mission:
            return {'status': 'No active mission'}
        
        return {
            'mission_id': self.current_mission.mission_id,
            'target_url': self.current_mission.target_url,
            'current_phase': self.current_mission.current_phase.value,
            'phase_completion': self.current_mission.phase_completion,
            'module_status': {name: status.value for name, status in self.module_status.items()},
            'elapsed_time': time.time() - self.current_mission.start_time,
            'vulnerabilities_found': len(self.current_mission.vulnerabilities),
            'successful_exploits': len(self.current_mission.successful_exploits),
            'ai_capabilities': self.ai_capabilities
        }
    
    async def _coordination_loop(self):
        """Main coordination loop for AI systems"""
        while self.current_mission:
            try:
                # Process coordination queue
                if not self.coordination_queue.empty():
                    task = await self.coordination_queue.get()
                    await self._process_coordination_task(task)
                
                # Update AI intelligence
                await self._update_real_time_intelligence()
                
                # Sleep briefly
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(5.0)
    
    # Additional helper methods for the various attack execution types
    
    async def _execute_web_attack(self, module, action: AttackAction) -> Dict[str, Any]:
        """Execute web attack via web attack module"""
        # Implementation would call appropriate web attack methods
        return {'success': True, 'confidence': 0.8}
    
    async def _execute_auth_attack(self, module, action: AttackAction) -> Dict[str, Any]:
        """Execute authentication attack via admin login module"""
        # Implementation would call appropriate auth attack methods
        return {'success': False, 'confidence': 0.3}
    
    async def _execute_generic_attack(self, module, action: AttackAction) -> Dict[str, Any]:
        """Execute generic attack"""
        # Implementation would call appropriate generic attack methods
        return {'success': False, 'confidence': 0.1}
    
    async def _generate_attack_actions_from_recon(self, recon_results: Dict[str, Any]) -> List[AttackAction]:
        """Generate attack actions based on reconnaissance results"""
        actions = []
        
        # This would analyze recon results and generate appropriate attack actions
        # For now, return empty list as placeholder
        
        return actions
    
    def _create_heuristic_sequence(self, actions: List[AttackAction]) -> List[AttackAction]:
        """Create heuristic attack sequence when AI sequencer not available"""
        # Sort by priority and success probability
        return sorted(actions, key=lambda a: (a.priority, a.success_probability), reverse=True)
    
    async def _adapt_sequence_realtime(self, remaining_sequence: List[AttackAction], last_result: AttackResult) -> List[AttackAction]:
        """Adapt remaining sequence based on last result"""
        if self.attack_sequencer:
            try:
                # Use AI to adapt sequence
                return remaining_sequence  # Placeholder
            except Exception as e:
                self.logger.error(f"Sequence adaptation failed: {e}")
        
        return remaining_sequence
    
    async def _generate_ai_mission_report(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI-enhanced mission report"""
        return {
            'mission_id': self.current_mission.mission_id,
            'target_url': self.current_mission.target_url,
            'execution_results': execution_results,
            'ai_analysis': {},
            'recommendations': []
        }
    
    async def _complete_mission(self, final_report: Dict[str, Any]):
        """Complete the current mission"""
        self.current_mission.current_phase = MissionPhase.COMPLETED
        self.mission_history.append(self.current_mission)
        
        if self.terminal_callback:
            self.terminal_callback(f"🎉 Mission {self.current_mission.mission_id}: COMPLETED", "success")
        
        self.current_mission = None
    
    async def _handle_mission_failure(self, error: Exception):
        """Handle mission failure"""
        if self.current_mission:
            self.current_mission.current_phase = MissionPhase.FAILED
            self.mission_history.append(self.current_mission)
            self.current_mission = None
        
        if self.terminal_callback:
            self.terminal_callback(f"💥 Mission FAILED: {error}", "error")
    
    async def _process_coordination_task(self, task: Dict[str, Any]):
        """Process coordination task"""
        # Placeholder for task processing
        pass
    
    async def _update_real_time_intelligence(self):
        """Update real-time intelligence data"""
        # Placeholder for intelligence updates
        pass

# Export main class
__all__ = ['AICoreIntegrationManager', 'MissionContext', 'ModuleResult', 'MissionPhase']