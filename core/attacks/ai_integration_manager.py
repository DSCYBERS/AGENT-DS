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
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from urllib.parse import urlparse

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
        self._enforce_scope_guard = os.getenv("AGENT_DS_ENFORCE_SCOPE_GUARD", "0").lower() in ("1", "true", "yes")
        self._global_allowed_domains = self._parse_allowed_domains(
            os.getenv("AGENT_DS_ALLOWED_DOMAINS", "")
        )
        
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

    def _parse_allowed_domains(self, domains_value: str) -> List[str]:
        """Parse comma-separated domain list into normalized host suffixes"""
        if not domains_value:
            return []
        return [d.strip().lower() for d in domains_value.split(",") if d.strip()]

    def _extract_host(self, target_url: str) -> str:
        """Extract hostname from URL/host input"""
        if not target_url:
            return ""
        parsed = urlparse(target_url if "://" in target_url else f"https://{target_url}")
        return (parsed.netloc or parsed.path).split("/")[0].split(":")[0].lower().strip()

    def _is_domain_allowed(self, host: str, allowed_domains: List[str]) -> bool:
        """Check if host matches any allowed domain/suffix"""
        for domain in allowed_domains:
            if host == domain or host.endswith(f".{domain}"):
                return True
        return False

    def _validate_mission_authorization(self, target_url: str, mission_config: Dict[str, Any]):
        """
        Validate mission authorization and scope controls for company-only usage.
        """
        config = mission_config or {}
        host = self._extract_host(target_url)
        requested_auth = bool(config.get("authorized", False))
        config_domains = config.get("allowed_domains", [])
        if isinstance(config_domains, str):
            allowed_domains = self._parse_allowed_domains(config_domains)
        else:
            allowed_domains = [str(d).strip().lower() for d in config_domains if str(d).strip()]
        combined_allowed_domains = list(dict.fromkeys(allowed_domains + self._global_allowed_domains))

        if self._enforce_scope_guard and not requested_auth:
            raise PermissionError(
                "Mission blocked: explicit authorization required (set `authorized=True` in mission config)."
            )

        if combined_allowed_domains and not self._is_domain_allowed(host, combined_allowed_domains):
            raise PermissionError(
                f"Mission blocked: target '{host}' is outside allowed domains: {combined_allowed_domains}"
            )
    
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
        self._validate_mission_authorization(target_url, mission_config or {})
        
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
                basic_recon = {}
                recon_methods = [
                    'comprehensive_recon',
                    'perform_advanced_enumeration',
                    'perform_reconnaissance'
                ]
                for method_name in recon_methods:
                    if hasattr(self.recon_engine, method_name):
                        basic_recon = await self._execute_module_with_ai(
                            self.recon_engine,
                            method_name,
                            target_url=self.current_mission.target_url
                        )
                        if basic_recon:
                            break

                if not basic_recon:
                    parsed = urlparse(self.current_mission.target_url)
                    basic_recon = {
                        'target': self.current_mission.target_url,
                        'domain': parsed.netloc,
                        'technologies': [],
                        'crawled_urls': []
                    }

                # Normalize endpoint keys expected by planners
                crawled = basic_recon.get('crawled_urls', [])
                endpoints = basic_recon.get('endpoints', [])
                if not endpoints and isinstance(crawled, list):
                    basic_recon['endpoints'] = crawled[:200]
                
                # AI-enhanced analysis
                if self.ai_thinking_model:
                    ai_analysis = await self.ai_thinking_model.analyze_reconnaissance_data(basic_recon)
                    basic_recon['ai_analysis'] = ai_analysis
                
                recon_results = basic_recon
                self.current_mission.target_info.update(recon_results)
                self.current_mission.discovered_endpoints = recon_results.get('endpoints', [])
                
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
                action_vector_name = self._vector_to_string(self._get_action_field(action, 'vector', None))
                if self.terminal_callback:
                    self.terminal_callback(f"🎯 Executing {i}/{len(attack_sequence)}: {action_vector_name}", "info")
                
                # Execute attack with AI coordination
                result = await self._execute_ai_coordinated_attack_action(action)
                
                # Learn from result
                if self.attack_sequencer:
                    await self.attack_sequencer.learn_from_execution(
                        action, result, self.current_mission.target_info
                    )
                
                # Update results
                if self._get_result_field(result, 'success', False):
                    execution_results['successful_attacks'] += 1
                    if self.terminal_callback:
                        self.terminal_callback(f"✅ {action_vector_name}: SUCCESS", "success")
                else:
                    execution_results['failed_attacks'] += 1
                    if self.terminal_callback:
                        self.terminal_callback(f"❌ {action_vector_name}: FAILED", "error")
                
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
        vector = self._get_action_field(action, 'vector', None)
        vector_name = self._vector_to_string(vector)

        web_vectors = {'sql_injection', 'xss', 'ssrf', 'directory_traversal', 'command_injection', 'file_upload'}
        auth_vectors = {'authentication_bypass', 'brute_force'}

        if vector_name in web_vectors:
            module = self.web_attack_engine
        elif vector_name in auth_vectors:
            module = self.admin_login_tester
        else:
            module = self.db_exploit_engine or self.web_attack_engine

        if not module:
            raise ValueError(f"No module available for attack vector: {vector_name}")
        
        # Generate AI-enhanced payload if needed
        if self._get_action_field(action, 'payload', '') == "AI_GENERATE":
            enhanced_payloads = await self._generate_ai_payload_for_action(action)
            if enhanced_payloads:
                first_payload = enhanced_payloads[0]
                payload_value = getattr(first_payload, 'payload', first_payload)
                self._set_action_field(action, 'payload', payload_value)
        
        # Execute attack using appropriate module
        start_time = time.time()
        
        try:
            if vector_name in {'sql_injection', 'xss', 'ssrf', 'directory_traversal', 'command_injection', 'file_upload'}:
                result_data = await self._execute_web_attack(module, action)
            elif vector_name in {'authentication_bypass', 'brute_force'}:
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
            
            vector_name = self._vector_to_string(self._get_action_field(action, 'vector', None))

            sql_payload_type = getattr(PayloadType, 'SQL_INJECTION', None)
            xss_payload_type = getattr(PayloadType, 'XSS', None)
            ssrf_payload_type = getattr(PayloadType, 'SSRF', None)

            payload_type_map = {
                'sql_injection': sql_payload_type,
                'xss': xss_payload_type,
                'ssrf': ssrf_payload_type
            }
            
            payload_type = payload_type_map.get(vector_name)
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
        actions: List[AttackAction] = []

        endpoints = recon_results.get('endpoints') or recon_results.get('crawled_urls') or []
        if not endpoints:
            target = self.current_mission.target_url if self.current_mission else ""
            endpoints = [target] if target else []

        technologies = [str(t).lower() for t in (recon_results.get('technologies') or [])]
        security_headers = recon_results.get('security_headers') or {}

        has_csp = bool(security_headers.get('Content-Security-Policy'))
        has_sql_stack = any(
            db_tech in technologies for db_tech in
            ('mysql', 'postgresql', 'postgres', 'mariadb', 'mssql', 'sql server', 'oracle', 'php')
        )
        has_template_stack = any(
            tmpl in technologies for tmpl in ('django', 'flask', 'jinja', 'twig', 'smarty')
        )

        for endpoint in endpoints[:120]:
            endpoint_str = str(endpoint)
            lower_ep = endpoint_str.lower()

            is_dynamic = any(marker in lower_ep for marker in ['?', '=', '&', '/api/', '.php', '.asp', '.aspx', '.jsp'])
            is_auth = any(marker in lower_ep for marker in ['/login', '/signin', '/auth', '/admin'])
            is_upload = any(marker in lower_ep for marker in ['/upload', '/import', '/avatar', '/file'])
            has_file_path = any(marker in lower_ep for marker in ['/download', '/file', '/view', '/template'])
            has_external_fetch = any(marker in lower_ep for marker in ['url=', 'redirect', 'proxy', 'fetch', 'callback'])
            has_exec_surface = any(marker in lower_ep for marker in ['cmd', 'exec', 'shell', 'ping', 'system'])
            high_value = any(marker in lower_ep for marker in ['/admin', '/api', '/dashboard', '/manage'])

            if is_dynamic or has_sql_stack:
                action = self._build_attack_action(
                    vector_name='sql_injection',
                    target_endpoint=endpoint_str,
                    priority=0.82 if has_sql_stack else 0.72,
                    success_probability=0.68 if has_sql_stack else 0.55,
                    risk_score=0.88,
                    metadata={'pattern': 'advanced_sqli', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

            if is_dynamic:
                action = self._build_attack_action(
                    vector_name='xss',
                    target_endpoint=endpoint_str,
                    priority=0.70 if not has_csp else 0.58,
                    success_probability=0.60 if not has_csp else 0.45,
                    risk_score=0.60,
                    metadata={'pattern': 'contextual_xss', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

            if has_external_fetch:
                action = self._build_attack_action(
                    vector_name='ssrf',
                    target_endpoint=endpoint_str,
                    priority=0.78,
                    success_probability=0.50,
                    risk_score=0.83,
                    metadata={'pattern': 'cloud_metadata_ssrf', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

            if has_file_path:
                action = self._build_attack_action(
                    vector_name='directory_traversal',
                    target_endpoint=endpoint_str,
                    priority=0.74,
                    success_probability=0.52,
                    risk_score=0.75,
                    metadata={'pattern': 'path_traversal_lfi', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

            if has_exec_surface:
                action = self._build_attack_action(
                    vector_name='command_injection',
                    target_endpoint=endpoint_str,
                    priority=0.80,
                    success_probability=0.46,
                    risk_score=0.90,
                    metadata={'pattern': 'os_command_injection', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

            if is_upload:
                action = self._build_attack_action(
                    vector_name='file_upload',
                    target_endpoint=endpoint_str,
                    priority=0.76,
                    success_probability=0.51,
                    risk_score=0.86,
                    metadata={'pattern': 'polyglot_upload', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

            if is_auth:
                auth_action = self._build_attack_action(
                    vector_name='authentication_bypass',
                    target_endpoint=endpoint_str,
                    priority=0.81,
                    success_probability=0.43,
                    risk_score=0.84,
                    metadata={'pattern': 'auth_logic_bypass', 'high_value_endpoint': True}
                )
                brute_action = self._build_attack_action(
                    vector_name='brute_force',
                    target_endpoint=endpoint_str,
                    priority=0.63,
                    success_probability=0.35,
                    risk_score=0.58,
                    metadata={'pattern': 'credential_stuffing', 'high_value_endpoint': True}
                )
                if auth_action:
                    actions.append(auth_action)
                if brute_action:
                    actions.append(brute_action)

            if has_template_stack and is_dynamic:
                action = self._build_attack_action(
                    vector_name='command_injection',
                    target_endpoint=endpoint_str,
                    priority=0.67,
                    success_probability=0.38,
                    risk_score=0.72,
                    metadata={'pattern': 'ssti_to_rce_path', 'high_value_endpoint': high_value}
                )
                if action:
                    actions.append(action)

        # De-duplicate by vector+endpoint and keep highest-priority variant
        deduped: Dict[Tuple[str, str], AttackAction] = {}
        for action in actions:
            vector_name = self._vector_to_string(self._get_action_field(action, 'vector', None))
            endpoint = str(self._get_action_field(action, 'target_endpoint', ''))
            key = (vector_name, endpoint)
            existing = deduped.get(key)
            if not existing:
                deduped[key] = action
                continue

            if self._get_action_field(action, 'priority', 0.0) > self._get_action_field(existing, 'priority', 0.0):
                deduped[key] = action

        return self._create_heuristic_sequence(list(deduped.values()))
    
    def _create_heuristic_sequence(self, actions: List[AttackAction]) -> List[AttackAction]:
        """Create heuristic attack sequence when AI sequencer not available"""
        # Sort by priority and success probability
        return sorted(
            actions,
            key=lambda a: (
                self._get_action_field(a, 'priority', 0.0),
                self._get_action_field(a, 'success_probability', 0.0)
            ),
            reverse=True
        )

    async def _adapt_sequence_realtime(self, remaining_sequence: List[AttackAction], last_result: AttackResult) -> List[AttackAction]:
        """Adapt remaining sequence based on last result"""
        if not remaining_sequence:
            return remaining_sequence

        try:
            last_success = self._get_result_field(last_result, 'success', False)
            last_action = self._get_result_field(last_result, 'action', None)
            last_vector = self._vector_to_string(self._get_action_field(last_action, 'vector', None))

            adapted = []
            for action in remaining_sequence:
                priority = float(self._get_action_field(action, 'priority', 0.0))
                vector_name = self._vector_to_string(self._get_action_field(action, 'vector', None))
                metadata = self._get_action_field(action, 'metadata', {}) or {}

                if last_success and vector_name == last_vector:
                    priority += 0.15
                elif (not last_success) and vector_name == last_vector:
                    priority -= 0.20

                if metadata.get('high_value_endpoint'):
                    priority += 0.05

                self._set_action_field(action, 'priority', max(0.05, min(1.0, priority)))
                adapted.append(action)

            return self._create_heuristic_sequence(adapted)

        except Exception as e:
            self.logger.error(f"Sequence adaptation failed: {e}")
            return remaining_sequence

    def _vector_to_string(self, vector: Any) -> str:
        """Normalize attack vector to string"""
        if vector is None:
            return ""
        return vector.value if hasattr(vector, 'value') else str(vector)

    def _get_action_field(self, action: Any, field_name: str, default: Any = None) -> Any:
        """Read action field from dataclass/object/dict"""
        if action is None:
            return default
        if isinstance(action, dict):
            return action.get(field_name, default)
        return getattr(action, field_name, default)

    def _set_action_field(self, action: Any, field_name: str, value: Any):
        """Set action field on dataclass/object/dict"""
        if action is None:
            return
        if isinstance(action, dict):
            action[field_name] = value
            return
        setattr(action, field_name, value)

    def _get_result_field(self, result: Any, field_name: str, default: Any = None) -> Any:
        """Read result field from dataclass/object/dict"""
        if result is None:
            return default
        if isinstance(result, dict):
            return result.get(field_name, default)
        return getattr(result, field_name, default)

    def _resolve_attack_vector(self, vector_name: str) -> Optional[Any]:
        """Resolve vector constant from name across enum/fallback types"""
        if not vector_name:
            return None

        if hasattr(AttackVector, '__members__'):
            return AttackVector.__members__.get(vector_name.upper())

        attr_name = vector_name.upper()
        return getattr(AttackVector, attr_name, None)

    def _resolve_attack_phase(self, phase_name: str) -> Optional[Any]:
        """Resolve attack phase constant from name across enum/fallback types"""
        if not phase_name:
            return None

        if hasattr(AttackPhase, '__members__'):
            return AttackPhase.__members__.get(phase_name.upper())

        attr_name = phase_name.upper()
        return getattr(AttackPhase, attr_name, None)

    def _build_attack_action(
        self,
        vector_name: str,
        target_endpoint: str,
        priority: float,
        success_probability: float,
        risk_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[AttackAction]:
        """Create AttackAction that works with both dataclass and fallback dict implementation"""
        vector = self._resolve_attack_vector(vector_name)
        phase = self._resolve_attack_phase("EXPLOITATION") or self._resolve_attack_phase("RECONNAISSANCE")
        if not vector or not phase:
            return None

        resource_requirements = {}
        cpu_key = getattr(ResourceType, 'CPU_THREADS', None)
        memory_key = getattr(ResourceType, 'MEMORY', None)
        if cpu_key is not None:
            resource_requirements[cpu_key] = 1.0
        if memory_key is not None:
            resource_requirements[memory_key] = 0.2

        return AttackAction(
            vector=vector,
            phase=phase,
            target_endpoint=target_endpoint,
            payload="AI_GENERATE",
            priority=max(0.05, min(1.0, float(priority))),
            estimated_time=30.0,
            resource_requirements=resource_requirements,
            success_probability=max(0.05, min(1.0, float(success_probability))),
            risk_score=max(0.05, min(1.0, float(risk_score))),
            dependencies=[],
            metadata=metadata or {}
        )
    
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