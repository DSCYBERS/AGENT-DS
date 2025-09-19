#!/usr/bin/env python3
"""
Agent DS v2.0 - AI Coordination Manager
======================================

Coordinates AI-enhanced attack modules for intelligent penetration testing:
- Manages enhanced attack modules with AI capabilities
- Provides unified interface for coordinated attacks  
- Handles mission planning and execution
- Integrates with AI systems for decision making

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .ai_module_enhancer import get_module_enhancer, enhance_attack_module

class MissionPhase(Enum):
    """Mission execution phases"""
    RECONNAISSANCE = "reconnaissance"
    VULNERABILITY_DISCOVERY = "vulnerability_discovery"
    EXPLOITATION = "exploitation" 
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"

@dataclass
class MissionContext:
    """Context for attack mission"""
    target_url: str
    target_ip: Optional[str] = None
    technologies: List[str] = None
    cms_type: Optional[str] = None
    waf_detected: bool = False
    web_server: Optional[str] = None
    database_type: Optional[str] = None
    admin_panels: List[str] = None
    open_ports: List[int] = None
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.technologies is None:
            self.technologies = []
        if self.admin_panels is None:
            self.admin_panels = []
        if self.open_ports is None:
            self.open_ports = []
        if self.custom_params is None:
            self.custom_params = {}

@dataclass
class ModuleResult:
    """Result from enhanced module execution"""
    module_name: str
    success: bool
    data: Dict[str, Any]
    execution_time: float
    ai_recommendations: List[str] = None
    confidence_score: float = 0.0
    
    def __post_init__(self):
        if self.ai_recommendations is None:
            self.ai_recommendations = []

class AICoordinationManager:
    """
    Coordinates AI-enhanced attack modules for intelligent penetration testing
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.module_enhancer = get_module_enhancer()
        self.enhanced_modules = {}
        self.mission_history = []
        
        # Initialize enhanced modules
        self._initialize_enhanced_modules()
    
    def _initialize_enhanced_modules(self):
        """Initialize and enhance attack modules"""
        module_configs = {
            'reconnaissance': {
                'module_path': 'core.attacks.recon',
                'class_name': 'ReconEngine'
            },
            'web_attacks': {
                'module_path': 'core.attacks.web_attack', 
                'class_name': 'WebAttackEngine'
            },
            'database_exploitation': {
                'module_path': 'core.attacks.db_exploit',
                'class_name': 'DatabaseExploitEngine'
            },
            'admin_login': {
                'module_path': 'core.attacks.admin_login',
                'class_name': 'AdminLoginTester'
            }
        }
        
        for module_name, config in module_configs.items():
            try:
                # Try to import and enhance module
                module = self._load_and_enhance_module(module_name, config)
                if module:
                    self.enhanced_modules[module_name] = module
                    self.logger.info(f"Successfully loaded and enhanced {module_name} module")
                else:
                    self.logger.warning(f"Failed to load {module_name} module - will use fallback")
                    
            except Exception as e:
                self.logger.error(f"Error loading {module_name} module: {e}")
    
    def _load_and_enhance_module(self, module_name: str, config: Dict[str, str]):
        """Load and enhance a specific module"""
        try:
            # Try to import the module
            import importlib
            module_path = config['module_path']
            class_name = config['class_name']
            
            try:
                module = importlib.import_module(module_path)
                module_class = getattr(module, class_name)
                
                # Create instance and enhance it
                instance = module_class()
                enhanced_instance = enhance_attack_module(instance, module_name)
                
                return enhanced_instance
                
            except (ImportError, AttributeError) as e:
                self.logger.warning(f"Module {module_name} not available ({e}), creating mock module")
                return self._create_mock_module(module_name)
                
        except Exception as e:
            self.logger.error(f"Failed to load module {module_name}: {e}")
            return None
    
    def _create_mock_module(self, module_name: str):
        """Create mock module for testing when real module not available"""
        
        class MockModule:
            def __init__(self):
                self.name = module_name
                self.logger = logging.getLogger(f"Mock{module_name}")
            
            async def scan(self, target_url: str, **kwargs):
                """Mock scan method"""
                await asyncio.sleep(0.1)  # Simulate work
                return {
                    'success': True,
                    'data': {
                        'mock_results': f"Mock {self.name} scan of {target_url}",
                        'findings': [f"Mock finding from {self.name}"]
                    },
                    'confidence': 0.5
                }
            
            async def attack(self, target_url: str, **kwargs):
                """Mock attack method"""
                await asyncio.sleep(0.2)  # Simulate work
                return {
                    'success': False,  # Mock attacks don't succeed
                    'data': {
                        'mock_attack': f"Mock {self.name} attack on {target_url}",
                        'reason': "Mock module - no real attack performed"
                    },
                    'confidence': 0.1
                }
            
            async def test(self, target_url: str, **kwargs):
                """Mock test method"""
                await asyncio.sleep(0.1)
                return {
                    'success': True,
                    'data': {
                        'mock_test': f"Mock {self.name} test of {target_url}",
                        'status': "Mock test completed"
                    },
                    'confidence': 0.3
                }
        
        # Create and enhance mock module
        mock = MockModule()
        enhanced_mock = enhance_attack_module(mock, module_name)
        return enhanced_mock
    
    async def execute_coordinated_mission(self, mission_context: MissionContext, 
                                        phases: List[MissionPhase] = None) -> Dict[str, Any]:
        """
        Execute a coordinated penetration testing mission
        """
        if phases is None:
            phases = [
                MissionPhase.RECONNAISSANCE,
                MissionPhase.VULNERABILITY_DISCOVERY,
                MissionPhase.EXPLOITATION
            ]
        
        mission_results = {
            'mission_id': f"mission_{len(self.mission_history)}",
            'target_url': mission_context.target_url,
            'phases_executed': [],
            'phase_results': {},
            'overall_success': False,
            'findings': [],
            'recommendations': [],
            'execution_time': 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            for phase in phases:
                self.logger.info(f"Executing mission phase: {phase.value}")
                
                phase_result = await self._execute_mission_phase(phase, mission_context)
                mission_results['phases_executed'].append(phase.value)
                mission_results['phase_results'][phase.value] = phase_result
                
                # Update mission context with findings
                self._update_mission_context(mission_context, phase_result)
                
                # Collect findings and recommendations
                if phase_result.get('findings'):
                    mission_results['findings'].extend(phase_result['findings'])
                if phase_result.get('recommendations'):
                    mission_results['recommendations'].extend(phase_result['recommendations'])
                
                # Check if we should continue based on AI recommendations
                if not await self._should_continue_mission(phase, phase_result, mission_context):
                    self.logger.info(f"AI recommends stopping mission after {phase.value}")
                    break
            
            # Determine overall success
            mission_results['overall_success'] = any(
                result.get('success', False) 
                for result in mission_results['phase_results'].values()
            )
            
            mission_results['execution_time'] = asyncio.get_event_loop().time() - start_time
            self.mission_history.append(mission_results)
            
            return mission_results
            
        except Exception as e:
            self.logger.error(f"Mission execution failed: {e}")
            mission_results['error'] = str(e)
            mission_results['execution_time'] = asyncio.get_event_loop().time() - start_time
            return mission_results
    
    async def _execute_mission_phase(self, phase: MissionPhase, context: MissionContext) -> Dict[str, Any]:
        """Execute a specific mission phase"""
        
        phase_result = {
            'phase': phase.value,
            'success': False,
            'modules_executed': [],
            'findings': [],
            'recommendations': [],
            'execution_time': 0
        }
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if phase == MissionPhase.RECONNAISSANCE:
                result = await self._execute_reconnaissance(context)
            elif phase == MissionPhase.VULNERABILITY_DISCOVERY:
                result = await self._execute_vulnerability_discovery(context)
            elif phase == MissionPhase.EXPLOITATION:
                result = await self._execute_exploitation(context)
            else:
                result = {'success': False, 'error': f"Unsupported phase: {phase.value}"}
            
            phase_result.update(result)
            phase_result['execution_time'] = asyncio.get_event_loop().time() - start_time
            
            return phase_result
            
        except Exception as e:
            self.logger.error(f"Phase {phase.value} execution failed: {e}")
            phase_result['error'] = str(e)
            phase_result['execution_time'] = asyncio.get_event_loop().time() - start_time
            return phase_result
    
    async def _execute_reconnaissance(self, context: MissionContext) -> Dict[str, Any]:
        """Execute reconnaissance phase"""
        results = {
            'success': False,
            'modules_executed': ['reconnaissance'],
            'findings': [],
            'recommendations': []
        }
        
        if 'reconnaissance' in self.enhanced_modules:
            recon_module = self.enhanced_modules['reconnaissance']
            
            try:
                # Use AI to determine what recon to perform
                if hasattr(recon_module, 'ai_get_recommendations'):
                    ai_recommendations = await recon_module.ai_get_recommendations({'phase': 'reconnaissance'})
                    results['recommendations'].extend(ai_recommendations)
                
                # Perform reconnaissance scan
                scan_result = await recon_module.scan(
                    context.target_url,
                    deep_scan=True,
                    ai_enhanced=True
                )
                
                if scan_result.get('success'):
                    results['success'] = True
                    results['findings'].append(f"Reconnaissance completed for {context.target_url}")
                    
                    # Extract findings from scan data
                    scan_data = scan_result.get('data', {})
                    if 'technologies' in scan_data:
                        context.technologies.extend(scan_data['technologies'])
                        results['findings'].append(f"Technologies detected: {scan_data['technologies']}")
                    
                    if 'admin_panels' in scan_data:
                        context.admin_panels.extend(scan_data['admin_panels'])
                        results['findings'].append(f"Admin panels found: {scan_data['admin_panels']}")
                
            except Exception as e:
                self.logger.error(f"Reconnaissance module error: {e}")
                results['findings'].append(f"Reconnaissance failed: {e}")
        else:
            results['findings'].append("Reconnaissance module not available")
        
        return results
    
    async def _execute_vulnerability_discovery(self, context: MissionContext) -> Dict[str, Any]:
        """Execute vulnerability discovery phase"""
        results = {
            'success': False,
            'modules_executed': [],
            'findings': [],
            'recommendations': []
        }
        
        # Use web attack module for vulnerability discovery
        if 'web_attacks' in self.enhanced_modules:
            web_module = self.enhanced_modules['web_attacks']
            results['modules_executed'].append('web_attacks')
            
            try:
                # Use AI to decide which vulnerabilities to test
                if hasattr(web_module, 'ai_should_attempt_attack'):
                    should_test_sql = await web_module.ai_should_attempt_attack(
                        'sql_injection', context.__dict__
                    )
                    should_test_xss = await web_module.ai_should_attempt_attack(
                        'xss', context.__dict__
                    )
                    
                    if should_test_sql:
                        results['findings'].append("AI recommends SQL injection testing")
                    if should_test_xss:
                        results['findings'].append("AI recommends XSS testing")
                
                # Perform vulnerability scan
                vuln_result = await web_module.scan(
                    context.target_url,
                    test_types=['sql_injection', 'xss', 'ssrf'],
                    ai_enhanced=True
                )
                
                if vuln_result.get('success'):
                    results['success'] = True
                    results['findings'].append("Vulnerability discovery completed")
                    
                    vuln_data = vuln_result.get('data', {})
                    if 'vulnerabilities' in vuln_data:
                        for vuln in vuln_data['vulnerabilities']:
                            results['findings'].append(f"Vulnerability found: {vuln}")
                
            except Exception as e:
                self.logger.error(f"Vulnerability discovery error: {e}")
                results['findings'].append(f"Vulnerability discovery failed: {e}")
        
        return results
    
    async def _execute_exploitation(self, context: MissionContext) -> Dict[str, Any]:
        """Execute exploitation phase"""
        results = {
            'success': False,
            'modules_executed': [],
            'findings': [],
            'recommendations': []
        }
        
        # Try web attacks first
        if 'web_attacks' in self.enhanced_modules:
            web_module = self.enhanced_modules['web_attacks']
            results['modules_executed'].append('web_attacks')
            
            try:
                # Use AI-generated smart payloads
                if hasattr(web_module, 'ai_generate_smart_payloads'):
                    sql_payloads = await web_module.ai_generate_smart_payloads(
                        'sql_injection', context.__dict__, count=5
                    )
                    results['findings'].append(f"Generated {len(sql_payloads)} AI-enhanced SQL payloads")
                
                # Attempt exploitation
                exploit_result = await web_module.attack(
                    context.target_url,
                    attack_types=['sql_injection'],
                    ai_enhanced=True
                )
                
                if exploit_result.get('success'):
                    results['success'] = True
                    results['findings'].append("Web exploitation successful")
                else:
                    results['findings'].append("Web exploitation attempts failed")
                
            except Exception as e:
                self.logger.error(f"Web exploitation error: {e}")
                results['findings'].append(f"Web exploitation failed: {e}")
        
        # Try database exploitation if web attacks suggest DB presence
        if context.database_type and 'database_exploitation' in self.enhanced_modules:
            db_module = self.enhanced_modules['database_exploitation']
            results['modules_executed'].append('database_exploitation')
            
            try:
                db_result = await db_module.attack(
                    context.target_url,
                    database_type=context.database_type,
                    ai_enhanced=True
                )
                
                if db_result.get('success'):
                    results['success'] = True
                    results['findings'].append("Database exploitation successful")
                
            except Exception as e:
                self.logger.error(f"Database exploitation error: {e}")
                results['findings'].append(f"Database exploitation failed: {e}")
        
        return results
    
    async def _should_continue_mission(self, current_phase: MissionPhase, 
                                     phase_result: Dict[str, Any], context: MissionContext) -> bool:
        """Use AI to decide if mission should continue"""
        
        # Simple heuristics for now - could be enhanced with AI decision making
        if not phase_result.get('success'):
            # If reconnaissance failed, probably should stop
            if current_phase == MissionPhase.RECONNAISSANCE:
                return False
        
        # Check if we have enough information to continue
        if current_phase == MissionPhase.RECONNAISSANCE:
            # Continue if we found some technologies or admin panels
            return len(context.technologies) > 0 or len(context.admin_panels) > 0
        
        # Generally continue unless explicitly told not to
        return True
    
    def _update_mission_context(self, context: MissionContext, phase_result: Dict[str, Any]):
        """Update mission context with findings from phase execution"""
        
        findings = phase_result.get('findings', [])
        
        for finding in findings:
            finding_lower = finding.lower()
            
            # Extract technology information
            if 'technologies detected:' in finding_lower:
                # This would be more sophisticated in real implementation
                pass
            
            # Extract admin panel information  
            if 'admin panels found:' in finding_lower:
                # This would be more sophisticated in real implementation
                pass
            
            # Extract database information
            if 'database' in finding_lower and 'mysql' in finding_lower:
                context.database_type = 'mysql'
            elif 'database' in finding_lower and 'postgres' in finding_lower:
                context.database_type = 'postgresql'
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'enhanced_modules': list(self.enhanced_modules.keys()),
            'mission_history_count': len(self.mission_history),
            'module_enhancer_status': self.module_enhancer.get_enhancement_summary(),
            'last_mission': self.mission_history[-1] if self.mission_history else None
        }

# Create default coordinator instance
_default_coordinator = None

def get_ai_coordinator() -> AICoordinationManager:
    """Get default AI coordination manager instance"""
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = AICoordinationManager()
    return _default_coordinator

# Export main classes and functions
__all__ = ['AICoordinationManager', 'MissionContext', 'MissionPhase', 'ModuleResult', 'get_ai_coordinator']