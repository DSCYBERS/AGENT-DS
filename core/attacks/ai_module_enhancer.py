#!/usr/bin/env python3
"""
Agent DS v2.0 - Module AI Enhancement System
===========================================

System to enhance existing attack modules with AI capabilities without modifying their core code:
- Injects AI Payload Generator integration
- Adds Adaptive Attack Sequencer coordination 
- Provides AI Thinking Model integration
- Enhances modules with intelligent decision making
- Maintains backward compatibility

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import functools
import logging
from typing import Dict, List, Optional, Any, Callable
import inspect

# Try to import AI components
try:
    from core.ai import (
        AIPayloadGenerator, get_sequencer_manager,
        PayloadType, TargetContext, get_ai_status
    )
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

class ModuleAIEnhancer:
    """
    Enhances attack modules with AI capabilities through method injection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enhanced_modules = {}
        
        # Initialize AI components if available
        if AI_AVAILABLE:
            self._initialize_ai_components()
        else:
            self.logger.warning("AI components not available")
            self._initialize_fallback()
    
    def _initialize_ai_components(self):
        """Initialize AI components"""
        try:
            self.payload_generator = AIPayloadGenerator()
            self.attack_sequencer = get_sequencer_manager(enable_training=False)
            self.ai_status = get_ai_status()
            self.logger.info("AI components initialized for module enhancement")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI components: {e}")
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback when AI not available"""
        self.payload_generator = None
        self.attack_sequencer = None
        self.ai_status = {}
    
    def enhance_module(self, module, module_name: str):
        """
        Enhance an attack module with AI capabilities
        """
        if module_name in self.enhanced_modules:
            self.logger.warning(f"Module {module_name} already enhanced")
            return module
        
        try:
            # Add AI attributes to module
            module._ai_enhanced = True
            module._ai_enhancer = self
            module._module_name = module_name
            
            # Inject AI methods
            self._inject_ai_payload_methods(module)
            self._inject_ai_decision_methods(module)
            self._inject_ai_learning_methods(module)
            self._inject_ai_coordination_methods(module)
            
            # Enhance existing methods with AI wrappers
            self._wrap_existing_methods(module)
            
            # Store enhanced module
            self.enhanced_modules[module_name] = module
            
            self.logger.info(f"Successfully enhanced {module_name} module with AI capabilities")
            return module
            
        except Exception as e:
            self.logger.error(f"Failed to enhance {module_name} module: {e}")
            return module
    
    def _inject_ai_payload_methods(self, module):
        """Inject AI payload generation methods"""
        
        async def ai_generate_smart_payloads(self, attack_type: str, target_context: Dict[str, Any], count: int = 5):
            """Generate AI-enhanced payloads for this attack type"""
            if not AI_AVAILABLE or not self._ai_enhancer.payload_generator:
                return self._generate_fallback_payloads(attack_type, count)
            
            try:
                # Map attack type to PayloadType
                payload_type_map = {
                    'sql_injection': PayloadType.SQL_INJECTION,
                    'xss': PayloadType.XSS,
                    'ssrf': PayloadType.SSRF,
                    'ssti': PayloadType.SSTI if hasattr(PayloadType, 'SSTI') else PayloadType.XSS
                }
                
                payload_type = payload_type_map.get(attack_type.lower())
                if not payload_type:
                    return self._generate_fallback_payloads(attack_type, count)
                
                # Create target context
                context = TargetContext(
                    technology_stack=target_context.get('technologies', []),
                    cms_type=target_context.get('cms_type'),
                    detected_waf=target_context.get('waf_detected', False),
                    web_server=target_context.get('web_server'),
                    database_type=target_context.get('database_type')
                )
                
                # Generate AI payloads
                payloads = await self._ai_enhancer.payload_generator.generate_payloads(
                    payload_type=payload_type,
                    target_context=context,
                    count=count
                )
                
                # Convert to simple strings for compatibility
                return [p.payload for p in payloads] if payloads else self._generate_fallback_payloads(attack_type, count)
                
            except Exception as e:
                self._ai_enhancer.logger.error(f"AI payload generation failed: {e}")
                return self._generate_fallback_payloads(attack_type, count)
        
        def _generate_fallback_payloads(self, attack_type: str, count: int):
            """Generate fallback payloads when AI not available"""
            fallback_payloads = {
                'sql_injection': [
                    "' OR 1=1 --",
                    "' UNION SELECT 1,2,3 --",
                    "'; DROP TABLE users; --",
                    "' OR 'a'='a",
                    "1' AND 1=1 --"
                ],
                'xss': [
                    "<script>alert('XSS')</script>",
                    "<img src=x onerror=alert('XSS')>",
                    "javascript:alert('XSS')",
                    "<svg onload=alert('XSS')>",
                    "<iframe src=javascript:alert('XSS')>"
                ],
                'ssrf': [
                    "http://169.254.169.254/latest/meta-data/",
                    "http://localhost:80",
                    "http://127.0.0.1:22",
                    "file:///etc/passwd",
                    "gopher://127.0.0.1:3306"
                ]
            }
            
            payloads = fallback_payloads.get(attack_type.lower(), ["test"])
            return payloads[:count]
        
        # Bind methods to module
        module.ai_generate_smart_payloads = ai_generate_smart_payloads.__get__(module, module.__class__)
        module._generate_fallback_payloads = _generate_fallback_payloads.__get__(module, module.__class__)
    
    def _inject_ai_decision_methods(self, module):
        """Inject AI decision-making methods"""
        
        async def ai_should_attempt_attack(self, attack_type: str, target_context: Dict[str, Any], success_threshold: float = 0.5):
            """Use AI to decide if attack should be attempted"""
            if not AI_AVAILABLE or not self._ai_enhancer.attack_sequencer:
                return self._heuristic_attack_decision(attack_type, target_context, success_threshold)
            
            try:
                # Create mock attack action for prediction
                from core.ai import AttackAction, AttackPhase, AttackVector
                
                vector_map = {
                    'sql_injection': AttackVector.SQL_INJECTION,
                    'xss': AttackVector.XSS,
                    'ssrf': AttackVector.SSRF,
                    'directory_traversal': AttackVector.DIRECTORY_TRAVERSAL,
                    'command_injection': AttackVector.COMMAND_INJECTION
                }
                
                vector = vector_map.get(attack_type.lower(), AttackVector.SQL_INJECTION)
                
                action = AttackAction(
                    vector=vector,
                    phase=AttackPhase.EXPLOITATION,
                    target_endpoint=target_context.get('endpoint', '/'),
                    payload="test",
                    priority=0.7,
                    success_probability=0.5
                )
                
                # Get AI prediction
                predicted_success = await self._ai_enhancer.attack_sequencer.predict_action_success(
                    action, target_context
                )
                
                return predicted_success >= success_threshold
                
            except Exception as e:
                self._ai_enhancer.logger.error(f"AI attack decision failed: {e}")
                return self._heuristic_attack_decision(attack_type, target_context, success_threshold)
        
        def _heuristic_attack_decision(self, attack_type: str, target_context: Dict[str, Any], success_threshold: float):
            """Heuristic decision when AI not available"""
            # Simple heuristics based on context
            base_probability = {
                'sql_injection': 0.7,
                'xss': 0.6,
                'ssrf': 0.4,
                'directory_traversal': 0.5,
                'command_injection': 0.3
            }.get(attack_type.lower(), 0.5)
            
            # Adjust based on WAF
            if target_context.get('waf_detected'):
                base_probability *= 0.6
            
            # Adjust based on technology stack
            if target_context.get('technologies'):
                techs = str(target_context['technologies']).lower()
                if 'wordpress' in techs:
                    base_probability *= 1.2  # WordPress often vulnerable
                if 'cloudflare' in techs:
                    base_probability *= 0.7  # CloudFlare protection
            
            return base_probability >= success_threshold
        
        # Bind methods to module
        module.ai_should_attempt_attack = ai_should_attempt_attack.__get__(module, module.__class__)
        module._heuristic_attack_decision = _heuristic_attack_decision.__get__(module, module.__class__)
    
    def _inject_ai_learning_methods(self, module):
        """Inject AI learning methods"""
        
        async def ai_learn_from_result(self, attack_type: str, payload: str, success: bool, 
                                     response_data: Dict[str, Any], target_context: Dict[str, Any]):
            """Learn from attack result for future improvement"""
            if not AI_AVAILABLE or not self._ai_enhancer.attack_sequencer:
                return self._log_result_for_heuristics(attack_type, payload, success, response_data)
            
            try:
                # Create attack action and result for learning
                from core.ai import AttackAction, AttackResult, AttackPhase, AttackVector
                
                vector_map = {
                    'sql_injection': AttackVector.SQL_INJECTION,
                    'xss': AttackVector.XSS,
                    'ssrf': AttackVector.SSRF
                }
                
                vector = vector_map.get(attack_type.lower(), AttackVector.SQL_INJECTION)
                
                action = AttackAction(
                    vector=vector,
                    phase=AttackPhase.EXPLOITATION,
                    target_endpoint=target_context.get('endpoint', '/'),
                    payload=payload,
                    priority=0.7,
                    success_probability=0.5
                )
                
                result = AttackResult(
                    action=action,
                    success=success,
                    response_time=response_data.get('response_time', 1.0),
                    status_code=response_data.get('status_code', 200 if success else 404),
                    response_size=response_data.get('response_size', 0),
                    confidence_score=response_data.get('confidence', 0.8 if success else 0.2)
                )
                
                # Send to AI for learning
                await self._ai_enhancer.attack_sequencer.learn_from_execution(
                    action, result, target_context
                )
                
            except Exception as e:
                self._ai_enhancer.logger.error(f"AI learning failed: {e}")
                self._log_result_for_heuristics(attack_type, payload, success, response_data)
        
        def _log_result_for_heuristics(self, attack_type: str, payload: str, success: bool, response_data: Dict[str, Any]):
            """Log result for heuristic learning when AI not available"""
            # Simple logging for future heuristic improvements
            log_entry = {
                'timestamp': asyncio.get_event_loop().time(),
                'attack_type': attack_type,
                'payload': payload[:100],  # Truncate long payloads
                'success': success,
                'status_code': response_data.get('status_code'),
                'response_time': response_data.get('response_time')
            }
            
            # Store in module's learning history
            if not hasattr(self, '_learning_history'):
                self._learning_history = []
            
            self._learning_history.append(log_entry)
            
            # Keep only recent entries
            if len(self._learning_history) > 1000:
                self._learning_history = self._learning_history[-500:]
        
        # Bind methods to module
        module.ai_learn_from_result = ai_learn_from_result.__get__(module, module.__class__)
        module._log_result_for_heuristics = _log_result_for_heuristics.__get__(module, module.__class__)
    
    def _inject_ai_coordination_methods(self, module):
        """Inject AI coordination methods"""
        
        async def ai_get_recommendations(self, current_state: Dict[str, Any]):
            """Get AI recommendations for current attack state"""
            if not AI_AVAILABLE:
                return self._get_heuristic_recommendations(current_state)
            
            try:
                # This would integrate with AI thinking model if available
                recommendations = []
                
                # Analyze current state and provide recommendations
                if current_state.get('failed_attempts', 0) > 3:
                    recommendations.append("Consider switching attack vector - high failure rate detected")
                
                if current_state.get('waf_detected'):
                    recommendations.append("WAF detected - recommend using evasion techniques")
                
                if current_state.get('slow_responses'):
                    recommendations.append("Slow responses detected - reduce request rate")
                
                return recommendations
                
            except Exception as e:
                self._ai_enhancer.logger.error(f"AI recommendations failed: {e}")
                return self._get_heuristic_recommendations(current_state)
        
        def _get_heuristic_recommendations(self, current_state: Dict[str, Any]):
            """Get heuristic recommendations when AI not available"""
            recommendations = []
            
            if current_state.get('success_rate', 0) < 0.2:
                recommendations.append("Low success rate - consider different attack approach")
            
            if current_state.get('response_time_avg', 0) > 5.0:
                recommendations.append("High response times - reduce concurrent requests")
            
            return recommendations
        
        def ai_get_status(self):
            """Get AI enhancement status for this module"""
            return {
                'ai_enhanced': getattr(self, '_ai_enhanced', False),
                'module_name': getattr(self, '_module_name', 'unknown'),
                'ai_components_available': AI_AVAILABLE,
                'capabilities': {
                    'smart_payload_generation': AI_AVAILABLE and self._ai_enhancer.payload_generator is not None,
                    'attack_decision_support': AI_AVAILABLE and self._ai_enhancer.attack_sequencer is not None,
                    'learning_from_results': AI_AVAILABLE,
                    'intelligent_recommendations': AI_AVAILABLE
                }
            }
        
        # Bind methods to module
        module.ai_get_recommendations = ai_get_recommendations.__get__(module, module.__class__)
        module._get_heuristic_recommendations = _get_heuristic_recommendations.__get__(module, module.__class__)
        module.ai_get_status = ai_get_status.__get__(module, module.__class__)
    
    def _wrap_existing_methods(self, module):
        """Wrap existing methods with AI enhancements"""
        
        # Get all methods in the module
        methods_to_wrap = []
        
        for attr_name in dir(module):
            if not attr_name.startswith('_') and not attr_name.startswith('ai_'):
                attr = getattr(module, attr_name)
                if callable(attr) and not inspect.isclass(attr):
                    methods_to_wrap.append((attr_name, attr))
        
        # Wrap attack-related methods
        attack_method_keywords = ['attack', 'exploit', 'test', 'scan', 'inject', 'brute']
        
        for method_name, method in methods_to_wrap:
            if any(keyword in method_name.lower() for keyword in attack_method_keywords):
                wrapped_method = self._create_ai_wrapped_method(method, method_name)
                setattr(module, f"_original_{method_name}", method)
                setattr(module, method_name, wrapped_method)
    
    def _create_ai_wrapped_method(self, original_method, method_name: str):
        """Create AI-enhanced wrapper for existing method"""
        
        @functools.wraps(original_method)
        async def ai_enhanced_wrapper(*args, **kwargs):
            # Pre-execution AI analysis
            start_time = asyncio.get_event_loop().time()
            
            try:
                # Call original method
                if asyncio.iscoroutinefunction(original_method):
                    result = await original_method(*args, **kwargs)
                else:
                    result = original_method(*args, **kwargs)
                
                # Post-execution AI learning (if result indicates success/failure)
                execution_time = asyncio.get_event_loop().time() - start_time
                
                # Try to extract meaningful data from result for learning
                if isinstance(result, dict) and 'success' in result:
                    await self._learn_from_method_result(
                        method_name, result, execution_time, args, kwargs
                    )
                
                return result
                
            except Exception as e:
                # Learn from failures too
                execution_time = asyncio.get_event_loop().time() - start_time
                await self._learn_from_method_failure(
                    method_name, str(e), execution_time, args, kwargs
                )
                raise
        
        return ai_enhanced_wrapper
    
    async def _learn_from_method_result(self, method_name: str, result: Dict[str, Any], 
                                      execution_time: float, args: tuple, kwargs: Dict[str, Any]):
        """Learn from method execution result"""
        if AI_AVAILABLE and self.attack_sequencer:
            try:
                # Create learning context
                learning_context = {
                    'method_name': method_name,
                    'execution_time': execution_time,
                    'success': result.get('success', False),
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                # This could be enhanced to provide more detailed learning
                # For now, just log the successful pattern
                self.logger.debug(f"Method {method_name} executed successfully in {execution_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to learn from method result: {e}")
    
    async def _learn_from_method_failure(self, method_name: str, error: str, 
                                       execution_time: float, args: tuple, kwargs: Dict[str, Any]):
        """Learn from method execution failure"""
        if AI_AVAILABLE:
            try:
                # Log failure pattern for learning
                self.logger.warning(f"Method {method_name} failed after {execution_time:.2f}s: {error}")
                
            except Exception as e:
                self.logger.error(f"Failed to learn from method failure: {e}")
    
    def get_enhancement_summary(self) -> Dict[str, Any]:
        """Get summary of all enhanced modules"""
        return {
            'enhanced_modules': list(self.enhanced_modules.keys()),
            'ai_available': AI_AVAILABLE,
            'total_enhanced': len(self.enhanced_modules),
            'ai_status': self.ai_status
        }

# Create default enhancer instance
_default_enhancer = None

def get_module_enhancer() -> ModuleAIEnhancer:
    """Get default module enhancer instance"""
    global _default_enhancer
    if _default_enhancer is None:
        _default_enhancer = ModuleAIEnhancer()
    return _default_enhancer

def enhance_attack_module(module, module_name: str):
    """Convenience function to enhance an attack module"""
    enhancer = get_module_enhancer()
    return enhancer.enhance_module(module, module_name)

# Export main classes and functions
__all__ = ['ModuleAIEnhancer', 'get_module_enhancer', 'enhance_attack_module']