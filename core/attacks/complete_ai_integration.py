#!/usr/bin/env python3
"""
Agent DS v2.0 - Complete AI Integration System  
==============================================

Final integration system that brings together all AI components:
- AI Thinking Model
- AI Payload Generator  
- Adaptive Attack Sequencer
- Module AI Enhancer
- AI Coordination Manager

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

# Import coordination components
from .ai_coordination_manager import get_ai_coordinator, MissionContext, MissionPhase
from .ai_module_enhancer import get_module_enhancer

class CompleteAIIntegrationSystem:
    """
    Complete AI integration system for Agent DS v2.0
    Provides unified interface to all AI capabilities
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize core AI systems
        try:
            self.coordination_manager = get_ai_coordinator()
            self.module_enhancer = get_module_enhancer()
            self.logger.info("AI Integration System initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI systems: {e}")
            self.coordination_manager = None
            self.module_enhancer = None
    
    async def execute_intelligent_pentest(self, target_url: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a complete AI-powered penetration test
        """
        self.logger.info(f"Starting intelligent penetration test for {target_url}")
        
        try:
            # Create mission context
            mission_context = MissionContext(
                target_url=target_url,
                target_ip=kwargs.get('target_ip'),
                technologies=kwargs.get('technologies', []),
                cms_type=kwargs.get('cms_type'),
                waf_detected=kwargs.get('waf_detected', False),
                web_server=kwargs.get('web_server'),
                database_type=kwargs.get('database_type'),
                admin_panels=kwargs.get('admin_panels', []),
                open_ports=kwargs.get('open_ports', []),
                custom_params=kwargs.get('custom_params', {})
            )
            
            # Execute coordinated mission
            if self.coordination_manager:
                mission_result = await self.coordination_manager.execute_coordinated_mission(
                    mission_context=mission_context,
                    phases=[
                        MissionPhase.RECONNAISSANCE,
                        MissionPhase.VULNERABILITY_DISCOVERY,
                        MissionPhase.EXPLOITATION
                    ]
                )
                
                return mission_result
            else:
                return {
                    'success': False,
                    'error': 'AI coordination manager not available',
                    'fallback_mode': True
                }
                
        except Exception as e:
            self.logger.error(f"Intelligent pentest failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'target_url': target_url
            }
    
    async def get_ai_recommendations(self, target_url: str, current_findings: Dict[str, Any]) -> List[str]:
        """
        Get AI-powered recommendations based on current findings
        """
        try:
            recommendations = []
            
            # Get recommendations from coordination manager
            if self.coordination_manager:
                status = self.coordination_manager.get_coordination_status()
                
                if current_findings.get('vulnerabilities_found'):
                    recommendations.append("Consider exploiting discovered vulnerabilities with AI-generated payloads")
                
                if current_findings.get('admin_panels'):
                    recommendations.append("Test admin panels with intelligent credential attacks")
                
                if current_findings.get('database_detected'):
                    recommendations.append("Attempt database exploitation using AI-enhanced SQL injection")
                
                if current_findings.get('waf_detected'):
                    recommendations.append("Use AI evasion techniques to bypass WAF protection")
                
                # Add module-specific recommendations
                for module_name in status.get('enhanced_modules', []):
                    recommendations.append(f"Leverage AI-enhanced {module_name} capabilities")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get AI recommendations: {e}")
            return ["Enable AI debugging mode for detailed analysis"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive AI system status
        """
        try:
            status = {
                'ai_integration_active': True,
                'systems': {}
            }
            
            # Coordination manager status
            if self.coordination_manager:
                status['systems']['coordination_manager'] = self.coordination_manager.get_coordination_status()
            else:
                status['systems']['coordination_manager'] = {'status': 'not_available'}
            
            # Module enhancer status  
            if self.module_enhancer:
                status['systems']['module_enhancer'] = self.module_enhancer.get_enhancement_summary()
            else:
                status['systems']['module_enhancer'] = {'status': 'not_available'}
            
            # AI component status
            try:
                from core.ai import get_ai_status
                status['systems']['ai_components'] = get_ai_status()
            except ImportError:
                status['systems']['ai_components'] = {'status': 'not_available'}
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {
                'ai_integration_active': False,
                'error': str(e)
            }
    
    async def demonstrate_ai_capabilities(self, target_url: str = "https://example.com") -> Dict[str, Any]:
        """
        Demonstrate AI capabilities with a safe target
        """
        self.logger.info("Demonstrating AI capabilities...")
        
        demo_results = {
            'demo_target': target_url,
            'capabilities_tested': [],
            'results': {}
        }
        
        try:
            # Test AI recommendation system
            demo_results['capabilities_tested'].append('ai_recommendations')
            recommendations = await self.get_ai_recommendations(target_url, {
                'vulnerabilities_found': True,
                'admin_panels': ['/admin', '/wp-admin'],
                'database_detected': True,
                'waf_detected': False
            })
            demo_results['results']['ai_recommendations'] = recommendations
            
            # Test system status
            demo_results['capabilities_tested'].append('system_status')
            status = self.get_system_status()
            demo_results['results']['system_status'] = status
            
            # Test module enhancement status
            if self.module_enhancer:
                demo_results['capabilities_tested'].append('module_enhancement')
                enhancement_summary = self.module_enhancer.get_enhancement_summary()
                demo_results['results']['module_enhancement'] = enhancement_summary
            
            demo_results['success'] = True
            return demo_results
            
        except Exception as e:
            self.logger.error(f"Demo failed: {e}")
            demo_results['success'] = False
            demo_results['error'] = str(e)
            return demo_results

# Create default integration system
_default_integration_system = None

def get_ai_integration_system() -> CompleteAIIntegrationSystem:
    """Get default AI integration system instance"""
    global _default_integration_system
    if _default_integration_system is None:
        _default_integration_system = CompleteAIIntegrationSystem()
    return _default_integration_system

# Convenience functions for easy access
async def execute_ai_pentest(target_url: str, **kwargs) -> Dict[str, Any]:
    """Execute AI-powered penetration test"""
    system = get_ai_integration_system()
    return await system.execute_intelligent_pentest(target_url, **kwargs)

async def get_ai_recommendations(target_url: str, findings: Dict[str, Any]) -> List[str]:
    """Get AI recommendations for current findings"""
    system = get_ai_integration_system()
    return await system.get_ai_recommendations(target_url, findings)

def get_ai_system_status() -> Dict[str, Any]:
    """Get AI system status"""
    system = get_ai_integration_system()
    return system.get_system_status()

async def demo_ai_capabilities(target_url: str = "https://example.com") -> Dict[str, Any]:
    """Demonstrate AI capabilities"""
    system = get_ai_integration_system()
    return await system.demonstrate_ai_capabilities(target_url)

# Export main classes and functions
__all__ = [
    'CompleteAIIntegrationSystem', 
    'get_ai_integration_system',
    'execute_ai_pentest',
    'get_ai_recommendations', 
    'get_ai_system_status',
    'demo_ai_capabilities'
]