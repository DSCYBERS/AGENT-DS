"""
Agent DS - Sandbox Environment
Isolated sandbox environment for safe payload testing and AI training
"""

import asyncio
import json
import time
import subprocess
import tempfile
import shutil
import docker
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import signal
import os

from core.config.settings import Config
from core.utils.logger import get_logger

logger = get_logger('sandbox_environment')

@dataclass
class SandboxContainer:
    """Container for sandbox execution environment"""
    container_id: str
    container_name: str
    environment_type: str
    status: str
    created_at: datetime
    last_activity: datetime
    resource_limits: Dict[str, Any]
    network_isolated: bool = True
    filesystem_isolated: bool = True

@dataclass
class SandboxExecution:
    """Result of sandbox execution"""
    execution_id: str
    container_id: str
    payload: str
    attack_type: str
    execution_time: float
    success: bool
    output: str
    error_output: str
    resource_usage: Dict[str, Any]
    security_violations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class DockerSandboxManager:
    """Manages Docker-based sandbox containers"""
    
    def __init__(self):
        self.logger = get_logger('docker_sandbox')
        try:
            self.client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            self.logger.warning(f"Docker not available: {str(e)}")
            self.docker_available = False
        
        # Sandbox configurations
        self.sandbox_configs = {
            'web_app': {
                'image': 'agent-ds/web-sandbox:latest',
                'ports': {'80/tcp': None, '443/tcp': None},
                'environment': {
                    'FLASK_ENV': 'development',
                    'MYSQL_ROOT_PASSWORD': 'sandbox123'
                },
                'mem_limit': '512m',
                'cpu_quota': 50000,  # 50% CPU
                'network_mode': 'none'
            },
            'database': {
                'image': 'agent-ds/db-sandbox:latest',
                'ports': {'3306/tcp': None, '5432/tcp': None},
                'environment': {
                    'MYSQL_ROOT_PASSWORD': 'sandbox123',
                    'POSTGRES_PASSWORD': 'sandbox123'
                },
                'mem_limit': '256m',
                'cpu_quota': 30000,
                'network_mode': 'none'
            },
            'api_server': {
                'image': 'agent-ds/api-sandbox:latest',
                'ports': {'8080/tcp': None, '8443/tcp': None},
                'environment': {
                    'NODE_ENV': 'development',
                    'API_KEY': 'sandbox_key_123'
                },
                'mem_limit': '384m',
                'cpu_quota': 40000,
                'network_mode': 'none'
            }
        }
        
        # Active containers
        self.active_containers = {}
        self.execution_history = []
        
        # Resource monitoring
        self.resource_monitor_active = False
        self.resource_monitor_thread = None
    
    async def create_sandbox_container(self, environment_type: str) -> Optional[SandboxContainer]:
        """Create a new sandbox container"""
        if not self.docker_available:
            self.logger.error("Docker not available for sandbox creation")
            return None
        
        if environment_type not in self.sandbox_configs:
            self.logger.error(f"Unknown environment type: {environment_type}")
            return None
        
        try:
            config = self.sandbox_configs[environment_type]
            container_name = f"agent-ds-sandbox-{environment_type}-{uuid.uuid4().hex[:8]}"
            
            self.logger.info(f"Creating sandbox container: {container_name}")
            
            # Create container with security constraints
            container = self.client.containers.run(
                image=config['image'],
                name=container_name,
                detach=True,
                remove=False,
                mem_limit=config['mem_limit'],
                cpu_quota=config.get('cpu_quota', 50000),
                network_mode=config.get('network_mode', 'none'),
                environment=config.get('environment', {}),
                ports=config.get('ports', {}),
                cap_drop=['ALL'],  # Drop all capabilities
                cap_add=['CHOWN', 'SETGID', 'SETUID'],  # Only essential capabilities
                security_opt=['no-new-privileges:true'],
                read_only=False,  # Allow writes to tmp directories
                tmpfs={'/tmp': 'noexec,nosuid,size=100m'},
                user='1000:1000'  # Non-root user
            )
            
            # Wait for container to be ready
            await asyncio.sleep(2)
            
            sandbox_container = SandboxContainer(
                container_id=container.id,
                container_name=container_name,
                environment_type=environment_type,
                status='running',
                created_at=datetime.now(),
                last_activity=datetime.now(),
                resource_limits={
                    'memory': config['mem_limit'],
                    'cpu_quota': config.get('cpu_quota', 50000)
                }
            )
            
            self.active_containers[container.id] = sandbox_container
            
            self.logger.info(f"Sandbox container created successfully: {container_name}")
            return sandbox_container
            
        except Exception as e:
            self.logger.error(f"Failed to create sandbox container: {str(e)}")
            return None
    
    async def execute_in_sandbox(self, container_id: str, payload: str, 
                                attack_type: str, timeout: int = 30) -> SandboxExecution:
        """Execute payload in sandbox container"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        if not self.docker_available:
            return self._create_mock_execution(execution_id, container_id, payload, attack_type)
        
        try:
            container = self.client.containers.get(container_id)
            
            # Prepare execution command based on attack type
            command = self._prepare_execution_command(payload, attack_type)
            
            self.logger.info(f"Executing in sandbox {container_id}: {attack_type}")
            
            # Execute with timeout
            exec_result = container.exec_run(
                cmd=command,
                timeout=timeout,
                user='1000',  # Non-root execution
                workdir='/tmp'
            )
            
            execution_time = time.time() - start_time
            
            # Parse execution results
            success = exec_result.exit_code == 0
            output = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else ""
            
            # Monitor resource usage
            resource_usage = await self._get_container_resource_usage(container_id)
            
            # Detect security violations
            security_violations = self._detect_security_violations(output, payload)
            
            execution = SandboxExecution(
                execution_id=execution_id,
                container_id=container_id,
                payload=payload,
                attack_type=attack_type,
                execution_time=execution_time,
                success=success,
                output=output[:1000],  # Limit output size
                error_output="",
                resource_usage=resource_usage,
                security_violations=security_violations
            )
            
            # Update container activity
            if container_id in self.active_containers:
                self.active_containers[container_id].last_activity = datetime.now()
            
            self.execution_history.append(execution)
            
            return execution
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Sandbox execution failed: {str(e)}")
            
            return SandboxExecution(
                execution_id=execution_id,
                container_id=container_id,
                payload=payload,
                attack_type=attack_type,
                execution_time=execution_time,
                success=False,
                output="",
                error_output=str(e),
                resource_usage={},
                security_violations=[]
            )
    
    def _prepare_execution_command(self, payload: str, attack_type: str) -> List[str]:
        """Prepare execution command based on attack type"""
        
        # Escape payload for safe execution
        escaped_payload = payload.replace('"', '\\"').replace('$', '\\$')
        
        if attack_type == 'sql_injection':
            return ['sh', '-c', f'echo "{escaped_payload}" | mysql -h localhost -u root -psandbox123 testdb']
        
        elif attack_type == 'xss':
            return ['sh', '-c', f'echo "{escaped_payload}" > /tmp/xss_test.html && cat /tmp/xss_test.html']
        
        elif attack_type == 'ssti':
            return ['python3', '-c', f'from jinja2 import Template; t = Template("{escaped_payload}"); print(t.render())']
        
        elif attack_type == 'command_injection':
            return ['sh', '-c', f'echo "Testing: {escaped_payload}"']
        
        elif attack_type == 'xxe':
            return ['python3', '-c', f'import xml.etree.ElementTree as ET; ET.fromstring("{escaped_payload}")']
        
        else:
            # Generic execution
            return ['sh', '-c', f'echo "Payload: {escaped_payload}"']
    
    async def _get_container_resource_usage(self, container_id: str) -> Dict[str, Any]:
        """Get resource usage statistics for container"""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            cpu_percent = 0.0
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # Memory usage
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_limit = stats['memory_stats'].get('limit', 0)
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'memory_percent': memory_percent,
                'network_rx_bytes': stats.get('networks', {}).get('eth0', {}).get('rx_bytes', 0),
                'network_tx_bytes': stats.get('networks', {}).get('eth0', {}).get('tx_bytes', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get resource usage: {str(e)}")
            return {}
    
    def _detect_security_violations(self, output: str, payload: str) -> List[str]:
        """Detect potential security violations in execution"""
        violations = []
        
        # Check for privilege escalation attempts
        if any(keyword in output.lower() for keyword in ['root', 'sudo', 'su ', '/etc/passwd']):
            violations.append('privilege_escalation_attempt')
        
        # Check for file system access
        if any(keyword in output.lower() for keyword in ['/etc/', '/proc/', '/sys/', '/dev/']):
            violations.append('system_file_access')
        
        # Check for network activity
        if any(keyword in output.lower() for keyword in ['connect', 'socket', 'bind', 'listen']):
            violations.append('network_activity')
        
        # Check for process manipulation
        if any(keyword in output.lower() for keyword in ['kill', 'killall', 'pkill', 'exec']):
            violations.append('process_manipulation')
        
        # Check for dangerous payload patterns
        if any(pattern in payload.lower() for pattern in ['rm -rf', 'format c:', 'del /f']):
            violations.append('destructive_payload')
        
        return violations
    
    def _create_mock_execution(self, execution_id: str, container_id: str, 
                             payload: str, attack_type: str) -> SandboxExecution:
        """Create mock execution when Docker is not available"""
        return SandboxExecution(
            execution_id=execution_id,
            container_id=container_id,
            payload=payload,
            attack_type=attack_type,
            execution_time=1.0 + (len(payload) / 100.0),
            success=True,
            output=f"Mock execution of {attack_type} payload",
            error_output="",
            resource_usage={'cpu_percent': 10.0, 'memory_usage_mb': 50.0},
            security_violations=[]
        )
    
    async def cleanup_container(self, container_id: str):
        """Clean up and remove sandbox container"""
        try:
            if container_id in self.active_containers:
                del self.active_containers[container_id]
            
            if self.docker_available:
                container = self.client.containers.get(container_id)
                container.stop(timeout=10)
                container.remove()
                self.logger.info(f"Sandbox container {container_id} cleaned up")
        
        except Exception as e:
            self.logger.error(f"Failed to cleanup container {container_id}: {str(e)}")
    
    async def cleanup_all_containers(self):
        """Clean up all active sandbox containers"""
        for container_id in list(self.active_containers.keys()):
            await self.cleanup_container(container_id)
    
    def get_sandbox_statistics(self) -> Dict[str, Any]:
        """Get sandbox usage statistics"""
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for exec in self.execution_history if exec.success)
        
        # Calculate average execution time
        avg_execution_time = sum(exec.execution_time for exec in self.execution_history) / total_executions if total_executions > 0 else 0.0
        
        # Security violations summary
        all_violations = []
        for exec in self.execution_history:
            all_violations.extend(exec.security_violations)
        
        violation_counts = {}
        for violation in all_violations:
            violation_counts[violation] = violation_counts.get(violation, 0) + 1
        
        return {
            'active_containers': len(self.active_containers),
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0.0,
            'average_execution_time': avg_execution_time,
            'security_violations': violation_counts,
            'docker_available': self.docker_available
        }

class LocalSandboxManager:
    """Manages local process-based sandbox for lightweight testing"""
    
    def __init__(self):
        self.logger = get_logger('local_sandbox')
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix='agent_ds_sandbox_'))
        self.active_processes = {}
        self.execution_history = []
        
        # Create sandbox directory structure
        self._setup_sandbox_directory()
    
    def _setup_sandbox_directory(self):
        """Setup sandbox directory structure"""
        try:
            # Create subdirectories
            (self.sandbox_dir / 'tmp').mkdir(exist_ok=True)
            (self.sandbox_dir / 'var').mkdir(exist_ok=True)
            (self.sandbox_dir / 'logs').mkdir(exist_ok=True)
            
            # Create dummy files for testing
            (self.sandbox_dir / 'index.html').write_text('<html><body>Test Page</body></html>')
            (self.sandbox_dir / 'test.php').write_text('<?php echo "Hello World"; ?>')
            
            self.logger.info(f"Local sandbox directory created: {self.sandbox_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup sandbox directory: {str(e)}")
    
    async def execute_in_local_sandbox(self, payload: str, attack_type: str, 
                                     timeout: int = 30) -> SandboxExecution:
        """Execute payload in local sandbox"""
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing in local sandbox: {attack_type}")
            
            # Prepare safe execution environment
            env = os.environ.copy()
            env['PATH'] = '/usr/bin:/bin'  # Limited PATH
            env['HOME'] = str(self.sandbox_dir)
            env['TMPDIR'] = str(self.sandbox_dir / 'tmp')
            
            # Prepare command based on attack type
            command = self._prepare_local_command(payload, attack_type)
            
            # Execute with resource limits
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.sandbox_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=self._set_process_limits
            )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError("Execution timeout")
            
            execution_time = time.time() - start_time
            
            output = stdout.decode('utf-8', errors='replace') if stdout else ""
            error_output = stderr.decode('utf-8', errors='replace') if stderr else ""
            
            # Monitor resource usage (simplified)
            resource_usage = {
                'execution_time': execution_time,
                'return_code': process.returncode
            }
            
            # Detect security violations
            security_violations = self._detect_local_security_violations(output, error_output, payload)
            
            execution = SandboxExecution(
                execution_id=execution_id,
                container_id='local_sandbox',
                payload=payload,
                attack_type=attack_type,
                execution_time=execution_time,
                success=process.returncode == 0,
                output=output[:1000],
                error_output=error_output[:500],
                resource_usage=resource_usage,
                security_violations=security_violations
            )
            
            self.execution_history.append(execution)
            return execution
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Local sandbox execution failed: {str(e)}")
            
            return SandboxExecution(
                execution_id=execution_id,
                container_id='local_sandbox',
                payload=payload,
                attack_type=attack_type,
                execution_time=execution_time,
                success=False,
                output="",
                error_output=str(e),
                resource_usage={},
                security_violations=[]
            )
    
    def _prepare_local_command(self, payload: str, attack_type: str) -> List[str]:
        """Prepare local execution command"""
        
        if attack_type == 'xss':
            # Test XSS in HTML context
            html_content = f'<html><body>{payload}</body></html>'
            html_file = self.sandbox_dir / 'xss_test.html'
            html_file.write_text(html_content)
            return ['cat', str(html_file)]
        
        elif attack_type == 'ssti':
            # Test SSTI with Python
            python_code = f"""
import sys
try:
    from jinja2 import Template
    t = Template('{payload}')
    print(t.render())
except Exception as e:
    print(f"Error: {{e}}")
"""
            python_file = self.sandbox_dir / 'ssti_test.py'
            python_file.write_text(python_code)
            return ['python3', str(python_file)]
        
        elif attack_type == 'command_injection':
            # Safe command injection test
            return ['echo', f'Testing payload: {payload}']
        
        elif attack_type == 'sql_injection':
            # Mock SQL execution
            return ['echo', f'SQL Query: {payload}']
        
        else:
            # Generic payload test
            return ['echo', f'Payload: {payload}']
    
    def _set_process_limits(self):
        """Set resource limits for sandbox process"""
        try:
            import resource
            
            # Limit CPU time (10 seconds)
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
            
            # Limit memory (64MB)
            resource.setrlimit(resource.RLIMIT_AS, (64 * 1024 * 1024, 64 * 1024 * 1024))
            
            # Limit file size (1MB)
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
            
            # Limit number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))
            
        except Exception as e:
            # Resource limits not available on this system
            pass
    
    def _detect_local_security_violations(self, output: str, error_output: str, payload: str) -> List[str]:
        """Detect security violations in local execution"""
        violations = []
        
        combined_output = (output + error_output).lower()
        
        # Check for system access attempts
        if any(keyword in combined_output for keyword in ['/etc/passwd', '/etc/shadow', 'root:']):
            violations.append('system_file_access')
        
        # Check for command execution attempts
        if any(keyword in combined_output for keyword in ['command not found', 'permission denied']):
            violations.append('unauthorized_command')
        
        # Check for error conditions that might indicate exploitation
        if 'traceback' in combined_output or 'exception' in combined_output:
            violations.append('execution_error')
        
        return violations
    
    def cleanup_local_sandbox(self):
        """Clean up local sandbox directory"""
        try:
            shutil.rmtree(self.sandbox_dir)
            self.logger.info("Local sandbox cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup local sandbox: {str(e)}")

class SandboxOrchestrator:
    """Main orchestrator for sandbox environments"""
    
    def __init__(self):
        self.logger = get_logger('sandbox_orchestrator')
        
        # Initialize sandbox managers
        self.docker_manager = DockerSandboxManager()
        self.local_manager = LocalSandboxManager()
        
        # Sandbox preference (Docker preferred, fallback to local)
        self.prefer_docker = True
        
        # Execution tracking
        self.total_executions = 0
        self.successful_executions = 0
        
        # Database for storing results
        self.db_path = Path("sandbox_results.db")
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database for storing sandbox results"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS sandbox_executions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        execution_id TEXT UNIQUE NOT NULL,
                        container_id TEXT,
                        payload TEXT NOT NULL,
                        attack_type TEXT NOT NULL,
                        execution_time REAL,
                        success INTEGER,
                        output TEXT,
                        security_violations TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            
            self.logger.info("Sandbox database initialized")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
    
    async def test_payload_safely(self, payload: str, attack_type: str, 
                                environment_type: str = 'web_app') -> SandboxExecution:
        """Test payload in safe sandbox environment"""
        self.total_executions += 1
        
        try:
            # Choose sandbox type
            if self.prefer_docker and self.docker_manager.docker_available:
                result = await self._test_in_docker_sandbox(payload, attack_type, environment_type)
            else:
                result = await self._test_in_local_sandbox(payload, attack_type)
            
            # Store result in database
            await self._store_execution_result(result)
            
            if result.success:
                self.successful_executions += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Sandbox testing failed: {str(e)}")
            
            # Return error result
            return SandboxExecution(
                execution_id=str(uuid.uuid4()),
                container_id='error',
                payload=payload,
                attack_type=attack_type,
                execution_time=0.0,
                success=False,
                output="",
                error_output=str(e),
                resource_usage={},
                security_violations=[]
            )
    
    async def _test_in_docker_sandbox(self, payload: str, attack_type: str, 
                                    environment_type: str) -> SandboxExecution:
        """Test payload in Docker sandbox"""
        # Create or reuse container
        container = await self._get_or_create_container(environment_type)
        
        if not container:
            raise Exception("Failed to create Docker container")
        
        # Execute in container
        result = await self.docker_manager.execute_in_sandbox(
            container.container_id, payload, attack_type
        )
        
        return result
    
    async def _test_in_local_sandbox(self, payload: str, attack_type: str) -> SandboxExecution:
        """Test payload in local sandbox"""
        return await self.local_manager.execute_in_local_sandbox(payload, attack_type)
    
    async def _get_or_create_container(self, environment_type: str) -> Optional[SandboxContainer]:
        """Get existing or create new container for environment type"""
        # Check for existing containers
        for container in self.docker_manager.active_containers.values():
            if (container.environment_type == environment_type and 
                container.status == 'running' and
                (datetime.now() - container.last_activity).seconds < 300):  # 5 minutes
                return container
        
        # Create new container
        return await self.docker_manager.create_sandbox_container(environment_type)
    
    async def _store_execution_result(self, result: SandboxExecution):
        """Store execution result in database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO sandbox_executions 
                    (execution_id, container_id, payload, attack_type, execution_time, 
                     success, output, security_violations)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.execution_id,
                    result.container_id,
                    result.payload,
                    result.attack_type,
                    result.execution_time,
                    1 if result.success else 0,
                    result.output,
                    json.dumps(result.security_violations)
                ))
        
        except Exception as e:
            self.logger.error(f"Failed to store execution result: {str(e)}")
    
    async def batch_test_payloads(self, payloads: List[Tuple[str, str]], 
                                environment_type: str = 'web_app') -> List[SandboxExecution]:
        """Test multiple payloads in batch"""
        results = []
        
        # Create semaphore to limit concurrent executions
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent tests
        
        async def test_payload_with_semaphore(payload_data):
            async with semaphore:
                payload, attack_type = payload_data
                return await self.test_payload_safely(payload, attack_type, environment_type)
        
        # Execute batch
        tasks = [test_payload_with_semaphore(payload_data) for payload_data in payloads]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, SandboxExecution)]
        
        self.logger.info(f"Batch test completed: {len(valid_results)}/{len(payloads)} successful")
        
        return valid_results
    
    async def cleanup_all_sandboxes(self):
        """Clean up all sandbox environments"""
        await self.docker_manager.cleanup_all_containers()
        self.local_manager.cleanup_local_sandbox()
        self.logger.info("All sandboxes cleaned up")
    
    def get_sandbox_metrics(self) -> Dict[str, Any]:
        """Get comprehensive sandbox metrics"""
        docker_stats = self.docker_manager.get_sandbox_statistics()
        
        return {
            'total_executions': self.total_executions,
            'successful_executions': self.successful_executions,
            'success_rate': self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0,
            'docker_available': self.docker_manager.docker_available,
            'docker_stats': docker_stats,
            'local_executions': len(self.local_manager.execution_history),
            'active_containers': len(self.docker_manager.active_containers)
        }
    
    async def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history from database"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('''
                    SELECT execution_id, container_id, payload, attack_type, execution_time,
                           success, output, security_violations, timestamp
                    FROM sandbox_executions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'execution_id': row[0],
                        'container_id': row[1],
                        'payload': row[2],
                        'attack_type': row[3],
                        'execution_time': row[4],
                        'success': bool(row[5]),
                        'output': row[6],
                        'security_violations': json.loads(row[7]) if row[7] else [],
                        'timestamp': row[8]
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get execution history: {str(e)}")
            return []

# Global instance for CLI access
sandbox_environment = SandboxOrchestrator()

if __name__ == "__main__":
    async def test_sandbox():
        """Test the sandbox environment"""
        orchestrator = SandboxOrchestrator()
        
        # Test single payload
        result = await orchestrator.test_payload_safely(
            "{{7*7}}", "ssti", "web_app"
        )
        
        print("Sandbox Test Result:")
        print(f"Success: {result.success}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        print(f"Output: {result.output}")
        print(f"Security Violations: {result.security_violations}")
        
        # Test batch payloads
        payloads = [
            ("' UNION SELECT 1,2,3-- ", "sql_injection"),
            ("<script>alert('xss')</script>", "xss"),
            ("{{config.__class__}}", "ssti")
        ]
        
        batch_results = await orchestrator.batch_test_payloads(payloads)
        print(f"\nBatch test completed: {len(batch_results)} results")
        
        # Get metrics
        metrics = orchestrator.get_sandbox_metrics()
        print("\nSandbox Metrics:")
        print(json.dumps(metrics, indent=2))
        
        # Cleanup
        await orchestrator.cleanup_all_sandboxes()
    
    # Run test
    asyncio.run(test_sandbox())