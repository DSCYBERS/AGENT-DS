"""
Agent DS Reconnaissance Module
Integrates multiple reconnaissance tools with AI-driven analysis
"""

import subprocess
import json
import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
import ipaddress
import socket
import concurrent.futures
import logging

from core.config.settings import Config
from core.database.manager import DatabaseManager
from core.utils.logger import get_logger, log_security_event

logger = get_logger('recon')

class ReconModule:
    """Main reconnaissance module for Agent DS"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        self.tools = {
            'nmap': NmapScanner(self.config),
            'masscan': MasscanScanner(self.config),
            'gobuster': GobusterScanner(self.config),
            'sublist3r': Sublist3rScanner(self.config),
            'amass': AmassScanner(self.config),
            'whatweb': WhatWebScanner(self.config)
        }
        
    async def run_full_recon(self, target: str, mission_id: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive reconnaissance on target"""
        logger.info(f"Starting full reconnaissance on {target}")
        
        results = {
            'target': target,
            'start_time': datetime.now().isoformat(),
            'tools_used': [],
            'hosts': [],
            'services': [],
            'subdomains': [],
            'technologies': [],
            'summary': {}
        }
        
        try:
            # Phase 1: Subdomain enumeration
            logger.info("Phase 1: Subdomain enumeration")
            subdomains = await self._run_subdomain_enum(target)
            results['subdomains'] = subdomains
            results['tools_used'].extend(['sublist3r', 'amass'])
            
            # Phase 2: Host discovery and port scanning
            logger.info("Phase 2: Host discovery and port scanning")
            all_targets = [target] + subdomains
            hosts_and_services = await self._run_host_discovery(all_targets)
            results['hosts'] = hosts_and_services['hosts']
            results['services'] = hosts_and_services['services']
            results['tools_used'].extend(['nmap', 'masscan'])
            
            # Phase 3: Web technology detection
            logger.info("Phase 3: Web technology detection")
            web_targets = self._extract_web_targets(results['services'])
            technologies = await self._run_tech_detection(web_targets)
            results['technologies'] = technologies
            results['tools_used'].append('whatweb')
            
            # Phase 4: Directory brute forcing
            logger.info("Phase 4: Directory enumeration")
            directories = await self._run_directory_enum(web_targets)
            results['directories'] = directories
            results['tools_used'].append('gobuster')
            
            # Store results in database
            if mission_id:
                self._store_recon_results(mission_id, results)
            
            # Generate summary
            results['summary'] = self._generate_summary(results)
            results['end_time'] = datetime.now().isoformat()
            
            log_security_event(
                'RECONNAISSANCE_COMPLETED',
                {'target': target, 'hosts_found': len(results['hosts']), 'services_found': len(results['services'])},
                mission_id=mission_id
            )
            
            logger.info(f"Reconnaissance completed for {target}")
            return results
            
        except Exception as e:
            logger.error(f"Reconnaissance failed: {str(e)}")
            raise
    
    async def run_custom_recon(self, target: str, modules: Optional[str] = None,
                              mission_id: Optional[str] = None) -> Dict[str, Any]:
        """Run specific reconnaissance modules"""
        if modules:
            requested_modules = [m.strip() for m in modules.split(',')]
        else:
            requested_modules = ['nmap', 'gobuster']
        
        logger.info(f"Running custom reconnaissance with modules: {requested_modules}")
        
        results = {
            'target': target,
            'start_time': datetime.now().isoformat(),
            'tools_used': requested_modules,
            'results': {}
        }
        
        # Run requested modules
        for module in requested_modules:
            if module in self.tools:
                logger.info(f"Running {module} scan")
                try:
                    module_result = await self.tools[module].scan(target)
                    results['results'][module] = module_result
                except Exception as e:
                    logger.error(f"Error running {module}: {str(e)}")
                    results['results'][module] = {'error': str(e)}
        
        # Store results
        if mission_id:
            for module, result in results['results'].items():
                self.db_manager.store_recon_result(
                    mission_id, target, module, 'custom_scan', result
                )
        
        results['end_time'] = datetime.now().isoformat()
        return results
    
    async def _run_subdomain_enum(self, domain: str) -> List[str]:
        """Run subdomain enumeration tools"""
        subdomains = set()
        
        # Run Sublist3r
        try:
            sublist3r_results = await self.tools['sublist3r'].scan(domain)
            subdomains.update(sublist3r_results.get('subdomains', []))
        except Exception as e:
            logger.error(f"Sublist3r error: {str(e)}")
        
        # Run Amass
        try:
            amass_results = await self.tools['amass'].scan(domain)
            subdomains.update(amass_results.get('subdomains', []))
        except Exception as e:
            logger.error(f"Amass error: {str(e)}")
        
        return list(subdomains)
    
    async def _run_host_discovery(self, targets: List[str]) -> Dict[str, List]:
        """Run host discovery and port scanning"""
        all_hosts = []
        all_services = []
        
        for target in targets:
            try:
                # Use Nmap for detailed scanning
                nmap_results = await self.tools['nmap'].scan(target)
                if 'hosts' in nmap_results:
                    all_hosts.extend(nmap_results['hosts'])
                if 'services' in nmap_results:
                    all_services.extend(nmap_results['services'])
            except Exception as e:
                logger.error(f"Nmap scanning error for {target}: {str(e)}")
        
        return {'hosts': all_hosts, 'services': all_services}
    
    async def _run_tech_detection(self, web_targets: List[str]) -> List[Dict]:
        """Run web technology detection"""
        technologies = []
        
        for target in web_targets:
            try:
                whatweb_results = await self.tools['whatweb'].scan(target)
                if 'technologies' in whatweb_results:
                    technologies.extend(whatweb_results['technologies'])
            except Exception as e:
                logger.error(f"WhatWeb error for {target}: {str(e)}")
        
        return technologies
    
    async def _run_directory_enum(self, web_targets: List[str]) -> List[Dict]:
        """Run directory enumeration"""
        directories = []
        
        for target in web_targets:
            try:
                gobuster_results = await self.tools['gobuster'].scan(target)
                if 'directories' in gobuster_results:
                    directories.extend(gobuster_results['directories'])
            except Exception as e:
                logger.error(f"Gobuster error for {target}: {str(e)}")
        
        return directories
    
    def _extract_web_targets(self, services: List[Dict]) -> List[str]:
        """Extract web targets from discovered services"""
        web_targets = []
        
        for service in services:
            if service.get('service_name') in ['http', 'https', 'http-alt', 'http-proxy']:
                port = service.get('port', 80)
                host = service.get('host', '')
                
                if port == 443 or service.get('service_name') == 'https':
                    web_targets.append(f"https://{host}:{port}")
                else:
                    web_targets.append(f"http://{host}:{port}")
        
        return web_targets
    
    def _store_recon_results(self, mission_id: str, results: Dict[str, Any]):
        """Store reconnaissance results in database"""
        try:
            # Store overall results
            self.db_manager.store_recon_result(
                mission_id, results['target'], 'full_recon', 'comprehensive', results
            )
            
            # Store discovered hosts
            for host in results.get('hosts', []):
                host_id = self.db_manager.store_discovered_host(
                    mission_id, host.get('ip', ''), host.get('hostname'), 
                    host.get('os_info'), host.get('confidence', 0.0)
                )
                
                # Store services for this host
                for service in results.get('services', []):
                    if service.get('host') == host.get('ip'):
                        self.db_manager.store_discovered_service(
                            host_id, service.get('port', 0), service.get('protocol', 'tcp'),
                            service.get('service_name'), service.get('version'), service.get('banner')
                        )
            
        except Exception as e:
            logger.error(f"Error storing recon results: {str(e)}")
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reconnaissance summary"""
        return {
            'total_hosts': len(results.get('hosts', [])),
            'total_services': len(results.get('services', [])),
            'total_subdomains': len(results.get('subdomains', [])),
            'web_technologies': len(results.get('technologies', [])),
            'open_ports': len([s for s in results.get('services', []) if s.get('state') == 'open']),
            'critical_services': self._identify_critical_services(results.get('services', [])),
            'attack_surface_score': self._calculate_attack_surface_score(results)
        }
    
    def _identify_critical_services(self, services: List[Dict]) -> List[str]:
        """Identify critical services that may be high-value targets"""
        critical_services = []
        high_value_services = ['ssh', 'ftp', 'telnet', 'smtp', 'pop3', 'imap', 'http', 'https', 
                              'mysql', 'postgresql', 'mssql', 'oracle', 'ldap', 'rdp', 'vnc']
        
        for service in services:
            if service.get('service_name', '').lower() in high_value_services:
                critical_services.append(f"{service.get('host')}:{service.get('port')}/{service.get('service_name')}")
        
        return critical_services
    
    def _calculate_attack_surface_score(self, results: Dict[str, Any]) -> float:
        """Calculate attack surface score based on discovered assets"""
        score = 0.0
        
        # Base score from number of hosts
        score += len(results.get('hosts', [])) * 10
        
        # Score from open services
        score += len(results.get('services', [])) * 5
        
        # Score from web applications
        score += len(results.get('technologies', [])) * 15
        
        # Score from subdomains
        score += len(results.get('subdomains', [])) * 3
        
        # Normalize to 0-100 scale
        return min(score / 10, 100.0)

class BaseScanner:
    """Base class for reconnaissance tools"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__.lower())
    
    async def scan(self, target: str) -> Dict[str, Any]:
        """Override in subclasses"""
        raise NotImplementedError
    
    def _is_valid_target(self, target: str) -> bool:
        """Validate target format"""
        try:
            # Check if it's an IP address
            ipaddress.ip_address(target)
            return True
        except ValueError:
            pass
        
        # Check if it's a valid domain
        if re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,}$', target):
            return True
        
        # Check if it's a URL
        if re.match(r'^https?://', target):
            return True
        
        return False

class NmapScanner(BaseScanner):
    """Nmap integration for network scanning"""
    
    async def scan(self, target: str) -> Dict[str, Any]:
        """Run Nmap scan on target"""
        if not self._is_valid_target(target):
            raise ValueError(f"Invalid target: {target}")
        
        nmap_config = self.config.get_tool_config('nmap')
        binary_path = nmap_config.get('binary_path', 'nmap')
        
        # Build Nmap command
        cmd = [binary_path, '-sS', '-sV', '-O', '--script=default', '-oX', '-', target]
        
        try:
            # Run Nmap asynchronously
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Nmap failed: {stderr.decode()}")
            
            # Parse XML output
            return self._parse_nmap_xml(stdout.decode())
            
        except Exception as e:
            self.logger.error(f"Nmap scan failed: {str(e)}")
            raise
    
    def _parse_nmap_xml(self, xml_output: str) -> Dict[str, Any]:
        """Parse Nmap XML output"""
        try:
            root = ET.fromstring(xml_output)
            
            hosts = []
            services = []
            
            for host in root.findall('host'):
                # Get host information
                address = host.find('address').get('addr')
                status = host.find('status').get('state')
                
                host_info = {
                    'ip': address,
                    'status': status,
                    'hostname': None,
                    'os_info': None
                }
                
                # Get hostname
                hostnames = host.find('hostnames')
                if hostnames is not None:
                    hostname = hostnames.find('hostname')
                    if hostname is not None:
                        host_info['hostname'] = hostname.get('name')
                
                # Get OS information
                os_elem = host.find('os')
                if os_elem is not None:
                    osmatch = os_elem.find('osmatch')
                    if osmatch is not None:
                        host_info['os_info'] = osmatch.get('name')
                
                hosts.append(host_info)
                
                # Get port information
                ports = host.find('ports')
                if ports is not None:
                    for port in ports.findall('port'):
                        port_num = int(port.get('portid'))
                        protocol = port.get('protocol')
                        
                        state = port.find('state').get('state')
                        service = port.find('service')
                        
                        service_info = {
                            'host': address,
                            'port': port_num,
                            'protocol': protocol,
                            'state': state,
                            'service_name': service.get('name') if service is not None else 'unknown',
                            'version': service.get('version') if service is not None else None,
                            'banner': service.get('product') if service is not None else None
                        }
                        
                        services.append(service_info)
            
            return {
                'hosts': hosts,
                'services': services,
                'scan_time': datetime.now().isoformat()
            }
            
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse Nmap XML: {str(e)}")
            return {'error': 'XML parsing failed'}

class MasscanScanner(BaseScanner):
    """Masscan integration for fast port scanning"""
    
    async def scan(self, target: str, ports: str = "1-65535") -> Dict[str, Any]:
        """Run Masscan scan on target"""
        if not self._is_valid_target(target):
            raise ValueError(f"Invalid target: {target}")
        
        masscan_config = self.config.get_tool_config('masscan')
        binary_path = masscan_config.get('binary_path', 'masscan')
        rate = masscan_config.get('rate_limit', 1000)
        
        cmd = [binary_path, target, '-p', ports, '--rate', str(rate), '-oJ', '-']
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Masscan failed: {stderr.decode()}")
            
            return self._parse_masscan_output(stdout.decode())
            
        except Exception as e:
            self.logger.error(f"Masscan scan failed: {str(e)}")
            raise
    
    def _parse_masscan_output(self, output: str) -> Dict[str, Any]:
        """Parse Masscan JSON output"""
        try:
            services = []
            
            for line in output.strip().split('\n'):
                if line.strip():
                    data = json.loads(line)
                    
                    services.append({
                        'host': data.get('ip'),
                        'port': data.get('port'),
                        'protocol': data.get('proto', 'tcp'),
                        'state': 'open',
                        'service_name': 'unknown',
                        'timestamp': data.get('timestamp')
                    })
            
            return {
                'services': services,
                'scan_time': datetime.now().isoformat()
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Masscan output: {str(e)}")
            return {'error': 'JSON parsing failed'}

class GobusterScanner(BaseScanner):
    """Gobuster integration for directory enumeration"""
    
    async def scan(self, target: str, wordlist: Optional[str] = None) -> Dict[str, Any]:
        """Run Gobuster directory scan"""
        if not target.startswith(('http://', 'https://')):
            target = f"http://{target}"
        
        gobuster_config = self.config.get_tool_config('gobuster')
        binary_path = gobuster_config.get('binary_path', 'gobuster')
        wordlist = wordlist or gobuster_config.get('wordlist_path', '/usr/share/wordlists/dirb/common.txt')
        threads = gobuster_config.get('threads', 50)
        
        cmd = [binary_path, 'dir', '-u', target, '-w', wordlist, '-t', str(threads), '-q']
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0 and process.returncode != 1:  # Gobuster returns 1 on no findings
                raise Exception(f"Gobuster failed: {stderr.decode()}")
            
            return self._parse_gobuster_output(stdout.decode(), target)
            
        except Exception as e:
            self.logger.error(f"Gobuster scan failed: {str(e)}")
            raise
    
    def _parse_gobuster_output(self, output: str, target: str) -> Dict[str, Any]:
        """Parse Gobuster output"""
        directories = []
        
        for line in output.strip().split('\n'):
            if line.strip() and not line.startswith('='):
                # Parse format: /path (Status: 200) [Size: 1234]
                match = re.match(r'(.+?)\s+\(Status:\s+(\d+)\)\s+\[Size:\s+(\d+)\]', line)
                if match:
                    path, status, size = match.groups()
                    
                    directories.append({
                        'url': f"{target.rstrip('/')}{path}",
                        'path': path,
                        'status_code': int(status),
                        'size': int(size),
                        'target': target
                    })
        
        return {
            'directories': directories,
            'scan_time': datetime.now().isoformat()
        }

class Sublist3rScanner(BaseScanner):
    """Sublist3r integration for subdomain enumeration"""
    
    async def scan(self, domain: str) -> Dict[str, Any]:
        """Run Sublist3r subdomain enumeration"""
        # Note: This is a simplified implementation
        # In practice, you'd integrate with the actual Sublist3r tool
        
        try:
            # Simulate subdomain discovery using DNS
            subdomains = await self._dns_subdomain_discovery(domain)
            
            return {
                'subdomains': subdomains,
                'scan_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Sublist3r scan failed: {str(e)}")
            raise
    
    async def _dns_subdomain_discovery(self, domain: str) -> List[str]:
        """Basic DNS-based subdomain discovery"""
        common_subdomains = ['www', 'mail', 'ftp', 'admin', 'api', 'dev', 'test', 'staging']
        discovered = []
        
        for subdomain in common_subdomains:
            full_domain = f"{subdomain}.{domain}"
            try:
                socket.gethostbyname(full_domain)
                discovered.append(full_domain)
            except socket.gaierror:
                pass
        
        return discovered

class AmassScanner(BaseScanner):
    """Amass integration for subdomain enumeration"""
    
    async def scan(self, domain: str) -> Dict[str, Any]:
        """Run Amass subdomain enumeration"""
        cmd = ['amass', 'enum', '-d', domain, '-json']
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                # Amass might not be installed, fall back to basic discovery
                return await Sublist3rScanner(self.config).scan(domain)
            
            return self._parse_amass_output(stdout.decode())
            
        except Exception as e:
            self.logger.error(f"Amass scan failed: {str(e)}")
            # Fall back to basic discovery
            return await Sublist3rScanner(self.config).scan(domain)
    
    def _parse_amass_output(self, output: str) -> Dict[str, Any]:
        """Parse Amass JSON output"""
        subdomains = []
        
        for line in output.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    subdomain = data.get('name')
                    if subdomain:
                        subdomains.append(subdomain)
                except json.JSONDecodeError:
                    continue
        
        return {
            'subdomains': list(set(subdomains)),
            'scan_time': datetime.now().isoformat()
        }

class WhatWebScanner(BaseScanner):
    """WhatWeb integration for web technology detection"""
    
    async def scan(self, target: str) -> Dict[str, Any]:
        """Run WhatWeb technology detection"""
        if not target.startswith(('http://', 'https://')):
            target = f"http://{target}"
        
        cmd = ['whatweb', '--log-json=-', target]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                # WhatWeb might not be installed, use basic detection
                return await self._basic_tech_detection(target)
            
            return self._parse_whatweb_output(stdout.decode())
            
        except Exception as e:
            self.logger.error(f"WhatWeb scan failed: {str(e)}")
            return await self._basic_tech_detection(target)
    
    def _parse_whatweb_output(self, output: str) -> Dict[str, Any]:
        """Parse WhatWeb JSON output"""
        technologies = []
        
        for line in output.strip().split('\n'):
            if line.strip():
                try:
                    data = json.loads(line)
                    
                    tech_info = {
                        'target': data.get('target'),
                        'technologies': []
                    }
                    
                    for plugin in data.get('plugins', {}):
                        tech_info['technologies'].append({
                            'name': plugin,
                            'version': data['plugins'][plugin].get('version', []),
                            'confidence': 'high'
                        })
                    
                    technologies.append(tech_info)
                    
                except json.JSONDecodeError:
                    continue
        
        return {
            'technologies': technologies,
            'scan_time': datetime.now().isoformat()
        }
    
    async def _basic_tech_detection(self, target: str) -> Dict[str, Any]:
        """Basic technology detection using HTTP headers"""
        import aiohttp
        
        technologies = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(target) as response:
                    headers = response.headers
                    
                    tech_info = {
                        'target': target,
                        'technologies': []
                    }
                    
                    # Detect server
                    if 'Server' in headers:
                        tech_info['technologies'].append({
                            'name': 'Server',
                            'version': headers['Server'],
                            'confidence': 'high'
                        })
                    
                    # Detect X-Powered-By
                    if 'X-Powered-By' in headers:
                        tech_info['technologies'].append({
                            'name': 'Framework',
                            'version': headers['X-Powered-By'],
                            'confidence': 'high'
                        })
                    
                    technologies.append(tech_info)
                    
        except Exception as e:
            self.logger.error(f"Basic tech detection failed: {str(e)}")
        
        return {
            'technologies': technologies,
            'scan_time': datetime.now().isoformat()
        }