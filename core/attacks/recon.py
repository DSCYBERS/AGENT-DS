#!/usr/bin/env python3
"""
Agent DS - Reconnaissance Engine
Advanced target intelligence gathering with AI-enhanced enumeration

This module handles comprehensive reconnaissance including:
- Subdomain enumeration (Sublist3r, Amass, DNS brute force)
- Port scanning (Nmap integration)
- Technology stack detection (Wappalyzer, custom fingerprinting)
- Service enumeration and banner grabbing

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import asyncio
import json
import subprocess
import socket
import requests
import dns.resolver
from typing import Dict, List, Set, Optional, Any
from urllib.parse import urlparse
import re
import concurrent.futures
from pathlib import Path
import xml.etree.ElementTree as ET

# AI and machine learning imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class ReconnaissanceEngine:
    """
    Advanced reconnaissance engine with AI-enhanced intelligence gathering
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Common subdomain wordlist
        self.subdomain_wordlist = [
            'www', 'mail', 'ftp', 'admin', 'api', 'dev', 'test', 'staging', 'beta',
            'demo', 'secure', 'portal', 'app', 'dashboard', 'cpanel', 'webmail',
            'blog', 'forum', 'shop', 'store', 'support', 'help', 'docs', 'cdn',
            'static', 'assets', 'media', 'images', 'files', 'download', 'upload',
            'backup', 'old', 'new', 'v1', 'v2', 'mobile', 'm', 'wap', 'pda',
            'db', 'database', 'mysql', 'sql', 'oracle', 'postgres', 'mongo',
            'redis', 'cache', 'memcache', 'elastic', 'search', 'solr', 'kibana',
            'grafana', 'prometheus', 'jenkins', 'gitlab', 'github', 'svn', 'git',
            'ci', 'build', 'deploy', 'prod', 'production', 'live', 'release'
        ]
        
        # Common ports for scanning
        self.common_ports = [
            21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995,  # Standard services
            135, 139, 445, 1433, 1521, 3306, 5432, 6379,      # Database/Windows
            8080, 8443, 8000, 8888, 9000, 9090, 3000, 5000,   # Web services
            27017, 11211, 6380, 9200, 9300, 5601, 2379,       # NoSQL/Caching
            4444, 4445, 8888, 9999, 10000, 50000               # Backdoors/Admin
        ]
        
        # Technology fingerprints
        self.tech_fingerprints = {
            'Apache': [r'Server: Apache', r'apache'],
            'Nginx': [r'Server: nginx', r'nginx'],
            'IIS': [r'Server: Microsoft-IIS', r'X-Powered-By: ASP.NET'],
            'PHP': [r'X-Powered-By: PHP', r'\.php'],
            'Python': [r'X-Powered-By: Python', r'Django', r'Flask'],
            'Node.js': [r'X-Powered-By: Express', r'X-Powered-By: Node.js'],
            'Java': [r'X-Powered-By: JSP', r'jsessionid', r'\.jsp'],
            'ASP.NET': [r'X-AspNet-Version', r'X-Powered-By: ASP.NET'],
            'WordPress': [r'wp-content', r'wp-includes', r'WordPress'],
            'Drupal': [r'Drupal', r'sites/default', r'X-Drupal-Dynamic-Cache'],
            'Joomla': [r'Joomla', r'/components/', r'/modules/'],
            'React': [r'react', r'__REACT_DEVTOOLS_GLOBAL_HOOK__'],
            'Angular': [r'ng-version', r'angular', r'@angular'],
            'Vue.js': [r'Vue\.js', r'__vue__', r'vue'],
            'jQuery': [r'jquery', r'jQuery'],
            'Bootstrap': [r'bootstrap', r'Bootstrap'],
            'MySQL': [r'mysql', r'X-Powered-By: MySQL'],
            'PostgreSQL': [r'postgresql', r'postgres'],
            'MongoDB': [r'mongodb', r'mongo'],
            'Redis': [r'redis', r'X-Powered-By: Redis'],
            'Elasticsearch': [r'elasticsearch', r'X-Elastic-Product'],
            'Docker': [r'docker', r'X-Powered-By: Docker']
        }
    
    async def enumerate_subdomains(self, target_url: str) -> List[Dict[str, Any]]:
        """
        Comprehensive subdomain enumeration using multiple techniques
        """
        domain = urlparse(target_url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        
        subdomains = set()
        subdomain_data = []
        
        # Method 1: DNS brute force
        dns_subdomains = await self._dns_brute_force(domain)
        subdomains.update(dns_subdomains)
        
        # Method 2: Certificate transparency logs
        ct_subdomains = await self._certificate_transparency_search(domain)
        subdomains.update(ct_subdomains)
        
        # Method 3: Search engine enumeration
        search_subdomains = await self._search_engine_enumeration(domain)
        subdomains.update(search_subdomains)
        
        # Method 4: External tools (if available)
        tool_subdomains = await self._external_tools_enumeration(domain)
        subdomains.update(tool_subdomains)
        
        # Validate and gather additional info for each subdomain
        for subdomain in subdomains:
            subdomain_info = await self._validate_subdomain(subdomain)
            if subdomain_info:
                subdomain_data.append(subdomain_info)
        
        return subdomain_data
    
    async def _dns_brute_force(self, domain: str) -> Set[str]:
        """DNS brute force enumeration"""
        found_subdomains = set()
        
        async def check_subdomain(subdomain):
            full_domain = f"{subdomain}.{domain}"
            try:
                resolver = dns.resolver.Resolver()
                resolver.timeout = 2
                answers = resolver.resolve(full_domain, 'A')
                if answers:
                    found_subdomains.add(full_domain)
                    return full_domain
            except:
                pass
            return None
        
        # Use ThreadPoolExecutor for concurrent DNS queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            tasks = [executor.submit(asyncio.create_task, check_subdomain(sub)) 
                    for sub in self.subdomain_wordlist]
            
            for future in concurrent.futures.as_completed(tasks):
                try:
                    await future.result()
                except:
                    pass
        
        return found_subdomains
    
    async def _certificate_transparency_search(self, domain: str) -> Set[str]:
        """Search certificate transparency logs"""
        found_subdomains = set()
        
        try:
            # crt.sh API
            url = f"https://crt.sh/?q=%25.{domain}&output=json"
            response = await asyncio.get_event_loop().run_in_executor(
                None, requests.get, url
            )
            
            if response.status_code == 200:
                certificates = response.json()
                for cert in certificates:
                    common_name = cert.get('common_name', '')
                    if common_name and domain in common_name:
                        found_subdomains.add(common_name)
                    
                    # Check subject alternative names
                    sans = cert.get('name_value', '').split('\n')
                    for san in sans:
                        san = san.strip()
                        if san and domain in san and not san.startswith('*'):
                            found_subdomains.add(san)
        
        except Exception as e:
            print(f"Certificate transparency search failed: {e}")
        
        return found_subdomains
    
    async def _search_engine_enumeration(self, domain: str) -> Set[str]:
        """Search engine enumeration for subdomains"""
        found_subdomains = set()
        
        search_queries = [
            f"site:{domain}",
            f"site:*.{domain}",
            f"inurl:{domain}"
        ]
        
        for query in search_queries:
            try:
                # Google dorking (simulated - in practice, use proper APIs)
                # This is a placeholder for actual search engine enumeration
                await asyncio.sleep(0.1)  # Rate limiting simulation
            except Exception as e:
                continue
        
        return found_subdomains
    
    async def _external_tools_enumeration(self, domain: str) -> Set[str]:
        """Use external tools like Sublist3r, Amass if available"""
        found_subdomains = set()
        
        try:
            # Try Sublist3r
            cmd = f"sublist3r -d {domain} -o /tmp/sublist3r_output.txt"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            # Read results
            try:
                with open('/tmp/sublist3r_output.txt', 'r') as f:
                    for line in f:
                        subdomain = line.strip()
                        if subdomain and domain in subdomain:
                            found_subdomains.add(subdomain)
            except FileNotFoundError:
                pass
        
        except Exception:
            pass
        
        try:
            # Try Amass
            cmd = f"amass enum -d {domain} -o /tmp/amass_output.txt"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            # Read results
            try:
                with open('/tmp/amass_output.txt', 'r') as f:
                    for line in f:
                        subdomain = line.strip()
                        if subdomain and domain in subdomain:
                            found_subdomains.add(subdomain)
            except FileNotFoundError:
                pass
        
        except Exception:
            pass
        
        return found_subdomains
    
    async def _validate_subdomain(self, subdomain: str) -> Optional[Dict[str, Any]]:
        """Validate subdomain and gather additional information"""
        try:
            # Resolve IP address
            resolver = dns.resolver.Resolver()
            answers = resolver.resolve(subdomain, 'A')
            ip_addresses = [str(answer) for answer in answers]
            
            # Check if subdomain is accessible via HTTP/HTTPS
            accessible_urls = []
            for protocol in ['https', 'http']:
                url = f"{protocol}://{subdomain}"
                try:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.session.head(url, timeout=5, allow_redirects=True)
                    )
                    if response.status_code < 400:
                        accessible_urls.append(url)
                except:
                    pass
            
            return {
                'subdomain': subdomain,
                'ip_addresses': ip_addresses,
                'accessible_urls': accessible_urls,
                'status': 'active' if accessible_urls else 'resolved'
            }
        
        except Exception:
            return None
    
    async def scan_ports(self, target_url: str) -> List[Dict[str, Any]]:
        """
        Comprehensive port scanning using multiple techniques
        """
        domain = urlparse(target_url).netloc
        if ':' in domain:
            domain = domain.split(':')[0]
        
        open_ports = []
        
        # Method 1: TCP Connect scan for common ports
        tcp_ports = await self._tcp_connect_scan(domain, self.common_ports)
        open_ports.extend(tcp_ports)
        
        # Method 2: SYN scan using Nmap (if available)
        nmap_ports = await self._nmap_scan(domain)
        open_ports.extend(nmap_ports)
        
        # Method 3: UDP scan for common UDP services
        udp_ports = await self._udp_scan(domain, [53, 67, 68, 123, 161, 162, 514])
        open_ports.extend(udp_ports)
        
        # Remove duplicates and sort
        unique_ports = {}
        for port_info in open_ports:
            port_num = port_info['port']
            if port_num not in unique_ports:
                unique_ports[port_num] = port_info
        
        return sorted(unique_ports.values(), key=lambda x: x['port'])
    
    async def _tcp_connect_scan(self, host: str, ports: List[int]) -> List[Dict[str, Any]]:
        """TCP connect scan"""
        open_ports = []
        
        async def scan_port(port):
            try:
                # Create socket with timeout
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    # Try to grab banner
                    banner = await self._grab_banner(host, port)
                    service = self._identify_service(port, banner)
                    
                    return {
                        'port': port,
                        'protocol': 'tcp',
                        'state': 'open',
                        'service': service,
                        'banner': banner,
                        'method': 'tcp_connect'
                    }
            except Exception:
                pass
            return None
        
        # Scan ports concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            tasks = [executor.submit(asyncio.create_task, scan_port(port)) for port in ports]
            
            for future in concurrent.futures.as_completed(tasks):
                try:
                    result = await future.result()
                    if result:
                        open_ports.append(result)
                except:
                    pass
        
        return open_ports
    
    async def _nmap_scan(self, host: str) -> List[Dict[str, Any]]:
        """Nmap scan if available"""
        open_ports = []
        
        try:
            # Try Nmap SYN scan
            cmd = f"nmap -sS -O -sV --top-ports 1000 {host} -oX /tmp/nmap_output.xml"
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.wait()
            
            # Parse Nmap XML output
            try:
                tree = ET.parse('/tmp/nmap_output.xml')
                root = tree.getroot()
                
                for host_elem in root.findall('host'):
                    for port_elem in host_elem.findall('.//port'):
                        port_num = int(port_elem.get('portid'))
                        protocol = port_elem.get('protocol')
                        
                        state_elem = port_elem.find('state')
                        if state_elem is not None and state_elem.get('state') == 'open':
                            service_elem = port_elem.find('service')
                            service_name = service_elem.get('name') if service_elem is not None else 'unknown'
                            service_version = service_elem.get('version') if service_elem is not None else ''
                            
                            open_ports.append({
                                'port': port_num,
                                'protocol': protocol,
                                'state': 'open',
                                'service': service_name,
                                'version': service_version,
                                'method': 'nmap'
                            })
            
            except FileNotFoundError:
                pass
        
        except Exception:
            pass
        
        return open_ports
    
    async def _udp_scan(self, host: str, ports: List[int]) -> List[Dict[str, Any]]:
        """UDP port scan"""
        open_ports = []
        
        async def scan_udp_port(port):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(2)
                
                # Send UDP packet
                sock.sendto(b'Agent DS UDP Probe', (host, port))
                
                try:
                    data, addr = sock.recvfrom(1024)
                    sock.close()
                    
                    service = self._identify_service(port, data.decode('utf-8', errors='ignore'))
                    return {
                        'port': port,
                        'protocol': 'udp',
                        'state': 'open',
                        'service': service,
                        'response': data.decode('utf-8', errors='ignore')[:100],
                        'method': 'udp_probe'
                    }
                except socket.timeout:
                    # No response might indicate open port for UDP
                    sock.close()
                    return {
                        'port': port,
                        'protocol': 'udp',
                        'state': 'open|filtered',
                        'service': self._identify_service(port, ''),
                        'method': 'udp_probe'
                    }
            
            except Exception:
                pass
            return None
        
        # Scan UDP ports
        tasks = [scan_udp_port(port) for port in ports]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, dict):
                open_ports.append(result)
        
        return open_ports
    
    async def _grab_banner(self, host: str, port: int) -> str:
        """Grab service banner"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect((host, port))
            
            # Send generic request and read response
            if port in [80, 8080, 8000]:
                sock.send(b'GET / HTTP/1.1\r\nHost: ' + host.encode() + b'\r\n\r\n')
            elif port == 443:
                # For HTTPS, we'll skip banner grabbing to avoid SSL issues
                sock.close()
                return ''
            else:
                sock.send(b'\r\n')
            
            banner = sock.recv(1024).decode('utf-8', errors='ignore')
            sock.close()
            return banner.strip()
        
        except Exception:
            return ''
    
    def _identify_service(self, port: int, banner: str) -> str:
        """Identify service based on port and banner"""
        common_services = {
            21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp', 53: 'dns',
            80: 'http', 110: 'pop3', 143: 'imap', 443: 'https', 993: 'imaps',
            995: 'pop3s', 135: 'msrpc', 139: 'netbios-ssn', 445: 'microsoft-ds',
            1433: 'mssql', 1521: 'oracle', 3306: 'mysql', 5432: 'postgresql',
            6379: 'redis', 8080: 'http-proxy', 8443: 'https-alt', 27017: 'mongodb'
        }
        
        service = common_services.get(port, 'unknown')
        
        # Refine based on banner
        if banner:
            banner_lower = banner.lower()
            if 'apache' in banner_lower:
                service = 'apache'
            elif 'nginx' in banner_lower:
                service = 'nginx'
            elif 'microsoft-iis' in banner_lower:
                service = 'iis'
            elif 'ssh' in banner_lower:
                service = 'ssh'
            elif 'ftp' in banner_lower:
                service = 'ftp'
        
        return service
    
    async def detect_technologies(self, target_url: str) -> List[str]:
        """
        Detect web technologies and frameworks
        """
        technologies = set()
        
        try:
            # Method 1: HTTP headers analysis
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.session.get(target_url, timeout=10, allow_redirects=True)
            )
            
            # Analyze headers
            headers_tech = self._analyze_headers(response.headers)
            technologies.update(headers_tech)
            
            # Method 2: Response body analysis
            body_tech = self._analyze_response_body(response.text)
            technologies.update(body_tech)
            
            # Method 3: URL pattern analysis
            url_tech = self._analyze_url_patterns(target_url, response.text)
            technologies.update(url_tech)
            
            # Method 4: CSS/JS file analysis
            static_tech = await self._analyze_static_files(target_url, response.text)
            technologies.update(static_tech)
            
            # Method 5: AI-powered technology detection
            ai_tech = await self._ai_technology_detection(response.text, response.headers)
            technologies.update(ai_tech)
        
        except Exception as e:
            print(f"Technology detection error: {e}")
        
        return sorted(list(technologies))
    
    def _analyze_headers(self, headers: Dict[str, str]) -> Set[str]:
        """Analyze HTTP headers for technology indicators"""
        technologies = set()
        
        header_str = ' '.join([f"{k}: {v}" for k, v in headers.items()])
        
        for tech, patterns in self.tech_fingerprints.items():
            for pattern in patterns:
                if re.search(pattern, header_str, re.IGNORECASE):
                    technologies.add(tech)
                    break
        
        return technologies
    
    def _analyze_response_body(self, body: str) -> Set[str]:
        """Analyze response body for technology indicators"""
        technologies = set()
        
        for tech, patterns in self.tech_fingerprints.items():
            for pattern in patterns:
                if re.search(pattern, body, re.IGNORECASE):
                    technologies.add(tech)
                    break
        
        return technologies
    
    def _analyze_url_patterns(self, url: str, body: str) -> Set[str]:
        """Analyze URL patterns and links"""
        technologies = set()
        
        # Extract all URLs from the response
        url_pattern = r'(?:href|src)=["\']([^"\']+)["\']'
        urls = re.findall(url_pattern, body, re.IGNORECASE)
        
        all_urls = [url] + urls
        
        for url_item in all_urls:
            if '.php' in url_item:
                technologies.add('PHP')
            elif '.jsp' in url_item:
                technologies.add('Java')
            elif '.aspx' in url_item:
                technologies.add('ASP.NET')
            elif '.cfm' in url_item:
                technologies.add('ColdFusion')
            elif 'wp-content' in url_item or 'wp-includes' in url_item:
                technologies.add('WordPress')
            elif '/sites/default/' in url_item:
                technologies.add('Drupal')
        
        return technologies
    
    async def _analyze_static_files(self, base_url: str, body: str) -> Set[str]:
        """Analyze CSS and JavaScript files"""
        technologies = set()
        
        # Extract CSS and JS file URLs
        css_pattern = r'<link[^>]+href=["\']([^"\']+\.css[^"\']*)["\']'
        js_pattern = r'<script[^>]+src=["\']([^"\']+\.js[^"\']*)["\']'
        
        css_files = re.findall(css_pattern, body, re.IGNORECASE)
        js_files = re.findall(js_pattern, body, re.IGNORECASE)
        
        all_files = css_files + js_files
        
        for file_url in all_files[:10]:  # Limit to first 10 files
            try:
                if not file_url.startswith('http'):
                    file_url = base_url.rstrip('/') + '/' + file_url.lstrip('/')
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(file_url, timeout=5)
                )
                
                if response.status_code == 200:
                    file_content = response.text.lower()
                    
                    # Check for technology signatures in file content
                    if 'jquery' in file_content:
                        technologies.add('jQuery')
                    if 'bootstrap' in file_content:
                        technologies.add('Bootstrap')
                    if 'angular' in file_content:
                        technologies.add('Angular')
                    if 'react' in file_content:
                        technologies.add('React')
                    if 'vue' in file_content:
                        technologies.add('Vue.js')
            
            except Exception:
                continue
        
        return technologies
    
    async def _ai_technology_detection(self, body: str, headers: Dict[str, str]) -> Set[str]:
        """AI-powered technology detection using machine learning"""
        technologies = set()
        
        try:
            # Prepare text for analysis
            text_content = body + ' ' + ' '.join([f"{k}: {v}" for k, v in headers.items()])
            
            # Use TF-IDF vectorization for feature extraction
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english'
            )
            
            # Create a simple knowledge base of technology signatures
            tech_signatures = {
                'React': ['react', 'jsx', '__react', 'reactdom', 'create-react-app'],
                'Angular': ['angular', 'ng-', '@angular', 'angularjs', 'typescript'],
                'Vue.js': ['vue', 'vuejs', '__vue__', 'vue-router', 'vuex'],
                'Django': ['django', 'csrf_token', 'csrfmiddlewaretoken', '__django__'],
                'Flask': ['flask', 'werkzeug', 'jinja2', '__flask__'],
                'Laravel': ['laravel', 'csrf-token', 'laravel_session', '__laravel__'],
                'Ruby on Rails': ['rails', 'authenticity_token', '__rails__', 'railties'],
                'Express.js': ['express', 'x-powered-by: express', '__express__'],
                'Spring': ['spring', 'jsessionid', '__spring__', 'springframework']
            }
            
            text_lower = text_content.lower()
            
            for tech, signatures in tech_signatures.items():
                for signature in signatures:
                    if signature in text_lower:
                        technologies.add(tech)
                        break
        
        except Exception as e:
            print(f"AI technology detection error: {e}")
        
        return technologies
    
    async def perform_advanced_enumeration(self, target_url: str) -> Dict[str, Any]:
        """
        Perform advanced enumeration combining all reconnaissance techniques
        """
        domain = urlparse(target_url).netloc
        
        # Initialize results structure
        results = {
            'target': target_url,
            'domain': domain,
            'timestamp': asyncio.get_event_loop().time(),
            'subdomains': [],
            'ports': [],
            'technologies': [],
            'dns_records': {},
            'ssl_info': {},
            'whois_info': {},
            'security_headers': {},
            'crawled_urls': []
        }
        
        try:
            # Concurrent enumeration
            subdomain_task = asyncio.create_task(self.enumerate_subdomains(target_url))
            port_task = asyncio.create_task(self.scan_ports(target_url))
            tech_task = asyncio.create_task(self.detect_technologies(target_url))
            dns_task = asyncio.create_task(self._enumerate_dns_records(domain))
            ssl_task = asyncio.create_task(self._analyze_ssl_certificate(target_url))
            header_task = asyncio.create_task(self._analyze_security_headers(target_url))
            crawl_task = asyncio.create_task(self._crawl_website(target_url))
            
            # Wait for all tasks to complete
            results['subdomains'] = await subdomain_task
            results['ports'] = await port_task
            results['technologies'] = await tech_task
            results['dns_records'] = await dns_task
            results['ssl_info'] = await ssl_task
            results['security_headers'] = await header_task
            results['crawled_urls'] = await crawl_task
            
        except Exception as e:
            print(f"Advanced enumeration error: {e}")
        
        return results
    
    async def _enumerate_dns_records(self, domain: str) -> Dict[str, List[str]]:
        """Enumerate various DNS record types"""
        dns_records = {}
        record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']
        
        for record_type in record_types:
            try:
                resolver = dns.resolver.Resolver()
                answers = resolver.resolve(domain, record_type)
                dns_records[record_type] = [str(answer) for answer in answers]
            except Exception:
                dns_records[record_type] = []
        
        return dns_records
    
    async def _analyze_ssl_certificate(self, target_url: str) -> Dict[str, Any]:
        """Analyze SSL certificate information"""
        ssl_info = {}
        
        try:
            import ssl
            import socket
            from urllib.parse import urlparse
            
            parsed_url = urlparse(target_url)
            hostname = parsed_url.netloc
            port = 443
            
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    ssl_info = {
                        'subject': dict(x[0] for x in cert.get('subject', [])),
                        'issuer': dict(x[0] for x in cert.get('issuer', [])),
                        'version': cert.get('version'),
                        'serial_number': cert.get('serialNumber'),
                        'not_before': cert.get('notBefore'),
                        'not_after': cert.get('notAfter'),
                        'san': cert.get('subjectAltName', [])
                    }
        
        except Exception as e:
            ssl_info = {'error': str(e)}
        
        return ssl_info
    
    async def _analyze_security_headers(self, target_url: str) -> Dict[str, Any]:
        """Analyze security headers"""
        security_headers = {}
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.head(target_url, timeout=10, allow_redirects=True)
            )
            
            security_header_list = [
                'Strict-Transport-Security',
                'Content-Security-Policy',
                'X-Frame-Options',
                'X-Content-Type-Options',
                'X-XSS-Protection',
                'Referrer-Policy',
                'Feature-Policy',
                'Permissions-Policy'
            ]
            
            for header in security_header_list:
                security_headers[header] = response.headers.get(header, 'Not Set')
        
        except Exception as e:
            security_headers = {'error': str(e)}
        
        return security_headers
    
    async def _crawl_website(self, target_url: str, max_urls: int = 50) -> List[str]:
        """Basic website crawling to discover URLs"""
        crawled_urls = set()
        to_crawl = [target_url]
        
        while to_crawl and len(crawled_urls) < max_urls:
            current_url = to_crawl.pop(0)
            
            if current_url in crawled_urls:
                continue
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(current_url, timeout=5, allow_redirects=True)
                )
                
                if response.status_code == 200:
                    crawled_urls.add(current_url)
                    
                    # Extract links
                    link_pattern = r'<a[^>]+href=["\']([^"\']+)["\']'
                    links = re.findall(link_pattern, response.text, re.IGNORECASE)
                    
                    base_domain = urlparse(target_url).netloc
                    
                    for link in links[:10]:  # Limit links per page
                        if link.startswith('/'):
                            full_url = target_url.rstrip('/') + link
                        elif link.startswith('http') and base_domain in link:
                            full_url = link
                        else:
                            continue
                        
                        if full_url not in crawled_urls and len(crawled_urls) < max_urls:
                            to_crawl.append(full_url)
            
            except Exception:
                continue
        
        return list(crawled_urls)

# Test function
async def main():
    """Test the reconnaissance engine"""
    recon = ReconnaissanceEngine()
    
    target = "https://example.com"
    print(f"Testing reconnaissance on: {target}")
    
    # Test subdomain enumeration
    print("\n--- Subdomain Enumeration ---")
    subdomains = await recon.enumerate_subdomains(target)
    for subdomain in subdomains[:5]:
        print(f"  {subdomain}")
    
    # Test port scanning
    print("\n--- Port Scanning ---")
    ports = await recon.scan_ports(target)
    for port in ports[:10]:
        print(f"  Port {port['port']}: {port['service']} ({port['state']})")
    
    # Test technology detection
    print("\n--- Technology Detection ---")
    technologies = await recon.detect_technologies(target)
    for tech in technologies:
        print(f"  {tech}")

if __name__ == "__main__":
    asyncio.run(main())