"""
Agent DS - Autonomous AI Learning Module
Core learning engine for continuous improvement and adaptive attack strategies
"""

import asyncio
import json
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import numpy as np
from dataclasses import dataclass, asdict
import hashlib
import uuid

# ML/AI imports
try:
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import joblib
except ImportError:
    # Fallback for systems without ML libraries
    torch = None
    nn = None

from core.config.settings import Config
from core.database.manager import DatabaseManager
from core.utils.logger import get_logger, log_security_event

logger = get_logger('ai_learning')

@dataclass
class AttackResult:
    """Structured attack result for learning"""
    attack_id: str
    mission_id: str
    attack_type: str
    target_info: Dict[str, Any]
    payload: str
    success: bool
    response_time: float
    response_code: int
    response_content: str
    error_message: Optional[str]
    timestamp: str
    tool_used: str
    evasion_techniques: List[str]
    target_tech_stack: Dict[str, str]

@dataclass
class LearningInsight:
    """Insights extracted from attack results"""
    insight_id: str
    pattern_type: str
    success_factors: List[str]
    failure_indicators: List[str]
    target_characteristics: Dict[str, Any]
    recommended_payloads: List[str]
    confidence_score: float
    applicable_scenarios: List[str]

class AutonomousLearningEngine:
    """Main AI learning engine for Agent DS"""
    
    def __init__(self):
        self.config = Config()
        self.db_manager = DatabaseManager()
        self.logger = get_logger('autonomous_learning')
        
        # Learning configuration
        self.learning_config = self.config.get('ai_learning', {})
        self.learning_enabled = self.learning_config.get('enabled', True)
        self.sandbox_mode = self.learning_config.get('sandbox_enabled', True)
        
        # AI model components
        self.payload_generator = None
        self.success_predictor = None
        self.attack_sequencer = None
        
        # Learning data storage
        self.learning_db_path = Path(self.config.get('ai_learning.database_path', 'data/ai_learning.db'))
        self.model_storage_path = Path(self.config.get('ai_learning.model_path', 'ai_models/learning'))
        
        # External intelligence sources
        self.threat_intel_sources = {
            'cve': CVEIntelligenceSource(),
            'otx': OTXIntelligenceSource(),
            'exploitdb': ExploitDBIntelligenceSource()
        }
        
        # Experimentation engine
        self.sandbox_experimenter = SandboxExperimenter()
        
        # Initialize learning database
        asyncio.create_task(self._initialize_learning_database())
    
    async def _initialize_learning_database(self):
        """Initialize AI learning database schema"""
        try:
            self.learning_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.learning_db_path))
            cursor = conn.cursor()
            
            # Attack results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attack_results (
                    attack_id TEXT PRIMARY KEY,
                    mission_id TEXT,
                    attack_type TEXT,
                    target_info TEXT,
                    payload TEXT,
                    success BOOLEAN,
                    response_time REAL,
                    response_code INTEGER,
                    response_content TEXT,
                    error_message TEXT,
                    timestamp TEXT,
                    tool_used TEXT,
                    evasion_techniques TEXT,
                    target_tech_stack TEXT
                )
            ''')
            
            # Learning insights table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_insights (
                    insight_id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    success_factors TEXT,
                    failure_indicators TEXT,
                    target_characteristics TEXT,
                    recommended_payloads TEXT,
                    confidence_score REAL,
                    applicable_scenarios TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            # Payload effectiveness table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS payload_effectiveness (
                    payload_hash TEXT PRIMARY KEY,
                    payload TEXT,
                    attack_type TEXT,
                    success_count INTEGER,
                    failure_count INTEGER,
                    effectiveness_score REAL,
                    last_used TEXT,
                    target_types TEXT
                )
            ''')
            
            # External intelligence cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS external_intelligence (
                    intel_id TEXT PRIMARY KEY,
                    source TEXT,
                    intel_type TEXT,
                    data TEXT,
                    created_at TEXT,
                    expires_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("AI learning database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning database: {str(e)}")
            raise
    
    async def learn_from_mission(self, mission_id: str) -> Dict[str, Any]:
        """Learn from completed mission results"""
        if not self.learning_enabled:
            return {'status': 'learning_disabled'}
        
        learning_results = {
            'mission_id': mission_id,
            'learning_started': datetime.now().isoformat(),
            'insights_generated': [],
            'patterns_discovered': [],
            'model_updates': [],
            'improvement_metrics': {}
        }
        
        try:
            # Extract mission data for learning
            mission_data = await self._extract_mission_data(mission_id)
            
            # Learn from attack results
            attack_insights = await self._learn_from_attack_results(mission_data['attacks'])
            learning_results['insights_generated'].extend(attack_insights)
            
            # Learn from reconnaissance patterns
            recon_insights = await self._learn_from_reconnaissance(mission_data['recon'])
            learning_results['insights_generated'].extend(recon_insights)
            
            # Update AI models with new learning
            model_updates = await self._update_ai_models(learning_results['insights_generated'])
            learning_results['model_updates'] = model_updates
            
            # Calculate improvement metrics
            improvement_metrics = await self._calculate_improvement_metrics(mission_id)
            learning_results['improvement_metrics'] = improvement_metrics
            
            # Store learning results
            await self._store_learning_session(learning_results)
            
            self.logger.info(f"Learning completed for mission {mission_id}: {len(learning_results['insights_generated'])} insights generated")
            
            return learning_results
            
        except Exception as e:
            self.logger.error(f"Learning from mission {mission_id} failed: {str(e)}")
            learning_results['error'] = str(e)
            return learning_results
    
    async def _extract_mission_data(self, mission_id: str) -> Dict[str, Any]:
        """Extract relevant data from mission for learning"""
        mission_data = {
            'mission_id': mission_id,
            'attacks': [],
            'recon': {},
            'vulnerabilities': [],
            'target_info': {}
        }
        
        # Get attack results from database
        attacks = self.db_manager.get_attack_attempts(mission_id)
        mission_data['attacks'] = attacks or []
        
        # Get reconnaissance results
        recon_results = self.db_manager.get_recon_results(mission_id)
        mission_data['recon'] = recon_results or {}
        
        # Get vulnerability data
        vulnerabilities = self.db_manager.get_vulnerabilities(mission_id)
        mission_data['vulnerabilities'] = vulnerabilities or []
        
        return mission_data
    
    async def _learn_from_attack_results(self, attack_results: List[Dict]) -> List[LearningInsight]:
        """Extract learning insights from attack results"""
        insights = []
        
        for attack in attack_results:
            try:
                # Create structured attack result
                attack_result = AttackResult(
                    attack_id=attack.get('id', str(uuid.uuid4())),
                    mission_id=attack.get('mission_id'),
                    attack_type=attack.get('attack_type'),
                    target_info=attack.get('target_info', {}),
                    payload=attack.get('payload', ''),
                    success=attack.get('success', False),
                    response_time=attack.get('response_time', 0.0),
                    response_code=attack.get('response_code', 0),
                    response_content=attack.get('response_content', ''),
                    error_message=attack.get('error_message'),
                    timestamp=attack.get('timestamp', datetime.now().isoformat()),
                    tool_used=attack.get('tool_name', ''),
                    evasion_techniques=attack.get('evasion_techniques', []),
                    target_tech_stack=attack.get('target_tech_stack', {})
                )
                
                # Store attack result for learning
                await self._store_attack_result(attack_result)
                
                # Generate insights from this attack
                attack_insights = await self._generate_attack_insights(attack_result)
                insights.extend(attack_insights)
                
            except Exception as e:
                self.logger.error(f"Failed to learn from attack result: {str(e)}")
        
        return insights
    
    async def _generate_attack_insights(self, attack_result: AttackResult) -> List[LearningInsight]:
        """Generate learning insights from individual attack result"""
        insights = []
        
        try:
            # Success pattern analysis
            if attack_result.success:
                success_insight = await self._analyze_success_patterns(attack_result)
                if success_insight:
                    insights.append(success_insight)
            
            # Failure pattern analysis
            else:
                failure_insight = await self._analyze_failure_patterns(attack_result)
                if failure_insight:
                    insights.append(failure_insight)
            
            # Payload effectiveness analysis
            payload_insight = await self._analyze_payload_effectiveness(attack_result)
            if payload_insight:
                insights.append(payload_insight)
            
            # Target response pattern analysis
            response_insight = await self._analyze_response_patterns(attack_result)
            if response_insight:
                insights.append(response_insight)
                
        except Exception as e:
            self.logger.error(f"Failed to generate insights from attack result: {str(e)}")
        
        return insights
    
    async def _analyze_success_patterns(self, attack_result: AttackResult) -> Optional[LearningInsight]:
        """Analyze patterns that lead to successful attacks"""
        if not attack_result.success:
            return None
        
        success_factors = []
        
        # Analyze payload characteristics
        if len(attack_result.payload) > 0:
            success_factors.append(f"payload_length:{len(attack_result.payload)}")
            if 'union' in attack_result.payload.lower():
                success_factors.append("sql_union_technique")
            if '<script' in attack_result.payload.lower():
                success_factors.append("xss_script_injection")
        
        # Analyze timing factors
        if attack_result.response_time > 5.0:
            success_factors.append("slow_response_indicator")
        
        # Analyze technology stack factors
        for tech, version in attack_result.target_tech_stack.items():
            success_factors.append(f"target_{tech}:{version}")
        
        # Analyze evasion techniques used
        for technique in attack_result.evasion_techniques:
            success_factors.append(f"evasion_{technique}")
        
        insight = LearningInsight(
            insight_id=str(uuid.uuid4()),
            pattern_type="success_pattern",
            success_factors=success_factors,
            failure_indicators=[],
            target_characteristics=attack_result.target_tech_stack,
            recommended_payloads=[attack_result.payload],
            confidence_score=0.8,
            applicable_scenarios=[attack_result.attack_type]
        )
        
        return insight
    
    async def _analyze_failure_patterns(self, attack_result: AttackResult) -> Optional[LearningInsight]:
        """Analyze patterns that lead to failed attacks"""
        if attack_result.success:
            return None
        
        failure_indicators = []
        
        # Analyze response codes
        if attack_result.response_code == 403:
            failure_indicators.append("access_forbidden")
        elif attack_result.response_code == 404:
            failure_indicators.append("endpoint_not_found")
        elif attack_result.response_code == 500:
            failure_indicators.append("server_error_potential_vuln")
        
        # Analyze error messages
        if attack_result.error_message:
            if 'blocked' in attack_result.error_message.lower():
                failure_indicators.append("waf_blocked")
            elif 'timeout' in attack_result.error_message.lower():
                failure_indicators.append("connection_timeout")
        
        # Analyze response content for defensive measures
        if 'blocked' in attack_result.response_content.lower():
            failure_indicators.append("content_filter_detected")
        
        insight = LearningInsight(
            insight_id=str(uuid.uuid4()),
            pattern_type="failure_pattern",
            success_factors=[],
            failure_indicators=failure_indicators,
            target_characteristics=attack_result.target_tech_stack,
            recommended_payloads=[],
            confidence_score=0.7,
            applicable_scenarios=[attack_result.attack_type]
        )
        
        return insight
    
    async def _store_attack_result(self, attack_result: AttackResult):
        """Store attack result in learning database"""
        try:
            conn = sqlite3.connect(str(self.learning_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO attack_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                attack_result.attack_id,
                attack_result.mission_id,
                attack_result.attack_type,
                json.dumps(attack_result.target_info),
                attack_result.payload,
                attack_result.success,
                attack_result.response_time,
                attack_result.response_code,
                attack_result.response_content[:1000],  # Truncate large responses
                attack_result.error_message,
                attack_result.timestamp,
                attack_result.tool_used,
                json.dumps(attack_result.evasion_techniques),
                json.dumps(attack_result.target_tech_stack)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store attack result: {str(e)}")
    
    async def generate_adaptive_payload(self, attack_type: str, target_context: Dict, 
                                      failed_payloads: List[str] = None) -> str:
        """Generate adaptive payload based on learning and context"""
        try:
            # Get successful payload patterns for this attack type
            successful_patterns = await self._get_successful_payload_patterns(attack_type, target_context)
            
            # Get failure patterns to avoid
            failure_patterns = await self._get_failure_patterns(attack_type, target_context)
            
            # Generate new payload using AI
            if self.payload_generator:
                payload = await self._ai_generate_payload(
                    attack_type, target_context, successful_patterns, failure_patterns, failed_payloads
                )
            else:
                # Fallback to pattern-based generation
                payload = await self._pattern_generate_payload(
                    attack_type, target_context, successful_patterns
                )
            
            # Log payload generation
            self.logger.info(f"Generated adaptive payload for {attack_type}: {payload[:50]}...")
            
            return payload
            
        except Exception as e:
            self.logger.error(f"Failed to generate adaptive payload: {str(e)}")
            return self._get_default_payload(attack_type)
    
    async def _get_successful_payload_patterns(self, attack_type: str, 
                                             target_context: Dict) -> List[str]:
        """Get successful payload patterns from learning database"""
        try:
            conn = sqlite3.connect(str(self.learning_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT payload FROM attack_results 
                WHERE attack_type = ? AND success = 1
                ORDER BY timestamp DESC LIMIT 10
            ''', (attack_type,))
            
            results = cursor.fetchall()
            conn.close()
            
            return [result[0] for result in results]
            
        except Exception as e:
            self.logger.error(f"Failed to get successful patterns: {str(e)}")
            return []
    
    async def _ai_generate_payload(self, attack_type: str, target_context: Dict,
                                 successful_patterns: List[str], failure_patterns: List[str],
                                 failed_payloads: List[str]) -> str:
        """Generate payload using AI models"""
        if not torch:
            return await self._pattern_generate_payload(attack_type, target_context, successful_patterns)
        
        try:
            # Create input context for AI model
            context = f"Attack type: {attack_type}\n"
            context += f"Target: {json.dumps(target_context)}\n"
            context += f"Successful patterns: {'; '.join(successful_patterns[:3])}\n"
            
            if failed_payloads:
                context += f"Failed payloads: {'; '.join(failed_payloads[:2])}\n"
            
            context += "Generate new payload:"
            
            # Use transformer model for payload generation
            if hasattr(self, 'transformer_model') and self.transformer_model:
                generated = self.transformer_model(context, max_length=200, num_return_sequences=1)
                payload = generated[0]['generated_text'].split("Generate new payload:")[-1].strip()
            else:
                # Fallback to pattern-based generation
                payload = await self._pattern_generate_payload(attack_type, target_context, successful_patterns)
            
            return payload
            
        except Exception as e:
            self.logger.error(f"AI payload generation failed: {str(e)}")
            return await self._pattern_generate_payload(attack_type, target_context, successful_patterns)
    
    async def _pattern_generate_payload(self, attack_type: str, target_context: Dict,
                                      successful_patterns: List[str]) -> str:
        """Generate payload based on successful patterns"""
        if not successful_patterns:
            return self._get_default_payload(attack_type)
        
        # Select best pattern and modify it
        base_pattern = successful_patterns[0]
        
        # Apply context-specific modifications
        if attack_type == 'sql_injection':
            # Modify SQL injection payload based on target database
            db_type = target_context.get('database', 'mysql')
            if db_type.lower() == 'postgresql':
                base_pattern = base_pattern.replace('@@version', 'version()')
            elif db_type.lower() == 'mssql':
                base_pattern = base_pattern.replace('version()', '@@version')
        
        elif attack_type == 'xss':
            # Modify XSS payload based on context filtering
            if 'script' in str(target_context.get('filters', [])):
                base_pattern = base_pattern.replace('<script>', '<img src=x onerror=')
        
        return base_pattern
    
    def _get_default_payload(self, attack_type: str) -> str:
        """Get default payload for attack type"""
        default_payloads = {
            'sql_injection': "' UNION SELECT 1,2,3,4,5--",
            'xss': '<script>alert("XSS")</script>',
            'rce': '; cat /etc/passwd',
            'lfi': '../../../etc/passwd',
            'ssrf': 'http://169.254.169.254/latest/meta-data/'
        }
        
        return default_payloads.get(attack_type, "test_payload")
    
    async def predict_attack_success(self, attack_type: str, payload: str, 
                                   target_context: Dict) -> float:
        """Predict probability of attack success based on learning"""
        try:
            # Extract features for prediction
            features = await self._extract_prediction_features(attack_type, payload, target_context)
            
            # Use ML model for prediction if available
            if self.success_predictor:
                probability = self.success_predictor.predict_proba([features])[0][1]
            else:
                # Fallback to heuristic-based prediction
                probability = await self._heuristic_predict_success(attack_type, payload, target_context)
            
            return min(max(probability, 0.0), 1.0)  # Ensure 0-1 range
            
        except Exception as e:
            self.logger.error(f"Failed to predict attack success: {str(e)}")
            return 0.5  # Default probability
    
    async def _extract_prediction_features(self, attack_type: str, payload: str,
                                         target_context: Dict) -> List[float]:
        """Extract features for success prediction"""
        features = []
        
        # Payload features
        features.append(len(payload))
        features.append(payload.count("'"))
        features.append(payload.count('"'))
        features.append(payload.count('<'))
        features.append(payload.count('script'))
        features.append(payload.count('union'))
        features.append(payload.count('select'))
        
        # Target context features
        features.append(1 if target_context.get('cms') == 'wordpress' else 0)
        features.append(1 if target_context.get('database') == 'mysql' else 0)
        features.append(1 if target_context.get('language') == 'php' else 0)
        features.append(1 if 'waf' in target_context.get('security', []) else 0)
        
        # Historical success rate for this attack type
        historical_success = await self._get_historical_success_rate(attack_type)
        features.append(historical_success)
        
        return features
    
    async def _get_historical_success_rate(self, attack_type: str) -> float:
        """Get historical success rate for attack type"""
        try:
            conn = sqlite3.connect(str(self.learning_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(CAST(success AS FLOAT)) FROM attack_results 
                WHERE attack_type = ?
            ''', (attack_type,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result[0] is not None else 0.5
            
        except Exception as e:
            self.logger.error(f"Failed to get historical success rate: {str(e)}")
            return 0.5
    
    async def update_external_intelligence(self) -> Dict[str, Any]:
        """Update AI knowledge with latest external threat intelligence"""
        update_results = {
            'update_started': datetime.now().isoformat(),
            'sources_updated': [],
            'new_intelligence_count': 0,
            'errors': []
        }
        
        for source_name, source in self.threat_intel_sources.items():
            try:
                intelligence = await source.fetch_latest_intelligence()
                
                if intelligence:
                    await self._integrate_external_intelligence(source_name, intelligence)
                    update_results['sources_updated'].append(source_name)
                    update_results['new_intelligence_count'] += len(intelligence)
                
            except Exception as e:
                error_msg = f"Failed to update {source_name}: {str(e)}"
                self.logger.error(error_msg)
                update_results['errors'].append(error_msg)
        
        update_results['update_completed'] = datetime.now().isoformat()
        return update_results
    
    async def experiment_with_novel_payloads(self, attack_type: str, 
                                           target_context: Dict) -> Dict[str, Any]:
        """Safely experiment with novel payloads in sandbox"""
        if not self.sandbox_mode:
            return {'status': 'sandbox_disabled'}
        
        experiment_results = {
            'experiment_started': datetime.now().isoformat(),
            'attack_type': attack_type,
            'novel_payloads_tested': [],
            'successful_discoveries': [],
            'experiment_metrics': {}
        }
        
        try:
            # Generate novel payloads for experimentation
            novel_payloads = await self._generate_novel_payloads(attack_type, target_context)
            
            for payload in novel_payloads:
                # Test payload in sandbox
                test_result = await self.sandbox_experimenter.test_payload(
                    payload, attack_type, target_context
                )
                
                experiment_results['novel_payloads_tested'].append({
                    'payload': payload,
                    'result': test_result
                })
                
                # Store successful discoveries
                if test_result.get('success', False):
                    experiment_results['successful_discoveries'].append(payload)
                    await self._store_novel_discovery(payload, attack_type, test_result)
            
            # Calculate experiment metrics
            success_rate = len(experiment_results['successful_discoveries']) / len(novel_payloads)
            experiment_results['experiment_metrics'] = {
                'total_tested': len(novel_payloads),
                'successful_discoveries': len(experiment_results['successful_discoveries']),
                'success_rate': success_rate
            }
            
            self.logger.info(f"Payload experimentation completed: {success_rate:.2%} success rate")
            
        except Exception as e:
            experiment_results['error'] = str(e)
            self.logger.error(f"Payload experimentation failed: {str(e)}")
        
        return experiment_results

class CVEIntelligenceSource:
    """CVE.org intelligence integration"""
    
    async def fetch_latest_intelligence(self) -> List[Dict]:
        """Fetch latest CVE intelligence"""
        # Implementation would integrate with CVE.org API
        # For now, return mock data
        return [
            {
                'cve_id': 'CVE-2023-12345',
                'description': 'SQL injection vulnerability in popular CMS',
                'attack_vector': 'sql_injection',
                'affected_products': ['wordpress', 'drupal'],
                'severity': 'high'
            }
        ]

class OTXIntelligenceSource:
    """AlienVault OTX intelligence integration"""
    
    async def fetch_latest_intelligence(self) -> List[Dict]:
        """Fetch latest OTX threat intelligence"""
        # Implementation would integrate with OTX API
        return []

class ExploitDBIntelligenceSource:
    """ExploitDB intelligence integration"""
    
    async def fetch_latest_intelligence(self) -> List[Dict]:
        """Fetch latest ExploitDB intelligence"""
        # Implementation would integrate with ExploitDB API
        return []

class SandboxExperimenter:
    """Safe payload experimentation in sandbox environment"""
    
    async def test_payload(self, payload: str, attack_type: str, 
                          target_context: Dict) -> Dict[str, Any]:
        """Test payload safely in sandbox"""
        # Mock sandbox testing - would implement actual sandbox
        return {
            'success': False,
            'response_time': 1.0,
            'response_code': 200,
            'safety_score': 0.9
        }