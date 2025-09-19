"""
Agent DS - AI Learning Database Integration
Database schema and integration for AI learning functionality
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.database.manager import DatabaseManager
from core.utils.logger import get_logger

logger = get_logger('ai_learning_db')

class AILearningDatabaseManager:
    """Database manager for AI learning functionality"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.base_db_manager = DatabaseManager()
        self.ai_db_path = Path(db_path) if db_path else Path("data/ai_learning.db")
        self.logger = get_logger('ai_learning_db')
        
        # Initialize AI learning database
        self._initialize_ai_database()
    
    def _initialize_ai_database(self):
        """Initialize AI learning database with extended schema"""
        try:
            self.ai_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            # Extended attack results for AI learning
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_attack_results (
                    id TEXT PRIMARY KEY,
                    mission_id TEXT,
                    attack_type TEXT,
                    payload TEXT,
                    target_info TEXT,
                    success BOOLEAN,
                    response_time REAL,
                    response_code INTEGER,
                    response_content TEXT,
                    response_headers TEXT,
                    error_message TEXT,
                    tool_used TEXT,
                    evasion_techniques TEXT,
                    target_tech_stack TEXT,
                    detection_probability REAL,
                    adaptation_used BOOLEAN,
                    ai_confidence REAL,
                    timestamp TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # AI learning insights
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id TEXT PRIMARY KEY,
                    insight_type TEXT,
                    pattern_data TEXT,
                    success_factors TEXT,
                    failure_indicators TEXT,
                    target_characteristics TEXT,
                    recommended_payloads TEXT,
                    confidence_score REAL,
                    applicable_scenarios TEXT,
                    validation_count INTEGER DEFAULT 0,
                    effectiveness_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Payload effectiveness tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS payload_effectiveness (
                    id TEXT PRIMARY KEY,
                    payload_hash TEXT UNIQUE,
                    payload TEXT,
                    attack_type TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    total_attempts INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    avg_response_time REAL DEFAULT 0.0,
                    last_successful_use TEXT,
                    last_failed_use TEXT,
                    target_types TEXT,
                    evasion_score REAL DEFAULT 0.0,
                    innovation_score REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # AI model training data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_training_data (
                    id TEXT PRIMARY KEY,
                    data_type TEXT,
                    input_features TEXT,
                    target_labels TEXT,
                    metadata TEXT,
                    data_source TEXT,
                    quality_score REAL,
                    used_for_training BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # External intelligence cache
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS external_intelligence (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    intel_type TEXT,
                    cve_id TEXT,
                    title TEXT,
                    description TEXT,
                    severity TEXT,
                    affected_products TEXT,
                    attack_vector TEXT,
                    exploit_data TEXT,
                    published_date TEXT,
                    last_modified TEXT,
                    confidence_score REAL,
                    relevance_score REAL,
                    integrated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            ''')
            
            # AI adaptation decisions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS adaptation_decisions (
                    id TEXT PRIMARY KEY,
                    mission_id TEXT,
                    attack_id TEXT,
                    decision_type TEXT,
                    trigger_reason TEXT,
                    original_payload TEXT,
                    adapted_payload TEXT,
                    reasoning TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    improvement_factor REAL,
                    execution_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Learning session tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id TEXT PRIMARY KEY,
                    session_type TEXT,
                    mission_ids TEXT,
                    insights_generated INTEGER DEFAULT 0,
                    patterns_discovered INTEGER DEFAULT 0,
                    models_updated TEXT,
                    performance_improvement REAL,
                    duration_seconds REAL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # AI performance metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_performance_metrics (
                    id TEXT PRIMARY KEY,
                    metric_type TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    baseline_value REAL,
                    improvement_percentage REAL,
                    measurement_period TEXT,
                    context_data TEXT,
                    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Experimental payload results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experimental_payloads (
                    id TEXT PRIMARY KEY,
                    payload TEXT,
                    payload_hash TEXT UNIQUE,
                    attack_type TEXT,
                    generation_method TEXT,
                    ai_model_used TEXT,
                    novelty_score REAL,
                    safety_score REAL,
                    test_results TEXT,
                    sandbox_tested BOOLEAN DEFAULT FALSE,
                    real_world_tested BOOLEAN DEFAULT FALSE,
                    success_in_sandbox BOOLEAN DEFAULT FALSE,
                    success_in_real_world BOOLEAN DEFAULT FALSE,
                    discovery_method TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_attack_results_mission ON ai_attack_results(mission_id)",
                "CREATE INDEX IF NOT EXISTS idx_attack_results_type ON ai_attack_results(attack_type)",
                "CREATE INDEX IF NOT EXISTS idx_attack_results_success ON ai_attack_results(success)",
                "CREATE INDEX IF NOT EXISTS idx_attack_results_timestamp ON ai_attack_results(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_insights_type ON learning_insights(insight_type)",
                "CREATE INDEX IF NOT EXISTS idx_insights_confidence ON learning_insights(confidence_score)",
                "CREATE INDEX IF NOT EXISTS idx_payload_effectiveness_type ON payload_effectiveness(attack_type)",
                "CREATE INDEX IF NOT EXISTS idx_payload_effectiveness_rate ON payload_effectiveness(success_rate)",
                "CREATE INDEX IF NOT EXISTS idx_intel_source ON external_intelligence(source)",
                "CREATE INDEX IF NOT EXISTS idx_intel_cve ON external_intelligence(cve_id)",
                "CREATE INDEX IF NOT EXISTS idx_adaptations_mission ON adaptation_decisions(mission_id)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_type ON ai_performance_metrics(metric_type)"
            ]
            
            for index in indexes:
                cursor.execute(index)
            
            conn.commit()
            conn.close()
            
            self.logger.info("AI learning database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI learning database: {str(e)}")
            raise
    
    def store_attack_result_for_learning(self, attack_result: Dict[str, Any]) -> str:
        """Store attack result with AI learning enhancements"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            attack_id = attack_result.get('id', f"attack_{datetime.now().timestamp()}")
            
            cursor.execute('''
                INSERT OR REPLACE INTO ai_attack_results VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                attack_id,
                attack_result.get('mission_id'),
                attack_result.get('attack_type'),
                attack_result.get('payload'),
                json.dumps(attack_result.get('target_info', {})),
                attack_result.get('success', False),
                attack_result.get('response_time', 0.0),
                attack_result.get('response_code', 0),
                attack_result.get('response_content', '')[:2000],  # Truncate
                json.dumps(attack_result.get('response_headers', {})),
                attack_result.get('error_message'),
                attack_result.get('tool_used'),
                json.dumps(attack_result.get('evasion_techniques', [])),
                json.dumps(attack_result.get('target_tech_stack', {})),
                attack_result.get('detection_probability', 0.0),
                attack_result.get('adaptation_used', False),
                attack_result.get('ai_confidence', 0.0),
                attack_result.get('timestamp', datetime.now().isoformat()),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update payload effectiveness
            self._update_payload_effectiveness(attack_result)
            
            return attack_id
            
        except Exception as e:
            self.logger.error(f"Failed to store attack result for learning: {str(e)}")
            raise
    
    def _update_payload_effectiveness(self, attack_result: Dict[str, Any]):
        """Update payload effectiveness metrics"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            payload = attack_result.get('payload', '')
            payload_hash = hash(payload)
            attack_type = attack_result.get('attack_type', '')
            success = attack_result.get('success', False)
            response_time = attack_result.get('response_time', 0.0)
            
            # Check if payload exists
            cursor.execute('SELECT * FROM payload_effectiveness WHERE payload_hash = ?', (str(payload_hash),))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing payload effectiveness
                if success:
                    new_success_count = existing[4] + 1
                    new_total = existing[6] + 1
                    last_successful = datetime.now().isoformat()
                    last_failed = existing[12]
                else:
                    new_success_count = existing[4]
                    new_failure_count = existing[5] + 1
                    new_total = existing[6] + 1
                    last_successful = existing[11]
                    last_failed = datetime.now().isoformat()
                
                new_success_rate = new_success_count / new_total if new_total > 0 else 0.0
                
                cursor.execute('''
                    UPDATE payload_effectiveness 
                    SET success_count = ?, failure_count = ?, total_attempts = ?, 
                        success_rate = ?, avg_response_time = ?, last_successful_use = ?, 
                        last_failed_use = ?, updated_at = ?
                    WHERE payload_hash = ?
                ''', (
                    new_success_count, 
                    new_failure_count if not success else existing[5],
                    new_total, new_success_rate,
                    (existing[7] + response_time) / 2,  # Running average
                    last_successful, last_failed,
                    datetime.now().isoformat(),
                    str(payload_hash)
                ))
            else:
                # Create new payload effectiveness record
                payload_id = f"payload_{datetime.now().timestamp()}"
                cursor.execute('''
                    INSERT INTO payload_effectiveness VALUES 
                    (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    payload_id, str(payload_hash), payload, attack_type,
                    1 if success else 0,  # success_count
                    0 if success else 1,  # failure_count
                    1,  # total_attempts
                    1.0 if success else 0.0,  # success_rate
                    response_time,  # avg_response_time
                    datetime.now().isoformat() if success else None,  # last_successful_use
                    datetime.now().isoformat() if not success else None,  # last_failed_use
                    json.dumps([attack_result.get('target_info', {}).get('type', 'unknown')]),  # target_types
                    0.5,  # evasion_score (default)
                    0.0,  # innovation_score (calculated later)
                    datetime.now().isoformat(),  # created_at
                    datetime.now().isoformat()   # updated_at
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update payload effectiveness: {str(e)}")
    
    def store_learning_insight(self, insight: Dict[str, Any]) -> str:
        """Store AI learning insight"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            insight_id = insight.get('id', f"insight_{datetime.now().timestamp()}")
            
            cursor.execute('''
                INSERT OR REPLACE INTO learning_insights VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                insight_id,
                insight.get('insight_type'),
                json.dumps(insight.get('pattern_data', {})),
                json.dumps(insight.get('success_factors', [])),
                json.dumps(insight.get('failure_indicators', [])),
                json.dumps(insight.get('target_characteristics', {})),
                json.dumps(insight.get('recommended_payloads', [])),
                insight.get('confidence_score', 0.0),
                json.dumps(insight.get('applicable_scenarios', [])),
                insight.get('validation_count', 0),
                insight.get('effectiveness_score', 0.0),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return insight_id
            
        except Exception as e:
            self.logger.error(f"Failed to store learning insight: {str(e)}")
            raise
    
    def get_successful_payload_patterns(self, attack_type: str, limit: int = 10) -> List[Dict]:
        """Get successful payload patterns for learning"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT payload, success_rate, total_attempts, target_types, evasion_score
                FROM payload_effectiveness 
                WHERE attack_type = ? AND success_rate > 0.5
                ORDER BY success_rate DESC, total_attempts DESC
                LIMIT ?
            ''', (attack_type, limit))
            
            results = cursor.fetchall()
            conn.close()
            
            patterns = []
            for result in results:
                patterns.append({
                    'payload': result[0],
                    'success_rate': result[1],
                    'total_attempts': result[2],
                    'target_types': json.loads(result[3]) if result[3] else [],
                    'evasion_score': result[4]
                })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Failed to get successful payload patterns: {str(e)}")
            return []
    
    def get_learning_insights(self, insight_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Get AI learning insights"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            if insight_type:
                cursor.execute('''
                    SELECT * FROM learning_insights 
                    WHERE insight_type = ?
                    ORDER BY confidence_score DESC, updated_at DESC
                    LIMIT ?
                ''', (insight_type, limit))
            else:
                cursor.execute('''
                    SELECT * FROM learning_insights 
                    ORDER BY confidence_score DESC, updated_at DESC
                    LIMIT ?
                ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            insights = []
            for result in results:
                insights.append({
                    'id': result[0],
                    'insight_type': result[1],
                    'pattern_data': json.loads(result[2]) if result[2] else {},
                    'success_factors': json.loads(result[3]) if result[3] else [],
                    'failure_indicators': json.loads(result[4]) if result[4] else [],
                    'target_characteristics': json.loads(result[5]) if result[5] else {},
                    'recommended_payloads': json.loads(result[6]) if result[6] else [],
                    'confidence_score': result[7],
                    'applicable_scenarios': json.loads(result[8]) if result[8] else [],
                    'validation_count': result[9],
                    'effectiveness_score': result[10],
                    'created_at': result[11],
                    'updated_at': result[12]
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get learning insights: {str(e)}")
            return []
    
    def store_external_intelligence(self, intel_data: Dict[str, Any]) -> str:
        """Store external threat intelligence"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            intel_id = intel_data.get('id', f"intel_{datetime.now().timestamp()}")
            
            # Calculate expiration date (default 30 days)
            expires_at = datetime.now() + timedelta(days=30)
            
            cursor.execute('''
                INSERT OR REPLACE INTO external_intelligence VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                intel_id,
                intel_data.get('source'),
                intel_data.get('intel_type'),
                intel_data.get('cve_id'),
                intel_data.get('title'),
                intel_data.get('description'),
                intel_data.get('severity'),
                json.dumps(intel_data.get('affected_products', [])),
                intel_data.get('attack_vector'),
                json.dumps(intel_data.get('exploit_data', {})),
                intel_data.get('published_date'),
                intel_data.get('last_modified'),
                intel_data.get('confidence_score', 0.5),
                intel_data.get('relevance_score', 0.5),
                False,  # integrated
                datetime.now().isoformat(),
                expires_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            return intel_id
            
        except Exception as e:
            self.logger.error(f"Failed to store external intelligence: {str(e)}")
            raise
    
    def get_ai_performance_metrics(self, metric_type: Optional[str] = None) -> List[Dict]:
        """Get AI performance metrics"""
        try:
            conn = sqlite3.connect(str(self.ai_db_path))
            cursor = conn.cursor()
            
            if metric_type:
                cursor.execute('''
                    SELECT * FROM ai_performance_metrics 
                    WHERE metric_type = ?
                    ORDER BY measured_at DESC
                    LIMIT 50
                ''', (metric_type,))
            else:
                cursor.execute('''
                    SELECT * FROM ai_performance_metrics 
                    ORDER BY measured_at DESC
                    LIMIT 100
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            metrics = []
            for result in results:
                metrics.append({
                    'id': result[0],
                    'metric_type': result[1],
                    'metric_name': result[2],
                    'metric_value': result[3],
                    'baseline_value': result[4],
                    'improvement_percentage': result[5],
                    'measurement_period': result[6],
                    'context_data': json.loads(result[7]) if result[7] else {},
                    'measured_at': result[8]
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to get AI performance metrics: {str(e)}")
            return []