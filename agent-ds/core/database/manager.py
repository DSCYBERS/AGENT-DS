"""
Agent DS Database Manager
Handles SQLite database operations, migrations, and data persistence
"""

import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import threading

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for Agent DS"""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager"""
        self.db_path = db_path or self._get_default_db_path()
        self.db_dir = Path(self.db_path).parent
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe database connections
        self._local = threading.local()
        
        self.initialize()
    
    def _get_default_db_path(self) -> str:
        """Get default database path"""
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        return str(data_dir / "agent_ds.db")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            
        return self._local.connection
    
    def initialize(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Missions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS missions (
                    id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    scope TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ACTIVE',
                    classification TEXT DEFAULT 'CONFIDENTIAL',
                    authorized_by TEXT,
                    authorization_document TEXT,
                    metadata TEXT  -- JSON
                )
            ''')
            
            # Reconnaissance results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recon_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    scan_type TEXT NOT NULL,
                    status TEXT DEFAULT 'RUNNING',
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    results TEXT,  -- JSON
                    raw_output TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            ''')
            
            # Discovered hosts and services
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS discovered_hosts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    hostname TEXT,
                    os_info TEXT,
                    status TEXT DEFAULT 'UP',
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.0,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (mission_id) REFERENCES missions (id),
                    UNIQUE(mission_id, ip_address)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS discovered_services (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    host_id INTEGER NOT NULL,
                    port INTEGER NOT NULL,
                    protocol TEXT NOT NULL,
                    service_name TEXT,
                    version TEXT,
                    banner TEXT,
                    state TEXT DEFAULT 'OPEN',
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (host_id) REFERENCES discovered_hosts (id),
                    UNIQUE(host_id, port, protocol)
                )
            ''')
            
            # Vulnerabilities
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vulnerabilities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    vuln_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    cvss_score REAL,
                    cve_id TEXT,
                    exploit_available BOOLEAN DEFAULT 0,
                    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified BOOLEAN DEFAULT 0,
                    false_positive BOOLEAN DEFAULT 0,
                    remediation TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            ''')
            
            # Attack attempts and results
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attack_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    vulnerability_id INTEGER,
                    target TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    tool_name TEXT,
                    payload TEXT,
                    success BOOLEAN DEFAULT 0,
                    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    result_data TEXT,  -- JSON
                    error_message TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (mission_id) REFERENCES missions (id),
                    FOREIGN KEY (vulnerability_id) REFERENCES vulnerabilities (id)
                )
            ''')
            
            # AI learning data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    input_data TEXT NOT NULL,  -- JSON
                    output_data TEXT NOT NULL,  -- JSON
                    model_type TEXT NOT NULL,
                    accuracy_score REAL,
                    feedback_score INTEGER,  -- 1-5 rating
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            ''')
            
            # Reports
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL,
                    report_type TEXT NOT NULL,
                    format TEXT NOT NULL,
                    classification TEXT NOT NULL,
                    file_path TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    generated_by TEXT,
                    metadata TEXT,  -- JSON
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            ''')
            
            # Audit logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT,
                    username TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT,
                    resource_id TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_recon_mission ON recon_results(mission_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_hosts_mission ON discovered_hosts(mission_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_vulns_mission ON vulnerabilities(mission_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attacks_mission ON attack_attempts(mission_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_logs(timestamp)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    # Mission Management
    def create_mission(self, mission_id: str, target: str, scope: Optional[str] = None, 
                      authorized_by: Optional[str] = None, authorization_document: Optional[str] = None,
                      classification: str = 'CONFIDENTIAL', metadata: Optional[Dict] = None) -> bool:
        """Create a new mission"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO missions (id, target, scope, authorized_by, authorization_document, 
                                        classification, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (mission_id, target, scope, authorized_by, authorization_document, 
                     classification, json.dumps(metadata or {})))
                
                # Log mission creation
                self.log_audit('MISSION_CREATE', 'mission', mission_id, 
                             f"Created mission for target: {target}")
                
                conn.commit()
                logger.info(f"Mission created: {mission_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating mission: {str(e)}")
            return False
    
    def get_mission(self, mission_id: str) -> Optional[Dict]:
        """Get mission by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM missions WHERE id = ?', (mission_id,))
                row = cursor.fetchone()
                
                if row:
                    mission = dict(row)
                    mission['metadata'] = json.loads(mission['metadata'] or '{}')
                    return mission
                    
        except Exception as e:
            logger.error(f"Error getting mission: {str(e)}")
            
        return None
    
    def update_mission_status(self, mission_id: str, status: str) -> bool:
        """Update mission status"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE missions SET status = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (status, mission_id))
                
                if cursor.rowcount > 0:
                    self.log_audit('MISSION_UPDATE', 'mission', mission_id, 
                                 f"Status updated to: {status}")
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Error updating mission status: {str(e)}")
            
        return False
    
    # Reconnaissance Data
    def store_recon_result(self, mission_id: str, target: str, tool_name: str, 
                          scan_type: str, results: Dict, raw_output: str = "",
                          metadata: Optional[Dict] = None) -> int:
        """Store reconnaissance results"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO recon_results (mission_id, target, tool_name, scan_type, 
                                             status, end_time, results, raw_output, metadata)
                    VALUES (?, ?, ?, ?, 'COMPLETED', CURRENT_TIMESTAMP, ?, ?, ?)
                ''', (mission_id, target, tool_name, scan_type, 
                     json.dumps(results), raw_output, json.dumps(metadata or {})))
                
                result_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"Recon result stored: {tool_name} for {target}")
                return result_id
                
        except Exception as e:
            logger.error(f"Error storing recon result: {str(e)}")
            return 0
    
    def get_recon_results(self, mission_id: str, tool_name: Optional[str] = None) -> List[Dict]:
        """Get reconnaissance results for mission"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if tool_name:
                    cursor.execute('''
                        SELECT * FROM recon_results 
                        WHERE mission_id = ? AND tool_name = ?
                        ORDER BY start_time DESC
                    ''', (mission_id, tool_name))
                else:
                    cursor.execute('''
                        SELECT * FROM recon_results 
                        WHERE mission_id = ?
                        ORDER BY start_time DESC
                    ''', (mission_id,))
                
                results = []
                for row in cursor.fetchall():
                    result = dict(row)
                    result['results'] = json.loads(result['results'] or '{}')
                    result['metadata'] = json.loads(result['metadata'] or '{}')
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recon results: {str(e)}")
            return []
    
    # Host and Service Discovery
    def store_discovered_host(self, mission_id: str, ip_address: str, hostname: Optional[str] = None,
                             os_info: Optional[str] = None, confidence: float = 0.0,
                             metadata: Optional[Dict] = None) -> int:
        """Store discovered host"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO discovered_hosts 
                    (mission_id, ip_address, hostname, os_info, confidence, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (mission_id, ip_address, hostname, os_info, confidence, 
                     json.dumps(metadata or {})))
                
                host_id = cursor.lastrowid
                conn.commit()
                return host_id
                
        except Exception as e:
            logger.error(f"Error storing discovered host: {str(e)}")
            return 0
    
    def store_discovered_service(self, host_id: int, port: int, protocol: str,
                                service_name: Optional[str] = None, version: Optional[str] = None,
                                banner: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """Store discovered service"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO discovered_services 
                    (host_id, port, protocol, service_name, version, banner, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (host_id, port, protocol, service_name, version, banner,
                     json.dumps(metadata or {})))
                
                service_id = cursor.lastrowid
                conn.commit()
                return service_id
                
        except Exception as e:
            logger.error(f"Error storing discovered service: {str(e)}")
            return 0
    
    # Vulnerability Management
    def store_vulnerability(self, mission_id: str, target: str, vuln_type: str, title: str,
                           description: str, severity: str, cvss_score: Optional[float] = None,
                           cve_id: Optional[str] = None, exploit_available: bool = False,
                           remediation: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """Store discovered vulnerability"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO vulnerabilities 
                    (mission_id, target, vuln_type, title, description, severity, cvss_score,
                     cve_id, exploit_available, remediation, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (mission_id, target, vuln_type, title, description, severity, cvss_score,
                     cve_id, exploit_available, remediation, json.dumps(metadata or {})))
                
                vuln_id = cursor.lastrowid
                conn.commit()
                
                self.log_audit('VULNERABILITY_DISCOVERED', 'vulnerability', str(vuln_id),
                              f"Discovered {severity} vulnerability: {title}")
                
                return vuln_id
                
        except Exception as e:
            logger.error(f"Error storing vulnerability: {str(e)}")
            return 0
    
    def get_vulnerabilities(self, mission_id: str, severity: Optional[str] = None) -> List[Dict]:
        """Get vulnerabilities for mission"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if severity:
                    cursor.execute('''
                        SELECT * FROM vulnerabilities 
                        WHERE mission_id = ? AND severity = ? AND false_positive = 0
                        ORDER BY cvss_score DESC, discovered_at DESC
                    ''', (mission_id, severity))
                else:
                    cursor.execute('''
                        SELECT * FROM vulnerabilities 
                        WHERE mission_id = ? AND false_positive = 0
                        ORDER BY cvss_score DESC, discovered_at DESC
                    ''', (mission_id,))
                
                vulnerabilities = []
                for row in cursor.fetchall():
                    vuln = dict(row)
                    vuln['metadata'] = json.loads(vuln['metadata'] or '{}')
                    vulnerabilities.append(vuln)
                
                return vulnerabilities
                
        except Exception as e:
            logger.error(f"Error getting vulnerabilities: {str(e)}")
            return []
    
    # Attack Management
    def store_attack_attempt(self, mission_id: str, target: str, attack_type: str,
                            vulnerability_id: Optional[int] = None, tool_name: Optional[str] = None,
                            payload: Optional[str] = None, success: bool = False,
                            result_data: Optional[Dict] = None, error_message: Optional[str] = None,
                            metadata: Optional[Dict] = None) -> int:
        """Store attack attempt result"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO attack_attempts 
                    (mission_id, vulnerability_id, target, attack_type, tool_name, payload,
                     success, completed_at, result_data, error_message, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
                ''', (mission_id, vulnerability_id, target, attack_type, tool_name, payload,
                     success, json.dumps(result_data or {}), error_message, json.dumps(metadata or {})))
                
                attack_id = cursor.lastrowid
                conn.commit()
                
                status = "SUCCESS" if success else "FAILED"
                self.log_audit('ATTACK_ATTEMPT', 'attack', str(attack_id),
                              f"{status}: {attack_type} against {target}")
                
                return attack_id
                
        except Exception as e:
            logger.error(f"Error storing attack attempt: {str(e)}")
            return 0
    
    # Audit Logging
    def log_audit(self, action: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None,
                  details: Optional[str] = None, mission_id: Optional[str] = None,
                  username: Optional[str] = None, ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None, success: bool = True):
        """Log audit event"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO audit_logs 
                    (mission_id, username, action, resource_type, resource_id, details,
                     ip_address, user_agent, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (mission_id, username, action, resource_type, resource_id, details,
                     ip_address, user_agent, success))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
    
    def get_audit_logs(self, mission_id: Optional[str] = None, days: int = 7) -> List[Dict]:
        """Get audit logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if mission_id:
                    cursor.execute('''
                        SELECT * FROM audit_logs 
                        WHERE mission_id = ? AND timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (mission_id, cutoff_date))
                else:
                    cursor.execute('''
                        SELECT * FROM audit_logs 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                    ''', (cutoff_date,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting audit logs: {str(e)}")
            return []
    
    # Database Maintenance
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create database backup"""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(self.db_dir / f"agent_ds_backup_{timestamp}.db")
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            raise
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data beyond retention period"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old audit logs
                cursor.execute('DELETE FROM audit_logs WHERE timestamp < ?', (cutoff_date,))
                audit_deleted = cursor.rowcount
                
                # Clean up old recon results for completed missions
                cursor.execute('''
                    DELETE FROM recon_results 
                    WHERE start_time < ? AND mission_id IN 
                    (SELECT id FROM missions WHERE status = 'COMPLETED')
                ''', (cutoff_date,))
                recon_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {audit_deleted} audit logs, {recon_deleted} recon results deleted")
                
        except Exception as e:
            logger.error(f"Data cleanup failed: {str(e)}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Count records in each table
                tables = ['missions', 'recon_results', 'discovered_hosts', 'discovered_services',
                         'vulnerabilities', 'attack_attempts', 'audit_logs', 'reports']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Database file size
                db_file = Path(self.db_path)
                stats['database_size_mb'] = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0
                
                # Active missions
                cursor.execute("SELECT COUNT(*) FROM missions WHERE status = 'ACTIVE'")
                stats['active_missions'] = cursor.fetchone()[0]
                
                # Recent activity (last 24 hours)
                yesterday = datetime.now() - timedelta(days=1)
                cursor.execute('SELECT COUNT(*) FROM audit_logs WHERE timestamp > ?', (yesterday,))
                stats['recent_activity'] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            
        return stats
    
    def close(self):
        """Close database connections"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()
            delattr(self._local, 'connection')