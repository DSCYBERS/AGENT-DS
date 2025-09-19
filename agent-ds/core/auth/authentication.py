"""
Agent DS Authentication Module
Phase-based authentication with admin credentials and session management
"""

import hashlib
import secrets
import time
import jwt
import sqlite3
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class AuthManager:
    """Manages authentication, session tokens, and phase-based access control"""
    
    # Hardcoded admin credentials as specified
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "8460Divy!@#$"
    
    # JWT secret key (in production, this should be environment variable)
    JWT_SECRET = "agent_ds_classified_operations_2024_nsa_approved"
    
    # Session timeout (4 hours for government operations)
    SESSION_TIMEOUT = 4 * 60 * 60  # 4 hours in seconds
    
    # Authentication phases
    PHASES = {
        'LOGIN': 'Initial authentication',
        'MISSION_START': 'Mission initialization',
        'RECONNAISSANCE': 'Target reconnaissance',
        'ANALYSIS': 'Vulnerability analysis',
        'ATTACK': 'Attack execution',
        'REPORTING': 'Report generation'
    }
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize authentication manager"""
        self.db_path = db_path or self._get_default_db_path()
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._initialize_database()
        
    def _get_default_db_path(self) -> str:
        """Get default database path"""
        project_root = Path(__file__).parent.parent.parent
        db_dir = project_root / "data"
        db_dir.mkdir(exist_ok=True)
        return str(db_dir / "auth.db")
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for sensitive data"""
        key_file = Path(__file__).parent.parent.parent / "data" / "encryption.key"
        key_file.parent.mkdir(exist_ok=True)
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
            
    def _initialize_database(self):
        """Initialize authentication database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    token TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    phase TEXT DEFAULT 'LOGIN',
                    is_active BOOLEAN DEFAULT 1,
                    ip_address TEXT,
                    user_agent TEXT
                )
            ''')
            
            # Authentication attempts table for audit
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    ip_address TEXT,
                    success BOOLEAN NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    failure_reason TEXT,
                    user_agent TEXT
                )
            ''')
            
            # MFA tokens table (optional)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS mfa_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    token TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    used BOOLEAN DEFAULT 0
                )
            ''')
            
            conn.commit()
            
    def authenticate(self, username: str, password: str, mfa_code: Optional[str] = None, 
                    ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Tuple[bool, Optional[str], str]:
        """
        Authenticate user with admin credentials
        
        Args:
            username: Username to authenticate
            password: Password to verify
            mfa_code: Optional MFA code
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (success, session_token, message)
        """
        try:
            # Log authentication attempt
            self._log_auth_attempt(username, ip_address, False, "", user_agent)
            
            # Verify admin credentials
            if username != self.ADMIN_USERNAME or password != self.ADMIN_PASSWORD:
                self._log_auth_attempt(username, ip_address, False, "Invalid credentials", user_agent)
                return False, None, "Invalid credentials"
            
            # Verify MFA if provided
            if mfa_code and not self._verify_mfa(username, mfa_code):
                self._log_auth_attempt(username, ip_address, False, "Invalid MFA code", user_agent)
                return False, None, "Invalid MFA code"
            
            # Generate session token
            session_id = secrets.token_urlsafe(32)
            session_token = self._generate_jwt_token(username, session_id)
            
            # Store session in database
            expires_at = datetime.now() + timedelta(seconds=self.SESSION_TIMEOUT)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO sessions (id, username, token, expires_at, phase, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, username, session_token, expires_at, 'LOGIN', ip_address, user_agent))
                conn.commit()
            
            # Log successful authentication
            self._log_auth_attempt(username, ip_address, True, "", user_agent)
            
            logger.info(f"Successful authentication for user: {username} from {ip_address}")
            return True, session_token, "Authentication successful"
            
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False, None, f"Authentication error: {str(e)}"
    
    def _generate_jwt_token(self, username: str, session_id: str) -> str:
        """Generate JWT session token"""
        payload = {
            'username': username,
            'session_id': session_id,
            'iat': time.time(),
            'exp': time.time() + self.SESSION_TIMEOUT,
            'iss': 'agent-ds-v1.0',
            'aud': 'government-authorized-operations'
        }
        
        return jwt.encode(payload, self.JWT_SECRET, algorithm='HS256')
    
    def verify_session(self, token: Optional[str] = None) -> bool:
        """Verify if session token is valid and active"""
        if not token:
            # Try to get token from config
            from core.config.settings import Config
            config = Config()
            token = config.get('session_token')
            
        if not token:
            return False
            
        try:
            # Decode JWT token
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=['HS256'])
            session_id = payload.get('session_id')
            username = payload.get('username')
            
            # Check session in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT expires_at, is_active FROM sessions 
                    WHERE id = ? AND username = ? AND token = ?
                ''', (session_id, username, token))
                
                result = cursor.fetchone()
                if not result:
                    return False
                    
                expires_at, is_active = result
                
                # Check if session is still active and not expired
                if not is_active or datetime.now() > datetime.fromisoformat(expires_at):
                    return False
                
                # Update last activity
                cursor.execute('''
                    UPDATE sessions SET last_activity = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (session_id,))
                conn.commit()
                
            return True
            
        except jwt.ExpiredSignatureError:
            logger.warning("Session token expired")
            return False
        except jwt.InvalidTokenError:
            logger.warning("Invalid session token")
            return False
        except Exception as e:
            logger.error(f"Session verification error: {str(e)}")
            return False
    
    def update_phase(self, token: str, new_phase: str) -> bool:
        """Update the current phase for authenticated session"""
        if new_phase not in self.PHASES:
            logger.error(f"Invalid phase: {new_phase}")
            return False
            
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions SET phase = ?, last_activity = CURRENT_TIMESTAMP 
                    WHERE id = ? AND is_active = 1
                ''', (new_phase, session_id))
                conn.commit()
                
            logger.info(f"Phase updated to {new_phase} for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Phase update error: {str(e)}")
            return False
    
    def get_current_phase(self, token: str) -> Optional[str]:
        """Get current phase for authenticated session"""
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT phase FROM sessions 
                    WHERE id = ? AND is_active = 1
                ''', (session_id,))
                
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Get phase error: {str(e)}")
            return None
    
    def logout(self, token: str) -> bool:
        """Logout and invalidate session"""
        try:
            payload = jwt.decode(token, self.JWT_SECRET, algorithms=['HS256'])
            session_id = payload.get('session_id')
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE sessions SET is_active = 0 
                    WHERE id = ?
                ''', (session_id,))
                conn.commit()
                
            logger.info(f"Session logged out: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Logout error: {str(e)}")
            return False
    
    def generate_mfa_token(self, username: str) -> str:
        """Generate MFA token (6-digit code)"""
        import random
        code = f"{random.randint(100000, 999999)}"
        expires_at = datetime.now() + timedelta(minutes=5)  # 5 minute expiry
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO mfa_tokens (username, token, expires_at)
                VALUES (?, ?, ?)
            ''', (username, code, expires_at))
            conn.commit()
            
        logger.info(f"MFA token generated for user: {username}")
        return code
    
    def _verify_mfa(self, username: str, code: str) -> bool:
        """Verify MFA token"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id FROM mfa_tokens 
                WHERE username = ? AND token = ? AND expires_at > CURRENT_TIMESTAMP AND used = 0
            ''', (username, code))
            
            result = cursor.fetchone()
            if result:
                # Mark token as used
                cursor.execute('''
                    UPDATE mfa_tokens SET used = 1 WHERE id = ?
                ''', (result[0],))
                conn.commit()
                return True
                
        return False
    
    def _log_auth_attempt(self, username: str, ip_address: Optional[str], 
                         success: bool, failure_reason: str, user_agent: Optional[str]):
        """Log authentication attempts for audit"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO auth_attempts (username, ip_address, success, failure_reason, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, ip_address, success, failure_reason, user_agent))
            conn.commit()
    
    def get_active_sessions(self) -> list:
        """Get all active sessions for monitoring"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT username, created_at, last_activity, phase, ip_address 
                FROM sessions 
                WHERE is_active = 1 AND expires_at > CURRENT_TIMESTAMP
                ORDER BY last_activity DESC
            ''')
            
            return cursor.fetchall()
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions SET is_active = 0 
                WHERE expires_at <= CURRENT_TIMESTAMP
            ''')
            
            expired_count = cursor.rowcount
            conn.commit()
            
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired sessions")
    
    def get_auth_audit_log(self, days: int = 7) -> list:
        """Get authentication audit log for the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT username, ip_address, success, timestamp, failure_reason
                FROM auth_attempts 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            ''', (cutoff_date,))
            
            return cursor.fetchall()
    
    def require_phase(self, required_phase: str):
        """Decorator to require specific phase for operations"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Get current session token
                from core.config.settings import Config
                config = Config()
                token = config.get('session_token')
                
                if not self.verify_session(token):
                    raise PermissionError("Authentication required")
                
                current_phase = self.get_current_phase(token)
                if current_phase != required_phase:
                    raise PermissionError(f"Phase {required_phase} required, current: {current_phase}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator