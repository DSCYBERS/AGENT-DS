"""
Agent DS Logging Utilities
Centralized logging configuration for security and audit compliance
"""

import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import sys

class SecurityAuditFormatter(logging.Formatter):
    """Custom formatter for security audit logs"""
    
    def format(self, record):
        """Format log record with security context"""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'process_id': os.getpid(),
            'thread_id': record.thread
        }
        
        # Add security context if available
        if hasattr(record, 'mission_id'):
            log_data['mission_id'] = record.mission_id
        if hasattr(record, 'username'):
            log_data['username'] = record.username
        if hasattr(record, 'ip_address'):
            log_data['ip_address'] = record.ip_address
        if hasattr(record, 'action'):
            log_data['action'] = record.action
        if hasattr(record, 'target'):
            log_data['target'] = record.target
        if hasattr(record, 'classification'):
            log_data['classification'] = record.classification
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)

class AgentDSLogger:
    """Main logger class for Agent DS"""
    
    def __init__(self, name: str = 'agent_ds'):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Clear existing handlers
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        
        # Create log directory
        log_dir = self._get_log_directory()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler for development
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for general logs
        general_log_file = log_dir / 'agent_ds.log'
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Security audit log handler
        audit_log_file = log_dir / 'security_audit.log'
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=10
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(SecurityAuditFormatter())
        
        # Create audit logger
        audit_logger = logging.getLogger(f'{self.name}.audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False
        
        # Attack log handler (separate for high-value events)
        attack_log_file = log_dir / 'attack_operations.log'
        attack_handler = logging.handlers.RotatingFileHandler(
            attack_log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=20  # Keep more attack logs
        )
        attack_handler.setLevel(logging.INFO)
        attack_handler.setFormatter(SecurityAuditFormatter())
        
        # Create attack logger
        attack_logger = logging.getLogger(f'{self.name}.attack')
        attack_logger.addHandler(attack_handler)
        attack_logger.setLevel(logging.INFO)
        attack_logger.propagate = False
        
        # Error log handler
        error_log_file = log_dir / 'errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=25 * 1024 * 1024,  # 25MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def _get_log_directory(self) -> Path:
        """Get log directory path"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / 'logs'
    
    def get_logger(self, module_name: Optional[str] = None) -> logging.Logger:
        """Get logger instance for specific module"""
        if module_name:
            return logging.getLogger(f'{self.name}.{module_name}')
        return self.logger
    
    def get_audit_logger(self) -> logging.Logger:
        """Get audit logger for security events"""
        return logging.getLogger(f'{self.name}.audit')
    
    def get_attack_logger(self) -> logging.Logger:
        """Get attack logger for attack operations"""
        return logging.getLogger(f'{self.name}.attack')

# Global logger instance
_agent_ds_logger = None

def setup_logger(name: str = 'agent_ds') -> logging.Logger:
    """Setup and return the main Agent DS logger"""
    global _agent_ds_logger
    
    if _agent_ds_logger is None:
        _agent_ds_logger = AgentDSLogger(name)
    
    return _agent_ds_logger.get_logger()

def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """Get logger for specific module"""
    if _agent_ds_logger is None:
        setup_logger()
    
    return _agent_ds_logger.get_logger(module_name)

def get_audit_logger() -> logging.Logger:
    """Get audit logger for security events"""
    if _agent_ds_logger is None:
        setup_logger()
    
    return _agent_ds_logger.get_audit_logger()

def get_attack_logger() -> logging.Logger:
    """Get attack logger for attack operations"""
    if _agent_ds_logger is None:
        setup_logger()
    
    return _agent_ds_logger.get_attack_logger()

def log_security_event(action: str, details: Dict[str, Any], 
                      mission_id: Optional[str] = None,
                      username: Optional[str] = None,
                      ip_address: Optional[str] = None,
                      target: Optional[str] = None,
                      classification: str = 'CONFIDENTIAL'):
    """Log security event to audit log"""
    audit_logger = get_audit_logger()
    
    # Create log record with extra context
    extra = {
        'action': action,
        'mission_id': mission_id,
        'username': username,
        'ip_address': ip_address,
        'target': target,
        'classification': classification
    }
    
    # Remove None values
    extra = {k: v for k, v in extra.items() if v is not None}
    
    message = f"Security Event: {action}"
    if details:
        message += f" - {json.dumps(details, default=str)}"
    
    audit_logger.info(message, extra=extra)

def log_attack_event(attack_type: str, target: str, success: bool,
                    details: Dict[str, Any],
                    mission_id: Optional[str] = None,
                    username: Optional[str] = None,
                    tool_name: Optional[str] = None,
                    payload: Optional[str] = None):
    """Log attack event to attack log"""
    attack_logger = get_attack_logger()
    
    extra = {
        'action': 'ATTACK_ATTEMPT',
        'mission_id': mission_id,
        'username': username,
        'target': target,
        'classification': 'SECRET'  # Attack logs are more sensitive
    }
    
    # Remove None values
    extra = {k: v for k, v in extra.items() if v is not None}
    
    status = "SUCCESS" if success else "FAILED"
    message = f"Attack {status}: {attack_type} against {target}"
    
    if tool_name:
        message += f" using {tool_name}"
    
    attack_details = {
        'attack_type': attack_type,
        'success': success,
        'tool_name': tool_name,
        'payload_preview': payload[:100] if payload else None,
        **details
    }
    
    message += f" - {json.dumps(attack_details, default=str)}"
    
    if success:
        attack_logger.warning(message, extra=extra)  # Successful attacks are warnings
    else:
        attack_logger.info(message, extra=extra)

def log_compliance_event(event_type: str, details: Dict[str, Any],
                        classification: str = 'CONFIDENTIAL'):
    """Log compliance-related events"""
    audit_logger = get_audit_logger()
    
    extra = {
        'action': f'COMPLIANCE_{event_type}',
        'classification': classification
    }
    
    message = f"Compliance Event: {event_type} - {json.dumps(details, default=str)}"
    audit_logger.info(message, extra=extra)

class SecurityContextFilter(logging.Filter):
    """Filter to add security context to log records"""
    
    def __init__(self, mission_id: Optional[str] = None, username: Optional[str] = None):
        super().__init__()
        self.mission_id = mission_id
        self.username = username
    
    def filter(self, record):
        """Add security context to record"""
        if self.mission_id:
            record.mission_id = self.mission_id
        if self.username:
            record.username = self.username
        return True

def add_security_context(logger: logging.Logger, mission_id: Optional[str] = None,
                        username: Optional[str] = None):
    """Add security context filter to logger"""
    security_filter = SecurityContextFilter(mission_id, username)
    logger.addFilter(security_filter)

def remove_security_context(logger: logging.Logger):
    """Remove security context filters from logger"""
    logger.filters = [f for f in logger.filters if not isinstance(f, SecurityContextFilter)]

# Utility functions for common logging patterns
def log_mission_start(mission_id: str, target: str, username: str):
    """Log mission start event"""
    log_security_event(
        'MISSION_START',
        {'target': target},
        mission_id=mission_id,
        username=username,
        classification='SECRET'
    )

def log_mission_complete(mission_id: str, summary: Dict[str, Any], username: str):
    """Log mission completion event"""
    log_security_event(
        'MISSION_COMPLETE',
        summary,
        mission_id=mission_id,
        username=username,
        classification='SECRET'
    )

def log_vulnerability_discovery(mission_id: str, vuln_details: Dict[str, Any]):
    """Log vulnerability discovery"""
    log_security_event(
        'VULNERABILITY_DISCOVERED',
        vuln_details,
        mission_id=mission_id,
        classification='CONFIDENTIAL'
    )

def log_unauthorized_access_attempt(username: str, ip_address: str, details: Dict[str, Any]):
    """Log unauthorized access attempts"""
    log_security_event(
        'UNAUTHORIZED_ACCESS_ATTEMPT',
        details,
        username=username,
        ip_address=ip_address,
        classification='SECRET'
    )