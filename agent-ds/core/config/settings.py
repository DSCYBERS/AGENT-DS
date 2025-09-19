"""
Agent DS Configuration Management
Handles application settings, environment variables, and configuration persistence
"""

import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration manager for Agent DS"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager"""
        self.config_dir = Path(config_dir) if config_dir else self._get_default_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "agent_ds.json"
        self.secrets_file = self.config_dir / "secrets.json"
        
        self._config = {}
        self._secrets = {}
        
        self.load()
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory"""
        project_root = Path(__file__).parent.parent.parent
        return project_root / "config"
    
    def load(self):
        """Load configuration from files"""
        # Load main configuration
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self._config = json.load(f)
            except Exception as e:
                logger.error(f"Error loading config: {str(e)}")
                self._config = {}
        
        # Load secrets
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    self._secrets = json.load(f)
            except Exception as e:
                logger.error(f"Error loading secrets: {str(e)}")
                self._secrets = {}
        
        # Load environment variables
        self._load_environment_defaults()
    
    def _load_environment_defaults(self):
        """Load default configuration values"""
        defaults = {
            # Database settings
            'database': {
                'path': str(self.config_dir.parent / "data" / "agent_ds.db"),
                'backup_interval': 24,  # hours
                'max_backups': 7
            },
            
            # Authentication settings
            'auth': {
                'session_timeout': 4 * 60 * 60,  # 4 hours
                'max_failed_attempts': 3,
                'lockout_duration': 30 * 60,  # 30 minutes
                'require_mfa': False
            },
            
            # AI/ML settings
            'ai': {
                'model_path': str(self.config_dir.parent / "models"),
                'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
                'max_context_length': 4096,
                'temperature': 0.7,
                'learning_enabled': True
            },
            
            # Tool settings
            'tools': {
                'nmap': {
                    'binary_path': 'nmap',
                    'default_args': ['-sS', '-sV', '-O', '--script=default'],
                    'timeout': 300
                },
                'masscan': {
                    'binary_path': 'masscan',
                    'rate_limit': 1000,
                    'timeout': 600
                },
                'gobuster': {
                    'binary_path': 'gobuster',
                    'wordlist_path': '/usr/share/wordlists/dirb/common.txt',
                    'threads': 50
                },
                'sqlmap': {
                    'binary_path': 'sqlmap',
                    'tamper_scripts': ['space2comment', 'randomcase'],
                    'timeout': 900
                }
            },
            
            # Reporting settings
            'reporting': {
                'output_dir': str(self.config_dir.parent / "reports"),
                'template_dir': str(self.config_dir.parent / "templates"),
                'classification_levels': ['UNCLASSIFIED', 'CONFIDENTIAL', 'SECRET', 'TOP SECRET'],
                'default_classification': 'CONFIDENTIAL'
            },
            
            # Security settings
            'security': {
                'audit_logging': True,
                'sandbox_mode': False,
                'target_validation': True,
                'authorized_domains': [],
                'blacklisted_ips': []
            },
            
            # Network settings
            'network': {
                'user_agent': 'Agent-DS/1.0 (Government-Authorized-Security-Assessment)',
                'timeout': 30,
                'max_redirects': 5,
                'proxy': None
            }
        }
        
        # Merge defaults with existing config
        for key, value in defaults.items():
            if key not in self._config:
                self._config[key] = value
            elif isinstance(value, dict) and isinstance(self._config[key], dict):
                for subkey, subvalue in value.items():
                    if subkey not in self._config[key]:
                        self._config[key][subkey] = subvalue
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to parent key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get secret value"""
        return self._secrets.get(key, default)
    
    def set_secret(self, key: str, value: Any):
        """Set secret value"""
        self._secrets[key] = value
    
    def save(self):
        """Save configuration to files"""
        try:
            # Save main configuration
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            # Save secrets
            if self._secrets:
                with open(self.secrets_file, 'w') as f:
                    json.dump(self._secrets, f, indent=2)
                
                # Set restrictive permissions on secrets file
                os.chmod(self.secrets_file, 0o600)
            
            logger.info("Configuration saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def load_yaml_config(self, yaml_path: str):
        """Load configuration from YAML file"""
        try:
            with open(yaml_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Merge with existing configuration
            self._merge_config(self._config, yaml_config)
            logger.info(f"Loaded YAML configuration from {yaml_path}")
            
        except Exception as e:
            logger.error(f"Error loading YAML config: {str(e)}")
    
    def _merge_config(self, base: dict, update: dict):
        """Recursively merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def validate_config(self) -> bool:
        """Validate configuration settings"""
        try:
            # Check required paths exist
            required_dirs = [
                self.get('database.path', '').rsplit('/', 1)[0],
                self.get('ai.model_path'),
                self.get('reporting.output_dir'),
                self.get('reporting.template_dir')
            ]
            
            for dir_path in required_dirs:
                if dir_path:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            # Validate tool paths
            tools = self.get('tools', {})
            for tool_name, tool_config in tools.items():
                binary_path = tool_config.get('binary_path')
                if binary_path and not self._is_tool_available(binary_path):
                    logger.warning(f"Tool not found: {tool_name} at {binary_path}")
            
            # Validate network settings
            timeout = self.get('network.timeout', 30)
            if not isinstance(timeout, int) or timeout <= 0:
                logger.error("Invalid network timeout value")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation error: {str(e)}")
            return False
    
    def _is_tool_available(self, tool_path: str) -> bool:
        """Check if a tool is available in the system"""
        import shutil
        return shutil.which(tool_path) is not None
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tool"""
        return self.get(f'tools.{tool_name}', {})
    
    def update_tool_config(self, tool_name: str, config: Dict[str, Any]):
        """Update configuration for a specific tool"""
        current_config = self.get_tool_config(tool_name)
        current_config.update(config)
        self.set(f'tools.{tool_name}', current_config)
    
    def export_config(self, export_path: str, include_secrets: bool = False):
        """Export configuration to file"""
        export_data = {
            'config': self._config
        }
        
        if include_secrets:
            export_data['secrets'] = self._secrets
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Configuration exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
    
    def import_config(self, import_path: str):
        """Import configuration from file"""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if 'config' in import_data:
                self._merge_config(self._config, import_data['config'])
            
            if 'secrets' in import_data:
                self._secrets.update(import_data['secrets'])
            
            logger.info(f"Configuration imported from {import_path}")
            
        except Exception as e:
            logger.error(f"Import error: {str(e)}")
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self._config = {}
        self._secrets = {}
        self._load_environment_defaults()
        logger.info("Configuration reset to defaults")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self._config.copy()
    
    def __str__(self) -> str:
        """String representation of configuration"""
        # Remove sensitive information for display
        display_config = self._config.copy()
        return json.dumps(display_config, indent=2)