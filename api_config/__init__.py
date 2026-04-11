"""
API Configuration Manager
Centralized configuration storage for all APIs
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class APIConfigManager:
    """Manage API configuration centrally"""
    
    CONFIG_DIR = Path(__file__).parent
    ENDPOINTS_FILE = CONFIG_DIR / "endpoints.json"
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON"""
        if self.ENDPOINTS_FILE.exists():
            with open(self.ENDPOINTS_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def get_endpoint(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get endpoint configuration by name"""
        return self.config.get("endpoints", {}).get(endpoint_name)
    
    def get_all_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Get all endpoint configurations"""
        return self.config.get("endpoints", {})
    
    def get_server(self, server_name: str = "primary") -> Optional[Dict[str, Any]]:
        """Get server configuration"""
        return self.config.get("api_servers", {}).get(server_name)
    
    def get_data_source(self, source_name: str) -> Optional[Dict[str, Any]]:
        """Get data source configuration"""
        return self.config.get("data_sources", {}).get(source_name)
    
    def add_endpoint(self, name: str, config: Dict[str, Any]) -> None:
        """Add new endpoint configuration"""
        if "endpoints" not in self.config:
            self.config["endpoints"] = {}
        self.config["endpoints"][name] = config
        self._save_config()
    
    def update_endpoint(self, name: str, config: Dict[str, Any]) -> None:
        """Update endpoint configuration"""
        if "endpoints" in self.config:
            self.config["endpoints"][name].update(config)
            self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to JSON"""
        with open(self.ENDPOINTS_FILE, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def validate_config(self) -> bool:
        """Validate configuration structure"""
        required_keys = ["version", "api_servers", "endpoints"]
        return all(key in self.config for key in required_keys)


# Singleton instance
_api_config_manager = None


def get_api_config_manager() -> APIConfigManager:
    """Get or create API config manager instance"""
    global _api_config_manager
    if _api_config_manager is None:
        _api_config_manager = APIConfigManager()
    return _api_config_manager
