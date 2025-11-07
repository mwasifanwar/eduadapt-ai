import yaml
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "configs/default.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default