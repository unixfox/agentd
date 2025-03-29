import json
import os
from appdirs import user_config_dir
from loguru import logger


class ConfigManager:
    def __init__(self, app_name: str, app_author: str):
        self.app_name = app_name
        self.app_author = app_author
        self.config_dir = user_config_dir(app_name, app_author)
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.config = {}
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
                logger.debug(f"Loaded config: {self.config}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        return self.config

    def save_config(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            logger.debug(f"Saved config: {self.config}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def update_agent_config_field(self, field, value, agent_type, agent_key):
        # Ensure the nested dictionaries exist
        agents_config = self.config.setdefault("agents", {})
        agent_group = agents_config.setdefault(agent_type, {})
        agent = agent_group.setdefault(agent_key, {})

        # Update the field
        agent[field] = value

        # Save the updated config
        self.save_config()