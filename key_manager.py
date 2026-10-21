"""
API Key Manager — Rotates OpenRouter keys per scan cycle.
"""
import logging
from typing import List

logger = logging.getLogger(__name__)

class KeyManager:
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("No API keys provided in .env!")
        self.keys = keys
        self.current_index = 0
        logger.info(f"🔑 KeyManager initialized with {len(keys)} keys.")

    def get_current_key(self) -> str:
        """Returns the currently active key."""
        return self.keys[self.current_index]

    def rotate(self) -> str:
        """Moves to the next key and returns it. Wraps around at the end."""
        self.current_index = (self.current_index + 1) % len(self.keys)
        key_preview = self.keys[self.current_index][:12] + "..." + self.keys[self.current_index][-4:]
        logger.info(f"🔄 Rotated API Key -> Index {self.current_index} ({key_preview})")
        return self.keys[self.current_index]
