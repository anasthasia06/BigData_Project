import time
from typing import Any, Dict, Optional, Tuple


class TTLCache:
	"""Cache mémoire minimaliste avec expiration TTL."""

	def __init__(self, default_ttl: int = 60) -> None:
		self.default_ttl = default_ttl
		self._store: Dict[str, Tuple[float, Any]] = {}

	def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
		ttl_value = self.default_ttl if ttl is None else ttl
		self._store[key] = (time.time() + ttl_value, value)

	def get(self, key: str) -> Optional[Any]:
		if key not in self._store:
			return None
		expires_at, value = self._store[key]
		if time.time() > expires_at:
			del self._store[key]
			return None
		return value

	def clear(self) -> None:
		self._store.clear()

