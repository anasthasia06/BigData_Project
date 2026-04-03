import time
from typing import Any, Dict, Optional, Tuple


class TTLCache:
	"""Cache mémoire minimaliste avec expiration TTL."""

	def __init__(self, default_ttl: int = 60) -> None:
		self.default_ttl = default_ttl
		self._store: Dict[str, Tuple[float, Any]] = {}
		self.hits = 0
		self.misses = 0

	def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
		ttl_value = self.default_ttl if ttl is None else ttl
		self._store[key] = (time.time() + ttl_value, value)

	def get(self, key: str) -> Optional[Any]:
		if key not in self._store:
			self.misses += 1
			return None
		expires_at, value = self._store[key]
		if time.time() > expires_at:
			del self._store[key]
			self.misses += 1
			return None
		self.hits += 1
		return value

	def get_or_set(self, key: str, builder, ttl: Optional[int] = None):
		cached = self.get(key)
		if cached is not None:
			return cached
		value = builder()
		self.set(key, value, ttl=ttl)
		return value

	def stats(self) -> Dict[str, int]:
		return {
			"hits": self.hits,
			"misses": self.misses,
			"size": len(self._store),
		}

	def clear(self) -> None:
		self._store.clear()

