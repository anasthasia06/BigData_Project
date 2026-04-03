def _mongo_client_factory(uri: str):
	from pymongo import MongoClient

	return MongoClient(uri, serverSelectionTimeoutMS=5000)


class MongoDB:
	def __init__(self, uri: str, db_name: str, collection_name: str):
		try:
			self.client = _mongo_client_factory(uri)
		except ModuleNotFoundError as exc:
			raise RuntimeError("pymongo package manquant") from exc
		self.db = self.client[db_name]
		self.collection = self.db[collection_name]

	def ping(self) -> bool:
		self.client.admin.command("ping")
		return True

	def close(self) -> None:
		self.client.close()

