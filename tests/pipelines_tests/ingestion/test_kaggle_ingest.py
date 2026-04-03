import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pipelines.ingestion.kaggle_ingest import download_steam_dataset


def test_download_calls_authenticate():
    """L'API Kaggle doit être authentifiée avant tout téléchargement."""
    with patch("pipelines.ingestion.kaggle_ingest.KaggleApi") as MockApi:
        instance = MockApi.return_value
        instance.dataset_download_files.return_value = None
        download_steam_dataset()
        instance.authenticate.assert_called_once()


def test_download_calls_dataset_download():
    """dataset_download_files doit être appelé avec le bon dataset."""
    with patch("pipelines.ingestion.kaggle_ingest.KaggleApi") as MockApi:
        instance = MockApi.return_value
        instance.dataset_download_files.return_value = None
        download_steam_dataset()
        args, kwargs = instance.dataset_download_files.call_args
        assert kwargs.get("unzip") is True or args[2] is True  # unzip activé


def test_download_uses_config_dataset():
    """Le dataset utilisé doit correspondre à KAGGLE_DATASET dans config."""
    import config
    with patch("pipelines.ingestion.kaggle_ingest.KaggleApi") as MockApi:
        instance = MockApi.return_value
        instance.dataset_download_files.return_value = None
        download_steam_dataset()
        _, kwargs = instance.dataset_download_files.call_args
        assert config.KAGGLE_DATASET in (kwargs.get("dataset") or instance.dataset_download_files.call_args[0][0])


def test_download_target_is_raw_data_dir():
    """Le répertoire de destination doit être RAW_DATA_DIR."""
    import config
    with patch("pipelines.ingestion.kaggle_ingest.KaggleApi") as MockApi:
        instance = MockApi.return_value
        instance.dataset_download_files.return_value = None
        download_steam_dataset()
        _, kwargs = instance.dataset_download_files.call_args
        assert str(config.RAW_DATA_DIR) == str(kwargs.get("path") or instance.dataset_download_files.call_args[0][1])


def test_download_handles_exception(caplog):
    """Une exception dans l'API doit être capturée et loggée sans crasher."""
    with patch("pipelines.ingestion.kaggle_ingest.KaggleApi") as MockApi:
        instance = MockApi.return_value
        instance.authenticate.side_effect = Exception("auth failed")
        import logging
        with caplog.at_level(logging.ERROR):
            download_steam_dataset()  # ne doit pas lever d'exception
        assert "auth failed" in caplog.text
