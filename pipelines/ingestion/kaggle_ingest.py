import os
import logging
from config import KAGGLE_DATASET, RAW_DATA_DIR

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    KaggleApi = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_steam_dataset():
    try:
        if KaggleApi is None:
            logging.error("Le package 'kaggle' n'est pas installe dans cet environnement.")
            return

        api = KaggleApi()
        api.authenticate()

        try:
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
        except OSError as exc:
            logging.warning(
                "Impossible de creer le repertoire cible %s (%s). Tentative de telechargement quand meme.",
                RAW_DATA_DIR,
                exc,
            )

        logging.info(f"Téléchargement du dataset Kaggle complet : {KAGGLE_DATASET}")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=RAW_DATA_DIR,
            unzip=True,
            force=True,
            quiet=False
        )
        logging.info(f"Dataset téléchargé et extrait dans {RAW_DATA_DIR}")
    except Exception as e:
        logging.error(f"Erreur lors du téléchargement du dataset : {e}")

if __name__ == "__main__":
    download_steam_dataset()