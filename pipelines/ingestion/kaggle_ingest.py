import os
import logging
from kaggle.api.kaggle_api_extended import KaggleApi
from config import KAGGLE_DATASET, RAW_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_steam_dataset():
    try:
        api = KaggleApi()
        api.authenticate()
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
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