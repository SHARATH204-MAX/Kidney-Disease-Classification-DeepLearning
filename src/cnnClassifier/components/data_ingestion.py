import os
import re
import zipfile

import gdown

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    @staticmethod
    def _extract_file_id(url: str) -> str:
        """Return the Google Drive file id from a share or download URL."""
        match = re.search(r"(?:/d/|id=)([A-Za-z0-9_-]{10,})", url)
        if not match:
            raise ValueError(
                f"Invalid Google Drive URL: {url}. "
                "Expected a normal Google Drive share link or a direct file URL."
            )
        return match.group(1)

    def download_file(self) -> str:
        """
        Fetch data from the configured Google Drive URL.
        """
        dataset_url = str(self.config.source_URL).strip()
        zip_download_dir = str(self.config.local_data_file)

        if os.path.exists(zip_download_dir):
            logger.info(f"Data already exists at {zip_download_dir}. Skipping download.")
            return zip_download_dir

        os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
        logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

        try:
            file_id = self._extract_file_id(dataset_url)
            download_urls = [
                dataset_url,
                f"https://drive.google.com/uc?export=download&id={file_id}",
                f"https://drive.google.com/uc?/export=download&id={file_id}",
            ]

            last_error = None
            for download_url in download_urls:
                try:
                    gdown.download(download_url, zip_download_dir, quiet=False)
                    logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
                    return zip_download_dir
                except Exception as exc:
                    last_error = exc

            if last_error is not None:
                raise last_error

        except Exception as e:
            message = str(e)
            if any(
                token in message.lower()
                for token in ["resolver", "gaierror", "connectionerror", "timed out", "temporary failure"]
            ):
                raise RuntimeError(
                    "Unable to reach Google Drive from this environment. "
                    "Check your internet connection or replace the source URL with a reachable dataset mirror. "
                    f"Original error: {e}"
                ) from e
            raise RuntimeError(
                f"Failed to download the dataset from Google Drive: {e}"
            ) from e

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)