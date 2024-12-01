import urllib.request
import zipfile
import os
from pathlib import Path

class SpamDataDownloader:
    def __init__(self, url, zip_path, extracted_path, data_file_name):
        self.url = url
        self.zip_path = zip_path
        self.extracted_path = extracted_path
        self.data_file_path = Path(extracted_path) / data_file_name

    def download_and_extract(self):
        """Download and extract the dataset."""
        if self.data_file_path.exists():
            print(f"{self.data_file_path} already exists. Skipping download and extraction.")
            return

        self._download_file()
        self._extract_zip()
        self._rename_file()
        print(f"File downloaded and saved as {self.data_file_path}")

    def _download_file(self):
        """Download the zip file from the URL."""
        print(f"Downloading dataset from {self.url}...")
        with urllib.request.urlopen(self.url) as response:
            with open(self.zip_path, "wb") as out_file:
                out_file.write(response.read())
        print("Download complete.")

    def _extract_zip(self):
        """Extract the downloaded zip file."""
        print(f"Extracting {self.zip_path}...")
        with zipfile.ZipFile(self.zip_path, "r") as zip_ref:
            zip_ref.extractall(self.extracted_path)
        print("Extraction complete.")

    def _rename_file(self):
        """Rename the extracted file to have a .tsv extension."""
        original_file_path = Path(self.extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, self.data_file_path)

# Main execution
if __name__ == "__main__":
    downloader = SpamDataDownloader(
        url="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        zip_path="sms_spam_collection.zip",
        extracted_path="sms_spam_collection",
        data_file_name="SMSSpamCollection.tsv"
    )
    downloader.download_and_extract()
