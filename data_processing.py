import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd

class SpamDataDownloader:
    """Class to handle downloading and extracting spam data."""
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


def load_and_check_data(data_file_path):
    """Load data into a DataFrame and display basic statistics."""
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print("Data loaded successfully.")
    print(df["Label"].value_counts())
    return df


def create_balanced_dataset(df):
    """Balance the dataset by matching the number of 'ham' and 'spam' entries."""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    print("Balanced dataset created.")
    print(balanced_df["Label"].value_counts())
    return balanced_df


def random_split(df, train_frac, validation_frac):
    """Split the dataset into train, validation, and test sets."""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
