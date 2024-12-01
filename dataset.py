import torch
from torch.utils.data import Dataset
import pandas as pd


class SpamDataset(Dataset):
    """Dataset class for SMS Spam data."""
    
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        Initialize the SpamDataset.
        
        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            tokenizer: Tokenizer for encoding text data.
            max_length (int): Maximum sequence length (optional).
            pad_token_id (int): Padding token ID.
        """
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence or specified max_length
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        """Retrieve an item at the given index."""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def _longest_encoded_length(self):
        """Find the longest encoded text sequence length."""
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
