import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
import config


def load_ner_dataframe(csv_path, max_samples=None):
    """Loads CSV containing string representation of tokens and ner_tags."""
    df = pd.read_csv(csv_path)
    if max_samples and max_samples < len(df):
        df = df.iloc[:max_samples].copy()

    processed_records = []
    for idx, row in df.iterrows():
        try:
            tokens = ast.literal_eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
            tags = ast.literal_eval(row['ner_tags']) if isinstance(row['ner_tags'], str) else row['ner_tags']
            if len(tokens) == len(tags) and len(tokens) > 0:
                processed_records.append({"tokens": tokens, "ner_tags": tags})
        except Exception:
            continue

    return processed_records


class MalayalamNERDataset(Dataset):
    """PyTorch Dataset with HuggingFace Fast Tokenizer Subword Alignment."""

    def __init__(self, records, tokenizer, max_length=config.MAX_LENGTH):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        word_list = record['tokens']
        label_list = record['ner_tags']

        # Tokenize with is_split_into_words=True
        encoding = self.tokenizer(
            word_list,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100 label
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of the word gets original label
                aligned_labels.append(label_list[word_idx])
            else:
                # Subsequent subword tokens get -100 so loss ignored
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

        return item
