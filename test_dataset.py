import sys
import pandas as pd
from transformers import AutoTokenizer
import config
from src.dataset import load_ner_dataframe, MalayalamNERDataset

sys.stdout.reconfigure(encoding='utf-8')

print("=== Running Malayalam NER Dataset Verification ===")
records = load_ner_dataframe(config.TEST_CSV, max_samples=10)
print(f"Successfully loaded {len(records)} test records.")

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
dataset = MalayalamNERDataset(records, tokenizer, max_length=config.MAX_LENGTH)

print(f"Dataset length: {len(dataset)}")
sample = dataset[0]

print("Sample Item Keys:", list(sample.keys()))
print("Input IDs shape:", sample["input_ids"].shape)
print("Attention Mask shape:", sample["attention_mask"].shape)
print("Labels shape:", sample["labels"].shape)

# Decode tokens & labels
tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
labels = sample["labels"].numpy()

print("\n--- Token & Subword Label Alignment Sample ---")
for t, l in zip(tokens[:20], labels[:20]):
    label_str = config.ID2LABEL[l] if l != -100 else "[MASKED -100]"
    print(f"{t:<25} -> {label_str}")

print("\nDataset Verification PASSED Cleanly!")
