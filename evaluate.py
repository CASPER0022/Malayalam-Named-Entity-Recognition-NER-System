import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

import config
from src.dataset import load_ner_dataframe, MalayalamNERDataset


def generate_confusion_matrix(all_labels, all_preds, save_path):
    """Generates and saves confusion matrix heatmap for entity tags."""
    flat_labels = [tag for sublist in all_labels for tag in sublist]
    flat_preds = [tag for sublist in all_preds for tag in sublist]

    tags_order = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]

    cm = confusion_matrix(flat_labels, flat_preds, labels=tags_order)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=tags_order,
        yticklabels=tags_order
    )
    plt.title('Malayalam NER Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Entity Tag', fontsize=12)
    plt.ylabel('Ground Truth Entity Tag', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix visualization to {save_path}")


def evaluate_test_set():
    print("=" * 60)
    print("Evaluating Malayalam NER Model on Test Set")
    print("=" * 60)

    if config.USE_CRF:
        model_dir = config.CRF_MODEL_SAVE_DIR if os.path.exists(os.path.join(config.CRF_MODEL_SAVE_DIR, "pytorch_model.bin")) else config.MODEL_NAME
        print(f"Loading CRF model checkpoint from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        from src.model import MalayalamBERTCRF
        model = MalayalamBERTCRF(model_name=config.MODEL_NAME)
        if model_dir != config.MODEL_NAME:
            model.load_pretrained(model_dir)
    else:
        model_dir = config.MODEL_SAVE_DIR if os.path.exists(os.path.join(config.MODEL_SAVE_DIR, "config.json")) else config.MODEL_NAME
        print(f"Loading Linear model checkpoint from: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        from src.model import build_ner_model
        model = build_ner_model(model_name=model_dir, use_crf=False)

    model.to(config.DEVICE)
    model.eval()

    print(f"Loading test set from {config.TEST_CSV}...")
    test_records = load_ner_dataframe(config.TEST_CSV)
    print(f"Loaded {len(test_records)} test sentences.")

    test_dataset = MalayalamNERDataset(test_records, tokenizer, max_length=config.MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            if config.USE_CRF:
                outputs_eval = model(input_ids=input_ids, attention_mask=attention_mask)
                preds_list = outputs_eval.predictions
                targets = labels.cpu().numpy()

                for i in range(len(targets)):
                    sentence_preds = []
                    sentence_labels = []
                    for j in range(len(targets[i])):
                        if targets[i][j] != -100 and j < len(preds_list[i]):
                            pred_id = preds_list[i][j]
                            sentence_preds.append(config.ID2LABEL[pred_id])
                            sentence_labels.append(config.ID2LABEL[targets[i][j]])
                    all_preds.append(sentence_preds)
                    all_labels.append(sentence_labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                targets = labels.cpu().numpy()

                for i in range(len(targets)):
                    sentence_preds = []
                    sentence_labels = []
                    for j in range(len(targets[i])):
                        if targets[i][j] != -100:
                            sentence_preds.append(config.ID2LABEL[preds[i][j]])
                            sentence_labels.append(config.ID2LABEL[targets[i][j]])
                    all_preds.append(sentence_preds)
                    all_labels.append(sentence_labels)

    print("\n" + "=" * 60)
    print("Seqeval Entity Classification Report:")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, digits=4)
    print(report)

    cm_save_path = os.path.join(config.BASE_DIR, "results", "confusion_matrix.png")
    generate_confusion_matrix(all_labels, all_preds, cm_save_path)

    return report


if __name__ == "__main__":
    evaluate_test_set()
