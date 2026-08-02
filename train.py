import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np

import config
from src.dataset import load_ner_dataframe, MalayalamNERDataset
from src.model import build_ner_model, get_class_weights


def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_val_loss = 0.0

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = criterion(logits.view(-1, config.NUM_LABELS), labels.view(-1))
            total_val_loss += loss.item()

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

    avg_loss = total_val_loss / max(len(val_loader), 1)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return avg_loss, precision, recall, f1, all_labels, all_preds


def train():
    print("=" * 60)
    print(f"Starting Malayalam NER Training Pipeline")
    print(f"Device: {config.DEVICE}")
    print(f"Model Backbone: {config.MODEL_NAME}")
    print(f"Mixed Precision (FP16): {config.USE_FP16}")
    print("=" * 60)

    # 1. Load Data
    print(f"Loading training data from {config.TRAIN_CSV} (max_samples={config.MAX_TRAIN_SAMPLES})...")
    train_records = load_ner_dataframe(config.TRAIN_CSV, max_samples=config.MAX_TRAIN_SAMPLES)
    print(f"Loaded {len(train_records)} training sentences.")

    print(f"Loading validation data from {config.VAL_CSV}...")
    val_records = load_ner_dataframe(config.VAL_CSV)
    print(f"Loaded {len(val_records)} validation sentences.")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = MalayalamNERDataset(train_records, tokenizer, max_length=config.MAX_LENGTH)
    val_dataset = MalayalamNERDataset(val_records, tokenizer, max_length=config.MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Model & Loss Weighting
    model = build_ner_model(model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS)
    model.to(config.DEVICE)

    class_weights = get_class_weights(train_records)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler('cuda') if config.USE_FP16 else None

    best_f1 = 0.0

    # 5. Training Loop
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            if config.USE_FP16 and scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits.view(-1, config.NUM_LABELS), labels.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits.view(-1, config.NUM_LABELS), labels.view(-1))
                loss.backward()
                optimizer.step()

            scheduler.step()
            running_loss += loss.item()

            if step % 200 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{config.EPOCHS} | Step {step}/{len(train_loader)} | Train Loss: {running_loss / step:.4f}")

        elapsed = time.time() - start_time
        val_loss, val_prec, val_rec, val_f1, _, _ = evaluate_model(model, val_loader, config.DEVICE)

        print(f"\n--- Epoch {epoch} Summary ---")
        print(f"Time: {elapsed:.2f}s | Train Loss: {running_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1-Score: {val_f1:.4f}")

        if val_f1 > best_f1 or epoch == 1:
            best_f1 = val_f1
            print(f"Saving best model checkpoint to {config.MODEL_SAVE_DIR} (F1: {best_f1:.4f})...\n")
            model.save_pretrained(config.MODEL_SAVE_DIR)
            tokenizer.save_pretrained(config.MODEL_SAVE_DIR)

    print("Training Completed Successfully!")


if __name__ == "__main__":
    train()
