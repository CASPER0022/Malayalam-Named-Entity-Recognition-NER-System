import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig
import config


def build_ner_model(model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS):
    """Initializes sequence token classification model with pretrained weights."""
    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=model_config
    )
    return model


def get_class_weights(dataset_records, num_labels=config.NUM_LABELS):
    """Computes inverse class frequencies to handle O-tag imbalance."""
    counts = torch.zeros(num_labels)
    for record in dataset_records:
        for tag in record['ner_tags']:
            if 0 <= tag < num_labels:
                counts[tag] += 1
    
    total = counts.sum()
    weights = total / (num_labels * (counts + 1e-5))
    weights = weights / weights.max()  # Normalize
    return weights.to(config.DEVICE)
