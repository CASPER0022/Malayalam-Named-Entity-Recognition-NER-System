import os
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification, AutoConfig, AutoModel
from torchcrf import CRF
import config


class MalayalamBERTCRF(nn.Module):
    """Transformer sequence encoder topped with a Conditional Random Field (CRF) layer."""
    def __init__(self, model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        # Create binary mask where 1 represents a valid token and 0 represents a masked token ([PAD] or subword)
        if labels is not None:
            mask = (labels != -100)
            # Ensure the first token [CLS] mask is always True to satisfy pytorch-crf validation
            mask[:, 0] = True
            
            # CRF cannot handle -100 labels directly, replace them with 0 (masked out anyway)
            clean_labels = labels.clone()
            clean_labels[clean_labels == -100] = 0
            
            # Compute negative log likelihood loss
            loss = -self.crf(emissions, clean_labels, mask=mask, reduction='mean')
            return type('CRFOutput', (object,), {"loss": loss, "logits": emissions})()
        else:
            # Mask based on attention mask (ignoring padding)
            mask = (attention_mask == 1)
            preds = self.crf.decode(emissions, mask=mask)
            return type('CRFOutput', (object,), {"logits": emissions, "predictions": preds})()

    def save_pretrained(self, save_dir):
        """Saves model weights state-dict and config."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
        self.config.save_pretrained(save_dir)

    def load_pretrained(self, save_dir):
        """Loads model weights state-dict."""
        self.load_state_dict(torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location=config.DEVICE))


def build_ner_model(model_name=config.MODEL_NAME, num_labels=config.NUM_LABELS, use_crf=config.USE_CRF):
    """Initializes standard Transformer sequence classifier or CRF-augmented model."""
    if use_crf:
        print("Configuring Model: IndicBERTv2 + Conditional Random Field (CRF) Head")
        return MalayalamBERTCRF(model_name=model_name, num_labels=num_labels)
    
    print("Configuring Model: IndicBERTv2 + Linear Token Classification Head")
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
