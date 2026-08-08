import os
import torch
import re
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import config


class MalayalamNERPredictor:
    def __init__(self, model_path=None):
        if config.USE_CRF:
            if model_path is None:
                model_path = config.CRF_MODEL_SAVE_DIR if os.path.exists(os.path.join(config.CRF_MODEL_SAVE_DIR, "pytorch_model.bin")) else config.MODEL_NAME
            
            print(f"Loading CRF Predictor checkpoint: {model_path}")
            from src.model import MalayalamBERTCRF
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
            self.model = MalayalamBERTCRF(model_name=config.MODEL_NAME)
            if model_path != config.MODEL_NAME:
                self.model.load_pretrained(model_path)
        else:
            if model_path is None:
                model_path = config.MODEL_SAVE_DIR if os.path.exists(os.path.join(config.MODEL_SAVE_DIR, "config.json")) else config.MODEL_NAME
            
            print(f"Loading Linear Predictor checkpoint: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)

        self.model.to(config.DEVICE)
        self.model.eval()

    def predict(self, text):
        """Predicts named entities for raw Malayalam input text."""
        # Split Malayalam text into words without breaking Indic glyph clusters
        words = re.findall(r'[^\s.,!?\(\)\x22\x27\-]+|[.,!?\(\)\x22\x27\-]', text)
        if not words:
            return [], []

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(config.DEVICE)
        attention_mask = encoding["attention_mask"].to(config.DEVICE)
        word_ids = encoding.word_ids(batch_index=0)

        word_tag_map = []
        previous_word_idx = None

        with torch.no_grad():
            if config.USE_CRF:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds_list = outputs.predictions[0]
                emissions = outputs.logits
                probs = torch.softmax(emissions, dim=-1).squeeze(0).cpu().numpy()
                
                for idx, word_idx in enumerate(word_ids):
                    if word_idx is not None and word_idx != previous_word_idx:
                        if word_idx < len(words) and idx < len(preds_list):
                            tag_id = preds_list[idx]
                            tag = config.ID2LABEL.get(tag_id, "O")
                            confidence = float(np.max(probs[idx]))
                            word_tag_map.append({
                                "word": words[word_idx],
                                "tag": tag,
                                "confidence": round(confidence, 4)
                            })
                        previous_word_idx = word_idx
            else:
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_tag_ids = torch.argmax(probabilities, dim=-1).squeeze(0).cpu().numpy()
                confidence_scores = torch.max(probabilities, dim=-1).values.squeeze(0).cpu().numpy()

                for idx, word_idx in enumerate(word_ids):
                    if word_idx is not None and word_idx != previous_word_idx:
                        if word_idx < len(words):
                            tag_id = predicted_tag_ids[idx]
                            tag = config.ID2LABEL.get(tag_id, "O")
                            confidence = float(confidence_scores[idx])
                            word_tag_map.append({
                                "word": words[word_idx],
                                "tag": tag,
                                "confidence": round(confidence, 4)
                            })
                        previous_word_idx = word_idx
        # --- Option B: Hybrid Rule-Based (Regex) Overlay ---
        months_pattern = re.compile(
            r"^(ജനുവരി|ഫെബ്രുവരി|മാർച്ച്|ഏപ്രിൽ|മേയ്|ജൂൺ|ജൂലൈ|ഓഗസ്റ്റ്|സെപ്റ്റംബർ|ഒക്ടോബർ|നവംബർ|ഡിസംബർ)$"
        )
        cardinal_pattern = re.compile(
            r"^[\d%\.\+\-,\u0D66-\u0D6F]+(?:ശതമാനം|രൂപ|ജിബി|ലക്ഷം|കോടി|വർഷം|മാസം|ദിവസം)?$"
        )

        for i, item in enumerate(word_tag_map):
            word = item["word"]
            # Skip if neural network already predicted a strong entity (PER, ORG, LOC)
            if item["tag"] != "O":
                continue
            
            # Check for months -> DATE
            if months_pattern.match(word):
                item["tag"] = "B-DATE"
                item["confidence"] = 1.0
                continue
                
            # Check for numeric patterns
            if cardinal_pattern.match(word) or any(c.isdigit() for c in word):
                # Is it adjacent to a month? (e.g. "8 ഓഗസ്റ്റ്" or "ഓഗസ്റ്റ് 8") -> DATE
                is_date = False
                for offset in [-2, -1, 1, 2]:
                    target_idx = i + offset
                    if 0 <= target_idx < len(word_tag_map):
                        adj_word = word_tag_map[target_idx]["word"]
                        if months_pattern.match(adj_word):
                            is_date = True
                            break
                
                if is_date:
                    item["tag"] = "I-DATE" if i > 0 and word_tag_map[i-1]["tag"] in ["B-DATE", "I-DATE"] else "B-DATE"
                else:
                    item["tag"] = "B-CARDINAL"
                item["confidence"] = 1.0

        # Group into entity spans
        entities = []
        current_entity = None

        for item in word_tag_map:
            word = item["word"]
            tag = item["tag"]
            conf = item["confidence"]

            if tag.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                ent_type = tag.split("-")[1]
                current_entity = {
                    "text": word,
                    "type": ent_type,
                    "confidence": conf,
                    "words": [word]
                }
            elif tag.startswith("I-") and current_entity and tag.split("-")[1] == current_entity["type"]:
                current_entity["words"].append(word)
                current_entity["text"] += " " + word
                current_entity["confidence"] = round((current_entity["confidence"] + conf) / 2, 4)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return word_tag_map, entities


if __name__ == "__main__":
    predictor = MalayalamNERPredictor()
    sample_text = "കോൺഗ്രസ് സംസ്ഥാന അധ്യക്ഷൻ ജി പരമേശ്വര ഉപമുഖ്യമന്ത്രിയാകും ."
    word_map, entities = predictor.predict(sample_text)
    
    print(f"Sample Input: {sample_text}\n")
    print("Extracted Entities:")
    for ent in entities:
        print(f" - [{ent['type']}] {ent['text']} (Confidence: {ent['confidence'] * 100:.1f}%)")
