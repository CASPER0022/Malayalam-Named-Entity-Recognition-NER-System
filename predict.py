import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
import config


class MalayalamNERPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = config.MODEL_SAVE_DIR if os.path.exists(os.path.join(config.MODEL_SAVE_DIR, "config.json")) else config.MODEL_NAME

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(config.DEVICE)
        self.model.eval()

    def predict(self, text):
        """Predicts named entities for raw Malayalam input text."""
        # Simple whitespace / punctuation tokenization for input Malayalam text
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
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

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_tag_ids = torch.argmax(probabilities, dim=-1).squeeze(0).cpu().numpy()
            confidence_scores = torch.max(probabilities, dim=-1).values.squeeze(0).cpu().numpy()

        word_tag_map = []
        previous_word_idx = None

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
