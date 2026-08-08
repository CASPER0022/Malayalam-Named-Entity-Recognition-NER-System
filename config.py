import os
import torch

# Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
MODEL_SAVE_DIR = os.path.join(BASE_DIR, "results", "malayalam_ner_model")
CRF_MODEL_SAVE_DIR = os.path.join(BASE_DIR, "results", "malayalam_ner_crf_model")
LOG_DIR = os.path.join(BASE_DIR, "results", "logs")

# Sequence Tagging Model Architecture Selector (Toggle between standard Linear head or CRF head)
USE_CRF = True

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(CRF_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hardware Acceleration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = torch.cuda.is_available()

# Pretrained Model Backbone
# Options: 'ai4bharat/IndicBERTv2-MLM-only', 'google/muril-base-cased', 'bert-base-multilingual-cased'
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"

# Tag Mappings for the Dataset
ID2LABEL = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC"
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

# Color Scheme for Entity Highlighting in Gradio / Web App
ENTITY_COLORS = {
    "PER": "#3B82F6",  # Blue
    "ORG": "#10B981",  # Emerald / Teal
    "LOC": "#F59E0B",  # Amber / Gold
    "MISC": "#8B5CF6", # Purple
    "DATE": "#EC4899", # Pink
    "CARDINAL": "#06B6D4" # Cyan / Teal-blue
}

# Training Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 32 if torch.cuda.is_available() else 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
EPOCHS = 3
SEED = 42
MAX_TRAIN_SAMPLES = 50000  # High convergence accuracy while keeping training swift
