# 🇮🇳 Malayalam Named Entity Recognition (NER) System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/Transformers-IndicBERTv2-yellow.svg)](https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A state-of-the-art Named Entity Recognition (NER) system specifically built for the **Malayalam** language, leveraging modern pre-trained Indic Transformer backbones (**IndicBERTv2** / **MuRIL**), subword token alignment, inverse class-weighted loss, and GPU acceleration.

---

## 🌟 Features

- **Indic Transformer Encoders**: Powered by `ai4bharat/IndicBERTv2-MLM-only` and `google/muril-base-cased`, specifically pre-trained on Indian languages.
- **Subword Alignment & Agglutination Handling**: Solves Malayalam's agglutinative morphology by aligning token subwords with `-100` label masking strategy.
- **GPU Acceleration**: Built-in PyTorch CUDA support with `fp16` mixed-precision training optimized for NVIDIA GPUs (RTX 3050+).
- **Comprehensive Evaluation**: Computes strict entity-level Precision, Recall, and F1-Score using `seqeval`, along with confusion matrix visualizations.
- **Interactive Web App**: Built with **Gradio**, featuring real-time entity prediction, color-coded entity highlighting (`PER`, `ORG`, `LOC`), and structured data extraction.

---

## 📁 Repository Structure

```
├── dataset/
│   ├── train.csv          # Training corpus (~716k Malayalam sentences)
│   ├── validation.csv     # Validation set (~3.6k sentences)
│   └── test.csv           # Benchmark test set (974 sentences)
├── src/
│   ├── __init__.py
│   ├── dataset.py         # PyTorch Dataset loader & fast subword alignment
│   └── model.py           # Transformer model builder & class weight calculator
├── config.py              # Central project configuration & hyperparameter settings
├── train.py               # Fine-tuning engine with mixed precision fp16
├── evaluate.py            # Test evaluation script & confusion matrix plotter
├── predict.py             # Inference predictor for raw Malayalam sentences
├── app.py                 # Gradio web UI application
├── requirements.txt       # Project dependencies
├── .gitignore             # Git exclusion rules
└── README.md              # Project documentation
```

---

## 🏷️ Entity Tag Mapping (IOB2 Scheme)

The dataset uses standard 7-class IOB2 sequence tagging:

| ID | Tag | Description | Color Highlight |
|---|---|---|---|
| `0` | `O` | Outside of any named entity | Default |
| `1` | `B-PER` | Beginning of a Person entity | Blue (`#3B82F6`) |
| `2` | `I-PER` | Inside of a Person entity | Blue (`#3B82F6`) |
| `3` | `B-ORG` | Beginning of an Organization entity | Teal/Emerald (`#10B981`) |
| `4` | `I-ORG` | Inside of an Organization entity | Teal/Emerald (`#10B981`) |
| `5` | `B-LOC` | Beginning of a Location entity | Amber/Gold (`#F59E0B`) |
| `6` | `I-LOC` | Inside of a Location entity | Amber/Gold (`#F59E0B`) |

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Clone the repository and activate virtual environment:

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Fine-Tuning the Model

To train the model on your GPU:

```bash
python train.py
```

The fine-tuned model checkpoint will be saved to `results/malayalam_ner_model`.

### 3. Model Evaluation

Run benchmark evaluation on `test.csv`:

```bash
python evaluate.py
```

Generates entity-level classification metrics and saves the confusion matrix graphic to `results/confusion_matrix.png`.

### 4. Interactive Web Application

Launch the Gradio web application for interactive real-time predictions:

```bash
python app.py
```

Open your browser at `http://127.0.0.1:7860`.

---

## 📚 References & Research Papers

1. *Fine-Tuned BERT-Based Multilingual Model for Named Entity Recognition in Native Indian Languages*
2. *Named Entity Recognition in Malayalam using Fuzzy Support Vector Machine*
3. *Efficient Text Analysis: A BERT-Based Approach to Named Entity Recognition (NER) and Classification for Malayalam Language* (Gopalakrishnan et al., 2024)

---

## 📄 License

This project is licensed under the MIT License.
