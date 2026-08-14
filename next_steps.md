# Project Next Steps (Completed Tasks & Future Goals)

## Completed Core Roadmap
- [x] **1. Clickable Wikipedia/Search Links (Highly Interactive)**
  - **What**: Automatically generate Malayalam Wikipedia and Google Search links for detected `PER`, `LOC`, and `ORG` entities in the UI.
  - **Why**: Boosts user experience and makes the extraction results immediately actionable.
- [x] **2. Batch File Uploader (TXT / CSV Processor)**
  - **What**: Added a dedicated tab to process entire text files, aggregate unique entity frequencies, and export a structured CSV report.
  - **Why**: Demonstrates pipeline throughput and processing utility for bulk datasets.
- [x] **3. Gradio Entity Distribution Visualizations**
  - **What**: Integrated `gr.BarPlot` to render interactive entity class distribution charts (color-coded, starting from zero).
  - **Why**: Adds a clean analytical dashboard layer that presents well during live project reviews.
- [x] **4. FastAPI REST API (server.py)**
  - **What**: Exposed a robust POST `/predict` endpoint returning JSON representations of word-level tags and parsed entity spans.
  - **Why**: Transforms the codebase into a production-ready microservice.

---

## 🚀 Recommended Next Steps (Training-Free / Run on Current Model)


### 📄 1. PDF / DOCX Document Entity Highlighter
- **What**: Extend the batch tab so users can upload standard formats (like `.pdf` or `.docx`). The backend extracts raw text, runs predictions with the *current* model, and returns a formatted HTML/markdown view with highlighted entities.
- **Why**: High utility feature for real-world document processing (legal documents, news archives, reports).

### 🕸️ 2. Entity Co-occurrence Network Graph (Relation Mapping)
- **What**: When processing a document, detect which entities appear together in the same sentences (e.g., `പിണറായി വിജയൻ` [PER] and `തിരുവനന്തപുരം` [LOC]). Use a simple visualization library (like `networkx` or `matplotlib`) to render a "Relation Map" of connected entities.
- **Why**: Looks incredibly advanced and visually stunning for presentations. Professors love relationship maps.

### 📊 3. Current Model Performance Evaluation Dashboard
- **What**: Run the existing model over the `dataset/test.csv` file. Generate a dashboard showing its final accuracy, F1-scores, precision, and recall, along with a list of "hard sentences" where the model made errors.
- **Why**: Demonstrates proper validation and testing rigor without requiring any training steps.

### ⚡ 4. CPU Inference Speed Optimization (ONNX Conversion)
- **What**: Export the existing PyTorch model to ONNX format. Use ONNX Runtime to run CPU inferences. Benchmark the speed of ONNX vs. native PyTorch on your CPU.
- **Why**: Real-world deployment on CPU is often done using ONNX to reduce latency and memory overhead without any re-training.
