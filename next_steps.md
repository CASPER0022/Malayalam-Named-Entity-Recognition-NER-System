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

## 🚀 Recommended Next Steps (Advanced/Portfolio Enhancements)


### ⚡ 1. Model Quantization & Optimization (Inference Speedup)
- **What**: Quantize the IndicBERTv2/MuRIL weights (e.g., using PyTorch's dynamic `INT8` quantization or converting to ONNX runtime) and measure the performance gains.
- **Why**: Proves you know how to reduce cloud hosting costs and optimize model latency (e.g., reducing inference time by 2-3x).

### 📊 2. Comprehensive Model Evaluation & Error Analysis Dashboard
- **What**: Create an evaluation script that calculates Precision, Recall, and F1-scores per entity class (`PER`, `LOC`, `ORG`) on the test dataset, and plot a confusion matrix.
- **Why**: Shows academic and industry rigor. Professors love detailed error analysis (e.g., identifying when the model confuses a Person name with an Organization).

### ✍️ 3. Active Learning & Human-in-the-Loop Entity Corrector
- **What**: Add a feature in the Gradio UI where the user can correct a misclassified entity, saving the corrected sample to a local `feedback.jsonl` file.
- **Why**: Simulates real-world production setups where systems gather user feedback to periodically fine-tune and improve the model over time.

### 📄 4. PDF / DOCX Document Entity Highlighter
- **What**: Upgrade the uploader to extract text directly from PDFs or Word documents, run NER, and highlight entities in a structured text view or generate a downloadable highlighted PDF.
- **Why**: Creates a highly practical tool for document scanning, legal tech, or financial audit use cases.
