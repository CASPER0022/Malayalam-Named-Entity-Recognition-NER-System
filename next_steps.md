# Project Next Steps

- [x] **1. Clickable Wikipedia/Search Links (Highly Interactive)**
  - **What**: In the Gradio web interface, when a name (PER), location (LOC), or organization (ORG) is extracted, dynamically generate a hyperlink that opens its Wikipedia or Google Search page in Malayalam.
  - **Why**: It makes the UI feel like a finished product (e.g., clicking on "പിണറായി വിജയൻ" immediately shows his bio).
  - **Status**: Completed.

- [ ] **2. Batch File Uploader (TXT / CSV Processor)**
  - **What**: Add an upload tab in the Gradio UI allowing users to upload a .txt file of Malayalam text. The system processes the file, counts all entities, and lets the user download a structured CSV report (e.g., "Top mentioned locations in this document").
  - **Why**: Demonstrates practical utility for processing large datasets (like news archives).

- [ ] **3. Gradio Entity Distribution Visualizations**
  - **What**: Use Gradio's built-in plotting component to render a simple bar chart or pie chart of the entity types detected in the text (e.g., 50% Persons, 30% Locations, 20% Dates).
  - **Why**: Adds a clean, visual analytics layer to the dashboard that professors love during presentations.

- [ ] **4. FastAPI REST API (server.py)**
  - **What**: Write a small `server.py` script (~30 lines using FastAPI) that exposes a POST endpoint `/predict`. Other software can send Malayalam text and receive JSON responses containing the extracted entities.
  - **Why**: Shows you know how to build machine learning microservices.
