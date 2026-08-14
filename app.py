import gradio as gr
import pandas as pd
from predict import MalayalamNERPredictor
import config


# Initialize Predictor
predictor = MalayalamNERPredictor()


def process_malayalam_ner(text, visible_entities=None):
    if not text or not text.strip():
        return [], pd.DataFrame(columns=["Entity Text", "Entity Class", "Confidence Score", "Quick Search"]), pd.DataFrame(columns=["Entity Class", "Count"])

    if visible_entities is None:
        visible_entities = ["PER", "ORG", "LOC", "DATE", "CARDINAL"]

    word_map, entities = predictor.predict(text)

    # Format for Gradio HighlightedText: list of (word, label) tuples
    highlighted_output = []
    for item in word_map:
        word = item["word"]
        tag = item["tag"]
        if tag == "O":
            highlighted_output.append((word + " ", None))
        else:
            ent_type = tag.split("-")[1]
            if ent_type in visible_entities:
                highlighted_output.append((word + " ", ent_type))
            else:
                highlighted_output.append((word + " ", None))

    # Format for Entity Data Table
    table_data = []
    for ent in entities:
        text_val = ent["text"]
        ent_type = ent["type"]
        confidence = f"{ent['confidence'] * 100:.1f}%"
        
        # Skip if not selected in filter
        if ent_type not in visible_entities:
            continue
            
        # Generate search links for PER, LOC, ORG in Malayalam
        if ent_type in ["PER", "LOC", "ORG"]:
            import urllib.parse
            encoded_text = urllib.parse.quote(text_val)
            wiki_url = f"https://ml.wikipedia.org/wiki/Special:Search?search={encoded_text}"
            google_url = f"https://www.google.com/search?q={encoded_text}+മലയാളം"
            
            # Format as clickable Markdown links
            entity_display = f"[{text_val}]({wiki_url})"
            quick_links = f"[Wikipedia 🌐]({wiki_url}) | [Google 🔍]({google_url})"
        else:
            entity_display = text_val
            quick_links = "—"
            
        table_data.append({
            "Entity Text": entity_display,
            "Entity Class": ent_type,
            "Confidence Score": confidence,
            "Quick Search": quick_links
        })

    df_results = pd.DataFrame(table_data) if table_data else pd.DataFrame(columns=["Entity Text", "Entity Class", "Confidence Score", "Quick Search"])
    
    # Calculate distributions
    class_counts = {}
    for ent in entities:
        ent_type = ent["type"]
        if ent_type in visible_entities:
            class_counts[ent_type] = class_counts.get(ent_type, 0) + 1
        
    df_plot = pd.DataFrame([
        {"Entity Class": k, "Count": v} for k, v in class_counts.items()
    ]) if class_counts else pd.DataFrame(columns=["Entity Class", "Count"])
    
    return highlighted_output, df_results, df_plot


def process_batch_file(file_obj, visible_entities=None):
    empty_results = pd.DataFrame(columns=["Entity Text", "Entity Class", "Occurrence Count", "Quick Search"])
    empty_plot = pd.DataFrame(columns=["Entity Class", "Total Detections"])
    
    if visible_entities is None:
        visible_entities = ["PER", "ORG", "LOC", "DATE", "CARDINAL"]
        
    if file_obj is None:
        return empty_results, None, "Please upload a valid text file.", empty_plot
    
    try:
        # Load file content
        with open(file_obj.name, "r", encoding="utf-8") as f:
            content = f.read()
            
        if not content.strip():
            return empty_results, None, "The uploaded file is empty.", empty_plot
            
        # Split into lines/sentences
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        
        entity_counts = {}
        for line in lines:
            _, entities = predictor.predict(line)
            for ent in entities:
                key = (ent["text"], ent["type"])
                entity_counts[key] = entity_counts.get(key, 0) + 1
                
        # Format results for Gradio DataFrame display
        table_data = []
        import urllib.parse
        for (text_val, ent_type), count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            if ent_type not in visible_entities:
                continue
                
            if ent_type in ["PER", "LOC", "ORG"]:
                encoded_text = urllib.parse.quote(text_val)
                wiki_url = f"https://ml.wikipedia.org/wiki/Special:Search?search={encoded_text}"
                google_url = f"https://www.google.com/search?q={encoded_text}+മലയാളം"
                quick_links = f"[Wikipedia 🌐]({wiki_url}) | [Google 🔍]({google_url})"
                entity_display = f"[{text_val}]({wiki_url})"
            else:
                entity_display = text_val
                quick_links = "—"
                
            table_data.append({
                "Entity Text": entity_display,
                "Entity Class": ent_type,
                "Occurrence Count": count,
                "Quick Search": quick_links
            })
            
        if not table_data:
            return empty_results, None, "No entities matching filters were detected in the uploaded file.", empty_plot
            
        df_results = pd.DataFrame(table_data)
        
        # Save a clean CSV for download
        download_data = []
        for (text_val, ent_type), count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            if ent_type not in visible_entities:
                continue
            download_data.append({
                "Entity Text": text_val,
                "Entity Class": ent_type,
                "Occurrence Count": count
            })
        df_download = pd.DataFrame(download_data)
        
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        csv_path = os.path.join(temp_dir, "extracted_malayalam_entities.csv")
        df_download.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        # Calculate entity class total detection count for visualization
        type_counts = {}
        for (text_val, ent_type), count in entity_counts.items():
            if ent_type in visible_entities:
                type_counts[ent_type] = type_counts.get(ent_type, 0) + count
            
        df_batch_plot = pd.DataFrame([
            {"Entity Class": k, "Total Detections": v} for k, v in type_counts.items()
        ])
        
        summary_text = f"✅ Processed {len(lines)} lines successfully. Extracted {len(entity_counts)} unique entities."
        return df_results, csv_path, summary_text, df_batch_plot
        
    except Exception as e:
        return empty_results, None, f"❌ Error processing file: {str(e)}", empty_plot


# Pre-defined sample Malayalam sentences for user quick testing
sample_examples = [
    ["കോൺഗ്രസ് സംസ്ഥാന അധ്യക്ഷൻ ജി പരമേശ്വര ഉപമുഖ്യമന്ത്രിയാകും ."],
    ["തിരുവനന്തപുരത്ത് നടന്ന ചടങ്ങിൽ കേരള മുഖ്യമന്ത്രി പിണറായി വിജയൻ സംസാരിച്ചു ."],
    ["ഇന്ത്യൻ ബഹിരാകാശ ഗവേഷണ സംഘടന ( ഐഎസ്ആർഒ ) ശ്രീഹരിക്കോട്ടയിൽ പുതിയ ഉപഗ്രഹം വിക്ഷേപിച്ചു ."],
    ["കൊച്ചി നഗരത്തിൽ കൊച്ചി മെട്രോ രണ്ടാം ഘട്ട നിർമ്മാണം ആരംഭിച്ചു ."]
]

# Custom CSS for modern displaCy-style theme
custom_css = """
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    max-width: 1200px !important;
    margin: auto;
}
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #312e81 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    text-align: center;
}
.app-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 10px 0;
    letter-spacing: -0.025em;
}
.app-header p {
    font-size: 1rem;
    margin: 0;
    opacity: 0.9;
}
"""

theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(title="Malayalam NER System", css=custom_css, theme=theme) as demo:
    # Custom HTML header matching displaCy banner
    gr.HTML(
        """
        <div class="app-header">
            <h1>🇮🇳 Malayalam Named Entity Visualizer</h1>
            <p>Fine-Tuned Deep Learning Transformer Model for Malayalam Entity Extraction & Discovery</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("📝 Single Text Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        lines=5,
                        placeholder="ഇവിടെ മലയാളം വാചകം നൽകുക (Enter Malayalam sentence here)...",
                        label="Input Text",
                        value="തിരുവനന്തപുരത്ത് നടന്ന ചടങ്ങിൽ കേരള മുഖ്യമന്ത്രി പിണറായി വിജയൻ സംസാരിച്ചു ."
                    )
                    
                    visible_entities_single = gr.CheckboxGroup(
                        choices=["PER", "ORG", "LOC", "DATE", "CARDINAL"],
                        value=["PER", "ORG", "LOC", "DATE", "CARDINAL"],
                        label="Entity Labels to Display"
                    )
                    
                    submit_btn = gr.Button("🔍 Visualise Entities", variant="primary", size="lg")

                    gr.Examples(
                        examples=sample_examples,
                        inputs=input_text,
                        label="Click a sample sentence to load:"
                    )
        
                with gr.Column(scale=3):
                    gr.Markdown("### 🏷️ Visualized Named Entities")
                    highlighted_output = gr.HighlightedText(
                        label="Interactive Entity Highlight",
                        color_map=config.ENTITY_COLORS,
                        combine_adjacent=True
                    )
        
                    gr.Markdown("### 📊 Extracted Entity Details")
                    entity_table = gr.Dataframe(
                        headers=["Entity Text", "Entity Class", "Confidence Score", "Quick Search"],
                        datatype=["markdown", "str", "str", "markdown"],
                        interactive=False,
                        wrap=True
                    )

                    single_plot = gr.BarPlot(
                        x="Entity Class",
                        y="Count",
                        title="Entity Class Count Distribution",
                        color="Entity Class",
                        color_map=config.ENTITY_COLORS,
                        y_lim=[0, None],
                        height=220
                    )
            
            # Allow live prediction when parameters or input changes
            submit_btn.click(
                fn=process_malayalam_ner,
                inputs=[input_text, visible_entities_single],
                outputs=[highlighted_output, entity_table, single_plot]
            )
            visible_entities_single.change(
                fn=process_malayalam_ner,
                inputs=[input_text, visible_entities_single],
                outputs=[highlighted_output, entity_table, single_plot]
            )

        with gr.Tab("📂 Batch File Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload Malayalam Document (.txt)",
                        file_types=[".txt"],
                        file_count="single"
                    )
                    
                    visible_entities_batch = gr.CheckboxGroup(
                        choices=["PER", "ORG", "LOC", "DATE", "CARDINAL"],
                        value=["PER", "ORG", "LOC", "DATE", "CARDINAL"],
                        label="Entity Labels to Filter"
                    )
                    
                    submit_btn_batch = gr.Button("⚙️ Process Document", variant="primary", size="lg")
                    status_output = gr.Markdown("Upload a file to start.")
                    
                with gr.Column(scale=3):
                    gr.Markdown("### 📈 Extracted Entities Summary")
                    batch_entity_table = gr.Dataframe(
                        headers=["Entity Text", "Entity Class", "Occurrence Count", "Quick Search"],
                        datatype=["markdown", "str", "number", "markdown"],
                        interactive=False,
                        wrap=True
                    )
                    
                    csv_download = gr.File(
                        label="📥 Download Extracted Entities CSV Report"
                    )

                    batch_plot = gr.BarPlot(
                        x="Entity Class",
                        y="Total Detections",
                        title="Overall Entity Class Distribution",
                        color="Entity Class",
                        color_map=config.ENTITY_COLORS,
                        y_lim=[0, None],
                        height=220
                    )
            
            submit_btn_batch.click(
                fn=process_batch_file,
                inputs=[file_input, visible_entities_batch],
                outputs=[batch_entity_table, csv_download, status_output, batch_plot]
            )
            visible_entities_batch.change(
                fn=process_batch_file,
                inputs=[file_input, visible_entities_batch],
                outputs=[batch_entity_table, csv_download, status_output, batch_plot]
            )

    with gr.Accordion("ℹ️ Technical Details & Model Architecture", open=False):
        gr.Markdown(
            """
            - **Model Encoder**: Fine-tuned sequence tagger built on `ai4bharat/IndicBERTv2-MLM-only` (with CRF decoder)
            - **Entity Classes**: 
              - `PER`: Persons (e.g. `പിണറായി വിജയൻ`)
              - `ORG`: Organizations (e.g. `ഐഎസ്ആർഒ`)
              - `LOC`: Locations (e.g. `ശ്രീഹരിക്കോട്ട`)
              - `DATE`: Dates & Months (e.g. `ഓഗസ്റ്റ്`)
              - `CARDINAL`: Numeric values and units (e.g. `രൂപ`, `100`)
            - **Interactive Lookup**: Uses dynamically generated links to search Malayalam Wikipedia or Google for instant details.
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
