import gradio as gr
import pandas as pd
from predict import MalayalamNERPredictor
import config


# Initialize Predictor
predictor = MalayalamNERPredictor()


def process_malayalam_ner(text):
    if not text or not text.strip():
        return [], []

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
            highlighted_output.append((word + " ", ent_type))

    # Format for Entity Data Table
    table_data = []
    for ent in entities:
        text = ent["text"]
        ent_type = ent["type"]
        confidence = f"{ent['confidence'] * 100:.1f}%"
        
        # Generate search links for PER, LOC, ORG in Malayalam
        if ent_type in ["PER", "LOC", "ORG"]:
            import urllib.parse
            encoded_text = urllib.parse.quote(text)
            wiki_url = f"https://ml.wikipedia.org/wiki/Special:Search?search={encoded_text}"
            google_url = f"https://www.google.com/search?q={encoded_text}+മലയാളം"
            
            # Format as clickable Markdown links
            entity_display = f"[{text}]({wiki_url})"
            quick_links = f"[Wikipedia 🌐]({wiki_url}) | [Google 🔍]({google_url})"
        else:
            entity_display = text
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
        class_counts[ent_type] = class_counts.get(ent_type, 0) + 1
        
    df_plot = pd.DataFrame([
        {"Entity Class": k, "Count": v} for k, v in class_counts.items()
    ]) if class_counts else pd.DataFrame(columns=["Entity Class", "Count"])
    
    return highlighted_output, df_results, df_plot


def process_batch_file(file_obj):
    empty_results = pd.DataFrame(columns=["Entity Text", "Entity Class", "Occurrence Count", "Quick Search"])
    empty_plot = pd.DataFrame(columns=["Entity Class", "Total Detections"])
    
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
        for (text, ent_type), count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            if ent_type in ["PER", "LOC", "ORG"]:
                encoded_text = urllib.parse.quote(text)
                wiki_url = f"https://ml.wikipedia.org/wiki/Special:Search?search={encoded_text}"
                google_url = f"https://www.google.com/search?q={encoded_text}+മലയാളം"
                quick_links = f"[Wikipedia 🌐]({wiki_url}) | [Google 🔍]({google_url})"
                entity_display = f"[{text}]({wiki_url})"
            else:
                entity_display = text
                quick_links = "—"
                
            table_data.append({
                "Entity Text": entity_display,
                "Entity Class": ent_type,
                "Occurrence Count": count,
                "Quick Search": quick_links
            })
            
        if not table_data:
            return empty_results, None, "No entities were detected in the uploaded file.", empty_plot
            
        df_results = pd.DataFrame(table_data)
        
        # Save a clean CSV for download
        download_data = []
        for (text, ent_type), count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
            download_data.append({
                "Entity Text": text,
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
        for (text, ent_type), count in entity_counts.items():
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

# Custom CSS for dark modern theme
custom_css = """
.gradio-container { font-family: 'Inter', system-ui, sans-serif; }
#title-header { text-align: center; margin-bottom: 20px; }
"""

with gr.Blocks(title="Malayalam NER System", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🇮🇳 Malayalam Named Entity Recognition (NER) System
        ### Fine-Tuned Transformer-Based Deep Learning Model for Malayalam Entity Extraction
        Extract and classify entities (**Person**, **Organization**, **Location**, etc.) from Malayalam text using **IndicBERTv2 / MuRIL** architecture.
        """
    )

    with gr.Tabs():
        with gr.Tab("📝 Single Text Analysis"):
            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        lines=4,
                        placeholder="ഇവിടെ മലയാളം വാചകം നൽകുക...",
                        label="Input Malayalam Sentence",
                        value="കോൺഗ്രസ് സംസ്ഥാന അധ്യക്ഷൻ ജി പരമേശ്വര ഉപമുഖ്യമന്ത്രിയാകും ."
                    )
                    submit_btn = gr.Button("🔍 Extract Named Entities", variant="primary", size="lg")

                    gr.Examples(
                        examples=sample_examples,
                        inputs=input_text,
                        label="Sample Inputs (Click to test)"
                    )

                with gr.Column(scale=3):
                    gr.Markdown("### Highlighted Named Entities")
                    highlighted_output = gr.HighlightedText(
                        label="Predicted Entities",
                        color_map=config.ENTITY_COLORS,
                        combine_adjacent=True
                    )

                    gr.Markdown("### Extracted Entity Table")
                    entity_table = gr.Dataframe(
                        headers=["Entity Text", "Entity Class", "Confidence Score", "Quick Search"],
                        datatype=["markdown", "str", "str", "markdown"],
                        interactive=False,
                        wrap=True
                    )

                    gr.Markdown("### Entity Distribution")
                    single_plot = gr.BarPlot(
                        x="Entity Class",
                        y="Count",
                        title="Entity Type Distribution",
                        color="Entity Class",
                        height=250
                    )

            submit_btn.click(
                fn=process_malayalam_ner,
                inputs=[input_text],
                outputs=[highlighted_output, entity_table, single_plot]
            )

        with gr.Tab("📂 Batch File Processing"):
            with gr.Row():
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Upload Malayalam Text File (.txt)",
                        file_types=[".txt"],
                        file_count="single"
                    )
                    submit_btn_batch = gr.Button("⚙️ Process Document", variant="primary", size="lg")
                    status_output = gr.Markdown("Upload a .txt file and click 'Process Document' to start.")
                    
                    # Provide link to download template or instruction
                    gr.Markdown("*(Note: The text file should contain one Malayalam sentence/paragraph per line.)*")
                    
                with gr.Column(scale=3):
                    gr.Markdown("### Extracted Entities Summary (Sorted by Occurrences)")
                    batch_entity_table = gr.Dataframe(
                        headers=["Entity Text", "Entity Class", "Occurrence Count", "Quick Search"],
                        datatype=["markdown", "str", "number", "markdown"],
                        interactive=False,
                        wrap=True
                    )
                    
                    csv_download = gr.File(
                        label="📥 Download Extracted Entities CSV Report"
                    )

                    gr.Markdown("### Overall Entity Distribution")
                    batch_plot = gr.BarPlot(
                        x="Entity Class",
                        y="Total Detections",
                        title="Overall Entity Class Distribution",
                        color="Entity Class",
                        height=250
                    )
            
            submit_btn_batch.click(
                fn=process_batch_file,
                inputs=[file_input],
                outputs=[batch_entity_table, csv_download, status_output, batch_plot]
            )

    with gr.Accordion("ℹ️ Model Architecture & Technical Details", open=False):
        gr.Markdown(
            """
            - **Model Encoder**: Pretrained `ai4bharat/IndicBERTv2-MLM-only` / `google/muril-base-cased`
            - **Tokenization**: Subword Fast Alignment with `-100` subword loss masking
            - **Supported Entity Classes**: `PER` (Person), `ORG` (Organization), `LOC` (Location), and all dataset entities.
            - **Dataset**: AI4Bharat / WikiAnn Malayalam Tagged Corpus (~716,000 sentences)
            """
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
