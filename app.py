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
    return highlighted_output, df_results


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

    submit_btn.click(
        fn=process_malayalam_ner,
        inputs=[input_text],
        outputs=[highlighted_output, entity_table]
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
