import html
import os
import tempfile
import urllib.parse

import gradio as gr
import pandas as pd

from predict import MalayalamNERPredictor


# Initialize Predictor
predictor = MalayalamNERPredictor()

ALL_LABELS = ["PER", "ORG", "LOC", "DATE", "CARDINAL"]

# Soft background tint + darker ink per label (displaCy-style inline marks)
LABEL_STYLES = {
    "PER":      {"bg": "#e8e4fd", "ink": "#4c3bb5", "name": "Person"},
    "ORG":      {"bg": "#d9f3f1", "ink": "#11706a", "name": "Organisation"},
    "LOC":      {"bg": "#fdecc8", "ink": "#8a5a00", "name": "Location"},
    "DATE":     {"bg": "#fbe1ea", "ink": "#a3264f", "name": "Date"},
    "CARDINAL": {"bg": "#e3eefc", "ink": "#1f5aa6", "name": "Number"},
}


def _needs_space(word_map, idx):
    """Decide whether a space follows the token at idx when re-joining text."""
    word = word_map[idx]["word"]
    if word in ["-", "/", "(", "[", "{"]:
        return False
    if idx < len(word_map) - 1:
        next_word = word_map[idx + 1]["word"]
        if next_word in ["-", "/", ".", ",", "!", "?", ")", "]", "}"]:
            return False
    return True


def _search_links(text_val, ent_type):
    if ent_type not in ["PER", "LOC", "ORG"]:
        return '<span class="muted">—</span>'
    encoded = urllib.parse.quote(text_val)
    wiki_url = f"https://ml.wikipedia.org/wiki/Special:Search?search={encoded}"
    google_url = f"https://www.google.com/search?q={encoded}+മലയാളം"
    return (
        f'<a href="{wiki_url}" target="_blank" rel="noopener">Wikipedia</a>'
        f'<span class="sep">·</span>'
        f'<a href="{google_url}" target="_blank" rel="noopener">Google</a>'
    )


def _label_tag(ent_type):
    style = LABEL_STYLES.get(ent_type, {"bg": "#eeeeee", "ink": "#333333"})
    return f'<span class="tag" style="background:{style["bg"]};color:{style["ink"]}">{ent_type}</span>'


def render_marked_text(word_map, visible_entities):
    """Render tokens as running text with entity spans wrapped in <mark> elements."""
    if not word_map:
        return '<div class="empty">Entities will appear here once you run the analysis.</div>'

    parts = []
    current_type = None
    buffer = []

    def flush():
        nonlocal buffer, current_type
        if not buffer:
            return
        span_text = "".join(buffer).rstrip()
        trailing = " " if "".join(buffer).endswith(" ") else ""
        style = LABEL_STYLES[current_type]
        parts.append(
            f'<mark class="ent" style="background:{style["bg"]}">{span_text}'
            f'<span class="ent-label" style="color:{style["ink"]}">{current_type}</span></mark>{trailing}'
        )
        buffer = []
        current_type = None

    for idx, item in enumerate(word_map):
        token = html.escape(item["word"]) + (" " if _needs_space(word_map, idx) else "")
        tag = item["tag"]
        ent_type = tag.split("-")[1] if tag != "O" else None
        if ent_type not in visible_entities or ent_type not in LABEL_STYLES:
            ent_type = None

        if ent_type and tag.startswith("I-") and ent_type == current_type:
            buffer.append(token)
            continue

        flush()
        if ent_type:
            current_type = ent_type
            buffer.append(token)
        else:
            parts.append(token)
    flush()

    return f'<div class="marked" lang="ml">{"".join(parts)}</div>'


def render_breakdown(counts):
    """Small horizontal bar breakdown of entity counts per label."""
    if not counts:
        return ""
    total = sum(counts.values())
    peak = max(counts.values())
    rows = []
    for label in ALL_LABELS:
        if label not in counts:
            continue
        style = LABEL_STYLES[label]
        width = counts[label] / peak * 100
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-name">{style["name"]}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.0f}%;background:{style["ink"]}"></span></span>'
            f'<span class="bar-count">{counts[label]}</span>'
            f'</div>'
        )
    noun = "entity" if total == 1 else "entities"
    return (
        f'<div class="breakdown"><div class="section-label">{total} {noun} found</div>'
        f'{"".join(rows)}</div>'
    )


def render_table(headers, rows, numeric_cols=()):
    if not rows:
        return '<div class="empty">No entities match the selected labels.</div>'
    head = "".join(
        f'<th class="{"num" if i in numeric_cols else ""}">{h}</th>' for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"num" if i in numeric_cols else ""}">{cell}</td>' for i, cell in enumerate(row)
        ) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table class="ent-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def process_malayalam_ner(text, visible_entities=None):
    if visible_entities is None:
        visible_entities = ALL_LABELS
    if not text or not text.strip():
        return render_marked_text([], visible_entities), "", ""

    word_map, entities = predictor.predict(text)
    marked = render_marked_text(word_map, visible_entities)

    rows = []
    counts = {}
    for ent in entities:
        ent_type = ent["type"]
        if ent_type not in visible_entities:
            continue
        counts[ent_type] = counts.get(ent_type, 0) + 1
        pct = ent["confidence"] * 100
        confidence = (
            f'<span class="conf"><span class="conf-track"><span class="conf-fill" style="width:{pct:.0f}%"></span></span>'
            f'{pct:.1f}%</span>'
        )
        rows.append([
            f'<span lang="ml">{html.escape(ent["text"])}</span>',
            _label_tag(ent_type),
            confidence,
            _search_links(ent["text"], ent_type),
        ])

    table = render_table(["Entity", "Label", "Confidence", "Look up"], rows)
    return marked, table, render_breakdown(counts)


def process_batch_file(file_obj, visible_entities=None):
    if visible_entities is None:
        visible_entities = ALL_LABELS

    def status(msg, kind="info"):
        return f'<div class="status status-{kind}">{msg}</div>'

    if file_obj is None:
        return "", None, status("Upload a .txt file to begin."), ""

    try:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return "", None, status("The uploaded file is empty.", "error"), ""

        lines = [line.strip() for line in content.split("\n") if line.strip()]

        entity_counts = {}
        for line in lines:
            _, entities = predictor.predict(line)
            for ent in entities:
                key = (ent["text"], ent["type"])
                entity_counts[key] = entity_counts.get(key, 0) + 1

        ranked = [
            (text_val, ent_type, count)
            for (text_val, ent_type), count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            if ent_type in visible_entities
        ]
        if not ranked:
            return "", None, status("No entities matching the selected labels were found.", "error"), ""

        rows = [
            [f'<span lang="ml">{html.escape(t)}</span>', _label_tag(e), str(c), _search_links(t, e)]
            for t, e, c in ranked
        ]
        table = render_table(["Entity", "Label", "Occurrences", "Look up"], rows, numeric_cols=(2,))

        # Save a clean CSV for download
        df_download = pd.DataFrame(
            [{"Entity Text": t, "Entity Class": e, "Occurrence Count": c} for t, e, c in ranked]
        )
        csv_path = os.path.join(tempfile.gettempdir(), "extracted_malayalam_entities.csv")
        df_download.to_csv(csv_path, index=False, encoding="utf-8-sig")

        type_counts = {}
        for _, ent_type, count in ranked:
            type_counts[ent_type] = type_counts.get(ent_type, 0) + count

        summary = status(
            f"Processed <strong>{len(lines)}</strong> lines and found "
            f"<strong>{len(ranked)}</strong> unique entities.",
            "ok",
        )
        return table, csv_path, summary, render_breakdown(type_counts)

    except Exception as e:
        return "", None, status(f"Could not process file: {html.escape(str(e))}", "error"), ""


# Pre-defined sample Malayalam sentences for quick testing (short label, full text)
sample_examples = [
    ("Political news", "കോൺഗ്രസ് സംസ്ഥാന അധ്യക്ഷൻ ജി പരമേശ്വര ഉപമുഖ്യമന്ത്രിയാകും ."),
    ("Public event", "തിരുവനന്തപുരത്ത് നടന്ന ചടങ്ങിൽ കേരള മുഖ്യമന്ത്രി പിണറായി വിജയൻ സംസാരിച്ചു ."),
    ("Space research", "ഇന്ത്യൻ ബഹിരാകാശ ഗവേഷണ സംഘടന ( ഐഎസ്ആർഒ ) ശ്രീഹരിക്കോട്ടയിൽ പുതിയ ഉപഗ്രഹം വിക്ഷേപിച്ചു ."),
    ("Infrastructure", "കൊച്ചി നഗരത്തിൽ കൊച്ചി മെട്രോ രണ്ടാം ഘട്ട നിർമ്മാണം ആരംഭിച്ചു ."),
]

DEFAULT_TEXT = "തിരുവനന്തപുരത്ത് നടന്ന ചടങ്ങിൽ കേരള മുഖ്യമന്ത്രി പിണറായി വിജയൻ സംസാരിച്ചു. 12-04-2023"

LEGEND_HTML = "".join(
    f'<span class="legend-item"><span class="swatch" style="background:{s["bg"]};border-color:{s["ink"]}33"></span>'
    f'<code>{label}</code> {s["name"]}</span>'
    for label, s in LABEL_STYLES.items()
)

HEADER_HTML = f"""
<nav class="topbar">
  <span class="brand"><span class="brand-dot"></span>Malayalam NER</span>
  <span class="nav-links">
    <a href="#model">Model</a>
    <a href="https://huggingface.co/ai4bharat/IndicBERTv2-MLM-only" target="_blank" rel="noopener">IndicBERTv2</a>
  </span>
</nav>
<header class="hero">
  <h1>Named entity recognition for Malayalam</h1>
  <p class="lede">Paste Malayalam text to find the people, organisations, places, dates and numbers in it.
  The tagger is IndicBERTv2 fine-tuned with a CRF layer, with rule-based date and number detection on top.</p>
  <div class="legend">{LEGEND_HTML}</div>
</header>
"""

MODEL_HTML = """
<section id="model" class="model">
  <h2>About the model</h2>
  <dl>
    <dt>Encoder</dt><dd><code>ai4bharat/IndicBERTv2-MLM-only</code>, fine-tuned for sequence labelling</dd>
    <dt>Decoder</dt><dd>Conditional random field (CRF) over BIO tags</dd>
    <dt>Neural labels</dt><dd><code>PER</code>, <code>ORG</code>, <code>LOC</code></dd>
    <dt>Rule-based labels</dt><dd><code>DATE</code> (months, DD-MM-YYYY) and <code>CARDINAL</code> (numbers, units)</dd>
    <dt>Look up</dt><dd>Person, organisation and location entities link to Malayalam Wikipedia and Google search</dd>
  </dl>
</section>
<footer class="foot">Malayalam NER · Semester 7 NLP project</footer>
"""

custom_css = """
:root, .dark { color-scheme: light; }
.gradio-container { max-width: 1040px !important; margin: 0 auto !important; padding: 0 20px 48px !important; }
footer:not(.foot) { display: none !important; }

/* Top bar + hero */
.topbar { display:flex; align-items:center; justify-content:space-between; padding:18px 0; border-bottom:1px solid #ececec; }
.brand { display:flex; align-items:center; gap:10px; font-weight:600; font-size:15px; color:#111; }
.brand-dot { width:10px; height:10px; border-radius:50%; background:#111; }
.nav-links { display:flex; gap:22px; }
.nav-links a { color:#555; text-decoration:none; font-size:14px; }
.nav-links a:hover { color:#111; }
.hero { padding:44px 0 8px; }
.hero h1 { font-size:34px; line-height:1.15; font-weight:600; letter-spacing:-0.02em; color:#111; margin:0 0 12px; }
.hero .lede { font-size:16px; line-height:1.6; color:#555; max-width:640px; margin:0 0 22px; }
.legend { display:flex; flex-wrap:wrap; gap:8px 18px; font-size:13px; color:#555; }
.legend-item { display:inline-flex; align-items:center; gap:6px; }
.legend code, .model code { font-family:'JetBrains Mono', ui-monospace, monospace; font-size:12px; color:#111; }
.swatch { width:12px; height:12px; border-radius:3px; border:1px solid; }

/* Tabs: plain underline style */
.tabs { margin-top:24px; }
.tab-wrapper { border-bottom:1px solid #ececec !important; }
.tab-wrapper button { font-size:14px !important; font-weight:500 !important; color:#666 !important; }
.tab-wrapper button.selected { color:#111 !important; }
.tab-wrapper button.selected::after { background:#111 !important; height:2px !important; }

/* Inputs */
#input-text textarea { font-family:'Noto Sans Malayalam', 'Inter', sans-serif !important; font-size:16px !important; line-height:1.8 !important; }
.samples { gap:8px !important; flex-wrap:wrap !important; align-items:center; }
.samples .samples-label { flex:0 0 auto !important; min-width:0 !important; }
.samples .samples-label p { margin:0 !important; font-size:13px; color:#777; }
.samples button { flex:0 0 auto !important; min-width:0 !important; border-radius:999px !important; font-weight:400 !important;
  font-size:13px !important; padding:4px 12px !important; color:#333 !important;
  border:1px solid #e2e2e2 !important; background:#fff !important; }
.samples button:hover { border-color:#bbb !important; }
.controls { align-items:flex-end !important; gap:16px !important; }
.label-filter label { font-size:13px !important; }
#analyze-btn { max-width:160px; }

/* Output card */
.section-label { font-size:12px; font-weight:500; text-transform:uppercase; letter-spacing:0.06em; color:#888; margin:0 0 10px; }
.output-card { border:1px solid #ececec; border-radius:10px; padding:20px 20px; margin-top:8px; background:#fff; }
.marked { font-family:'Noto Sans Malayalam', sans-serif; font-size:18px; line-height:2.5; color:#1a1a1a; }
mark.ent { padding:0.3em 0.45em; margin:0 0.1em; border-radius:5px; color:inherit; box-decoration-break:clone; -webkit-box-decoration-break:clone; }
.ent-label { font-family:'Inter', sans-serif; font-size:10.5px; font-weight:600; letter-spacing:0.04em; margin-left:0.5em; vertical-align:middle; }
.empty { color:#999; font-size:14px; padding:12px 0; }

/* Table */
.table-wrap { overflow-x:auto; max-height:560px; overflow-y:auto; }
.ent-table { width:100%; border-collapse:collapse; font-size:14px; border:none !important; margin:0 !important; }
.ent-table tr { border:none !important; }
.ent-table th, .ent-table td { border-left:none !important; border-right:none !important; border-top:none !important; }
.ent-table th { text-align:left; font-weight:500; color:#888; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;
  padding:8px 8px; border-bottom:1px solid #ececec !important; background:none !important; white-space:nowrap; }
.ent-table td { padding:10px 8px; border-bottom:1px solid #f3f3f3 !important; color:#1a1a1a; vertical-align:middle; }
.ent-table td [lang="ml"] { font-family:'Noto Sans Malayalam', sans-serif; font-size:15px; }
.ent-table tr:last-child td { border-bottom:none !important; }
.ent-table td:last-child, .conf { white-space:nowrap; }
.ent-table .num { text-align:right; font-variant-numeric:tabular-nums; }
.ent-table a { color:#1f5aa6; text-decoration:none; }
.ent-table a:hover { text-decoration:underline; }
.ent-table .sep { color:#ccc; margin:0 8px; }
.tag { display:inline-block; font-family:'JetBrains Mono', ui-monospace, monospace; font-size:11px; font-weight:600;
  padding:2px 7px; border-radius:4px; }
.conf { display:inline-flex; align-items:center; gap:10px; font-variant-numeric:tabular-nums; color:#444; }
.conf-track { width:44px; height:4px; background:#f0f0f0; border-radius:2px; overflow:hidden; }
.conf-fill { display:block; height:100%; background:#111; }
.muted { color:#bbb; }

/* Breakdown bars */
.breakdown { padding-top:2px; }
.bar-row { display:grid; grid-template-columns:90px 1fr 24px; align-items:center; gap:12px; padding:7px 0; font-size:13px; color:#444; }
.bar-track { height:6px; background:#f3f3f3; border-radius:3px; overflow:hidden; }
.bar-fill { display:block; height:100%; border-radius:3px; }
.bar-count { text-align:right; font-variant-numeric:tabular-nums; color:#111; font-weight:500; }

/* Batch status */
.status { font-size:14px; padding:10px 14px; border-radius:8px; background:#f7f7f7; color:#444; }
.status-ok { background:#effaf3; color:#1e6b3a; }
.status-error { background:#fdf0f0; color:#a12b2b; }

/* Model section + footer */
.model { margin-top:56px; padding-top:32px; border-top:1px solid #ececec; }
.model h2 { font-size:20px; font-weight:600; color:#111; margin:0 0 18px; letter-spacing:-0.01em; }
.model dl { display:grid; grid-template-columns:180px 1fr; gap:12px 24px; margin:0; font-size:14px; }
.model dt { color:#888; }
.model dd { margin:0; color:#222; }
.foot { margin-top:48px; padding-top:20px; border-top:1px solid #ececec; font-size:13px; color:#999; }

@media (max-width: 640px) {
  .hero h1 { font-size:26px; }
  .model dl { grid-template-columns:1fr; gap:4px; }
  .model dd { margin-bottom:12px; }
  .output-card { padding:18px; }
}
"""

theme = gr.themes.Base(
    primary_hue="zinc",
    neutral_hue="zinc",
    radius_size="md",
    font=[gr.themes.GoogleFont("Inter"), gr.themes.GoogleFont("Noto Sans Malayalam"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#ffffff",
    body_text_color="#1a1a1a",
    background_fill_primary="#ffffff",
    background_fill_secondary="#fafafa",
    block_background_fill="#ffffff",
    block_border_color="#ececec",
    block_border_width="1px",
    block_shadow="none",
    block_radius="10px",
    block_label_background_fill="#ffffff",
    block_label_text_color="#666666",
    block_label_text_weight="500",
    block_title_text_color="#666666",
    block_title_text_weight="500",
    border_color_primary="#ececec",
    input_background_fill="#ffffff",
    input_border_color="#e2e2e2",
    input_border_color_focus="#111111",
    input_shadow="none",
    input_shadow_focus="0 0 0 3px rgba(17,17,17,0.06)",
    button_primary_background_fill="#111111",
    button_primary_background_fill_hover="#333333",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#111111",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_hover="#f5f5f5",
    button_secondary_border_color="#e2e2e2",
    button_secondary_text_color="#222222",
    button_primary_shadow="none",
    button_primary_shadow_hover="none",
    button_secondary_shadow="none",
    button_secondary_shadow_hover="none",
    checkbox_label_background_fill="#ffffff",
    checkbox_label_background_fill_hover="#f7f7f7",
    checkbox_label_background_fill_selected="#f4f4f5",
    checkbox_label_border_color="#e2e2e2",
    checkbox_label_border_color_selected="#111111",
    checkbox_label_text_color_selected="#111111",
    checkbox_background_color_selected="#111111",
    checkbox_border_color_selected="#111111",
)

# Keep the page in light mode regardless of the visitor's OS setting:
# mirror every light theme variable into its dark counterpart
for _attr in list(vars(theme)):
    if _attr.endswith("_dark") and hasattr(theme, _attr[: -len("_dark")]):
        setattr(theme, _attr, getattr(theme, _attr[: -len("_dark")]))

with gr.Blocks(title="Malayalam NER") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_classes="tabs"):
        with gr.Tab("Analyze text"):
            input_text = gr.Textbox(
                lines=5,
                placeholder="Type or paste Malayalam text…",
                show_label=False,
                value=DEFAULT_TEXT,
                elem_id="input-text",
                container=False,
            )
            with gr.Row(elem_classes="samples"):
                gr.Markdown("Try an example:", elem_classes="samples-label")
                sample_buttons = [
                    (gr.Button(label, size="sm", variant="secondary"), text) for label, text in sample_examples
                ]
            with gr.Row(elem_classes="controls"):
                visible_entities_single = gr.CheckboxGroup(
                    choices=ALL_LABELS,
                    value=ALL_LABELS,
                    label="Show labels",
                    elem_classes="label-filter",
                    scale=4,
                )
                submit_btn = gr.Button("Analyze", variant="primary", elem_id="analyze-btn", scale=1)

            with gr.Column(elem_classes="output-card"):
                gr.HTML('<div class="section-label">Result</div>')
                marked_output = gr.HTML(render_marked_text([], ALL_LABELS))

            with gr.Row(equal_height=False):
                with gr.Column(scale=2, elem_classes="output-card"):
                    gr.HTML('<div class="section-label">Entities</div>')
                    entity_table = gr.HTML()
                with gr.Column(scale=1, elem_classes="output-card"):
                    single_breakdown = gr.HTML()

        with gr.Tab("Process a document"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    file_input = gr.File(
                        label="Malayalam document (.txt, one sentence per line)",
                        file_types=[".txt"],
                        file_count="single",
                    )
                    visible_entities_batch = gr.CheckboxGroup(
                        choices=ALL_LABELS,
                        value=ALL_LABELS,
                        label="Show labels",
                        elem_classes="label-filter",
                    )
                    submit_btn_batch = gr.Button("Process document", variant="primary")
                    status_output = gr.HTML('<div class="status">Upload a .txt file to begin.</div>')
                    batch_breakdown = gr.HTML()
                with gr.Column(scale=3):
                    with gr.Column(elem_classes="output-card"):
                        gr.HTML('<div class="section-label">Extracted entities</div>')
                        batch_entity_table = gr.HTML('<div class="empty">Results will be listed here.</div>')
                    csv_download = gr.File(label="Download CSV")

    gr.HTML(MODEL_HTML)

    # Event Handlers
    single_outputs = [marked_output, entity_table, single_breakdown]
    submit_btn.click(process_malayalam_ner, [input_text, visible_entities_single], single_outputs, show_progress="minimal")
    visible_entities_single.change(process_malayalam_ner, [input_text, visible_entities_single], single_outputs, show_progress="minimal")
    demo.load(process_malayalam_ner, [input_text, visible_entities_single], single_outputs, show_progress="minimal")

    for button, text in sample_buttons:
        button.click(lambda t=text: t, None, input_text).then(
            process_malayalam_ner, [input_text, visible_entities_single], single_outputs, show_progress="minimal"
        )

    batch_outputs = [batch_entity_table, csv_download, status_output, batch_breakdown]
    submit_btn_batch.click(process_batch_file, [file_input, visible_entities_batch], batch_outputs, show_progress="minimal")
    visible_entities_batch.change(process_batch_file, [file_input, visible_entities_batch], batch_outputs, show_progress="minimal")

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=theme,
        css=custom_css,
    )
