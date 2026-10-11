"""Evaluates the Linear (softmax) and CRF checkpoints on the test set and writes results/eval_results.json (numbers used in Report/)."""
import os, sys, json, time
PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import torch
from collections import Counter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import classification_report as sk_report, confusion_matrix
import config
from src.dataset import load_ner_dataframe, MalayalamNERDataset
from src.model import MalayalamBERTCRF, build_ner_model

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJ, "results")
dev = config.DEVICE
recs = load_ner_dataframe(config.TEST_CSV)
tr = load_ner_dataframe(config.TRAIN_CSV, max_samples=config.MAX_TRAIN_SAMPLES)
va = load_ner_dataframe(config.VAL_CSV)

def stats(records):
    c = Counter(t for r in records for t in r["ner_tags"])
    ents = Counter(config.ID2LABEL[t][2:] for r in records for t in r["ner_tags"] if config.ID2LABEL[t].startswith("B-"))
    return {"sentences": len(records), "tokens": sum(c.values()),
            "tag_counts": {config.ID2LABEL[k]: v for k, v in sorted(c.items())},
            "entities": dict(ents),
            "avg_len": sum(len(r["tokens"]) for r in records) / max(len(records), 1)}

result = {"stats": {"train_subset": stats(tr), "val": stats(va), "test": stats(recs)}}
tok = AutoTokenizer.from_pretrained(config.MODEL_NAME)
# subword fertility on test set
nw = ns = 0
for r in recs:
    for w in r["tokens"]:
        nw += 1; ns += len(tok.tokenize(w))
result["fertility"] = ns / nw
loader = DataLoader(MalayalamNERDataset(recs, tok), batch_size=32)

def run(name, model, crf):
    model.to(dev).eval()
    P, L = [], []
    t0 = time.time()
    with torch.no_grad():
        for b in loader:
            ii, am, lab = b["input_ids"].to(dev), b["attention_mask"].to(dev), b["labels"].numpy()
            if crf:
                preds = model(input_ids=ii, attention_mask=am).predictions
            else:
                preds = model(input_ids=ii, attention_mask=am).logits.argmax(-1).cpu().tolist()
            for i in range(len(lab)):
                sp, sl = [], []
                for j in range(len(lab[i])):
                    if lab[i][j] != -100 and j < len(preds[i]):
                        sp.append(config.ID2LABEL[preds[i][j]]); sl.append(config.ID2LABEL[lab[i][j]])
                P.append(sp); L.append(sl)
    el = time.time() - t0
    # count invalid IOB2 transitions (I-X not preceded by B-X / I-X)
    bad = 0
    for s in P:
        prev = "O"
        for t in s:
            if t.startswith("I-") and prev[2:] != t[2:]:
                bad += 1
            prev = t
    flatL = [t for s in L for t in s]; flatP = [t for s in P for t in s]
    tags = [config.ID2LABEL[i] for i in range(config.NUM_LABELS)]
    result[name] = {
        "seqeval": classification_report(L, P, digits=4, output_dict=False),
        "seqeval_dict": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in classification_report(L, P, output_dict=True).items()},
        "token": sk_report(flatL, flatP, labels=tags, digits=4, output_dict=True, zero_division=0),
        "token_acc": sum(a == b for a, b in zip(flatL, flatP)) / len(flatL),
        "cm": confusion_matrix(flatL, flatP, labels=tags).tolist(),
        "invalid_transitions": bad, "seconds": el,
    }
    print(name, "\n", result[name]["seqeval"], "acc", result[name]["token_acc"], "invalid", bad, "time", el, flush=True)

lin = build_ner_model(model_name=config.MODEL_SAVE_DIR, use_crf=False)
run("linear", lin, False)
del lin; torch.cuda.empty_cache()
crf = MalayalamBERTCRF(model_name=config.MODEL_NAME); crf.load_pretrained(config.CRF_MODEL_SAVE_DIR)
run("crf", crf, True)
json.dump(result, open(os.path.join(out_dir, "eval_results.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("fertility", result["fertility"]); print(json.dumps(result["stats"], ensure_ascii=False, indent=1))
