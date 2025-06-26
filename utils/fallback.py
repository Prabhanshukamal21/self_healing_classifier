from transformers import pipeline
_zs = None

def zero_shot_fallback(text):
    global _zs
    if _zs is None:
        _zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    out = _zs(text, candidate_labels=["positive", "negative"])
    return out["labels"][0]  # highest score
