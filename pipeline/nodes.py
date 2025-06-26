"""import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from config import CONFIDENCE_THRESHOLD
import logging

logging.basicConfig(filename="logs/pipeline.log", level=logging.INFO)

model = AutoModelForSequenceClassification.from_pretrained("./model/fine_tuned")
tokenizer = AutoTokenizer.from_pretrained("./model/fine_tuned")


def inference_node(state):
    inputs = tokenizer(state["input"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = softmax(logits, dim=-1)[0]
        confidence, pred_label = torch.max(probs, dim=0)
    label = "Positive" if pred_label.item() == 1 else "Negative"
    state.update({"label": label, "confidence": confidence.item()})
    logging.info(f"Inference: {state['input']} -> {label} ({confidence.item():.2f})")
    return state


def confidence_check_node(state):
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        return "fallback"
    return "accept"


def fallback_node(state):
    print(f"[FallbackNode] Model was {state['confidence']*100:.2f}% confident. Was this {state['label']}?")
    user_input = input("Your answer (Yes/No or actual label): ").strip().lower()
    if user_input in ["no", "negative"]:
        state["label"] = "Negative"
    elif user_input in ["yes", "positive"]:
        state["label"] = "Positive"
    logging.info(f"Fallback used. Final label: {state['label']}")
    return state"""

"""import torch, logging
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from utils.fallback import zero_shot_fallback
from torch.nn.functional import softmax

logging.basicConfig(
    filename="logs/run_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

tokenizer = DistilBertTokenizerFast.from_pretrained("model_output")
model = DistilBertForSequenceClassification.from_pretrained("model_output")
CONF_THRESH = 0.7

def inference_node(state):
    inputs = tokenizer(state["input"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=-1)[0].tolist()
    state["confidence"], state["label_id"] = max(probs), int(torch.argmax(logits, -1))
    state["label"] = ["negative", "positive"][state["label_id"]]
    logging.info(f"Model ➜ '{state['input']}' = {state['label']} ({state['confidence']:.2f})")
    return state

def confidence_check_node(state):
    return "fallback" if state["confidence"] < CONF_THRESH else "accept"

def fallback_node(state):
    logging.info("Fallback activated")
    fallback_label = zero_shot_fallback(state["input"])
    state["label"], state["confidence"], state["fallback_used"] = fallback_label, None, True
    logging.info(f"Fallback ➜ '{state['input']}' = {fallback_label}")
    return state"""
# pipeline/nodes.py
import torch
import logging
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load model and tokenizer (make sure model_output is the correct local path)
model = DistilBertForSequenceClassification.from_pretrained("model_output")
tokenizer = DistilBertTokenizerFast.from_pretrained("model_output")
model.eval()

# Setup logging
logging.basicConfig(filename="logs/run_log.txt", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Inference node

def inference_node(state):
    logging.info("Running inference_node")
    inputs = tokenizer(state.input_text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=1)

    result = {
        "prediction": str(predicted_class.item()),
        "confidence": float(confidence.item())
    }
    logging.info(f"Prediction: {result['prediction']} with confidence {result['confidence']:.2f}")
    return result

# Confidence check node

def confidence_check_node(state):
    logging.info("Running confidence_check_node")
    if state.confidence is not None and state.confidence < 0.7:
        logging.info("Confidence below threshold. Triggering fallback.")
    else:
        logging.info("Confidence above threshold. Proceeding without fallback.")
    return {}

# Fallback node

def fallback_node(state):
    logging.info("Running fallback_node")
    # Replace with a proper fallback strategy. Here we just tag it.
    return {
        "prediction": "fallback_prediction",
        "confidence": 0.5,
        "fallback_used": True
    }

# Test block (optional)
if __name__ == "__main__":
    class DummyState:
        input_text = "The movie was great!"
    print(inference_node(DummyState()))
