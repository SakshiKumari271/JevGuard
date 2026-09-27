import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model import JevGuardClassifier
import torch
from transformers import AutoTokenizer

# 1. Device and Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "microsoft/deberta-v3-small"

print("JevGuard Engine load ho raha hai...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = JevGuardClassifier(model_name=MODEL_NAME).to(device)

# Trained weights load karna
WEIGHTS_PATH = "weights/jevguard_weights.pt"
if os.path.exists(WEIGHTS_PATH):
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    print("Trained weights successfully load ho gaye!")
else:
    print("Warning: Weights file nahi mili, random weights use ho rahe hain.")

model.eval()

# MNLI class index mapping
# 0: Entailment (True), 1: Neutral (Unsupported), 2: Contradiction (Lie)
LABELS = ["ENTAILMENT (Match)", "NEUTRAL (Unverified)", "CONTRADICTION (Lie)"]


def arbitrate(context: str, claim: str):
    start_time = time.time()

    # Tokenize single pair
    inputs = tokenizer(
        context,
        claim,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    # Fast forward pass (Zero gradients)
    with torch.no_grad():
        logits = model(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs.get("token_type_ids", None),
        )
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    latency_ms = (time.time() - start_time) * 1000
    entailment_prob = probs[0].item()
    neutral_prob = probs[1].item()
    contradiction_prob = probs[2].item()

    # System-1 Fast Decision Logic
    if contradiction_prob > 0.40:
        decision = "CIRCUIT_BREAKER_TRIGGERED (BLOCK)"
        action = "Stop text stream immediately!"
    elif entailment_prob > 0.50:
        decision = "ALLOW_PASS"
        action = "Render text to user screen."
    else:
        decision = "FLAG_CAUTION"
        action = "Append warning tag: [Unsupported fact]"

    return {
        "decision": decision,
        "action": action,
        "latency_ms": f"{latency_ms:.2f} ms",
        "scores": {
            "entailment": f"{entailment_prob:.3f}",
            "neutral": f"{neutral_prob:.3f}",
            "contradiction": f"{contradiction_prob:.3f}",
        },
    }


# --- Live Testing Demonstration ---
if __name__ == "__main__":
    # Test Context (Maan lo yeh retrieved document chunk hai)
    sample_context = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 in California."

    print("\n" + "=" * 60)
    print("DOCUMENT CONTEXT:")
    print(sample_context)
    print("=" * 60)

    # Test Cases: Factual Claim vs Hallucinated Claim
    test_claims = [
        "Steve Jobs co-founded Apple in 1976.",  # Entailment
        "Apple was founded by Elon Musk in 2010.",  # Contradiction
        "Apple makes the best smartphones in the world.",  # Neutral / Extrapolation
    ]

    for i, claim in enumerate(test_claims, 1):
        print(f"\n[Test Case {i}] Generated Sentence: \"{claim}\"")
        result = arbitrate(sample_context, claim)
        print(f" -> Decision : {result['decision']}")
        print(f" -> Latency  : {result['latency_ms']}")
        print(f" -> Scores   : {result['scores']}")
        print(f" -> Action   : {result['action']}")