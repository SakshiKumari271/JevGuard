import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
ONNX_PATH = "weights/jevguard_engine.onnx"
os.makedirs("weights", exist_ok=True)

print("1. Optimized NLI Tokenizer aur Model Load ho raha hai...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Agar ONNX model pehle se nahi bana, toh export karenge
if not os.path.exists(ONNX_PATH):
    print("2. Model ko ONNX High-Speed Format mein export kiya ja raha hai...")
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    pt_model.eval()

    dummy_input = tokenizer(
        "Document context",
        "Generated claim",
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True,
    )

    torch.onnx.export(
        pt_model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        ONNX_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print("ONNX Model successfully export ho gaya!")

# 3. ONNX Runtime Session Initialize (Optimized for Multi-Core CPU)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)
sess_options.intra_op_num_threads = 4  # CPU threads

ort_session = ort.InferenceSession(
    ONNX_PATH, sess_options, providers=["CPUExecutionProvider"]
)
print("3. JevGuard Fast Engine Ready hai!\n")


def fast_arbitrate(context: str, claim: str):
    t0 = time.perf_counter()

    inputs = tokenizer(
        context,
        claim,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="np",
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    # High-speed ONNX Inference
    ort_outputs = ort_session.run(None, ort_inputs)
    logits = ort_outputs[0][0]

    # Numerically stable Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    latency_ms = (time.perf_counter() - t0) * 1000

    # Label indexing for cross-encoder/nli-deberta-v3-xsmall:
    # 0: Contradiction, 1: Entailment, 2: Neutral
    p_contradiction = float(probs[0])
    p_entailment = float(probs[1])
    p_neutral = float(probs[2])

    if p_contradiction > 0.60:
        decision = "CIRCUIT_BREAKER (REJECT/HALT)"
        action = "Stop text generation! Hallucination caught."
    elif p_entailment > 0.60:
        decision = "SAFE (ALLOW)"
        action = "Emit token stream to user screen."
    else:
        decision = "UNVERIFIED (FLAG)"
        action = "Tag output as unverified speculation."

    return {
        "decision": decision,
        "latency": f"{latency_ms:.2f} ms",
        "scores": {
            "entailment": round(p_entailment, 3),
            "contradiction": round(p_contradiction, 3),
            "neutral": round(p_neutral, 3),
        },
        "action": action,
    }


# --- Verification Tests ---
if __name__ == "__main__":
    context = "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs to enable the colonization of Mars."

    print("=" * 65)
    print("CONTEXT:", context)
    print("=" * 65)

    test_claims = [
        "Elon Musk established SpaceX in 2002.",  # Sahi fact
        "SpaceX was founded by Jeff Bezos in 1994.",  # Jhooth fact
        "SpaceX rockets are the coolest machines ever built.",  # Subjective claim
    ]

    for idx, claim in enumerate(test_claims, 1):
        res = fast_arbitrate(context, claim)
        print(f"\n[Claim {idx}]: \"{claim}\"")
        print(f" -> Decision: {res['decision']}")
        print(f" -> Latency : {res['latency']}")
        print(f" -> Scores  : {res['scores']}")
        print(f" -> Action  : {res['action']}")
        