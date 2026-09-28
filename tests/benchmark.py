import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
ONNX_PATH = "weights/jevguard_engine.onnx"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
session = ort.InferenceSession(
    ONNX_PATH, sess_options, providers=["CPUExecutionProvider"]
)

# Test Suite covering different domains
test_suite = [
    {
        "domain": "Finance",
        "context": "NVIDIA revenue for Q4 fiscal 2024 was $22.1 billion, up 265% from a year ago.",
        "claim": "NVIDIA reported over $22 billion in Q4 2024 revenue.",
        "expected": "PASS",
    },
    {
        "domain": "Finance",
        "context": "NVIDIA revenue for Q4 fiscal 2024 was $22.1 billion, up 265% from a year ago.",
        "claim": "NVIDIA Q4 revenue crashed by 50% year-over-year.",
        "expected": "HALT",
    },
    {
        "domain": "Medical",
        "context": "Paracetamol is commonly used to treat fever and mild headaches.",
        "claim": "Paracetamol is an FDA approved cure for malignant diabetes.",
        "expected": "WARN",
    },
    {
        "domain": "SpaceTech",
        "context": "The James Webb Space Telescope operates in a halo orbit around the Sun-Earth L2 Lagrange point.",
        "claim": "JWST orbits the Earth in low Earth orbit just like Hubble.",
        "expected": "HALT",
    },
    {
        "domain": "Legal/Policy",
        "context": "Employees are entitled to 20 days of paid annual leave after completing one year of service.",
        "claim": "New employees get 20 days of paid leave immediately on day one.",
        "expected": "HALT",
    },
]

print("\n" + "=" * 95)
print(
    f"{'Domain':<12} | {'Expected':<8} | {'Verdict':<8} | {'Latency':<10} | {'Contradiction':<14} | {'Entailment':<12} | Status"
)
print("=" * 95)

latencies = []
correct_matches = 0

for test in test_suite:
    t0 = time.perf_counter()

    inputs = tokenizer(
        test["context"],
        test["claim"],
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="np",
    )

    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }

    outputs = session.run(None, ort_inputs)
    logits = outputs[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()

    dt = (time.perf_counter() - t0) * 1000
    latencies.append(dt)

    p_contra = probs[0]
    p_entail = probs[1]
    p_neut = probs[2]

    if p_contra > 0.50:
        verdict = "HALT"
    elif p_entail > 0.60:
        verdict = "PASS"
    else:
        verdict = "WARN"

    match = "MATCH" if verdict == test["expected"] else "MISMATCH"
    if match == "MATCH":
        correct_matches += 1

    print(
        f"{test['domain']:<12} | {test['expected']:<8} | {verdict:<8} | {dt:>6.2f} ms  | {p_contra*100:>12.1f}% | {p_entail*100:>10.1f}% | {match}"
    )

print("=" * 95)
print(f"Total Benchmark Accuracy : {(correct_matches / len(test_suite))*100:.1f}%")
print(f"Average CPU Latency      : {np.mean(latencies):.2f} ms")
print(f"P95 Latency              : {np.percentile(latencies, 95):.2f} ms")
print("=" * 95)