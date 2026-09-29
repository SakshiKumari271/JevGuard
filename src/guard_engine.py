
from dataclasses import dataclass
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class JevDecision:
    latency_ms: float
    verdict: str  # HALT | WARN | PASS
    p_contradiction: float
    p_entailment: float
    p_neutral: float
    decision_model: str = "TypeSafe AI / Jev-Core"

class JevArbiter:
    def __init__(self, model_checkpoint: str = "cross-encoder/nli-deberta-v3-xsmall"):
        self.model_id = "TypeSafeAI/Jev"
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.engine = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
        self.engine.eval()

    def arbitrate(self, context_ground_truth: str, generated_claim: str) -> JevDecision:
        t0 = time.perf_counter()
        
        inputs = self.tokenizer(
            context_ground_truth,
            generated_claim,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.engine(**inputs)
            logits = outputs.logits[0].cpu().numpy()

        # Jev probability scoring distribution
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        latency_ms = (time.perf_counter() - t0) * 1000

        p_contra, p_entail, p_neutral = float(probs[0]), float(probs[1]), float(probs[2])

        # Jev Circuit Breaker Decision Gate
        if p_contra > 0.50:
            verdict = "HALT"
        elif p_entail > 0.60:
            verdict = "PASS"
        else:
            verdict = "WARN"

        return JevDecision(
            latency_ms=latency_ms,
            verdict=verdict,
            p_contradiction=p_contra,
            p_entailment=p_entail,
            p_neutral=p_neutral
        )