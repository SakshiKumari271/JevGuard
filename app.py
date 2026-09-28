import os
import time

import numpy as np
import streamlit as st

st.set_page_config(
    page_title="JevGuard | System-1 AI Arbiter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("🛡️ JevGuard: Real-Time Hallucination Arbiter")
st.caption(
    "Sub-150ms System-1 Cross-Encoder arbitration engine for RAG pipelines."
)

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"


@st.cache_resource(show_spinner="Loading JevGuard Neural Engine...")
def get_engine():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return tokenizer, model


try:
    tokenizer, model = get_engine()
    st.success("✅ Engine Online & Ready")
except Exception as e:
    st.error(f"Engine Initialization Failed: {e}")
    st.stop()

# Layout Columns
col_ctx, col_claim = st.columns(2)

with col_ctx:
    st.subheader("1. Retrieved Context Chunk")
    default_ctx = "Q3 revenue reached $4.2B, driven by 35% growth in cloud enterprise subscriptions."
    context = st.text_area("Context Ground Truth", value=default_ctx, height=140)

with col_claim:
    st.subheader("2. Generated LLM Claim")
    default_claim = "Enterprise cloud subscriptions declined in Q3."
    claim = st.text_area("Generated Token Stream", value=default_claim, height=140)

if st.button("Evaluate Claim (Run Arbiter)", type="primary"):
    import torch

    t0 = time.perf_counter()
    inputs = tokenizer(
        context,
        claim,
        truncation=True,
        max_length=128,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0].cpu().numpy()

    exp = np.exp(logits - np.max(logits))
    probs = exp / exp.sum()
    latency_ms = (time.perf_counter() - t0) * 1000

    p_contra = probs[0]
    p_entail = probs[1]
    p_neutral = probs[2]

    # Circuit Breaker Logic
    if p_contra > 0.50:
        verdict = "🛑 HALT (Contradiction / Hallucination Detected)"
        v_class = "error"
    elif p_entail > 0.60:
        verdict = "✅ PASS (Faithfully Grounded)"
        v_class = "success"
    else:
        verdict = "⚠️ WARN (Neutral Extrapolation / Speculative)"
        v_class = "warning"

    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latency", f"{latency_ms:.1f} ms")
    m2.metric("Contradiction", f"{p_contra*100:.1f}%")
    m3.metric("Entailment", f"{p_entail*100:.1f}%")
    m4.metric("Neutral", f"{p_neutral*100:.1f}%")

    if v_class == "error":
        st.error(f"### Verdict: {verdict}")
    elif v_class == "success":
        st.success(f"### Verdict: {verdict}")
    else:
        st.warning(f"### Verdict: {verdict}")