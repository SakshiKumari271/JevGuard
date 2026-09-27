import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import onnxruntime as ort
import streamlit as st
from transformers import AutoTokenizer

# Page Config
st.set_page_config(
    page_title="JevGuard | System-1 AI Arbiter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom High-End Cyberpunk / Modern Dark CSS
st.markdown(
    """
<style>
    .reportview-container, .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #161b22 !important;
        color: #f0f6fc !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 12px;
    }
    .alert-box {
        border-radius: 8px;
        padding: 16px;
        font-weight: 600;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .halt-alert {
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        color: #ff7b72;
    }
    .safe-alert {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        color: #3fb950;
    }
    .warn-alert {
        background: rgba(210, 153, 34, 0.15);
        border: 1px solid #d29922;
        color: #e3b341;
    }
    .telemetry-chip {
        display: inline-block;
        background: #21262d;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 12px;
        border: 1px solid #30363d;
        color: #58a6ff;
        margin-right: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
ONNX_PATH = "weights/jevguard_engine.onnx"


@st.cache_resource
def load_engine():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        ONNX_PATH, sess_options, providers=["CPUExecutionProvider"]
    )
    return tokenizer, session


tokenizer, ort_session = load_engine()

# Header
st.markdown(
    """
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
    <h1 style="margin: 0; font-size: 2.2rem; color: #f0f6fc;">🛡️ JevGuard</h1>
    <span class="telemetry-chip">System-1 Arbiter</span>
    <span class="telemetry-chip">ONNX INT8 Multi-Core</span>
    <span class="telemetry-chip">Sub-50ms Target</span>
</div>
<p style="color: #8b949e; margin-top: -5px; margin-bottom: 25px;">
    Autonomous in-flight hallucination interceptor & circuit-breaker for RAG inference pipelines.
</p>
""",
    unsafe_allow_html=True,
)

# Presets for fast testing
st.markdown("##### ⚡ Quick Presets (Click to Auto-fill):")
preset_cols = st.columns(3)

default_ctx = "SpaceX was founded in 2002 by Elon Musk with the goal of reducing space transportation costs to enable the colonization of Mars."
default_claim = "SpaceX was founded by Jeff Bezos in 1994."

if "context_val" not in st.session_state:
    st.session_state.context_val = default_ctx
if "claim_val" not in st.session_state:
    st.session_state.claim_val = default_claim

with preset_cols[0]:
    if st.button("🚨 Test Hallucination (Lie)", use_container_width=True):
        st.session_state.claim_val = "SpaceX was founded by Jeff Bezos in 1994."
with preset_cols[1]:
    if st.button("✅ Test Factual Match (Truth)", use_container_width=True):
        st.session_state.claim_val = "Elon Musk established SpaceX in 2002."
with preset_cols[2]:
    if st.button("⚠️ Test Unverified Speculation", use_container_width=True):
        st.session_state.claim_val = (
            "SpaceX rockets are the coolest machines ever built in history."
        )

st.write("")

# Main Interface
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown("#### 1. Retrieved Document Context")
    ctx_text = st.text_area(
        "Source Knowledge Base:",
        value=st.session_state.context_val,
        height=130,
        help="Ground truth chunk fetched from Vector DB.",
    )

    st.markdown("#### 2. Generated Token Stream")
    claim_text = st.text_input(
        "In-Flight LLM Sentence:",
        value=st.session_state.claim_val,
        help="Streaming output from generative LLM to be intercepted.",
    )

    arbitrate_btn = st.button(
        "⚡ Intercept & Verify Sentence", type="primary", use_container_width=True
    )

with col_right:
    st.markdown("#### 3. Real-Time Telemetry & Judgment")

    # Run automatically on button click OR first load
    if arbitrate_btn or claim_text:
        t0 = time.perf_counter()

        inputs = tokenizer(
            ctx_text,
            claim_text,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="np",
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }

        outputs = ort_session.run(None, ort_inputs)
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        latency_ms = (time.perf_counter() - t0) * 1000

        # Mapping: 0: Contradiction, 1: Entailment, 2: Neutral
        p_contra = float(probs[0])
        p_entail = float(probs[1])
        p_neut = float(probs[2])

        # Latency Metric Cards
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric("Judgment Latency", f"{latency_ms:.2f} ms")
        with m_col2:
            verdict_badge = (
                "PASS"
                if p_entail > 0.6
                else ("HALT" if p_contra > 0.5 else "WARN")
            )
            st.metric("System-1 Verdict", verdict_badge)

        # Decision Banners
        if p_contra > 0.50:
            st.markdown(
                f"""
            <div class="alert-box halt-alert">
                🛑 <div><b>CIRCUIT BREAKER TRIGGERED</b><br>
                <small>Direct contradiction detected (Confidence: {p_contra*100:.1f}%). Text generation stream severed.</small></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif p_entail > 0.60:
            st.markdown(
                f"""
            <div class="alert-box safe-alert">
                ✅ <div><b>STREAM PERMITTED (VERIFIED)</b><br>
                <small>Full entailment with retrieved context (Confidence: {p_entail*100:.1f}%). Rendering to client.</small></div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
            <div class="alert-box warn-alert">
                ⚠️ <div><b>FLAGGED AS UNVERIFIED SPECULATION</b><br>
                <small>Claim not supported by knowledge base (Neutral: {p_neut*100:.1f}%). Prepending warning chip.</small></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Probability Bars
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Confidence Distribution:**")

        st.caption(f"Entailment (Factual): {p_entail*100:.1f}%")
        st.progress(min(p_entail, 1.0))

        st.caption(f"Contradiction (Hallucination): {p_contra*100:.1f}%")
        st.progress(min(p_contra, 1.0))

        st.caption(f"Neutral (Extrapolation): {p_neut*100:.1f}%")
        st.progress(min(p_neut, 1.0))
        st.markdown("</div>", unsafe_allow_html=True)