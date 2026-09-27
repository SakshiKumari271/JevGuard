import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import onnxruntime as ort
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(
    page_title="JevGuard | System-1 AI Arbiter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
ONNX_PATH = "weights/jevguard_engine.onnx"


@st.cache_resource
def load_engine():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    os.makedirs("weights", exist_ok=True)

    # Cloud par agar file na ho toh on-the-fly export karega
    if not os.path.exists(ONNX_PATH):
        with st.spinner("Downloading weights & initializing ONNX engine..."):
            pt_model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_ID
            )
            pt_model.eval()
            dummy_input = tokenizer(
                "context",
                "claim",
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

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        ONNX_PATH, sess_options, providers=["CPUExecutionProvider"]
    )
    return tokenizer, session