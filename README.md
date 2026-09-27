# 🛡️ JevGuard: Real-Time Hallucination Arbiter

JevGuard is a high-speed, System-1 decision arbiter designed to intercept hallucinations in Retrieval-Augmented Generation (RAG) pipelines in real time.



## ⚡ Key Highlights
- **Sub-150ms Latency:** Optimized via ONNX Runtime graph execution.
- **Circuit Breaker:** Automatically halts LLM response generation when contradiction probability $P_C > 0.50$.
- **Tri-State Logic:** Accurately classifies tokens into Entailment (True), Neutral (Speculative), or Contradiction (Hallucination).



## 📊 Benchmark Results

| Domain | Expected | Verdict | Latency (CPU) | Contradiction % | Entailment % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Finance** | `PASS` | `PASS` | 114 ms | 0.0% | 99.8% |
| **Finance** | `HALT` | `HALT` | 120 ms | 99.9% | 0.0% |
| **Medical** | `WARN` | `WARN` | 138 ms | 3.7% | 0.1% |
| **SpaceTech** | `HALT` | `HALT` | 119 ms | 99.7% | 0.1% |

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Export optimized ONNX graph
python fast_jevguard.py

# 3. Launch interactive UI
streamlit run app.py
