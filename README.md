# 🛡️ JevGuard: Real-Time Hallucination Arbiter

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jevguard-5.streamlit.app)
[![CI/CD Pipeline](https://github.com/SakshiKumari271/JevGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/SakshiKumari271/JevGuard/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**JevGuard** is a low-latency **System-1 Decision Arbiter** designed to intercept factual hallucinations in Retrieval-Augmented Generation (RAG) pipelines in real time. Unlike LLM-as-a-judge evaluators that introduce 2–5 seconds of latency, JevGuard verifies or halts token streams against retrieved context chunks with sub-150ms inference times.

---

## 💡 Origin: The Jevons Paradox in AI

The project is named after the **Jevons Paradox** (formulated by economist William Stanley Jevons):
> *As technological progress increases the efficiency with which a resource is used, total consumption of that resource tends to rise rather than fall.*

In modern enterprise AI, as LLM inference costs and latency drop, consumption scales exponentially—and with it, the blast radius of factual hallucinations. Inspired by Daniel Kahneman’s **System-1 (Fast, Instinctive Reflex)** paradigm, JevGuard avoids heavyweight autoregressive evaluation loops, executing direct cross-attention classification to stop ungrounded claims before they reach end users.

---

## ⚡ Key Capabilities

- **Sub-150ms Reflex:** Delivers ~51ms inference latency on standard multi-threaded CPU environments.
- **Circuit-Breaker Pattern:** Triggers an immediate execution **HALT** whenever contradiction probability exceeds threshold ($P_C > 0.50$).
- **Tri-State Operational Logic:**
  - `✅ PASS (Entailment)`: Factual claim is grounded in retrieved context.
  - `⚠️ WARN (Neutral)`: Speculative or harmless extrapolations absent from context.
  - `🛑 HALT (Contradiction)`: Factually inconsistent claim detected.
- **Production-Ready Architecture:** Clean separation of concerns (`src/`, `tests/`) backed by GitHub Actions automated regression testing.

---

## 📊 Evaluation Benchmarks

Evaluated across production scenarios in Finance, Medical, and Enterprise Tech:

| Domain | Retrieved Ground Truth Context | Generated LLM Claim | Expected | Verdict | Latency (CPU) | Contradiction % | Entailment % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Finance** | Cloud enterprise subscriptions grew by 35% in Q3. | Enterprise cloud subscriptions declined in Q3. | `HALT` | `🛑 HALT` | **51.2 ms** | 99.9% | 0.0% |
| **Finance** | Q3 revenue reached $4.2B driven by cloud growth. | Cloud enterprise revenue increased in Q3. | `PASS` | `✅ PASS` | **48.6 ms** | 0.0% | 99.8% |
| **Medical** | Patient exhibits elevated liver enzymes with no fever. | Patient is diagnosed with acute bacterial meningitis. | `HALT` | `🛑 HALT` | **53.1 ms** | 98.7% | 0.1% |
| **Tech** | Model training completed in 4 epochs. | Model will be updated again next Monday. | `WARN` | `⚠️ WARN` | **49.8 ms** | 3.4% | 0.1% |

---

## 🏗️ System Architecture

```text
JevGuard/
├── .github/workflows/         # Automated CI/CD pipelines
│   └── ci.yml
├── src/                       # Production core modules
│   ├── __init__.py
│   ├── model.py               # Cross-encoder classification head
│   ├── fast_jevguard.py       # Graph optimization pipeline
│   └── guard_engine.py        # Circuit-breaker arbitration engine
├── tests/                     # Evaluation & test suites
│   ├── __init__.py
│   └── benchmark.py           # Automated benchmark runner
├── app.py                     # Streamlit application entrypoint
├── requirements.txt           # Dependency specifications
└── README.md

🚀 Getting Started
1. Installation
git clone [https://github.com/SakshiKumari271/JevGuard.git](https://github.com/SakshiKumari271/JevGuard.git)
cd JevGuard
pip install -r requirements.txt

2. Run Domain Benchmarks
python tests/benchmark.py

3. Run Streamlit Application
streamlit run app.py

🔗 Project Links
Live Application: https://jevguard-5.streamlit.app

Source Repository: https://github.com/SakshiKumari271/JevGuard

Base Architecture: DeBERTa-v3 NLI Cross-Encoder