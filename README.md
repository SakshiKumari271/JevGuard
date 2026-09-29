# 🛡️ JevGuard: Real-Time Arbitration Harness for Jev (TypeSafe AI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jevguard-5.streamlit.app)
[![CI/CD Pipeline](https://github.com/SakshiKumari271/JevGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/SakshiKumari271/JevGuard/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**JevGuard** is a low-latency, real-time System-1 circuit-breaker runtime built explicitly around **Jev** — the specialized non-generative AI decision model developed by **TypeSafe AI**.

While conversational LLMs are designed to generate text (introducing 2–5 seconds of latency when used as evaluators), **Jev** is architected purely for software systems to output structured decision vectors and probability scores. JevGuard operationalizes Jev to arbitrate streaming RAG outputs against retrieved ground-truth context with sub-150ms execution times.

---

## 🧠 Core Foundation: Jev by TypeSafe AI

Released by TypeSafe AI on September 15, 2026, **Jev** shifts the paradigm from generative text inspection to deterministic, calibrated probability scoring:

- **Non-Generative Output:** Jev does not generate prose or free-form chat tokens. It outputs structured probability triples: $P_{\text{Contradiction}}$, $P_{\text{Entailment}}$, and $P_{\text{Neutral}}$.
- **Software-Native Interfacing:** Purpose-built for direct integration into production software pipelines, automated guardrails, and type-safe systems.
- **System-1 Latency Profile:** Operates at ~51ms CPU inference speeds, bypassing the autoregressive overhead of heavy LLM judges to stop rogue tokens in flight.

---

## ⚡ JevGuard Operational Architecture

JevGuard wraps the Jev decision engine with a zero-delay circuit-breaker pattern:


[Retrieved Ground-Truth Context] ──┐
                                   ├──► [TypeSafe AI: Jev Core Model]
[Streaming RAG Generated Token]  ──┘          │
                                              ▼ Structured Probabilities
                                     [P_C, P_E, P_N Vectors]
                                              │
                                              ▼
                                 [JevGuard Circuit-Breaker Gate]
                                    ├─ P_C > 0.50 ──► 🛑 HALT (Sever Stream)
                                    ├─ P_E > 0.60 ──► ✅ PASS (Verified)
                                    └─ Otherwise  ──► ⚠️ WARN (Speculative)


📊 Evaluation Benchmarks (Jev Decision Engine)

Evaluated across production scenarios in Finance, Medical, and Enterprise Tech:
DomainRetrieved Ground Truth ContextGenerated LLM ClaimCore ModelVerdictLatency (CPU)Contradiction %Entailment %FinanceCloud enterprise subscriptions grew by 35% in Q3.Enterprise cloud subscriptions declined in Q3.TypeSafe Jev🛑 HALT51.2 ms99.9%0.0%FinanceQ3 revenue reached $4.2B driven by cloud growth.Cloud enterprise revenue increased in Q3.TypeSafe Jev✅ PASS48.6 ms0.0%99.8%MedicalPatient exhibits elevated liver enzymes with no fever.Patient is diagnosed with acute bacterial meningitis.TypeSafe Jev🛑 HALT53.1 ms98.7%0.1%TechModel training completed in 4 epochs.Model will be updated again next Monday.TypeSafe Jev⚠️ WARN

🏗️ Repository Architecture
JevGuard/
├── .github/workflows/         # Automated CI/CD regression workflows
│   └── ci.yml
├── src/                       # Jev decision core & arbitration logic
│   ├── __init__.py
│   ├── model.py               # Underlying NLI cross-attention backbone
│   ├── fast_jevguard.py       # Execution graph optimizations
│   └── guard_engine.py        # JevArbiter implementation & probability routing
├── tests/                     # Multi-domain evaluation suite
│   ├── __init__.py
│   └── benchmark.py           # Automated Jev regression harness
├── app.py                     # Streamlit live telemetry dashboard
├── requirements.txt           # Production dependencies
└── README.md

🚀 Quickstart & Reproduction

1. Clone & Install Dependencies
git clone [https://github.com/SakshiKumari271/JevGuard.git](https://github.com/SakshiKumari271/JevGuard.git)
cd JevGuard
pip install -r requirements.txt

2. Run Jev Decision Benchmarks
python tests/benchmark.py

3. Launch Local Telemetry UI
streamlit run app.py

🔗 Live Application & Links
Interactive Telemetry Dashboard: https://jevguard-5.streamlit.app

Source Repository: https://github.com/SakshiKumari271/JevGuard

Core Architecture: TypeSafe AI Jev Decision Calibration Framework
