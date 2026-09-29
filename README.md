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


## 📊 Evaluation Benchmarks (Jev Decision Engine)

Evaluated across production scenarios in Finance, Medical, and Enterprise Tech:

<table>
  <thead>
    <tr>
      <th>Domain</th>
      <th>Retrieved Ground Truth Context</th>
      <th>Generated LLM Claim</th>
      <th>Core Model</th>
      <th>Verdict</th>
      <th>Latency (CPU)</th>
      <th>Contradiction %</th>
      <th>Entailment %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Finance</b></td>
      <td>Cloud enterprise subscriptions grew by 35% in Q3.</td>
      <td>Enterprise cloud subscriptions declined in Q3.</td>
      <td><b>TypeSafe Jev</b></td>
      <td>🛑 HALT</td>
      <td><b>51.2 ms</b></td>
      <td>99.9%</td>
      <td>0.0%</td>
    </tr>
    <tr>
      <td><b>Finance</b></td>
      <td>Q3 revenue reached $4.2B driven by cloud growth.</td>
      <td>Cloud enterprise revenue increased in Q3.</td>
      <td><b>TypeSafe Jev</b></td>
      <td>✅ PASS</td>
      <td><b>48.6 ms</b></td>
      <td>0.0%</td>
      <td>99.8%</td>
    </tr>
    <tr>
      <td><b>Medical</b></td>
      <td>Patient exhibits elevated liver enzymes with no fever.</td>
      <td>Patient is diagnosed with acute bacterial meningitis.</td>
      <td><b>TypeSafe Jev</b></td>
      <td>🛑 HALT</td>
      <td><b>53.1 ms</b></td>
      <td>98.7%</td>
      <td>0.1%</td>
    </tr>
    <tr>
      <td><b>Tech</b></td>
      <td>Model training completed in 4 epochs.</td>
      <td>Model will be updated again next Monday.</td>
      <td><b>TypeSafe Jev</b></td>
      <td>⚠️ WARN</td>
      <td><b>49.8 ms</b></td>
      <td>3.4%</td>
      <td>0.1%</td>
    </tr>
  </tbody>
</table>


## 🏗️ Repository Architecture

<pre><code>JevGuard/
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
└── README.md</code></pre>

---

## 🚀 Quickstart & Reproduction

<p><b>1. Clone & Install Dependencies:</b></p>
<pre><code>git clone https://github.com/SakshiKumari271/JevGuard.git
cd JevGuard
pip install -r requirements.txt</code></pre>

<p><b>2. Run Jev Decision Benchmarks:</b></p>
<pre><code>python tests/benchmark.py</code></pre>

<p><b>3. Launch Local Telemetry Dashboard:</b></p>
<pre><code>streamlit run app.py</code></pre>

---

## 🔗 Live Application & Links

<ul>
  <li><b>Interactive Dashboard:</b> <a href="https://jevguard-5.streamlit.app">https://jevguard-5.streamlit.app</a></li>
  <li><b>Source Repository:</b> <a href="https://github.com/SakshiKumari271/JevGuard">https://github.com/SakshiKumari271/JevGuard</a></li>
  <li><b>Core Engine:</b> TypeSafe AI Jev Decision Calibration Framework</li>
</ul>
