"""
Multi-Domain Benchmark & Regression Suite for TypeSafe AI Jev Decision Architecture
"""
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.guard_engine import JevArbiter

BENCHMARK_CASES = [
    {
        "domain": "Finance",
        "context": "Cloud enterprise subscriptions grew by 35% in Q3.",
        "claim": "Enterprise cloud subscriptions declined in Q3.",
        "expected": "HALT"
    },
    {
        "domain": "Finance",
        "context": "Q3 revenue reached $4.2B driven by cloud growth.",
        "claim": "Cloud enterprise revenue increased in Q3.",
        "expected": "PASS"
    },
    {
        "domain": "Medical",
        "context": "Patient exhibits elevated liver enzymes with no fever.",
        "claim": "Patient is diagnosed with acute bacterial meningitis.",
        "expected": "HALT"
    },
    {
        "domain": "Tech",
        "context": "Model training completed in 4 epochs.",
        "claim": "Model will be updated again next Monday.",
        "expected": "WARN"
    }
]

def run_benchmarks():
    print("=" * 65)
    print("🚀 Running JevGuard Benchmark Suite (Engine: TypeSafe AI / Jev)")
    print("=" * 65)

    arbiter = JevArbiter()
    passed = 0

    for i, test in enumerate(BENCHMARK_CASES, 1):
        decision = arbiter.arbitrate(test["context"], test["claim"])
        status = "✅ PASS" if decision.verdict == test["expected"] else "❌ FAIL"
        
        if decision.verdict == test["expected"]:
            passed += 1

        print(f"\n[Case {i}] Domain: {test['domain']} | Status: {status}")
        print(f"Context: {test['context']}")
        print(f"Claim:   {test['claim']}")
        print(f"Jev Latency: {decision.latency_ms:.2f} ms")
        print(f"Contradiction: {decision.p_contradiction * 100:.2f}% | Entailment: {decision.p_entailment * 100:.2f}%")
        print(f"Decision: {decision.verdict} (Expected: {test['expected']})")

    print("\n" + "=" * 65)
    print(f"Results: {passed}/{len(BENCHMARK_CASES)} tests passed.")
    print("=" * 65)

    if passed != len(BENCHMARK_CASES):
        sys.exit(1)

if __name__ == "__main__":
    run_benchmarks()