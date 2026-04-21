import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    limit = int(os.getenv("BENCHMARK_LIMIT", "0"))
    if limit > 0:
        dataset = dataset[:limit]
        print(f"🔢 BENCHMARK_LIMIT={limit} → chạy {len(dataset)} case đầu tiên.")

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    # Khởi tạo các component thật
    agent = MainAgent(version=agent_version)
    evaluator = RetrievalEvaluator()
    judge = LLMJudge()
    
    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "mrr": sum(r["ragas"]["retrieval"]["mrr"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total,
            "avg_latency": sum(r["latency"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    # V1: Baseline
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # V2: Optimized (Có thể thay đổi logic trong MainAgent dựa trên version)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    v1_score = v1_summary["metrics"]["avg_score"]
    v2_score = v2_summary["metrics"]["avg_score"]
    delta = v2_score - v1_score
    
    print(f"V1 Score: {v1_score:.2f}")
    print(f"V2 Score: {v2_score:.2f}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")
    print(f"V1 Hit Rate: {v1_summary['metrics']['hit_rate']:.2f}")
    print(f"V2 Hit Rate: {v2_summary['metrics']['hit_rate']:.2f}")

    os.makedirs("reports", exist_ok=True)
    
    # Lưu report V2 làm bản mới nhất
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        # Ghi cả thông tin so sánh vào summary
        v2_summary["regression"] = {
            "v1_score": v1_score,
            "delta": delta,
            "status": "PASS" if delta >= 0 else "FAIL"
        }
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
        
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta >= 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
