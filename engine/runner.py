import asyncio
import time
from typing import List, Dict

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge, max_concurrent: int = 2):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def run_single_test(self, test_case: Dict) -> Dict:
        async with self.semaphore:
            start_time = time.perf_counter()
            
            # 1. Gọi Agent
            response = await self.agent.query(test_case["question"])
            latency = time.perf_counter() - start_time
            
            # 2. Chạy Retrieval Eval
            expected_ids = test_case.get("expected_retrieval_ids", [])
            retrieved_ids = response.get("retrieved_ids", []) # Đảm bảo Agent trả về trường này
            
            hit_rate = self.evaluator.calculate_hit_rate(expected_ids, retrieved_ids)
            mrr = self.evaluator.calculate_mrr(expected_ids, retrieved_ids)
            
            # 3. Chạy Multi-Judge
            judge_result = await self.judge.evaluate_multi_judge(
                test_case["question"], 
                response["answer"], 
                test_case["expected_answer"]
            )
            
            return {
                "test_case": test_case["question"],
                "agent_response": response["answer"],
                "latency": latency,
                "ragas": {
                    "retrieval": {
                        "hit_rate": hit_rate,
                        "mrr": mrr
                    },
                    "faithfulness": 0.0, # Placeholder nếu không dùng RAGAS thực
                    "relevancy": 0.0     # Placeholder nếu không dùng RAGAS thực
                },
                "judge": judge_result,
                "status": "fail" if judge_result["final_score"] < 3 else "pass"
            }

    async def run_all(self, dataset: List[Dict]) -> List[Dict]:
        """
        Chạy song song tất cả các test cases với giới hạn bởi semaphore.
        """
        tasks = [self.run_single_test(case) for case in dataset]
        results = await asyncio.gather(*tasks)
        return results
