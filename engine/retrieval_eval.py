from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán Hit Rate. Hỗ trợ so khớp mờ: nếu bất kỳ expected_id nào chứa 
        tên file trong retrieved_ids (hoặc ngược lại) thì coi là Hit.
        """
        if not expected_ids:
            return 0.0
        top_retrieved = retrieved_ids[:top_k]
        
        hit = False
        for exp_id in expected_ids:
            for ret_id in top_retrieved:
                # So khớp mờ: hr_leave_policy_0 khớp với hr_leave_policy.txt
                clean_exp = exp_id.replace(".txt", "").split("_")[0]
                clean_ret = ret_id.replace(".txt", "").split("_")[0]
                if clean_exp in clean_ret or clean_ret in clean_exp:
                    hit = True
                    break
            if hit: break
            
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank với logic so khớp mờ.
        """
        if not expected_ids:
            return 0.0
        for i, ret_id in enumerate(retrieved_ids):
            for exp_id in expected_ids:
                clean_exp = exp_id.replace(".txt", "").split("_")[0]
                clean_ret = ret_id.replace(".txt", "").split("_")[0]
                if clean_exp in clean_ret or clean_ret in clean_exp:
                    return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có các trường:
        - 'expected_retrieval_ids': List[str]
        - 'retrieved_ids': List[str]
        """
        total_hit_rate = 0.0
        total_mrr = 0.0
        count = len(dataset)
        
        if count == 0:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}

        for item in dataset:
            expected = item.get("expected_retrieval_ids", [])
            retrieved = item.get("retrieved_ids", [])
            total_hit_rate += self.calculate_hit_rate(expected, retrieved)
            total_mrr += self.calculate_mrr(expected, retrieved)

        return {
            "avg_hit_rate": total_hit_rate / count,
            "avg_mrr": total_mrr / count
        }
