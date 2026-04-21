from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        if not expected_ids:
            return 0.0
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        if not expected_ids:
            return 0.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
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
