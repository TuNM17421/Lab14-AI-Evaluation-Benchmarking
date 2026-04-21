import asyncio
import os
import json
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class LLMJudge:
    def __init__(self, judge_a_model: str = "gpt-4o", judge_b_model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.judge_a_model = judge_a_model
        self.judge_b_model = judge_b_model
        
        self.rubrics = {
            "accuracy": "Chấm điểm từ 1-5 dựa trên độ chính xác so với Ground Truth. 5: Hoàn hảo, 1: Sai hoàn toàn.",
            "tone": "Chấm điểm từ 1-5 dựa trên sự chuyên nghiệp. 5: Chuyên nghiệp, 1: Thiếu tôn trọng/Cợt nhả."
        }
        
        self.prompt_template = """
        Bạn là một chuyên gia đánh giá chất lượng AI Agent.
        Nhiệm vụ: Đánh giá câu trả lời của AI dựa trên Câu hỏi và Câu trả lời mẫu (Ground Truth).
        
        Câu hỏi: {question}
        Câu trả lời của AI: {answer}
        Câu trả lời mẫu (Ground Truth): {ground_truth}
        
        Tiêu chí đánh giá:
        1. Accuracy (1-5): {accuracy_rubric}
        2. Tone (1-5): {tone_rubric}
        
        Yêu cầu trả về định dạng JSON duy nhất như sau:
        {{
            "accuracy_score": <số từ 1-5>,
            "tone_score": <số từ 1-5>,
            "reasoning": "<Giải thích ngắn gọn lý do chấm điểm>"
        }}
        """

    async def get_judge_score(self, model: str, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            accuracy_rubric=self.rubrics["accuracy"],
            tone_rubric=self.rubrics["tone"]
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            result = json.loads(response.choices[0].message.content)
            # Tính điểm tổng hợp cho model này (trung bình accuracy và tone)
            result["total_score"] = (result["accuracy_score"] + result["tone_score"]) / 2
            return result
        except Exception as e:
            print(f"Error calling judge model {model}: {e}")
            return {"accuracy_score": 1, "tone_score": 1, "reasoning": f"Error: {e}", "total_score": 1}

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        Gọi 2 model Judge song song.
        Tính toán sự sai lệch và độ đồng thuận.
        """
        results = await asyncio.gather(
            self.get_judge_score(self.judge_a_model, question, answer, ground_truth),
            self.get_judge_score(self.judge_b_model, question, answer, ground_truth)
        )
        
        score_a = results[0]["total_score"]
        score_b = results[1]["total_score"]
        
        avg_score = (score_a + score_b) / 2
        
        # Agreement rate: 1.0 nếu lệch <= 1, 0.0 nếu lệch > 1
        diff = abs(score_a - score_b)
        agreement = 1.0 if diff <= 1.0 else 0.0
        
        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": {
                self.judge_a_model: results[0],
                self.judge_b_model: results[1]
            },
            "reasoning": f"Judge A: {results[0]['reasoning']} | Judge B: {results[1]['reasoning']}"
        }

    async def check_position_bias(self, response_a: str, response_b: str):
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        """
        pass
