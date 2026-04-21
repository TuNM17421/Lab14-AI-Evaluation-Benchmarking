import asyncio
from typing import Dict

from rag_answer import rag_answer

VERSION_CONFIGS = {
    "Agent_V1_Base": {
        "retrieval_mode": "sparse",
        "top_k_search": 5,
        "top_k_select": 3,
        "use_rerank": False,
    },
    "Agent_V2_Optimized": {
        "retrieval_mode": "dense",
        "top_k_search": 5,
        "top_k_select": 3,
        "use_rerank": False,
    },
}


class MainAgent:
    """
    RAG Agent với 2 phiên bản cho regression test:
      - Agent_V1_Base: dense retrieval, không rerank (baseline)
      - Agent_V2_Optimized: hybrid retrieval (RRF) + LLM rerank
    """

    def __init__(self, version: str = "Agent_V1_Base"):
        if version not in VERSION_CONFIGS:
            raise ValueError(
                f"Unknown agent version: {version}. "
                f"Supported: {list(VERSION_CONFIGS.keys())}"
            )
        self.version = version
        self.config = VERSION_CONFIGS[version]

    async def query(self, question: str) -> Dict:
        result = await asyncio.to_thread(
            rag_answer,
            question,
            retrieval_mode=self.config["retrieval_mode"],
            top_k_search=self.config["top_k_search"],
            top_k_select=self.config["top_k_select"],
            use_rerank=self.config["use_rerank"],
            verbose=False,
        )

        chunks = result.get("chunks_used", [])
        retrieved_ids = [c["id"] for c in chunks if "id" in c]
        contexts = [c.get("text", "") for c in chunks]

        return {
            "answer": result.get("answer", ""),
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "version": self.version,
                "config": self.config,
                "sources": result.get("sources", []),
            },
        }


if __name__ == "__main__":
    async def _demo():
        for v in VERSION_CONFIGS:
            agent = MainAgent(version=v)
            resp = await agent.query("SLA xử lý ticket P1 là bao lâu?")
            print(f"\n=== {v} ===")
            print(f"Answer: {resp['answer'][:200]}...")
            print(f"Retrieved IDs: {resp['retrieved_ids']}")

    asyncio.run(_demo())
