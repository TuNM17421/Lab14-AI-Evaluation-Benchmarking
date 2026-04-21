"""
app.py — FastAPI backend for RAG Chatbot with step-by-step thinking UI
=======================================================================
Streams each pipeline step via Server-Sent Events (SSE) so the frontend
can render the "thinking" process in real time.
"""

import os
import json
import time
import asyncio
from typing import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot — Thinking UI")


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(
        content=html_path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# SSE endpoint — streams each thinking step
# ---------------------------------------------------------------------------

def _send_step(step_id: str, title: str, status: str, detail: str = "", data: dict = None):
    """Helper to build an SSE event payload."""
    payload = {
        "step_id": step_id,
        "title": title,
        "status": status,  # "running" | "done" | "error"
        "detail": detail,
        "timestamp": time.time(),
    }
    if data:
        payload["data"] = data
    return json.dumps(payload, ensure_ascii=False)


async def rag_stream(query: str, retrieval_mode: str, use_rerank: bool) -> AsyncGenerator[str, None]:
    """
    Generator that yields SSE events for each RAG pipeline step.
    Runs the blocking RAG functions in a thread pool to keep the event loop alive.
    """
    loop = asyncio.get_event_loop()

    # ── Step 1: Query Analysis ─────────────────────────────────────────────
    yield _send_step("query_analysis", "Phân tích câu hỏi", "running",
                      f"Đang phân tích: \"{query}\"")
    await asyncio.sleep(0.3)

    transformed_queries = [query]
    try:
        from rag_answer import transform_query
        transformed_queries = await loop.run_in_executor(
            None, lambda: transform_query(query, strategy="expansion")
        )
        yield _send_step("query_analysis", "Phân tích câu hỏi", "done",
                          f"Đã mở rộng thành {len(transformed_queries)} biến thể",
                          {"original": query, "expanded": transformed_queries})
    except Exception as e:
        yield _send_step("query_analysis", "Phân tích câu hỏi", "done",
                          f"Sử dụng query gốc (expansion không khả dụng)",
                          {"original": query, "expanded": [query]})

    # ── Step 2: Retrieval ──────────────────────────────────────────────────
    retrieval_label = {"dense": "Dense (Vector)", "sparse": "Sparse (BM25)", "hybrid": "Hybrid (RRF)"}
    yield _send_step("retrieval", f"Truy xuất tài liệu — {retrieval_label.get(retrieval_mode, retrieval_mode)}", "running",
                      f"Đang tìm kiếm trong ChromaDB...")

    candidates = []
    try:
        from rag_answer import retrieve_dense, retrieve_sparse, retrieve_hybrid, TOP_K_SEARCH

        if retrieval_mode == "dense":
            retrieve_fn = retrieve_dense
        elif retrieval_mode == "sparse":
            retrieve_fn = retrieve_sparse
        elif retrieval_mode == "hybrid":
            retrieve_fn = retrieve_hybrid
        else:
            retrieve_fn = retrieve_dense

        # Use first expanded query for retrieval
        search_query = transformed_queries[0] if transformed_queries else query
        candidates = await loop.run_in_executor(
            None, lambda: retrieve_fn(search_query, top_k=TOP_K_SEARCH)
        )

        chunks_summary = []
        for i, c in enumerate(candidates[:5]):
            chunk_info = {
                "rank": i + 1,
                "source": c["metadata"].get("source", "?"),
                "section": c["metadata"].get("section", ""),
                "score": round(c.get("score", 0), 4),
                "preview": c["text"][:120] + "...",
            }
            # Include sub-scores and ranks for hybrid mode
            if "dense_score" in c:
                chunk_info["dense_score"] = round(c["dense_score"], 4)
                chunk_info["sparse_score"] = round(c["sparse_score"], 2)
                chunk_info["dense_rank"] = c.get("dense_rank")
                chunk_info["sparse_rank"] = c.get("sparse_rank")
            chunks_summary.append(chunk_info)

        yield _send_step("retrieval", f"Truy xuất tài liệu — {retrieval_label.get(retrieval_mode, retrieval_mode)}", "done",
                          f"Tìm thấy {len(candidates)} chunks liên quan",
                          {"chunks": chunks_summary, "total": len(candidates), "mode": retrieval_mode})
    except Exception as e:
        yield _send_step("retrieval", "Truy xuất tài liệu", "error",
                          f"Lỗi: {str(e)}")
        return

    # ── Step 3: Reranking ──────────────────────────────────────────────────
    if use_rerank:
        yield _send_step("rerank", "Rerank — Đánh giá lại độ liên quan", "running",
                          f"Đang chấm điểm {len(candidates)} chunks bằng LLM...")
        try:
            from rag_answer import rerank, TOP_K_SELECT
            candidates = await loop.run_in_executor(
                None, lambda: rerank(query, candidates, top_k=TOP_K_SELECT)
            )
            reranked_summary = []
            for i, c in enumerate(candidates):
                reranked_summary.append({
                    "rank": i + 1,
                    "source": c["metadata"].get("source", "?"),
                    "section": c["metadata"].get("section", ""),
                    "score": round(c.get("score", 0), 4),
                })
            yield _send_step("rerank", "Rerank — Đánh giá lại độ liên quan", "done",
                              f"Đã chọn top {len(candidates)} chunks chất lượng nhất",
                              {"reranked": reranked_summary, "mode": retrieval_mode})
        except Exception as e:
            yield _send_step("rerank", "Rerank", "error", f"Lỗi: {str(e)}")
    else:
        from rag_answer import TOP_K_SELECT
        candidates = candidates[:TOP_K_SELECT]
        selected_summary = []
        for i, c in enumerate(candidates):
            item = {
                "rank": i + 1,
                "source": c["metadata"].get("source", "?"),
                "section": c["metadata"].get("section", ""),
                "score": round(c.get("score", 0), 4),
            }
            if "dense_score" in c:
                item["dense_score"] = round(c["dense_score"], 4)
                item["sparse_score"] = round(c["sparse_score"], 2)
                item["dense_rank"] = c.get("dense_rank")
                item["sparse_rank"] = c.get("sparse_rank")
            selected_summary.append(item)
        yield _send_step("rerank", f"Chọn top {len(candidates)} chunks", "done",
                          f"Lấy top {len(candidates)} chunks (không rerank)",
                          {"reranked": selected_summary, "mode": retrieval_mode})

    # ── Step 4: Context Building ───────────────────────────────────────────
    yield _send_step("context", "Xây dựng ngữ cảnh", "running",
                      "Đang đóng gói context cho LLM...")
    await asyncio.sleep(0.2)

    try:
        from rag_answer import build_context_block, build_grounded_prompt
        context_block = build_context_block(candidates)
        prompt = build_grounded_prompt(query, context_block)

        context_chunks = []
        for i, c in enumerate(candidates, 1):
            meta = c.get("metadata", {})
            chunk_info = {
                "index": i,
                "source": meta.get("source", "unknown"),
                "section": meta.get("section", ""),
                "text": c["text"][:300] + ("..." if len(c["text"]) > 300 else ""),
                "score": round(c.get("score", 0), 4),
            }
            if "dense_score" in c:
                chunk_info["dense_score"] = round(c["dense_score"], 4)
                chunk_info["sparse_score"] = round(c["sparse_score"], 2)
                chunk_info["dense_rank"] = c.get("dense_rank")
                chunk_info["sparse_rank"] = c.get("sparse_rank")
            context_chunks.append(chunk_info)

        yield _send_step("context", "Xây dựng ngữ cảnh", "done",
                          f"Đã tạo context block với {len(candidates)} chunks",
                          {"context_chunks": context_chunks, "prompt_length": len(prompt), "mode": retrieval_mode})
    except Exception as e:
        yield _send_step("context", "Xây dựng ngữ cảnh", "error", f"Lỗi: {str(e)}")
        return

    # ── Step 5: LLM Generation ─────────────────────────────────────────────
    yield _send_step("generation", "Sinh câu trả lời", "running",
                      f"Đang gọi {os.getenv('LLM_MODEL', 'gpt-4o-mini')}...")

    try:
        from rag_answer import call_llm
        answer = await loop.run_in_executor(None, lambda: call_llm(prompt))

        sources = list({c["metadata"].get("source", "unknown") for c in candidates})

        yield _send_step("generation", "Sinh câu trả lời", "done",
                          "Hoàn tất!",
                          {"answer": answer, "sources": sources, "model": os.getenv("LLM_MODEL", "gpt-4o-mini")})
    except Exception as e:
        yield _send_step("generation", "Sinh câu trả lời", "error", f"Lỗi: {str(e)}")
        return

    # ── Final: Complete ────────────────────────────────────────────────────
    yield _send_step("complete", "Hoàn tất", "done", "",
                      {"answer": answer, "sources": sources})


@app.get("/api/chat")
async def chat_stream(
    q: str,
    mode: str = "dense",
    rerank: str = "false",
):
    """SSE endpoint: streams thinking steps for a RAG query."""
    use_rerank = rerank.lower() == "true"

    async def event_generator():
        async for step_data in rag_stream(q, mode, use_rerank):
            yield {"event": "step", "data": step_data}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "model": os.getenv("LLM_MODEL", "gpt-4o-mini")}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
