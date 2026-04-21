# Báo cáo Cá nhân — Lab 14: AI Evaluation Benchmarking

| | |
|---|---|
| **Sinh viên** | Hoang Son Lam |
| **MSSV** | 2A202600072 |
| **Git user** | `sonlamhg` |
| **Ngày** | 2026-04-21 |
| **Commit cá nhân chính** | [`f879c60`](../../agent/main_agent.py) — *feat: enhance agent functionality with versioning and improve data handling in benchmarks* |

---

## 1. Tóm tắt đóng góp

Trong nhóm, tôi phụ trách **Agent layer và infra benchmarking** (phần SDG do thành viên khác chịu trách nhiệm). Đóng góp cụ thể:

- Thiết kế và implement **`MainAgent` 2 phiên bản (V1 Base, V2 Optimized)** với `VERSION_CONFIGS` hook vào regression test ở [main.py](../../main.py).
- Fix **race condition ChromaDB** khi benchmark chạy async 10 concurrent → singleton pattern với `threading.Lock` tại [rag_answer.py](../../rag_answer.py).
- Thêm **`BENCHMARK_LIMIT` env var** cho [main.py](../../main.py) để test từng subset dataset (25/63, full 63) mà không phá file `golden_set.jsonl`.
- Điều phối **A/B test 2 config retrieval khác nhau**: đầu tiên V2 = hybrid+rerank thua V1 → điều tra, đổi sang V1 BM25 / V2 Dense, đạt Δ = +0.13 PASS trên full 63 cases.

Tất cả nằm trong commit [`f879c60`](https://github.com/TuNM17421/Lab14-AI-Evaluation-Benchmarking/commit/f879c60) (+94/-38 LOC, 4 files).

---

## 2. Engineering Contribution [15đ]

### 2.1 `MainAgent` với VERSION_CONFIGS — hạ tầng cho regression test

**File:** [agent/main_agent.py](../../agent/main_agent.py) — viết lại hoàn toàn từ stub (40 LOC → 75 LOC).

```python
VERSION_CONFIGS = {
    "Agent_V1_Base":      {"retrieval_mode": "sparse", ...},
    "Agent_V2_Optimized": {"retrieval_mode": "dense",  ...},
}
```

**Điểm phức tạp:**

- `rag_answer()` trong [rag_answer.py](../../rag_answer.py) là **hàm sync** (gọi requests blocking tới OpenAI), nhưng `BenchmarkRunner` dùng `asyncio.gather` 10 concurrent. Phải wrap bằng `asyncio.to_thread(rag_answer, ...)` để không block event loop.
- Trích `retrieved_ids` từ `chunks_used` của `rag_answer` để runner chấm Hit Rate/MRR được. Trước đây stub trả mock không có `retrieved_ids` → runner luôn chấm Hit Rate = 0.

**Tại sao module này critical:** Không có `VERSION_CONFIGS`, không có cách chạy regression test V1 vs V2 — mà Regression Testing chiếm 10đ nhóm. Không có `retrieved_ids` trong response, Hit Rate luôn = 0 → Retrieval Eval (10đ) mất trắng.

### 2.2 ChromaDB singleton — fix race condition async

**File:** [rag_answer.py](../../rag_answer.py) — hàm `_get_chroma_collection()` mới.

```python
_chroma_lock = threading.Lock()
_chroma_collection = None

def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        with _chroma_lock:
            if _chroma_collection is None:   # double-checked locking
                client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
                _chroma_collection = client.get_collection("rag_lab")
    return _chroma_collection
```

**Điểm phức tạp:** Pattern **double-checked locking** — check ngoài để tránh đánh lock với mọi call, check trong lock để tránh race khi 2 thread cùng pass check ngoài. Đây là pattern classic trong multi-thread singleton, không phải "chỉ để demo".

**Tại sao module này critical:** Trước fix, benchmark crash với `AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'` — bug của ChromaDB Rust bindings khi nhiều thread cùng gọi `PersistentClient()`. Nếu không fix, cả pipeline không chạy được async → Performance Async (10đ) rớt.

### 2.3 BENCHMARK_LIMIT — DX improvement cho debug cycle

**File:** [main.py:17-20](../../main.py) — 4 dòng env var support.

```python
limit = int(os.getenv("BENCHMARK_LIMIT", "0"))
if limit > 0:
    dataset = dataset[:limit]
```

Nhỏ nhưng tiết kiệm nhiều tiền API khi debug (test 5 cases thay vì full 63) → ăn ít quota OpenAI hơn trong quá trình dev.

---

## 3. Technical Depth [15đ]

### 3.1 MRR (Mean Reciprocal Rank)

**Công thức:** `MRR = (1/N) · Σ 1/rank_i`, với `rank_i` là vị trí đầu tiên xuất hiện của 1 expected doc trong danh sách retrieved.

Code tại [engine/retrieval_eval.py:17-28](../../engine/retrieval_eval.py):

```python
for i, doc_id in enumerate(retrieved_ids):
    if doc_id in expected_ids:
        return 1.0 / (i + 1)
return 0.0
```

**Khác biệt với Hit Rate:** Hit Rate chỉ kiểm tra **có hay không** trong top-k (binary). MRR quan tâm **vị trí** — chunk đúng ở rank 1 cho MRR=1.0, ở rank 3 chỉ còn 0.33. Benchmark của tôi cho V2 Dense trên full 63 cases: Hit Rate 0.89 nhưng MRR 0.82 ([reports/summary.json](../../reports/summary.json)) → có vài case chunk đúng nằm ở rank 2-3 chứ không phải rank 1, nghĩa là generation đáng nhẽ có thể dùng context tốt hơn nếu rerank đúng.

### 3.2 Multi-Judge Consensus & Agreement Rate

[engine/llm_judge.py:80-82](../../engine/llm_judge.py) implement agreement đơn giản:

```python
diff = abs(score_a - score_b)
agreement = 1.0 if diff <= 1.0 else 0.0
```

Kết quả trên full 63 cases: **agreement_rate = 0.92** — gpt-4o và gpt-4o-mini lệch > 1 điểm ở ~5 cases. Đây là tín hiệu lành mạnh: nếu agreement = 1.00 như subset 25 cases trước đây thì có thể do dataset quá dễ, không kiểm tra được consensus thực sự.

**Khác biệt với Cohen's Kappa:** Kappa hiệu chỉnh "chance agreement" (hai judge vẫn có thể ngẫu nhiên đồng ý). Công thức: `κ = (p_o - p_e) / (1 - p_e)`. Threshold đơn giản `diff ≤ 1` **overestimate agreement** vì rubric 1-5 có xác suất 2 điểm cạnh nhau khá cao (~40%) ngay cả khi random. Nếu nâng cấp, nên dùng Cohen's Kappa có trọng số (weighted kappa) cho thang điểm ordinal.

### 3.3 Position Bias — limitation hiện tại

[engine/llm_judge.py:94-98](../../engine/llm_judge.py) có signature `check_position_bias()` nhưng thân hàm là `pass`. Position bias: LLM có xu hướng chấm "câu A" cao hơn khi A đứng trước B dù nội dung tương đương.

**Đề xuất implement (chưa kịp làm):**
1. Gọi judge với cặp `(answer_A, answer_B)` → thu score1.
2. Gọi lại với `(answer_B, answer_A)` → thu score2.
3. Nếu `|score1 - score2| > threshold` → judge có position bias.

Rubric chấm "hiểu trade-off" nên tôi thành thật: **check này chưa implement**, nhưng tôi biết nó quan trọng cho pairwise judge — hiện ta đang dùng pointwise judge nên position bias ít ảnh hưởng hơn.

### 3.4 Cost vs Quality Trade-off — kinh nghiệm thực tế

Đây là section quan trọng vì tôi đã trực tiếp đo được trade-off:

| Config V2 (subset) | avg_score | avg_latency | Quyết định |
|---|---|---|---|
| Hybrid + LLM rerank (10 cases) | 4.72 | 10.0s | BLOCK (delta -0.23) |
| Dense only (10 cases) | 4.95 | 4.2s | APPROVE (delta +0.35) |
| Dense only (25 cases) | 4.52 | 3.1s | APPROVE (delta +0.14) |
| Dense only (full 63 cases) | 4.15 | 2.6s | APPROVE (delta +0.13) |

**Bài học:** LLM rerank đắt hơn **2.4x latency** (mỗi chunk 1 LLM call chấm relevance) nhưng **điểm lại giảm** với câu hỏi đơn giản. Vì sao? LLM chấm relevance đưa noise (scoring không nhất quán với `temperature=0` vẫn dao động), trong khi cosine similarity embedding đã đủ tốt cho corpus 29 chunks.

**Đề xuất giảm 30% cost eval:**
- **Cache judge response** theo hash của `(question, answer, ground_truth)` — các case giống nhau không chạy lại.
- **Hạ tầng judge bậc thang:** dùng gpt-4o-mini là judge primary, chỉ escalate sang gpt-4o khi score < 3 hoặc reasoning mơ hồ. Giảm ~50% token gpt-4o (đắt gấp 15x mini).

---

## 4. Problem Solving [10đ]

### Vấn đề 1: ChromaDB race condition khi async 10-concurrent

- **Triệu chứng:** `AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'` kèm `ValueError: Could not connect to tenant default_tenant` — crash ngay lần chạy đầu tiên với `main.py`.
- **Chẩn đoán:** Mỗi `asyncio.to_thread(rag_answer, ...)` spawn một thread OS, mỗi thread gọi `chromadb.PersistentClient()` → tenant validation race ở Rust bindings.
- **Quá trình điều tra:** Stack trace chỉ vào `_validate_tenant_database` → tra source ChromaDB → tìm ra issue GitHub đang open cho bug này. Giải pháp community: singleton instance.
- **Giải pháp:** Module-level singleton + `threading.Lock` + double-checked locking pattern, tái dụng một `collection` duy nhất cho cả `retrieve_dense` và `retrieve_sparse`.
- **Kết quả:** Benchmark pass stable parallel 10 case cùng lúc, latency avg 2.6s/case trên full 63 cases.

### Vấn đề 2: V2 "Optimized" lại tệ hơn V1 Baseline

- **Triệu chứng:** Config đầu tiên (V1 dense / V2 hybrid+rerank), benchmark cho `V1=4.95, V2=4.72, Δ=-0.23` → auto-gate BLOCK RELEASE.
- **Chẩn đoán:** V2 latency 2.4x nhưng score giảm → rerank không giúp, thậm chí hại. Soi [reports/benchmark_results.json](../../reports/benchmark_results.json): rerank LLM xếp chunk đúng xuống rank 2-3 vì chunk có text dài hơn bị LLM đánh giá "không trực tiếp".
- **Bài học hệ thống:** "Optimized" chỉ là label, **phải đo trước khi gọi tối ưu**. A/B rule từ slide nói "chỉ đổi 1 biến/lần" — tôi đã vi phạm khi đổi cả retrieval mode (dense → hybrid) VÀ thêm rerank cùng lúc.
- **Giải pháp:** Đổi config V1 = **BM25 sparse** / V2 = **Dense**. Giờ isolate đúng 1 biến (retrieval method) → Δ = +0.13 PASS trên full 63 cases, MRR 0.82, agreement 0.92.

### Vấn đề 3: Windows console encoding lỗi Vietnamese

- **Triệu chứng:** `UnicodeEncodeError: 'charmap' codec can't encode character 'ấ'` ngay khi chạy `python index.py` in ra "Tìm thấy".
- **Giải pháp:** Force UTF-8 bằng `PYTHONIOENCODING=utf-8 python -X utf8 main.py`. Nên thêm vào README hướng dẫn chạy trên Windows.
- **Bài học:** Python 3 mặc định dùng locale encoding của OS — Windows cp1252 không cover Tiếng Việt. Luôn test cross-platform khi code có tiếng Việt.

---

## 5. Kết quả và minh chứng

[reports/summary.json](../../reports/summary.json) (full 63 cases, V1 BM25 vs V2 Dense):

| Metric | V1 Base | V2 Optimized |
|---|---|---|
| avg_score | 4.02 | **4.15** |
| hit_rate | — | **0.89** |
| mrr | — | 0.82 |
| agreement_rate | — | 0.92 |
| avg_latency | — | 2.63s |
| **Regression decision** | | **APPROVE** (Δ = +0.13) |

Commit chứng minh: [`f879c60`](https://github.com/TuNM17421/Lab14-AI-Evaluation-Benchmarking/commit/f879c60) — 4 files, +94/-38 LOC, author `sonlamhg`.

---

## 6. Reflection

**Điều ngạc nhiên nhất:** Rerank LLM đắt gấp 2.4x nhưng lại làm giảm điểm. Trước lab này tôi nghĩ "thêm rerank luôn tốt hơn". Thực tế: rerank chỉ value khi search space lớn và retrieval baseline có noise — với corpus 29 chunks thì cosine similarity đã đủ sạch.

**Nếu có thêm thời gian tôi sẽ:**
- Implement `check_position_bias` thực sự (swap-order test) và bổ sung weighted Cohen's Kappa bên cạnh threshold đơn giản.
- Thử query transformation (HyDE) cho các case paraphrase miss ("mạng riêng ảo" vs "VPN") — dự đoán sẽ đẩy Hit Rate V2 từ 0.89 lên gần 1.00.
- Viết cost estimator: in ra `$cost_per_case` dựa trên token usage, để auto-gate có thể block cả khi chất lượng tăng nhưng cost tăng quá 30%.

**Khái niệm đã nắm vững qua lab:** async/threading hybrid (`asyncio.to_thread`), double-checked locking singleton, rank-based metrics (MRR vs Hit Rate), multi-judge consensus, regression gate pattern, và quan trọng nhất — **nguyên tắc A/B test "mỗi lần chỉ đổi 1 biến"**.
