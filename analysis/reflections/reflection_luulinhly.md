# Báo cáo Cá nhân — Lab 14: AI Evaluation Benchmarking

| **Sinh viên** | Lưu Linh Ly |
| **MSSV** | 2A202600119 |
| **Vai trò** | Data Team — SDG & Retrieval Eval |
| **Commit chính** | [`7fde070`](https://github.com/TuNM17421/Lab14-AI-Evaluation-Benchmarking/commit/7fde070) — *update track data team* (+935 LOC) |

---

## 1. Engineering Contribution

**Module phụ trách:** [`data/synthetic_gen.py`](../../data/synthetic_gen.py) — viết lại hoàn toàn từ stub (40 LOC → 894 LOC).

**Async SDG pipeline:** 45 cases L1–L4 được sinh đồng thời bằng `asyncio.gather`, kiểm soát tốc độ bằng `asyncio.Semaphore(5)` để tránh lỗi 429 rate limit từ OpenAI. Kết quả nhanh hơn ~5x so với chạy tuần tự.

**Schema `expected_retrieval_ids`:** Thiết kế trường này trong mỗi record là ground truth để `retrieval_eval.py` tính Hit Rate và MRR. L3 cases có `requires_chunks=2`, L5 `out_of_scope` có `[]` (retriever phải trả về rỗng). Đây là field kết nối trực tiếp dataset của tôi với Retrieval Metrics của nhóm.

**25 L5 Adversarial Cases:** Mở rộng từ 5 lên 25 cases với 20 sub-type (prompt injection, out-of-scope, conflicting info, boundary cases, temporal confusion, false assumption, v.v.). Hard-coded thay vì dùng LLM để kiểm soát chính xác `expected_behavior` cho từng case.

**Fix chunk ID schema:** Phát hiện mismatch giữa generator (`sla_p1_2026__sec_2`) và vector DB thực tế (`sla_p1_2026_1`). Sửa 3 dòng trong `load_and_chunk_docs()` để dùng 0-indexed, khiến toàn bộ `expected_retrieval_ids` của 45 cases L1–L4 tự động đúng.

---

## 2. Technical Depth

**MRR (Mean Reciprocal Rank):** Đo vị trí chunk đúng trong danh sách retrieved — `MRR = 1/rank`. Khác với Hit Rate (binary: có/không), MRR phản ánh chất lượng ranking: chunk đúng ở rank 1 cho MRR=1.0, rank 3 chỉ còn 0.33. Quan trọng vì generation LLM dùng context theo thứ tự — chunk đứng đầu có trọng số cao hơn.

**Cohen's Kappa:** Đo độ đồng thuận giữa các judge, có hiệu chỉnh xác suất ngẫu nhiên: $\kappa = (p_o - p_e)/(1 - p_e)$. Agreement rate đơn giản (`diff ≤ 1`) bị overestimate vì trên thang 1–5, hai judge random vẫn có ~40% cơ hội nằm trong khoảng cách 1 điểm. Weighted Kappa chuẩn hơn cho thang điểm ordinal.

**Position Bias:** LLM judge có xu hướng cho điểm cao hơn cho câu trả lời đứng đầu tiên trong prompt, bất kể nội dung. Phát hiện bằng swap-order test: gọi judge 2 lần với thứ tự đảo, nếu score lệch quá ngưỡng thì có bias. Hệ thống hiện tại chưa implement (`check_position_bias()` vẫn là `pass`).

**Cost vs Quality trade-off:** Dùng `gpt-4.1` để sinh dataset (đắt hơn ~13x so với mini) vì dataset sinh 1 lần nhưng dùng mãi — model yếu thường tạo L3/L4 cases thực chất chỉ cần 1 chunk, làm sai mục đích test. Ngược lại, LLM Judge chạy mỗi lần benchmark nên nên dùng `gpt-4o-mini` làm primary và chỉ escalate khi score thấp — giảm ~60% chi phí.

---

## 3. Problem Solving

**Chunk ID mismatch (nghiêm trọng nhất):** Generator tạo `__sec_N` (1-indexed), vector DB dùng `_N` (0-indexed). Hậu quả: Hit Rate = 0% toàn bộ dù retriever hoạt động đúng — lỗi silent, code không báo error. Phát hiện bằng cách đối chiếu tay `list_chunks_in_db.md` với output của `load_and_chunk_docs()`. Fix 3 dòng, ảnh hưởng toàn bộ pipeline.

**L5 context/retrieval_ids rỗng sai chỗ:** Copy template từ cases intentionally-empty (`out_of_scope`, `goal_hijacking`) sang cases có tài liệu nguồn, quên điền. Cách phân biệt: cases không có chunk liên quan trong corpus thì để `[]`, còn lại phải có đủ chunk ID và nội dung chunk thực.

**LLM trả về JSON bọc markdown fence:** `json.loads()` throw `JSONDecodeError` khi GPT-4.1 thêm ` ```json ` vào response. Fix bằng 2 dòng regex strip trước khi parse, thêm retry logic cho transient API errors.
