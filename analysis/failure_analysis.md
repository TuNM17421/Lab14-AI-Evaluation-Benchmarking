
# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark
- **Tổng số cases:** 63
- **Tỉ lệ Pass/Fail:** 54/63
- **Điểm RAGAS trung bình:**
    - Faithfulness: 0.82
    - Relevancy: 0.82
- **Điểm LLM-Judge trung bình:** 4.15 / 5.0
- **Hit Rate:** 0.89
- **MRR:** 0.82
- **Agreement Rate:** 0.92
- **Latency trung bình:** 2.63s

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi        | Số lượng | Nguyên nhân dự kiến                        |
|-----------------|----------|--------------------------------------------|
| Hallucination   | 5        | Retriever lấy sai context                  |
| Incomplete      | 3        | Prompt quá ngắn, không yêu cầu chi tiết    |
| Tone Mismatch   | 2        | Agent trả lời quá suồng sã                 |
| Missing Detail  | 4        | Agent bỏ sót thông tin phụ trong Ground Truth |
| Out of Scope    | 2        | Câu hỏi ngoài phạm vi tài liệu             |

## 3. Phân tích 5 Whys (Chọn 3 case tệ nhất)

### Case #1: "Nếu gặp sự cố thường xuyên ngắt khi truy cập mạng riêng ảo, tôi nên xử lý ra sao?"
1. **Symptom:** Agent trả lời "Tôi không biết."
2. **Why 1:** LLM không tìm thấy thông tin phù hợp trong context.
3. **Why 2:** Câu hỏi phrasing khác biệt so với tài liệu gốc.
4. **Why 3:** Retriever không match được các từ khóa liên quan.
5. **Why 4:** Thiếu bước synonym expansion trong pipeline.
6. **Root Cause:** Pipeline chưa tối ưu cho các truy vấn phrasing lạ.

### Case #2: "Sau bao lâu kể từ lúc có sự cố P1 mà chưa có phản hồi thì hệ thống sẽ tự động chuyển vụ việc lên kỹ sư cấp cao hơn?"
1. **Symptom:** Agent trả lời "Tôi không biết."
2. **Why 1:** LLM không thấy thông tin trong context.
3. **Why 2:** Retriever không lấy đúng chunk về escalation.
4. **Why 3:** Câu hỏi phrasing khác biệt với tài liệu.
5. **Why 4:** Thiếu paraphrase augmentation cho câu hỏi.
6. **Root Cause:** Thiếu paraphrase data và retriever chưa tối ưu.

### Case #3: "Nếu thông tin trên hai tài liệu về số ngày phép khác nhau thì tôi nên làm gì?"
1. **Symptom:** Agent trả lời không hướng dẫn xác minh thông tin.
2. **Why 1:** LLM không được huấn luyện để xử lý conflicting information.
3. **Why 2:** Không có logic detect mâu thuẫn giữa các chunk.
4. **Why 3:** Prompt chưa nhấn mạnh xử lý trường hợp mâu thuẫn.
5. **Why 4:** Thiếu hướng dẫn rõ ràng trong system prompt.
6. **Root Cause:** Chưa có module xử lý conflicting information.

## 4. Kế hoạch cải tiến (Action Plan)
- [ ] Thay đổi Chunking strategy từ Fixed-size sang Semantic Chunking.
- [ ] Cập nhật System Prompt để nhấn mạnh vào việc "Chỉ trả lời dựa trên context".
- [ ] Thêm bước Reranking vào Pipeline.
- [ ] Bổ sung paraphrase augmentation cho test cases.
- [ ] Thêm logic detect và xử lý conflicting information.
- [ ] Tối ưu retriever cho các phrasing lạ và synonym.
