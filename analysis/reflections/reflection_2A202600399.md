
# Reflection - Nguyen Manh Tu

## 1. Engineering Contribution
Trong quá trình thực hiện Lab, mình đảm nhận việc setup repository cho cả nhóm, cấu hình database và lựa chọn techstack sử dụng Chroma. Mình cũng là người chọn phương pháp tách chunks cho document, review các câu hỏi trong golden_set.jsonl và hỗ trợ các thành viên hiểu, định nghĩa rõ ý nghĩa các attribute. Khi xây dựng agent v1 và v2, mình đóng vai trò hỗ trợ chọn kiến trúc phù hợp cho từng phiên bản: v1 sử dụng BM25, v2 sử dụng cosine_score. Ngoài ra, mình trực tiếp viết báo cáo failure_analysis.md và hoàn thiện các tài liệu report cho nhóm.

## 2. Technical Depth
Mình nắm vững các khái niệm như MRR, Hit Rate, cũng như hiểu rõ về các chỉ số đánh giá chất lượng hệ thống retrieval. Trong quá trình review và xây dựng bộ golden set, mình giải thích cho các thành viên về ý nghĩa của từng thuộc tính, đảm bảo mọi người đều hiểu rõ về mapping ground truth và cách đánh giá. Khi lựa chọn mô hình và phương pháp tách chunk, mình cân nhắc giữa chi phí và chất lượng, ví dụ như trade-off giữa tốc độ truy vấn (BM25) và độ chính xác (cosine_score).

## 3. Problem Solving
Trong quá trình phát triển, nhóm gặp phải vấn đề khi kết quả của agent v1 lại tốt hơn v2 dù v2 dùng phương pháp hiện đại hơn. Mình đã chủ động phân tích nguyên nhân, kiểm tra lại pipeline, kiến trúc và tham số, từ đó đề xuất giữ lại v1 với BM25 cho baseline và tiếp tục tối ưu v2. Ngoài ra, mình cũng giải quyết các vấn đề liên quan đến việc ingest dữ liệu, mapping ground truth, và hoàn thiện báo cáo phân tích lỗi (failure_analysis.md) để nhóm có hướng cải tiến rõ ràng.