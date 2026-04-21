Họ và tên: Lê Tuấn Đạt
MSSV: 2A202600382

1. Các đóng góp kỹ thuật cụ thể (Engineering Contribution)
  Trong dự án này, tôi đã trực tiếp triển khai các module cốt lõi sau:
   - Xây dựng Multi-Judge Engine (engine/llm_judge.py): Triển khai hệ thống đánh giá song song sử dụng đồng thời 2 model LLM (gpt-4o và gpt-4o-mini).
     Thiết kế logic tính toán agreement_rate để đo lường độ tin cậy của Judge.
   - Phát triển Retrieval Metrics (engine/retrieval_eval.py): Viết mã nguồn tính toán Hit Rate và MRR (Mean Reciprocal Rank). Đặc biệt, tôi đã tối ưu
     logic so khớp (Fuzzy Matching) để hệ thống có thể đánh giá chính xác ngay cả khi có sự lệch định dạng giữa Chunk ID và Filename.
   - Tối ưu hiệu năng Async (engine/runner.py): Sử dụng asyncio.Semaphore để kiểm soát concurrency. Điều này giúp hệ thống chạy nhanh nhưng vẫn kiểm soát
     được Rate Limit (TPM/RPM) của OpenAI API, tránh lỗi 429.

2. Thấu hiểu kỹ thuật (Technical Depth)
  Qua quá trình thực hiện, tôi đã nắm vững các khái niệm:
   - MRR (Mean Reciprocal Rank): Hiểu rằng MRR quan trọng hơn Hit Rate vì nó đánh giá được "vị trí" của tài liệu đúng. Nếu tài liệu đúng nằm ở top 1,
     Agent sẽ trả lời tốt hơn so với khi nó nằm ở top 5.
   - Multi-Judge Consensus: Nhận thức được rằng một Judge đơn lẻ có thể bị Position Bias hoặc Length Bias. Việc sử dụng agreement_rate giúp chúng ta lọc
     ra được các case mà Judge đang phân vân để con người vào can thiệp.
   - Regression Testing trong AI: Cách thiết lập một "Release Gate" để so sánh V1 vs V2, đảm bảo phiên bản mới không gây ra sự sụt giảm về chất lượng
     (Regression).

3. Giải quyết vấn đề (Problem Solving)
  Trong quá trình chạy hệ thống, tôi đã xử lý thành công 2 vấn đề lớn:
   - Vấn đề 1 (Rate Limit): Khi chạy 50 cases song song, API bị lỗi 429. Tôi đã giải quyết bằng cách giảm max_concurrent xuống mức an toàn (2-5) thông qua
     Semaphore.
   - Vấn đề 2 (Data Mismatch): File Golden Dataset của đồng nghiệp sử dụng ID dạng chunk, trong khi Agent trả về ID dạng file. Tôi đã viết lại logic so   
     khớp mờ để "cứu" chỉ số Hit Rate từ 0% lên >90% mà không cần yêu cầu đồng nghiệp làm lại dữ liệu.

  ---