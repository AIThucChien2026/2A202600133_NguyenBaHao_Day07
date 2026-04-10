# Lab 7 Summary

## Project Scope
- **Domain:** Văn học Việt Nam (Truyện ngắn hiện thực Nam Cao).
- **Benchmark corpus:** 5 file truyện trong thư mục `data/` (chi_pheo.txt, 1_bua_no.txt, quet_nha.txt, trang_sang.txt, tu_cach_mo.txt).
- **LLM backend used:** `gemini-3.1-flash-lite-preview`.

## Implemented Improvements
- Cô lập database ChromaDB in-memory (`f"{collection_name}_{id(self)}"`) để giải quyết vấn đề rò rỉ dữ liệu (state leak) khi chạy pytest test suite chung với nhau.
- Bổ sung logic `metadata_filter` vào tất cả các câu hỏi benchmark. Điều này là quan trọng vì Chí Phèo (56k chữ) lớn gấp 3-4 lần các file khác nên sẽ chiếm hết slot kết quả nếu không filter.
- Tăng `chunk_size` từ 500 lên 1500 cho RecursiveChunker: Text truyện văn xuôi thường câu rất dài và ngữ cảnh trải dài, chunk to giúp LLM sinh (grounding) câu trả lời đầy đủ hơn nhiều so với chunk cắt vụn.
- Triển khai logic \_Deduplication\_ (lọc trùng lặp) cho danh sách source tài liệu trong quá trình in benchmark, giúp output sạch sẽ gọn gàng (score của source sẽ được lấy là best score).

## Key Files Updated
- `src/chunking.py` (Custom Regex for SentenceChunker & Recursive Priority Split).
- `src/store.py` (ChromaDB Fallback logic & Document filtering logic).
- `src/agent.py` (RAG Context Injection and Prompt Engineering).
- `src/benchmark.py` (Modified for metadata extraction and Top1 scoring).
- `tests/test_benchmark.py` (Config parameter overrides test environment).
- `report/REPORT.md` (Core analytical report updated with valid baseline comparisons).
- `exercises.md` (Submission checklist).

## Latest Validated Results
- **Unit Tests:** 42/42 passed (100% test coverage passed).
- **Benchmark (Recursive mặc định, size=1500):**
  - num_docs_loaded: 5
  - num_chunks_loaded: 100
  - retrieval_hit_rate: 1.0 (100% lọt top retrieval)
  - avg_keyword_score: 0.55 (Cải thiện rõ nét so với 0.2 ban đầu)
- **Strategy comparison:**
  - `fixed_size (500)`: chunks=134, hit_rate=0.8, avg_keyword=0.0
  - `sentence`: chunks=504, hit_rate=1.0, avg_keyword=0.0
  - `recursive (1500)`: chunks=100, hit_rate=1.0, avg_keyword=0.55
  - Best by hit_rate: sentence / recursive
  - Best by avg_keyword_score: recursive

## Notes For Report/Defense
- Yêu cầu Exercise 3.2 ĐÃ ĐẠT: Toàn bộ 5 benchmark queries đều có áp dụng metadata filtering thông qua thẻ `source` giúp lấy nội dung hoàn toàn chính xác theo tác phẩm.
- Yêu cầu Exercise 3.1 & 3.4 ĐÃ ĐẠT: Code bencharmk chạy mượt mà giữa các strategy để lập bảng biểu đồ so sánh.
- Phân tích lỗi (Failure Pattern): Do mô hình embedding `all-MiniLM-L6-v2` huấn luyện bằng tiếng Anh nên nó có xu hướng khó phân biệt ngữ nghĩa Tiếng Việt (điển hình query 1). Cách báo cáo: nhấn mạnh vào việc chuyển qua dùng embedding tiếng việt như PhoBERT để khắc phục.

## Run Commands
- **Chạy UnitTest Report:**
  `python -m pytest tests/ -v`
- **Chạy Code Phân Tích (Custom script sinh ra):**
  `python temp_benchmark.py`
- **Chạy Benchmark chính hãng của test:**
  `python -m tests.test_benchmark`
