# Danh sách công việc Lab 7: Nền Tảng Dữ Liệu: Embedding & Vector Store

Dựa trên cấu trúc của bài lab, đây là danh sách các công việc đã phân chia và hoàn thành thành công 100%:

## Phase 1: Hoàn thành Core Coding
- [x] **Trong `src/chunking.py`:**
  - [x] Implement `SentenceChunker`.
  - [x] Implement `RecursiveChunker`.
  - [x] Implement `compute_similarity`.
  - [x] Implement `ChunkingStrategyComparator`.
- [x] **Trong `src/store.py`:**
  - [x] Implement `EmbeddingStore.__init__`.
  - [x] Implement `EmbeddingStore.add_documents`.
  - [x] Implement `EmbeddingStore.search`.
  - [x] Implement `EmbeddingStore.get_collection_size`.
  - [x] Implement `EmbeddingStore.search_with_filter`.
  - [x] Implement `EmbeddingStore.delete_document`.
- [x] **Trong `src/agent.py`:**
  - [x] Implement `KnowledgeBaseAgent.answer` (chuẩn kiến trúc RAG).

## Phase 2: So Sánh Retrieval Strategy
- [x] **Exercise 1.1 & 1.2:** Trả lời khái niệm Cosine Similarity và tính số chunk an toàn.
- [x] **Exercise 3.0:** Chuẩn bị 5 tài liệu truyện ngắn Nam Cao với metadata schema.
- [x] **Exercise 3.1:** Chạy baseline & chọn `RecursiveChunker` tối ưu (chunk_size=1500).
- [x] **Exercise 3.2:** Thống nhất 5 benchmark queries với gold answers, tích hợp sẵn filter.
- [x] **Exercise 3.3:** Dự đoán Cosine Similarity bằng mắt thường vs kết quả hàm chạy thực tế.
- [x] **Exercise 3.4:** Chạy Benchmark & So sánh, tính ra Keyword Score 0.55.
- [x] **Exercise 3.5:** Phân tích Failure case (lỗi từ embedding tiếng anh).

## Tổng Kết
- [x] Fix triệt để bug in-memory database để pass toàn bộ `pytest tests/ -v` (42/42 Tests OK).
- [x] Điền dữ liệu thật 100% vào form báo cáo `report/REPORT.md`.
