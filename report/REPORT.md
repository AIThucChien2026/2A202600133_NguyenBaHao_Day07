# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Bá hào
**Nhóm:** 24
**Ngày:** 11/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* là thể hiện sự tương đồng cao về mặt ngữ nghĩa giữa 2 sentence.Khi nó càng cao thì mức độ liên quan càng lớn.

**Ví dụ HIGH similarity:**
- Sentence A:" con mèo ở trên sofa"
- Sentence B:"con mèo đang nằm  trên sofa"
- Tại sao tương đồng:2 câu đều chỉ cùng đối tượng nên sự tương đồng cao 

**Ví dụ LOW similarity:**
- Sentence A: "Chương trình này hay thế"
- Sentence B: "bạn ăn gì hôm nay"
- Tại sao khác: 2 câu này không có mối liên quan đến nhau tu context đến đối tượng

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:*
Vì text embeddings thường có độ dài khác nhau phụ thuộc vào độ dài của chunking. Euclidean distance đo khoảng cách bề mặt nên dễ bị sai lệch khi hai chunking có kích thước khác nhau. Cosine similarity tốt hơn vì nó chỉ đo góc (angle) bỏ qua sự bất đồng về độ dài của chunking.
### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* số chunking = (10000-50)/(500-50)+1
> *Đáp án:* 23

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:*: chunking count = (10000-100)/(500-100)+1 = 26 => chunking tăng lên. Vì khi nó nhiều hơn thì độ bảo phù ngữ cảnh giữa các chunk sẽ tốt hơn. giúp chunking không bị "gãy" ngữ cảnh.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & lý do chọn

**Domain:** Văn học Việt Nam (truyện ngắn Nam Cao).
Lý do: dữ liệu có chiều sâu ngữ nghĩa, phù hợp để kiểm tra retrieval theo nội dung và khả năng grounding.

### Data Inventory (bộ benchmark)

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | chi_pheo.txt | data nội bộ lab | 56,378 | source, extension, title, author, category, year |
| 2 | 1_bua_no.txt | data nội bộ lab | 15,811 | source, extension, title, author, category, year |
| 3 | quet_nha.txt | data nội bộ lab | 12,419 | source, extension, title, author, category, year |
| 4 | trang_sang.txt | data nội bộ lab | 15,929 | source, extension, title, author, category, year |
| 5 | tu_cach_mo.txt | data nội bộ lab | 11,769 | source, extension, title, author, category, year |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ | Vai trò |
|----------------|------|-------|---------|
| source | string | data/chi_pheo.txt | Truy vết nguồn chunk, dùng cho metadata filter |
| extension | string | .txt | Lọc theo loại file |
| title | string | Chí Phèo | Lọc theo tác phẩm |
| author | string | Nam Cao | Lọc theo tác giả |
| category | string | truyện ngắn | Lọc theo thể loại |
| year | string | 1941 | Lọc theo mốc thời gian |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu (chunk_size=500):

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| chi pheo.txt (56K) | FixedSizeChunker (`fixed_size`) | 126 | 497.0 | ❌ Cắt giữa câu |
| chi pheo.txt (56K) | SentenceChunker (`by_sentences`) | 313 | 178.9 | ✅ Giữ câu trọn vẹn |
| chi pheo.txt (56K) | RecursiveChunker (`recursive`) | 154 | 364.1 | ✅ Ưu tiên ranh giới tự nhiên |
| 1 bua no.txt (16K) | FixedSizeChunker (`fixed_size`) | 36 | 487.8 | ❌ Cắt giữa câu |
| 1 bua no.txt (16K) | SentenceChunker (`by_sentences`) | 109 | 143.9 | ✅ Giữ câu trọn vẹn |
| 1 bua no.txt (16K) | RecursiveChunker (`recursive`) | 45 | 349.4 | ✅ Ưu tiên ranh giới tự nhiên |
| trang sang.txt (16K) | FixedSizeChunker (`fixed_size`) | 36 | 491.1 | ❌ Cắt giữa câu |
| trang sang.txt (16K) | SentenceChunker (`by_sentences`) | 105 | 150.6 | ✅ Giữ câu trọn vẹn |
| trang sang.txt (16K) | RecursiveChunker (`recursive`) | 44 | 360.1 | ✅ Ưu tiên ranh giới tự nhiên |

### Strategy Của Tôi

**Loại:** RecursiveChunker (chunk_size=1500)

**Mô tả cách hoạt động:**
> RecursiveChunker thử cắt tài liệu theo thứ tự ưu tiên separator: `\n\n` (đoạn văn) → `\n` (xuống dòng) → `. ` (câu) → ` ` (từ) → `""` (ký tự). Khi một đoạn nhỏ hơn chunk_size thì giữ nguyên, nếu lớn hơn thì thử separator tiếp theo. Tôi tăng chunk_size từ 500 lên 1500 vì truyện ngắn Nam Cao có câu văn dài, mô tả chi tiết — chunk nhỏ dễ bị cắt giữa ý, mất ngữ cảnh quan trọng.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Truyện ngắn Nam Cao có cấu trúc đoạn văn rõ ràng (phân tách bằng `\n\n`), nên RecursiveChunker khai thác được ranh giới đoạn tự nhiên thay vì cắt cơ học theo số ký tự. Với chunk_size=1500, mỗi chunk chứa đủ ngữ cảnh (2-3 đoạn văn) giúp LLM hiểu và trả lời chính xác hơn. Benchmark cho thấy tăng chunk_size cải thiện keyword_score từ 0.20 lên 0.55.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 5 truyện | FixedSizeChunker (best baseline, hit_rate=0.8) | 134 | ~490 | Trung bình — cắt giữa câu |
| 5 truyện | SentenceChunker (hit_rate=1.0) | 504 | ~160 | Tốt nhưng chunk quá nhỏ |
| 5 truyện | **RecursiveChunker (1500) — của tôi** | 100 | ~1120 | **Tốt nhất** — avg_keyword=0.55 |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker (1500) + metadata filter | 8/10 | Chunk lớn giữ ngữ cảnh, filter chính xác | Chunk lớn có thể chứa noise |
| Hậu | RecursiveChunker (chunk_size=1000, overlap=150) | 9.5 | Giữ ngữ cảnh tốt, linh hoạt | Chunk nhỏ thiếu bao quát |
| Tú | Recursive | 9 | Giữ ngữ cảnh tốt, linh hoạt | Chunk nhỏ thiếu bao quát |


**Strategy nào tốt nhất cho domain này? Tại sao?**
> RecursiveChunker với chunk_size lớn (1500) kết hợp metadata filter theo `source` cho kết quả tốt nhất. Lý do: (1) Truyện ngắn cần ngữ cảnh lớn để LLM hiểu — chunk nhỏ mất ý. (2) Metadata filter giải quyết vấn đề Chí Phèo (56K chars) chiếm hết top-k vì document lớn hơn 4x các truyện khác. (3) RecursiveChunker tôn trọng ranh giới đoạn văn tự nhiên trong truyện.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Dùng regex gì để detect sentence? Xử lý edge case nào?*

(`src/chunking.py`, dòng 49–77):
- Em dùng `re.split()` với pattern: `(?:\. |\! |\? |\.\n|\." |\?" |\!" )` để tách text thành các câu
- Em thêm **Negative Lookbehind** `(?<!Mr)(?<!Dr)(?<!Mrs)(?<!Ms)(?<!v\.v)` để không cắt nhầm tại từ viết tắt
- Sau khi tách, em ghép các piece lại thành câu hoàn chỉnh bằng vòng lặp: nối piece vào `current_sentence`, khi gặp piece khớp `delimiter_pattern` thì cắt câu
- Cuối cùng em nhóm các câu theo `max_sentences_per_chunk` bằng `" ".join(sentences[i : i + max])` và trả về list chunks

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Algorithm hoạt động thế nào? Base case là gì?*

(`src/chunking.py`, dòng 98–145):
- Em dùng hàm `chunk()` gọi `_split(text, self.separators)` — separator mặc định: `["\n\n", "\n", ". ", " ", ""]`
- **Base case 1:** `len(current_text) <= chunk_size` → trả về nguyên đoạn `[current_text]`
- **Base case 2:** `remaining_separators` rỗng → cắt cứng từng `chunk_size` ký tự
- **Recursive step:** lấy separator đầu tiên, gọi `current_text.split(separator)` để tách thành parts
- Với mỗi part: nếu `len(part) <= chunk_size` thì em gộp vào `current_chunk`; nếu gộp sẽ vượt chunk_size thì em flush `current_chunk` ra `final_chunks` và bắt đầu chunk mới
- Nếu part quá lớn → gọi đệ quy `_split(part, next_separators)` với separator tiếp theo
- Nếu separator không tồn tại trong text → bỏ qua, thử separator tiếp: `_split(current_text, next_separators)`

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Lưu trữ thế nào? Tính similarity ra sao?*

**`__init__`** (`src/store.py`, dòng 18–39):
- Em nhận `embedding_fn` (mặc định fallback: `_mock_embed` → `_local_embedd`)
- Em khởi tạo `chromadb.Client()` (ephemeral in-memory), tạo collection với tên duy nhất `f"{collection_name}_{id(self)}"`
- Nếu ChromaDB lỗi → fallback sang in-memory store (`self._store = []`)

**`add_documents`** (`src/store.py`, dòng 69–96):
- Em duyệt từng `Document`, gọi `self._embedding_fn(doc.content)` để tạo embedding vector
- **ChromaDB mode:** gom ids, documents, embeddings, metadatas vào list → gọi `self._collection.add()`
- Nếu metadata rỗng → gán `{"_empty_": True}` để tránh ChromaDB lỗi
- **In-memory mode:** em gọi `_make_record(doc)` tạo dict `{id, content, metadata, embedding}` rồi `append` vào `self._store`

**`search`** (`src/store.py`, dòng 98–125):
- Em gọi `self._embedding_fn(query)` để embed câu hỏi thành vector
- **ChromaDB mode:** em gọi `self._collection.query(query_embeddings=[query_vec], n_results=top_k)`, chuyển distance sang score bằng `1.0 / (1.0 + distance)`
- **In-memory mode:** em gọi `_search_records()` — duyệt tất cả records, tính `compute_similarity(query_vec, record["embedding"])`, sort giảm dần theo score, trả `top_k` đầu tiên

**`search_with_filter` + `delete_document`** — approach:
> *Filter trước hay sau? Delete bằng cách nào?*

**`search_with_filter`** (`src/store.py`, dòng 134–177):
- Em dùng chiến lược **Pre-filtering** — lọc trước, search sau
- **ChromaDB mode:** em truyền `where=metadata_filter` vào `self._collection.query()` — ChromaDB tự filter ở server-side
- **In-memory mode:** em duyệt `self._store`, chỉ giữ record nào `record["metadata"][k] == v` với mọi `(k, v)` trong `metadata_filter` → rồi em gọi `_search_records()` trên tập đã lọc

**`delete_document`** (`src/store.py`, dòng 179–205):
- **ChromaDB mode:** em thử 2 cách xóa: `delete(ids=[doc_id])` và `delete(where={"doc_id": doc_id})`, so sánh `count()` trước/sau để return True/False
- **In-memory mode:** dùng list comprehension lọc bỏ record có `id == doc_id` hoặc `metadata["doc_id"] == doc_id`, so sánh `len` trước/sau

### KnowledgeBaseAgent

**`answer`** — approach:
> *Prompt structure? Cách inject context?*

(`src/agent.py`, dòng 21–59) — kiến trúc RAG 3 bước:
- **Bước 1 — Retrieve:** em gọi `self.store.search(question, top_k=15)`, lọc kết quả qua `score_threshold=0.5` (chỉ giữ chunk có score ≥ 0.5)
- **Bước 2 — Build prompt:** em ghép các chunk vào chuỗi `context`, mỗi chunk hiển thị `[Chunk i - Nguồn: source - Độ khớp: score]`. Prompt gồm 4 thẻ XML:
  - `<persona>`: vai trò chuyên gia phân tích văn bản
  - `<rules>`: chỉ dùng thông tin trong `<context>`, không dùng kiến thức ngoài, trả "Tôi không biết" nếu không có thông tin
  - `<constraints>`: trả lời rõ ràng, liệt kê nguồn cuối câu trả lời
  - `<context>` + `<question>`: dữ liệu retrieved + câu hỏi
- **Bước 3 — Generate:** em gọi `self.llm_fn(prompt)` và trả về kết quả

### Bonus task (ChromaDB)

**Tích hợp ChromaDB thay thế in-memory test (`src/store.py`):**
- **Khởi tạo thông minh:** Khi tạo `EmbeddingStore`, hệ thống thử khởi tạo `chromadb.Client()` với collection động `f"{self._collection_name}_{id(self)}"`. Thiết lập động này giúp test chạy độc lập hoàn toàn không bị rò rỉ state. Nếu máy thiếu ChromaDB, tự động rơi vào `_use_chroma = False`.
- **Thêm Vector Native:** Hàm `add_documents` được nâng cấp. Thay vì đưa dict vào Local List, logic bóc tách `ids`, `documents`, `embeddings` và `metadatas` và đưa vào database qua API `self._collection.add()`.
- **Pre-filtering với Vector DB:** Nếu list thường phải gõ for loop kiểm tra từng `record["metadata"]`, em tận dụng luôn tham số `where=metadata_filter` của hàm `self._collection.query(...)`. ChromaDB tự động lọc thẻ bên trong C++ server-side trước khi trích xuất vector, tăng tốc độ đáng kể. Similarity được parse khéo léo qua `1.0 / (1.0 + res["distances"])`.
- **Xóa cực nhanh:** Hàm `delete_document` gọi thẳng 2 lệnh `self._collection.delete(ids=[doc_id])` và `delete(where={"doc_id": doc_id})`. Chức năng xoá DB native giúp drop an toàn document.
### Test Results

```bash
test_chunker_classes_exist (tests.test_solution.TestClassBasedInterfaces) ... ok
test_mock_embedder_exists (tests.test_solution.TestClassBasedInterfaces) ... ok      
test_counts_are_positive (tests.test_solution.TestCompareChunkingStrategies) ... ok  
test_each_strategy_has_count_and_avg_length (tests.test_solution.TestCompareChunkingStrategies) ... ok
test_returns_three_strategies (tests.test_solution.TestCompareChunkingStrategies) ... ok
test_identical_vectors_return_1 (tests.test_solution.TestComputeSimilarity) ... ok   
test_opposite_vectors_return_minus_1 (tests.test_solution.TestComputeSimilarity) ... ok
test_orthogonal_vectors_return_0 (tests.test_solution.TestComputeSimilarity) ... ok  
test_zero_vector_returns_0 (tests.test_solution.TestComputeSimilarity) ... ok        
test_add_documents_increases_size (tests.test_solution.TestEmbeddingStore) ... ok    
test_add_more_increases_further (tests.test_solution.TestEmbeddingStore) ... ok
test_initial_size_is_zero (tests.test_solution.TestEmbeddingStore) ... ok
test_search_results_have_content_key (tests.test_solution.TestEmbeddingStore) ... ok
test_search_results_have_score_key (tests.test_solution.TestEmbeddingStore) ... ok
test_search_results_sorted_by_score_descending (tests.test_solution.TestEmbeddingStore) ... ok
test_search_returns_at_most_top_k (tests.test_solution.TestEmbeddingStore) ... ok
test_search_returns_list (tests.test_solution.TestEmbeddingStore) ... ok
test_delete_reduces_collection_size (tests.test_solution.TestEmbeddingStoreDeleteDocument) ... ok
test_delete_returns_false_for_nonexistent_doc (tests.test_solution.TestEmbeddingStoreDeleteDocument) ... ok
test_delete_returns_true_for_existing_doc (tests.test_solution.TestEmbeddingStoreDeleteDocument) ... ok
test_filter_by_department (tests.test_solution.TestEmbeddingStoreSearchWithFilter) ... ok
test_no_filter_returns_all_candidates (tests.test_solution.TestEmbeddingStoreSearchWithFilter) ... ok
test_returns_at_most_top_k (tests.test_solution.TestEmbeddingStoreSearchWithFilter) ... ok
test_chunks_respect_size (tests.test_solution.TestFixedSizeChunker) ... ok
test_correct_number_of_chunks_no_overlap (tests.test_solution.TestFixedSizeChunker) ... ok
test_empty_text_returns_empty_list (tests.test_solution.TestFixedSizeChunker) ... ok 
test_no_overlap_no_shared_content (tests.test_solution.TestFixedSizeChunker) ... ok  
test_overlap_creates_shared_content (tests.test_solution.TestFixedSizeChunker) ... ok
test_returns_list (tests.test_solution.TestFixedSizeChunker) ... ok
test_single_chunk_if_text_shorter (tests.test_solution.TestFixedSizeChunker) ... ok  
test_answer_non_empty (tests.test_solution.TestKnowledgeBaseAgent) ... ok
test_answer_returns_string (tests.test_solution.TestKnowledgeBaseAgent) ... ok
test_root_main_entrypoint_exists (tests.test_solution.TestProjectStructure) ... ok
test_src_package_exists (tests.test_solution.TestProjectStructure) ... ok
test_chunks_within_size_when_possible (tests.test_solution.TestRecursiveChunker) ... ok
test_empty_separators_falls_back_gracefully (tests.test_solution.TestRecursiveChunker) ... ok
test_handles_double_newline_separator (tests.test_solution.TestRecursiveChunker) ... ok
test_returns_list (tests.test_solution.TestRecursiveChunker) ... ok
test_chunks_are_strings (tests.test_solution.TestSentenceChunker) ... ok
test_respects_max_sentences (tests.test_solution.TestSentenceChunker) ... ok
test_returns_list (tests.test_solution.TestSentenceChunker) ... ok
test_single_sentence_max_gives_many_chunks (tests.test_solution.TestSentenceChunker) ... ok

----------------------------------------------------------------------
Ran 42 tests in 1.517s

OK
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán (Bằng mắt) | Actual Score (Cosine) | Đoán Đúng Không? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Con mèo đang ngủ trên ghế sofa. | Mèo con nằm say giấc trên ghế dài. | High | 0.6352 | ✅ Đúng |
| 2 | Thời tiết hôm nay rất đẹp và nắng ấm. | Hôm nay trời nắng và ấm áp lạ thường. | High | 0.6270 | ✅ Đúng |
| 3 | Trí tuệ nhân tạo đang thay đổi thế giới. | Buổi tối tôi thường ăn cơm với thịt gà. | Low | 0.5322 | ❌ Sai (Điểm cao bất thường) |
| 4 | Ngân hàng trung ương quyết định tăng lãi suất. | Bitcoin vượt mốc 100k đô la. | Low | 0.3378 | ✅ Đúng |
| 5 | Hello world | Hello world | High | 1.0000 | ✅ Đúng (Chính xác tuyệt đối) |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:*
Bất ngờ nhất là cặp số 3: hai câu hoàn toàn khác chủ đề mà model vẫn cho 0.53 — khá cao. Nguyên nhân là do `all-MiniLM-L6-v2` được train chủ yếu trên tiếng Anh, nên khi gặp tiếng Việt nó phân biệt ngữ nghĩa kém hơn và dễ cho kết quả "giống nhau ảo". Kết luận thu được: chọn model embedding phù hợp với ngôn ngữ rất quan trọng.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

**Cấu hình benchmark:** `RecursiveChunker(chunk_size=1500)` + `LocalEmbedder(all-MiniLM-L6-v2)` + `GeminiFlashLiteLLM` + metadata filter theo `source`

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Nhân vật chính mở đầu truyện bằng hành động gì? | Chí Phèo vừa đi vừa chửi (chửi trời, chửi đời, chửi làng Vũ Đại) trong lúc say rượu |
| 2 | Trong truyện Một bữa no, bà lão sống bằng cách nào khi tuổi già sức yếu? | Bà lão đi ở cho nhiều nhà, xin ăn, gạ gẫm để kiếm bữa — đói triền miên |
| 3 | Trong truyện Quét nhà, vì sao cha mẹ của Hồng hay cáu gắt? | Vì gia đình túng thiếu, nợ nần, kinh tế khó khăn khiến tâm trạng nóng nảy |
| 4 | Trong truyện Trăng sáng, Điền thay đổi quan niệm nghệ thuật thế nào? | Từ mơ mộng nghệ thuật "ánh trăng lừa dối" sang nghệ thuật là "tiếng đau khổ" từ kiếp lầm than |
| 5 | Trong truyện Tư cách mõ, Lộ bị biến đổi tính cách do tác động xã hội nào? | Do bị khinh bỉ, làm nhục liên tục khiến Lộ mất tự trọng, trở nên đê tiện |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Nhân vật chính mở đầu truyện? | Chunk từ chi pheo.txt | 0.631 | ✅ | "Tôi không biết" (chunk không chứa đoạn mở đầu) |
| 2 | Bà lão sống bằng cách nào? | Chunk từ 1 bua no.txt | 0.625 | ✅ | Bà gạ gẫm ở cho nhiều nhà, bị ngã vỡ lọ, về hưu trí |
| 3 | Cha mẹ Hồng hay cáu gắt vì sao? | Chunk từ quet nha.txt | 0.655 | ✅ | Vì túng thiếu, khó khăn kinh tế, tâm trạng nóng như lửa |
| 4 | Điền thay đổi quan niệm nghệ thuật? | Chunk từ trang sang.txt | 0.614 | ✅ | Từ "ánh trăng lừa dối" sang "tiếng đau khổ từ kiếp lầm than" |
| 5 | Lộ bị biến đổi tính cách do gì? | Chunk từ tu cach mo.txt | 0.598 | ✅ | Bị khinh bỉ, làm nhục → mất tự trọng |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

**Bao nhiêu queries agent trả lời chính xác?** 4 / 5 (keyword_score trung bình: 0.55)

**Failure case:** Query 1 (Chí Phèo mở đầu) — retrieval đúng truyện nhưng chunk được lấy không chứa đoạn mở đầu truyện ("Hắn vừa đi vừa chửi"). Nguyên nhân: embedding model tiếng Anh không hiểu ngữ nghĩa "mở đầu truyện" trong tiếng Việt, nên có thể đã xếp hạng sai thứ tự chunk.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Thành viên dùng SentenceChunker cho thấy rằng chunk nhỏ (giữ ranh giới câu) tuy retrieval hit_rate cao (1.0) nhưng keyword_score thấp vì context quá ít cho LLM. Bài học: retrieval hit ≠ answer quality — cần cân bằng giữa precision và context size.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm sử dụng domain FAQ/SOP có kết quả tốt hơn nhiều vì cấu trúc câu hỏi-đáp tự nhiên phù hợp với RAG pattern. Điều này cho thấy việc chọn domain ảnh hưởng lớn đến chất lượng retrieval — văn xuôi tự sự khó hơn FAQ vì thông tin phân tán, không tập trung trong 1 chunk.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> (1) Sử dụng embedding model hỗ trợ tiếng Việt (ví dụ: `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`) thay vì `all-MiniLM-L6-v2` chỉ train trên tiếng Anh. (2) Thêm tiền xử lý metadata phong phú hơn: gắn thêm `chapter`, `paragraph_idx`, `characters_mentioned` để filter chính xác hơn. (3) Thử hybrid search (BM25 + semantic) vì keyword matching hoạt động tốt hơn cho tiếng Việt khi embedding model kém.

---



## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 4 / 5 |
| Bonus task (ChromaDB) | Cá nhân | 3 / 5 |
| **Tổng** | | **91 / 100** |
