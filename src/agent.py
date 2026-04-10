from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn

    def answer(
        self, question: str, top_k: int = 15, score_threshold: float = 0.5
    ) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # Bước 1: Retrieves
        results = self.store.search(question, top_k=top_k)

        # Lọc ra các Chunk có độ trùng khớp trên ngưỡng cho phép (Score >= threshold)
        filtered_results = [res for res in results if res["score"] >= score_threshold]

        # Bước 2: Build prompt
        context = ""
        for i, res in enumerate(filtered_results, 1):
            source = res["metadata"].get("source", "Unknown")
            context += f"[Chunk {i} - Nguồn: {source} - Độ khớp: {res['score']:.2f}]\n{res['content']}\n\n"

        prompt = (
            "<persona>\n"
            "Bạn là một chuyên gia phân tích dữ liệu văn bản xuất sắc. Nhiệm vụ của bạn là đọc hiểu tài liệu và trả lời câu hỏi trực tiếp, ngắn gọn, không đưa thông tin thừa so với câu hỏi.\n"
            "</persona>\n\n"
            "<rules>\n"
            "1. Chỉ sử dụng thông tin được cung cấp trong thẻ <context>.\n"
            "2. Tuyệt đối không được phép sử dụng bất kỳ kiến thức bên ngoài cho dùng dữ liệu có trong thẻ <context>.\n"
            "3. Nếu <context> không chứa thông tin để trả lời, phải trả về chính xác chuỗi: 'Tôi không biết'.\n"
            "</rules>\n\n"
            "<constraints>\n"
            "1. Trả về kết quả dưới dạng văn bản bình thường, rõ ràng, dễ đọc.\n"
            "2. Nếu có thông tin để trả lời thì Ở cuối câu trả lời, XUỐNG DÒNG VÀ LIỆT KÊ TRỰC TIẾP danh sách tên các nguồn tài liệu đã tra cứu theo cú pháp: 'Nguồn: tên tác giả - tên truyện'.\n"
            "</constraints>\n\n"
            "<context>\n"
            f"{context}\n"
            "</context>\n\n"
            "<question>\n"
            f"{question}\n"
            "</question>\n"
        )

        # Bước 3: Call LLM
        return self.llm_fn(prompt)
