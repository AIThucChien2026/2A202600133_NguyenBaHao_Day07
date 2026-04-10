import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.embeddings import _local_embedd
from src.chunking import compute_similarity

pairs = [
    ("Con mèo đang ngủ trên ghế sofa.", "Mèo con nằm say giấc trên ghế dài.", "High"),
    (
        "Thời tiết hôm nay rất đẹp và nắng ấm.",
        "Hôm nay trời nắng và ấm áp lạ thường.",
        "High",
    ),
    (
        "Trí tuệ nhân tạo đang thay đổi thế giới.",
        "Buổi tối tôi thường ăn cơm với thịt gà.",
        "Low",
    ),
    (
        "Ngân hàng trung ương quyết định tăng lãi suất.",
        "Bitcoin vượt mốc 100k đô la.",
        "Low",
    ),
    ("Hello world", "Hello world", "Tuyệt đối"),
]

print("| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |")
print("|------|-----------|-----------|---------|--------------|-------|")

for i, (a, b, pred) in enumerate(pairs, 1):
    vec_a = _local_embedd(a)
    vec_b = _local_embedd(b)
    score = compute_similarity(vec_a, vec_b)

    # Nếu điểm > 0.5 thì coi là High, dưới 0.5 là Low
    actual = "high" if score > 0.5 else "low"
    is_correct = "✅" if actual.lower() == pred.lower() or score == 1.0 else "❌"

    print(f"| {i} | {a} | {b} | {pred} | {score:.4f} | {is_correct} |")
