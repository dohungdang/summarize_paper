"""
BARTpho Batch & Stream Inference Script

Script này hỗ trợ hai chế độ chạy:

1) **Batch mode (JSON input file)**
   - Cú pháp:
         python infer.py <model_dir> <input_json>
   - File JSON phải là danh sách object dạng:
         [
           {"id": 1, "text": "..."},
           {"id": 2, "text": "..."}
         ]
   - Trả về JSON chứa summary tương ứng cho từng item.

2) **Streaming mode (stdin)**
   - Cú pháp:
         echo "văn bản cần tóm tắt" | python infer.py <model_dir>
   - Mỗi dòng nhập sẽ được mô hình tóm tắt và in ngay.

Mục đích: hỗ trợ triển khai inference linh hoạt cho mô hình Seq2Seq.
"""
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json

# Configuration
model_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/bartpho-finetuned"
input_file = sys.argv[2] if len(sys.argv) > 2 else None

# Tải tokenizer và mô hình
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Hàm tóm tắt
def summarize(text):
    """
    Sinh tóm tắt cho một đoạn văn bản đầu vào.

    Args:
        text (str): nội dung cần tóm tắt

    Returns:
        str: bản tóm tắt sau khi decode
    """
    # Tokenize input, đảm bảo cắt bớt khi vượt độ dài 1024 token
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    # Sinh kết quả bằng beam search (num_beams=4 để cân bằng chất lượng & tốc độ)
    out = model.generate(inputs["input_ids"], attention_mask=inputs.get("attention_mask"), max_length=256, num_beams=4)
    # Decode sang text (bỏ token đặc biệt)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# Batch Mode
if input_file:
    with open(input_file, "r", encoding="utf-8") as f:
        items = json.load(f)
    outputs = []
    # Xử lý từng document trong danh sách
    for item in items:
        s = summarize(item.get("text",""))
        outputs.append({"id": item.get("id"), "summary": s})
    # In toàn bộ output dưới dạng JSON format đẹp
    print(json.dumps(outputs, ensure_ascii=False, indent=2))
else:
    # Chế độ đọc từng dòng từ terminal
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        print(summarize(text))
