from pathlib import Path
from typing import Literal
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Thư mục model local: <project_root>/model
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"

if not MODEL_DIR.exists():
    raise RuntimeError(f"Model directory not found: {MODEL_DIR}")

MODEL_NAME = MODEL_DIR.name

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)

device = torch.device("cpu")
model.to(device)
model.eval()

LengthType = Literal["short", "medium", "long"]

LENGTH_TO_MAX_TOKENS = {
    "short": 64,
    "medium": 128,
    "long": 256,
}


def summarize_text(text: str, length: LengthType = "medium") -> str:
    """Tóm tắt 1 đoạn text ngắn/trung bình ,không dùng RAG, chỉ là seq2seq thuần """
    text = (text or "").strip()
    if not text:
        return ""

    max_new_tokens = LENGTH_TO_MAX_TOKENS.get(length, 128)

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            length_penalty=1.0,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()
