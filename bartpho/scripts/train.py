"""
Fine-tuning BARTpho model for Vietnamese text summarization.
1. Nhận tham số dòng lệnh (model, đường dẫn dữ liệu, output).
2. Load tokenizer + model BARTpho.
3. Tiền xử lý dữ liệu tóm tắt.
4. Thiết lập Trainer của HuggingFace.
5. Huấn luyện và đánh giá bằng ROUGE.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset
import torch
import evaluate
import numpy as np
import pandas as pd
from typing import Dict

# Thiết lập mô hình – lấy từ argv hoặc dùng giá trị mặc định
model_name = sys.argv[1] if len(sys.argv) > 1 else "vinai/bartpho-word"
data_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
output_dir = sys.argv[3] if len(sys.argv) > 3 else "outputs/bartpho-finetuned"
os.makedirs(output_dir, exist_ok=True)

# Khởi tạo tokenizer và mô hình BARTpho
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to("cuda")

# Hàm load file CSV và chuyển thành HuggingFace Dataset
def load_csv(path):
    """
    Load file CSV và chuyển đổi sang Dataset.
    
    Args:
        path (str): đường dẫn tới file CSV
    
    Returns:
        Dataset: Dataset HuggingFace chứa dữ liệu dạng bảng
    """
    df = pd.read_csv(path)
    return Dataset.from_pandas(df)

# Load tập train/val/test
train_ds = load_csv(os.path.join(data_dir,"train.csv"))
val_ds = load_csv(os.path.join(data_dir,"val.csv"))
test_ds = load_csv(os.path.join(data_dir,"test.csv"))

# Tiền xử lý dữ liệu – Tokenize input & summary
max_input_length = 1024
max_target_length = 256
def preprocess_function(examples):
    """
    Tokenize văn bản đầu vào và summary.
    
    Args:
        examples (Dict): batch dữ liệu gồm "text" và "summary"
    
    Returns:
        Dict: tokenized input và labels
    """
    inputs = examples["text"]
    targets = examples["summary"]
    # Tokenize phần văn bản gốc
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Tokenize phần summary (mục tiêu)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Chuyển đổi dữ liệu đã tokenize
train_block = train_ds.map(preprocess_function, batched=True, remove_columns=train_ds.column_names)
val_block = val_ds.map(preprocess_function, batched=True, remove_columns=val_ds.column_names)
test_block = test_ds.map(preprocess_function, batched=True, remove_columns=test_ds.column_names)

# Data collator – padding động cho Seq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Hàm tính ROUGE cho tóm tắt
rouge = evaluate.load("rouge")
def postprocess_text(preds, labels):
    """
    Làm sạch kết quả sinh và nhãn (loại khoảng trắng thừa).
    
    Args:
        preds (list[str]): danh sách summary dự đoán
        labels (list[str]): danh sách summary thật
    
    Returns:
        tuple: (preds_clean, labels_clean)
    """
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def compute_metrics(eval_pred):
    """
    Hàm đánh giá mô hình bằng ROUGE.

    Args:
        eval_pred (tuple): (predictions, labels)

    Returns:
        dict: kết quả ROUGE và độ dài trung bình của câu sinh
    """
    predictions, labels = eval_pred
    # Decode dự đoán
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Thay -100 bằng token PAD để decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Làm sạch text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    # Tính ROUGE
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    # Độ dài trung bình của chuỗi sinh
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# Cấu hình tham số training của HuggingFace Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch", #mỗi khi kết thúc 1 epoch, mô hình sẽ tự động đánh giá trên tập valid 
    per_device_train_batch_size=2, #batch_size khi train
    per_device_eval_batch_size=2, #batch size khi val
    predict_with_generate=True, #Khi evaluate, mô hình tự sinh ra văn bản
    do_train=True, #cho phép huấn luyện
    do_eval=True, #cho phép đánh giá mô hình trong lúc huấn luyện
    logging_strategy="steps", #ghi log sau mỗi bước cố định
    logging_steps=200, #cứ 200 bước train sẽ ghi log 1 lần
    save_strategy="epoch", #mỗi epoch sẽ lưu checkpoint 1 lần
    save_total_limit=3, #chỉ giữ lại tối đa 3 checkpoint gần nhất(tốt nhất)
    num_train_epochs=4, #huấn luyện mô hình trong 4 epochs
    learning_rate=3e-5, #tốc độ học
    weight_decay=0.01, #hệ số regularization -> giảm overfitting
    warmup_steps=500, #trong 500 bước đầu tiên, learning rate sẽ tăng dần từ 0 -> learning rate gốc
    fp16=True, #công cụ hỗ trợ train nhanh và tiết kiệm dung lượng hơn
    gradient_accumulation_steps=4, #giả lập batch size lớn hơn
    generation_max_length=256, #chiều dài tối đa khi sinh văn bản tóm tắt
    generation_num_beams=4 #Dùng Beam Search với 4 beam để tạo tóm tắt chất lượng hơn so với greedy search.
)

# Khởi tạo Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_block, #dataset đã xử lý
    eval_dataset=val_block, #dataset valid 
    tokenizer=tokenizer, # khi train thì chuyển văn bản thành input_ids(ID), khi evaluate thì chuyển ID ngược lại thành câu chữ
    data_collator=data_collator, #gom dữ liệu thành batch đúng kích thước batch size
    compute_metrics=compute_metrics #hàm để tính các chỉ số sau khi tạo tóm tắt
)

trainer.train()
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
