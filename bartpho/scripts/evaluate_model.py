"""
Evaluation Script for Text Summarization Model

1. Sinh tóm tắt cho toàn bộ mẫu trong test.csv.
2. Tính các chỉ số đánh giá:
      - ROUGE-1, ROUGE-2, ROUGE-L
      - BERTScore-F1
      - Cosine Similarity giữa TF-IDF vector
3. Lưu kết quả bao gồm summary dự đoán + các metric vào CSV.

Output cuối:
    data/evaluate/output_predicted_evaluated.csv

Yêu cầu dữ liệu đầu vào:
    test.csv chứa các cột ["text", "summary"]
"""
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Load model & Tokenizer
model_path = "outputs/bartpho-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
model = model.to(device)

# Hàm tóm tắt
def summarize_text(text):
    """
    Sinh bản tóm tắt cho một đoạn văn bản đầu vào.

    Args:
        text (str): nội dung cần sinh tóm tắt.

    Returns:
        str: câu tóm tắt sau khi mô hình decode.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512*2).to(device)
    ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128*2,
        min_length=30,
        num_beams=4
    )
    return tokenizer.decode(ids[0], skip_special_tokens=True)

# Thông số cosine
def compute_cosine(a, b):
    """
    Tính cosine similarity giữa 2 văn bản dựa trên TF-IDF.

    Args:
        a (str): văn bản 1
        b (str): văn bản 2

    Returns:
        float: giá trị similarity từ 0→1
    """
    tfidf = TfidfVectorizer()
    vecs = tfidf.fit_transform([a, b])
    return cosine_similarity(vecs[0], vecs[1])[0][0]

# Load data
df = pd.read_csv("data/processed/test.csv")

# Sinh tóm tắt
preds = []
for text in df["text"]:
    preds.append(summarize_text(text))

df["summary_pred"] = preds

# Rouge Score
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

rouge_1_list = []
rouge_2_list = []
rouge_l_list = []

# Tính ROUGE cho từng mẫu
for pred, ref in zip(df["summary_pred"], df["summary"]):
    scores = scorer.score(ref, pred)
    rouge_1_list.append(scores["rouge1"].fmeasure)
    rouge_2_list.append(scores["rouge2"].fmeasure)
    rouge_l_list.append(scores["rougeL"].fmeasure)

df["ROUGE-1"] = rouge_1_list
df["ROUGE-2"] = rouge_2_list
df["ROUGE-L"] = rouge_l_list

# BertScore
P, R, F1 = bert_score(df["summary_pred"].tolist(),
                      df["summary"].tolist(),
                      lang="vi",
                      verbose=True)

df["BERTScore_F1"] = F1.tolist()

# Thông số Cosine
cos_list = []
for i in range(len(df)):
    cos_list.append(
        compute_cosine(df["text"][i], df["summary_pred"][i])
    )

df["Cosine_Similarity"] = cos_list


df.to_csv("data/evaluate/output_predicted_evaluated.csv", index=False)
print("DONE → data/evaluate/output_predicted_evaluated.csv")

