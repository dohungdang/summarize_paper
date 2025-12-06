# Summarize Paper – Hệ thống tóm tắt văn bản đa mô hình

Dự án Summarize Paper được xây dựng nhằm đánh giá và so sánh hiệu quả giữa nhiều phương pháp tóm tắt văn bản, bao gồm ba thuật toán truyền thống (TextRank, KL-Sum, LSA) và một mô hình học sâu (PhoBERT fine-tune). Hệ thống hỗ trợ toàn bộ quy trình từ tiền xử lý, sinh tóm tắt, đánh giá chất lượng đến xuất báo cáo kết quả.

## 1. Cấu trúc dự án
Tham khảo file /bartpho/run_project.txt

## 2. Cách sử dụng mô hình BartPho

### 2.1. Yêu cầu hệ thống
1. Python >= 3.8
2. CUDA (khuyến nghị)

### 2.2. Cách chạy mô hình
1. Open cmd -> cd '/thu muc project/'
2. python -m venv bartpho_env
3. bartpho_env\Scripts\activate
4. pip install -r requirements.txt
5. python scripts/prepapre_data.py
6. python scripts/train.py vinai/bartpho-word data/processed outputs/bartpho-finetuned
7. python scripts/evaluate_model.py
8. python scripts/infer.py outputs/bartpho-finetuned "TEXT"

## 3. Đánh giá mô hình
Dự án sử dụng ba nhóm chỉ số chính:
1. ROUGE-1, ROUGE-2, ROUGE-L: đo mức độ trùng khớp giữa tóm tắt dự đoán và tóm tắt chuẩn
2. BERTScore-F1: đo mức độ tương đồng ngữ nghĩa trên không gian embedding
3. Cosine Similarity (TF-IDF): đo sự gần gũi của phân phối từ

