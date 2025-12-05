"""
Tiền xử lý dữ liệu

1. Đọc file CSV chứa các cột: text, summary.
2. Loại bỏ các dòng thiếu dữ liệu.
3. Chuẩn hóa kiểu dữ liệu.
4. Tính độ dài văn bản và phân loại theo bins độ dài.
5. Chia dữ liệu thành train/validation/test theo tỷ lệ 80/10/10.
6. Ghi output ra thư mục data/processed.

"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/data.csv"
output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_csv)
# Kiểm tra bắt buộc có 2 cột: text và summary
if 'text' not in df.columns or 'summary' not in df.columns:
    raise SystemExit("Require columns: text, summary")
# Loại bỏ dòng thiếu dữ liệu
df = df.dropna(subset=['text','summary']).reset_index(drop=True)
df['text'] = df['text'].astype(str)
df['summary'] = df['summary'].astype(str)
# FEATURE ENGINEERING
# Tính số từ của text và summary
df['len_text'] = df['text'].str.split().str.len()
df['len_summary'] = df['summary'].str.split().str.len()
# Phân loại văn bản thành nhóm theo độ dài
# Ví dụ: 0–200 từ, 200–400 từ, 400–800 từ, >800 từ
bins = [0,200,400,800,10000]
df['len_bin'] = pd.cut(df['len_text'], bins=bins, labels=False, include_lowest=True)
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)
train[['text','summary']].to_csv(os.path.join(output_dir,'train.csv'), index=False)
val[['text','summary']].to_csv(os.path.join(output_dir,'val.csv'), index=False)
test[['text','summary']].to_csv(os.path.join(output_dir,'test.csv'), index=False)
