
## Cài đặt và chạy dự án

### Bước 1: Cài đặt môi trường ảo cho Backend
1. Khởi động backend :
   ```bash
   cd backend
   python -m venv venv ( lần đầu thì chạy )
   Windows: .\venv\Scripts\activate
   Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt ( lần đầu thì chạy )
   uvicorn main:app --reload
Sau khi chạy xong backend sẽ chạy ở cổng http://127.0.0.1:8000/docs
2. Khởi động frontend :
   python -m http.server 5500
Sau khi chạy xong frontend sẽ chạy ở cổng http://127.0.0.1:5500



