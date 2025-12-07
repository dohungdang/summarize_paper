## 🎨 Frontend - Txt

Phần Frontend của dự án **Txt** được xây dựng bằng **HTML, CSS và JavaScript thuần**.

Chức năng chính:
- Nhập văn bản cần tóm tắt
- Hiển thị kết quả tóm tắt
- Sao chép và xoá nội dung nhanh chóng
- Hiển thị lịch sử tóm tắt
- Giao diện đăng nhập (UI phía frontend)

Giao diện đơn giản, dễ sử dụng và hoạt động tốt trên nhiều thiết bị.
Hướng dẫn triển khai Frontend trên local bằng cách: 
- Sau khi tải file Frontend về máy, ta sử dụng Bash: python -m http.server 5500 để đưa frontend lên local http://127.0.0.1:5500/. Việc đưa lên local thực hiện tại thư mục frontend đã tải về từ git
- Khi đó ta kết nối với Backend đã được chạy trên local bằng 2 API http://127.0.0.1:8000/upload_paper và http://127.0.0.1:8000/summarize_text để thực hiện xử lý tóm tắt văn bản upload_paper là xử lý file pdf và summarize_text là để xử lý văn bản


