document.addEventListener('DOMContentLoaded', () => {
    const inputText = document.getElementById('inputText');
    const summaryText = document.getElementById('summaryText');
    const summarizeBtn = document.getElementById('summarizeBtn');

    const copyBtn = document.getElementById('copyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const summaryLength = document.getElementById('summaryLength');

    const loginLink = document.querySelector('nav a[href="login"]');
    const miniLogin = document.getElementById('miniLogin');
    const closeMini = document.getElementById('closeMini');

    const pdfFileInput = document.getElementById('pdfFileInput');
    const uploadPdfBtn = document.getElementById('uploadPdfBtn');
    const pdfSummaryText = document.getElementById('pdfSummaryText');

    const historyList = document.getElementById('historyList');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');


    loginLink.addEventListener('click', (e) => {
        e.preventDefault();
        miniLogin.style.display = 'block';
    });

    closeMini.addEventListener('click', () => {
        miniLogin.style.display = 'none';
    });
     // ====== LỊCH SỬ ======
    function saveToHistory(original, result) {
        const history = JSON.parse(localStorage.getItem('history')) || [];

        history.unshift({
            input: original,
            summary: result,
            time: new Date().toLocaleString()
        });

        localStorage.setItem('history', JSON.stringify(history));
        renderHistory();
    }

    function renderHistory() {
        const history = JSON.parse(localStorage.getItem('history')) || [];
        historyList.innerHTML = '';

        history.forEach(item => {
            const li = document.createElement('li');
            li.classList.add("history-item");

            li.innerHTML = `
                <div><b>${item.time}</b></div>
                <div>📄 Gốc: ${item.input.substring(0, 60)}...</div>
                <div>✂️ Tóm tắt: ${item.summary.substring(0, 60)}...</div>
            `;

            li.addEventListener('click', () => {
                inputText.value = item.input;
                summaryText.value = item.summary;
            });

            historyList.appendChild(li);
        });
    }

    if (historyList) renderHistory();

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            localStorage.removeItem('history');
            renderHistory();
        });
    }
    //uploadpdf
    uploadPdfBtn.addEventListener('click', async () => {
        const file = pdfFileInput.files[0];
        if (!file) return alert('Vui lòng chọn file PDF.');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://127.0.0.1:8000/upload_paper', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Backend error:', errorText);
                pdfSummaryText.value = 'Lỗi khi tóm tắt PDF!';
                return;
            }

            const data = await response.json();
            pdfSummaryText.value = data.summary;
            saveToHistory(file.name, data.summary, 'pdf');

        } catch (error) {
            console.error('Network error:', error);
            pdfSummaryText.value = 'Không thể kết nối server!';
        }
    });

    summarizeBtn.addEventListener('click', async() => {
        const originalText = inputText.value.trim();
        if (originalText === '') {
            alert('Vui lòng nhập văn bản để tóm tắt.');
            return;
        }
        
        
        // --- Đây là nơi bạn sẽ tích hợp API hoặc thuật toán tóm tắt thực tế ---
        // Trong ví dụ này, tôi sẽ mô phỏng việc tóm tắt bằng cách cắt bớt văn bản
         const lengthValue = summaryLength.value; // từ <select id="summaryLength">
            if (!["short", "medium", "long"].includes(lengthValue)) {
                alert('Độ dài tóm tắt không hợp lệ!');
                return;
            }

     

            try {
                const response = await fetch('http://127.0.0.1:8000/summarize_text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: originalText, length: lengthValue })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('Backend error:', errorText);
                    alert('Lỗi khi gọi API tóm tắt!\n' + errorText);
                    return;
                }

                const data = await response.json();

                // --------- Giải pháp frontend ---------
                let summary = data.summary;

                // Cắt bớt nếu quá dài
                const maxWords = lengthValue === "short" ? 20
                                : lengthValue === "medium" ? 40
                                : 80;

                const words = summary.split(/\s+/);
                if (words.length > maxWords) {
                    summary = words.slice(0, maxWords).join(' ') + '...';
                }

                summaryText.value = summary;

                // Lưu vào lịch sử
                saveToHistory(originalText, summary);

            } catch (error) {
                console.error('Network error:', error);
                alert('Có lỗi xảy ra khi kết nối với server!');
            }
    });

    copyBtn.addEventListener('click', () => {
        if (summaryText.value.trim() === '') {
            alert('Không có văn bản tóm tắt để sao chép.');
            return;
        }
        summaryText.select();
        summaryText.setSelectionRange(0, 99999); // For mobile devices
        document.execCommand('copy');
        alert('Văn bản tóm tắt đã được sao chép!');
    });

    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        summaryText.value = '';
        alert('Đã xóa tất cả văn bản.');
    });

   
        const numWordsToKeep = Math.floor(words.length * percentage);
        
        return words.slice(0, numWordsToKeep).join(' ') + '... (Văn bản tóm tắt mô phỏng)';
    }

);