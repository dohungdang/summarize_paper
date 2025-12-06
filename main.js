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

    summarizeBtn.addEventListener('click', async() => {
        const originalText = inputText.value.trim();
        if (originalText === '') {
            alert('Vui lòng nhập văn bản để tóm tắt.');
            return;
        }
        
        
        // --- Đây là nơi bạn sẽ tích hợp API hoặc thuật toán tóm tắt thực tế ---
        // Trong ví dụ này, tôi sẽ mô phỏng việc tóm tắt bằng cách cắt bớt văn bản
        
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
