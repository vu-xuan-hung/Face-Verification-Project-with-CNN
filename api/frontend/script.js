const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const imageUpload = document.getElementById('imageUpload');
const resultBox = document.getElementById('result');
const previewImg = document.getElementById('previewImg');
const previewContainer = document.getElementById('previewContainer');
const loading = document.getElementById('loading');

const API_URL = 'http://localhost:8000/predict';

// Initialize webcam
async function setupWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 640, height: 480, facingMode: "user" } 
        });
        video.srcObject = stream;
    } catch (err) {
        console.error("Camera access error: ", err);
        resultBox.textContent = "Không thể truy cập Webcam. Vui lòng upload ảnh.";
        showResult('error');
    }
}
setupWebcam();

// Process image before request
function captureAndResize() {
    // Generate square crop
    const size = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - size) / 2;
    const startY = (video.videoHeight - size) / 2;
    
    ctx.drawImage(video, startX, startY, size, size, 0, 0, 128, 128);
    const base64Image = canvas.toDataURL('image/jpeg', 0.9);
    
    previewImg.src = base64Image;
    previewContainer.classList.remove('hidden');
    
    return base64Image;
}

// Request to FastAPI Backend via fetch()
async function sendToBackend(base64Image) {
    loading.classList.remove('hidden');
    resultBox.classList.add('hidden');
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: base64Image })
        });
        
        const data = await response.json();
        
        loading.classList.add('hidden');
        
        if (data.success) {
            resultBox.textContent = `Đăng nhập thành công! Vai trò: ${data.role}`;
            showResult('success');
            
            // Save info to local storage
            localStorage.setItem('username', data.username);
            localStorage.setItem('role', data.role);
            
            const timeNow = new Date().toLocaleString();
            localStorage.setItem('last_login', timeNow);

            // Redirect to appropriate dashboard based on role
            setTimeout(() => {
                if (data.role === 'admin') {
                    window.location.href = 'admin.html';
                } else {
                    window.location.href = 'user.html';
                }
            }, 1000);

        } else {
            resultBox.textContent = `Đăng nhập thất bại: ${data.message || 'Unknown'}`;
            showResult('error');
        }
    } catch (err) {
        console.error("API error: ", err);
        loading.classList.add('hidden');
        resultBox.textContent = "Lỗi kết nối Server. Vui lòng bật FastAPI Backend.";
        showResult('error');
    }
}

function showResult(type) {
    resultBox.classList.remove('hidden', 'success', 'error');
    resultBox.classList.add(type);
}

captureBtn.addEventListener('click', () => {
    if(!video.srcObject) return;
    const base64Img = captureAndResize();
    sendToBackend(base64Img);
});

imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                const size = Math.min(img.width, img.height);
                const startX = (img.width - size) / 2;
                const startY = (img.height - size) / 2;
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, startX, startY, size, size, 0, 0, 128, 128);
                
                const base64Image = canvas.toDataURL('image/jpeg', 0.9);
                previewImg.src = base64Image;
                previewContainer.classList.remove('hidden');
                
                sendToBackend(base64Image);
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});
