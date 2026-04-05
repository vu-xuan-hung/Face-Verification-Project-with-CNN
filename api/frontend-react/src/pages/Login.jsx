import { useState, useRef, useEffect } from 'react';
import { Camera, Upload } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

export default function Login() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null); 
  const [preview, setPreview] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    let stream = null;
    const startWebcam = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" }
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setResult({ type: 'error', msg: 'Cannot access camera. Please upload an image.' });
      }
    };
    startWebcam();
    return () => {
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, []);

  const captureAndResize = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Crop center square
    const size = Math.min(video.videoWidth, video.videoHeight);
    const startX = (video.videoWidth - size) / 2;
    const startY = (video.videoHeight - size) / 2;
    
    ctx.drawImage(video, startX, startY, size, size, 0, 0, 128, 128);
    const base64Image = canvas.toDataURL('image/jpeg', 0.9);
    setPreview(base64Image);
    return base64Image;
  };

  const sendToBackend = async (base64Image) => {
    setLoading(true);
    setResult(null);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image })
      });
      const data = await response.json();
      setLoading(false);
      
      if (data.success) {
        setResult({ type: 'success', msg: `Thành công! Quyền: ${data.role}` });
        localStorage.setItem('username', data.username);
        localStorage.setItem('role', data.role);
        localStorage.setItem('last_login', new Date().toLocaleString());
        
        setTimeout(() => {
          navigate(data.role === 'admin' ? '/admin' : '/user');
        }, 1200);
      } else {
        setResult({ type: 'error', msg: `Thất bại: ${data.message}` });
      }
    } catch (err) {
      setLoading(false);
      setResult({ type: 'error', msg: 'Lỗi kết nối tới Server (FastAPI backend).' });
    }
  };

  const handleCapture = () => {
    if (!videoRef.current || !videoRef.current.srcObject) return;
    const base64 = captureAndResize();
    sendToBackend(base64);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        const size = Math.min(img.width, img.height);
        const startX = (img.width - size) / 2;
        const startY = (img.height - size) / 2;
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, startX, startY, size, size, 0, 0, 128, 128);
        
        const base64Image = canvas.toDataURL('image/jpeg', 0.9);
        setPreview(base64Image);
        sendToBackend(base64Image);
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="card-container login-container">
      <h1>Face Login System</h1>
      <p className="subtitle">Position your face clearly in the camera</p>
      
      <div className="video-wrapper">
        <video ref={videoRef} className="webcam" autoPlay playsInline muted></video>
        <canvas ref={canvasRef} width="128" height="128" className="hidden"></canvas>
        <div className="scanning-frame"></div>
      </div>

      <div className="controls">
        <button className="primary-btn" onClick={handleCapture} disabled={loading}>
          <Camera size={20} /> Capture & Login
        </button>
        
        <div className="divider"><span>OR</span></div>
        
        <label htmlFor="imageUpload" className="secondary-btn">
          <Upload size={18} /> Upload Image
        </label>
        <input type="file" id="imageUpload" accept="image/*" className="hidden" onChange={handleFileUpload} />
      </div>
      
      {loading && (
        <div className="spinner-container">
          <div className="spinner"></div>
          <p>Verifying face...</p>
        </div>
      )}

      {result && (
        <div className={`result-box ${result.type}`}>
          {result.msg}
        </div>
      )}
      
      {preview && (
        <div className="preview-container">
          <p>Image request payload (128x128):</p>
          <img src={preview} alt="Preview" className="preview-img" />
        </div>
      )}
    </div>
  );
}
