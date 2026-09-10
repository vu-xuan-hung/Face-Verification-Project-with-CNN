import { useState, useRef, useEffect } from 'react';
import { Camera, Upload } from 'lucide-react';
import { Navigate } from 'react-router-dom';
import { apiRequest } from '../auth-api';
import { useAuth } from '../auth-context';
import { isManager } from '../role-permissions';

export default function Login() {
  const { user, loading: restoring, error, login } = useAuth();
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const aliveRef = useRef(false);
  const busyRef = useRef(false);
  const requestRef = useRef(null);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      requestRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    if (restoring || user) return undefined;
    let disposed = false;
    let stream;
    const startWebcam = async () => {
      try {
        const media = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });
        if (disposed) {
          media.getTracks().forEach(track => track.stop());
          return;
        }
        stream = media;
        if (videoRef.current) videoRef.current.srcObject = media;
      } catch {
        if (!disposed) setResult({ type: 'error', msg: 'Cannot access camera. Please upload an image.' });
      }
    };
    startWebcam();
    return () => {
      disposed = true;
      stream?.getTracks().forEach(track => track.stop());
    };
  }, [restoring, user]);

  const sendToBackend = async (base64Image) => {
    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const response = await apiRequest('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image }),
        signal: controller.signal
      });
      const data = await response.json();
      if (!aliveRef.current || controller.signal.aborted) return;
      if (data.success) {
        await login(data.access_token, controller.signal);
      } else {
        setResult({ type: 'error', msg: data.message || 'Authentication failed. Contact your administrator to enroll.' });
      }
    } catch (err) {
      if (aliveRef.current && err.name !== 'AbortError') {
        setResult({ type: 'error', msg: err.message || 'Cannot connect to authentication server.' });
      }
    } finally {
      busyRef.current = false;
      if (aliveRef.current) setLoading(false);
    }
  };

  const beginAttempt = () => {
    if (busyRef.current) return false;
    busyRef.current = true;
    setLoading(true);
    setResult(null);
    return true;
  };
  const failImage = () => {
    busyRef.current = false;
    if (aliveRef.current) {
      setLoading(false);
      setResult({ type: 'error', msg: 'Cannot read this image. Please choose a valid image (maximum 10 MB).' });
    }
  };
  const capture = (source, width, height) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scale = Math.min(1, 1024 / Math.max(width, height));
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
    const encoded = canvas.toDataURL('image/jpeg', 0.9);
    setPreview(encoded);
    sendToBackend(encoded);
  };
  const handleCapture = () => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) {
      setResult({ type: 'error', msg: 'Camera is not ready. Please wait or upload an image.' });
      return;
    }
    if (!beginAttempt()) return;
    try { capture(video, video.videoWidth, video.videoHeight); } catch { failImage(); }
  };
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    event.target.value = '';
    if (!file || !beginAttempt()) return;
    if (!file.type.startsWith('image/') || file.size > 10 * 1024 * 1024) {
      failImage();
      return;
    }
    const reader = new FileReader();
    reader.onerror = failImage;
    reader.onload = () => {
      if (!aliveRef.current) return;
      const img = new Image();
      img.onerror = failImage;
      img.onload = () => {
        if (!aliveRef.current) return;
        try { capture(img, img.width, img.height); } catch { failImage(); }
      };
      img.src = reader.result;
    };
    reader.readAsDataURL(file);
  };

  if (restoring) return <div className="card-container" role="status">Verifying session...</div>;
  if (user) return <Navigate to={isManager(user) ? '/admin' : '/user'} replace />;

  return (
    <div className="card-container login-container">
      <h1>Face Login System</h1>
      <p className="subtitle">Position your face clearly in the camera</p>
      <p className="subtitle">Chưa đăng ký khuôn mặt? Liên hệ quản trị viên để đăng ký với sự đồng ý của bạn.</p>
      {error && <div className="result-box error" role="alert">{error}</div>}
      <div className="video-wrapper">
        <video ref={videoRef} className="webcam" autoPlay playsInline muted />
        <canvas ref={canvasRef} className="hidden" />
        <div className="scanning-frame" />
      </div>
      <div className="controls">
        <button className="primary-btn" onClick={handleCapture} disabled={loading}>
          <Camera size={20} /> Capture &amp; Login
        </button>
        <div className="divider"><span>OR</span></div>
        <label htmlFor="imageUpload" className="secondary-btn" aria-disabled={loading}>
          <Upload size={18} /> Upload Image
        </label>
        <input type="file" id="imageUpload" accept="image/*" className="hidden" disabled={loading} onChange={handleFileUpload} />
      </div>
      {loading && <div className="spinner-container" role="status"><div className="spinner" /><p>Verifying face...</p></div>}
      {result && <div className={`result-box ${result.type}`} role="alert">{result.msg}</div>}
      {preview && <div className="preview-container"><p>Detected-face input preview:</p><img src={preview} alt="Preview" className="preview-img" /></div>}
    </div>
  );
}
