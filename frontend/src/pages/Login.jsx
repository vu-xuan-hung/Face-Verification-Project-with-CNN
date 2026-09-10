import { useEffect, useRef, useState } from 'react';
import { Camera, Database, LockKeyhole, ShieldCheck } from 'lucide-react';
import { Navigate } from 'react-router-dom';
import { apiRequest } from '../auth-api';
import { useAuth } from '../auth-context';
import { Brand } from '../components/app-shell';
import LoginStatus, { loginFeedback } from '../components/login-status';
import { ErrorBanner, StatusBadge } from '../components/shared-ui';
import { isManager } from '../role-permissions';

export default function Login() {
  const { user, loading: restoring, error: authError, login } = useAuth();
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState(null);
  const [preview, setPreview] = useState(null);
  const [localError, setLocalError] = useState('');
  const [cameraState, setCameraState] = useState('STARTING');
  const [cameraAttempt, setCameraAttempt] = useState(0);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const aliveRef = useRef(false);
  const busyRef = useRef(false);
  const requestRef = useRef(null);

  useEffect(() => {
    aliveRef.current = true;
    return () => { aliveRef.current = false; requestRef.current?.abort(); };
  }, []);

  useEffect(() => {
    if (restoring || user) return undefined;
    let disposed = false;
    let stream;
    setCameraState('STARTING');
    setLocalError('');
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API is unavailable.');
        const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        if (disposed) { media.getTracks().forEach(track => track.stop()); return; }
        stream = media;
        if (videoRef.current) videoRef.current.srcObject = media;
        setCameraState('READY');
      } catch { if (!disposed) { setCameraState('UNAVAILABLE'); setLocalError('Camera access is required for secure sign in. Check browser permission and try again.'); } }
    })();
    return () => { disposed = true; stream?.getTracks().forEach(track => track.stop()); };
  }, [cameraAttempt, restoring, user]);

  const sendToBackend = async image => {
    const controller = new AbortController();
    requestRef.current = controller;
    setFeedback(null);
    try {
      const response = await apiRequest('/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image }), signal: controller.signal });
      const data = await response.json();
      if (!aliveRef.current || controller.signal.aborted) return;
      const status = loginFeedback(data.code, data.success, data.message, data.name || data.username);
      setFeedback(status);
      if (data.success) await login(data.access_token, controller.signal);
    } catch (err) {
      if (aliveRef.current && err.name !== 'AbortError') {
        const code = err.code || err.body?.code || (err.status === 503 ? 'MODEL_ERROR' : null);
        setFeedback(loginFeedback(code, false, err.message));
      }
    } finally {
      busyRef.current = false;
      requestRef.current = null;
      if (aliveRef.current) setLoading(false);
    }
  };

  const beginAttempt = () => {
    if (busyRef.current) return false;
    busyRef.current = true; setLoading(true); setLocalError(''); setFeedback(null);
    return true;
  };
  const failImage = message => {
    busyRef.current = false;
    if (aliveRef.current) { setLoading(false); setLocalError(message || 'Cannot capture the camera frame. Please try again.'); }
  };
  const captureSource = (source, width, height) => {
    const canvas = canvasRef.current;
    const scale = Math.min(1, 1024 / Math.max(width, height));
    canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
    canvas.getContext('2d').drawImage(source, 0, 0, canvas.width, canvas.height);
    const encoded = canvas.toDataURL('image/jpeg', 0.9);
    setPreview(encoded); sendToBackend(encoded);
  };
  const capture = () => {
    const video = videoRef.current;
    if (!video?.videoWidth || !video.videoHeight) { setLocalError('Camera is not ready. Wait a moment and try again.'); return; }
    if (!beginAttempt()) return;
    try { captureSource(video, video.videoWidth, video.videoHeight); } catch { failImage(); }
  };

  if (restoring) return <div className="route-loader" role="status"><ShieldCheck size={30} /><span>Verifying secure session…</span></div>;
  if (user) return <Navigate to={isManager(user) ? '/admin' : '/user'} replace />;
  return <main className="login-page">
    <section className="login-brand-panel">
      <Brand />
      <div><p className="eyebrow">Biometric security platform</p><h1>Intelligent access.<br />Verified identity.</h1><p>AI-powered face recognition access control with passive liveness protection and server-enforced roles.</p></div>
      <ul className="trust-list"><li><ShieldCheck /><span><strong>Anti-spoofing first</strong><small>Recognition runs only after liveness passes.</small></span></li><li><LockKeyhole /><span><strong>Fail-closed security</strong><small>Unavailable models never bypass verification.</small></span></li><li><Database /><span><strong>Auditable access</strong><small>Security decisions are recorded for administrators.</small></span></li></ul>
      <p className="login-brand-panel__foot">VShield · FaceNet identity · MiniFASNet liveness</p>
    </section>
    <section className="login-workspace">
      <div className="login-card">
        <header><div><p className="eyebrow">Secure sign in</p><h2>Verify your identity</h2><p>Position one live face inside the guide and keep the camera image clear.</p></div><StatusBadge value={cameraState === 'READY' ? 'ACTIVE' : cameraState === 'STARTING' ? undefined : 'DISABLED'}>{cameraState === 'READY' ? 'Camera ready' : cameraState === 'STARTING' ? 'Starting camera' : 'Camera unavailable'}</StatusBadge></header>
        <ErrorBanner message={authError || localError} />
        <div className="camera-stage">
          <video ref={videoRef} autoPlay playsInline muted />
          <canvas ref={canvasRef} className="hidden" />
          <div className="face-guide"><span /><span /><span /><span /></div>
          <div className="camera-stage__label"><Camera size={15} />Live camera</div>
        </div>
        <div className="login-actions">
          <button className="button button--primary button--large" onClick={capture} disabled={loading || cameraState !== 'READY'}><Camera size={19} />Capture &amp; Login</button>
          {cameraState === 'UNAVAILABLE' && <button className="button button--secondary button--large" onClick={() => setCameraAttempt(value => value + 1)} disabled={loading}><Camera size={19} />Retry camera</button>}
        </div>
        <LoginStatus loading={loading} feedback={feedback} />
        {preview && <details className="capture-preview"><summary>View submitted frame</summary><img src={preview} alt="Submitted authentication frame" /></details>}
        <p className="privacy-note"><LockKeyhole size={15} />Not enrolled? Ask an administrator. Face data requires explicit biometric consent.</p>
      </div>
    </section>
  </main>;
}
