import { useEffect, useRef, useState } from 'react';
import { Camera, ImagePlus, LoaderCircle, ShieldCheck, Trash2 } from 'lucide-react';
import { ErrorBanner, StatusBadge } from './components/shared-ui';

export default function FaceEnrollment({ images, onChange, disabled, verifying, onReadingChange }) {
  const [camera, setCamera] = useState(false);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState('');
  const video = useRef(null);
  const alive = useRef(false);
  const busy = useRef(false);

  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  useEffect(() => {
    if (!camera || disabled) return undefined;
    let disposed = false;
    let stream;
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera API unavailable');
        const media = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
        if (disposed) { media.getTracks().forEach(track => track.stop()); return; }
        stream = media;
        if (video.current) video.current.srcObject = stream;
      } catch { if (!disposed) { setError('Cannot access camera. Upload photos instead.'); setCamera(false); } }
    })();
    return () => { disposed = true; stream?.getTracks().forEach(track => track.stop()); };
  }, [camera, disabled]);

  const capture = () => {
    if (disabled || reading || images.length >= 10) return;
    if (!video.current?.videoWidth) { setError('Camera is not ready.'); return; }
    const canvas = document.createElement('canvas');
    canvas.width = video.current.videoWidth; canvas.height = video.current.videoHeight;
    canvas.getContext('2d').drawImage(video.current, 0, 0);
    const image = canvas.toDataURL('image/jpeg', 0.9);
    if (images.includes(image)) { setError('Capture a different pose.'); return; }
    onChange([...images, image]); setError('');
  };

  const upload = async event => {
    const files = Array.from(event.target.files || []); event.target.value = '';
    if (!files.length || busy.current || disabled) return;
    if (files.length + images.length > 10) { setError('A maximum of 10 samples is allowed.'); return; }
    if (files.some(file => !['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 8 * 1024 * 1024)) {
      setError('Use JPEG, PNG or WebP images up to 8 MiB each.'); return;
    }
    busy.current = true; setReading(true); onReadingChange?.(true);
    try {
      const added = await Promise.all(files.map(file => new Promise((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Cannot read one of the selected images.'));
        reader.readAsDataURL(file);
      })));
      if (!alive.current) return;
      if (new Set([...images, ...added]).size !== images.length + added.length) throw new Error('Choose distinct face samples.');
      onChange([...images, ...added]); setError('');
    } catch (err) { if (alive.current) setError(err.message); }
    finally { busy.current = false; if (alive.current) { setReading(false); onReadingChange?.(false); } }
  };

  return <div className="form-section face-enrollment">
    <div className="form-section__heading"><span><Camera size={20} /></span><div><h3>3. Capture face samples</h3><p>Provide 2–10 clear images of the same consenting person with small pose and lighting variations.</p></div><span className="sample-count">{images.length}/10</span></div>
    <div className="capture-toolbar"><button type="button" className="button button--secondary" onClick={() => setCamera(value => !value)} disabled={disabled}>{camera ? 'Stop camera' : 'Start camera'}</button><label className={`button button--secondary file-button ${disabled ? 'is-disabled' : ''}`}><ImagePlus size={18} />Upload samples<input className="file-input" aria-label="Face photos" type="file" accept="image/jpeg,image/png,image/webp" multiple disabled={disabled} onChange={upload} /></label><p>JPEG, PNG or WebP · max 8 MiB each</p></div>
    {camera && <div className="enrollment-camera"><video ref={video} autoPlay muted playsInline /><div className="face-guide"><span /><span /><span /><span /></div><button type="button" className="button button--primary" disabled={disabled || images.length >= 10} onClick={capture}><Camera size={18} />Capture sample</button></div>}
    {reading && <div className="inline-loader" role="status"><LoaderCircle className="spin" size={18} />Reading selected images…</div>}
    <ErrorBanner message={error} />
    {images.length ? <div className="sample-grid">{images.map((image, index) => <article className="sample-card" key={`${image.slice(-24)}-${index}`}><div className="sample-card__image"><img src={image} alt={`Enrollment sample ${index + 1}`} /><span>#{index + 1}</span></div><div className="sample-card__footer"><StatusBadge variant={verifying ? 'info' : 'neutral'}>{verifying ? <><LoaderCircle className="spin" size={12} />Verifying</> : <><ShieldCheck size={12} />Queued</>}</StatusBadge><button type="button" className="icon-btn icon-btn--danger" aria-label={`Remove sample ${index + 1}`} disabled={disabled} onClick={() => onChange(images.filter((_, itemIndex) => itemIndex !== index))}><Trash2 size={15} /></button></div></article>)}</div> : <div className="sample-empty"><ImagePlus size={26} /><strong>No samples added</strong><p>Start the camera or upload at least two images.</p></div>}
    <div className="server-verification-note"><ShieldCheck size={18} /><p><strong>Server verification is authoritative.</strong> Every sample must independently pass face detection and MiniFASNet liveness before FaceNet creates an embedding.</p></div>
  </div>;
}
