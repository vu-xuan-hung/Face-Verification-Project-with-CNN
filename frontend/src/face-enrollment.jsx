import { useEffect, useRef, useState } from 'react';

export default function FaceEnrollment({ images, onChange, disabled, onReadingChange }) {
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
    canvas.width = video.current.videoWidth;
    canvas.height = video.current.videoHeight;
    canvas.getContext('2d').drawImage(video.current, 0, 0);
    const image = canvas.toDataURL('image/jpeg', 0.9);
    if (images.includes(image)) { setError('Capture a different pose.'); return; }
    onChange([...images, image]);
    setError('');
  };
  const upload = async event => {
    const files = Array.from(event.target.files || []);
    event.target.value = '';
    if (!files.length || busy.current || disabled) return;
    if (files.length + images.length > 10) { setError('Maximum 10 photos.'); return; }
    if (files.some(file => !['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 8 * 1024 * 1024)) {
      setError('Use JPEG, PNG or WebP photos, maximum 8 MiB each.'); return;
    }
    busy.current = true;
    setReading(true);
    onReadingChange(true);
    try {
      const added = await Promise.all(files.map(file => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(new Error('Cannot read photo.'));
        reader.readAsDataURL(file);
      })));
      if (!alive.current) return;
      if (new Set([...images, ...added]).size !== images.length + added.length) throw new Error('Choose distinct photos.');
      onChange([...images, ...added]);
      setError('');
    } catch (err) { if (alive.current) setError(err.message); }
    finally { busy.current = false; if (alive.current) { setReading(false); onReadingChange(false); } }
  };
  return <fieldset disabled={disabled || reading} className="enrollment-panel">
    <legend>Face photos ({images.length}/10)</legend>
    <p>Provide 2–10 distinct photos of one consenting person, with different poses and lighting. No model retraining.</p>
    <div className="management-actions">
      <button type="button" className="secondary-btn" onClick={() => setCamera(value => !value)}>{camera ? 'Stop camera' : 'Start camera'}</button>
      <label>Upload photos <input aria-label="Face photos" type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={upload} /></label>
    </div>
    {camera && <><div className="video-wrapper"><video ref={video} autoPlay muted playsInline className="webcam" /></div>
      <button type="button" className="secondary-btn" disabled={images.length >= 10} onClick={capture}>Capture another photo</button></>}
    <div className="face-previews">{images.map((image, index) => <div key={image}>
      <img src={image} className="preview-img" alt={`Enrollment photo ${index + 1}`} />
      <button type="button" className="secondary-btn small-btn" onClick={() => onChange(images.filter((_, i) => i !== index))}>Remove {index + 1}</button>
    </div>)}</div>
    {reading && <p role="status">Reading photos…</p>}
    {error && <p role="alert" className="result-box error">{error}</p>}
  </fieldset>;
}
