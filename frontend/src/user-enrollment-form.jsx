import { useEffect, useRef, useState } from 'react';
import { useAuth } from './auth-context';
import { canCreateAdmin, enrollmentPayload, isManager } from './role-permissions';
import FaceEnrollment from './face-enrollment';

export default function UserEnrollmentForm({ onCreated }) {
  const { user, request } = useAuth();
  const [values, setValues] = useState({ username: '', name: '', email: '' });
  const [kind, setKind] = useState('USER');
  const [images, setImages] = useState([]);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [reading, setReading] = useState(false);
  const [error, setError] = useState('');
  const pending = useRef(null);
  useEffect(() => () => pending.current?.abort(), []);
  const submit = async event => {
    event.preventDefault();
    if (pending.current || reading || !isManager(user)) return;
    if (kind === 'ADMIN' && !canCreateAdmin(user)) { setError('Only SUPER_ADMIN may enroll an ADMIN.'); return; }
    const controller = new AbortController();
    pending.current = controller;
    setBusy(true);
    setError('');
    try {
      const payload = enrollmentPayload(values, images, consent);
      const response = await request(kind === 'ADMIN' ? '/admins' : '/users', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal,
      });
      const account = await response.json();
      if (controller.signal.aborted) return;
      setValues({ username: '', name: '', email: '' }); setImages([]); setConsent(false);
      onCreated(account);
    } catch (err) { if (!controller.signal.aborted) setError(err.message); }
    finally { pending.current = null; if (!controller.signal.aborted) setBusy(false); }
  };
  return <form onSubmit={submit} className="management-form">
    <h3>Enroll a new account</h3>
    <fieldset disabled={busy}>
      <div className="management-fields">{['username', 'name', 'email'].map(field => <label key={field}>
        {field}<input className="input-field" type={field === 'email' ? 'email' : 'text'} value={values[field]} required
          maxLength={field === 'username' ? 64 : field === 'email' ? 254 : 150}
          onChange={event => setValues(current => ({ ...current, [field]: event.target.value }))} />
      </label>)}</div>
      {canCreateAdmin(user) ? <label>Account type <select className="input-field" value={kind} onChange={event => setKind(event.target.value)}>
        <option value="USER">USER</option><option value="ADMIN">ADMIN</option>
      </select></label> : <p>Account type: USER</p>}
      <FaceEnrollment images={images} onChange={setImages} disabled={busy} onReadingChange={setReading} />
      <label className="consent-label"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} required />
        This person explicitly consents to storing their face photos and biometric embeddings for VShield authentication.
      </label>
      <button className="primary-btn" disabled={busy || reading || images.length < 2 || !consent}>{busy ? 'Enrolling…' : `Create ${kind}`}</button>
    </fieldset>
    {error && <p className="result-box error" role="alert">{error}</p>}
  </form>;
}
