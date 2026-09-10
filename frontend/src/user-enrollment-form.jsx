import { useEffect, useRef, useState } from 'react';
import { Check, Fingerprint, LoaderCircle, ShieldCheck, UserRoundPlus } from 'lucide-react';
import { useAuth } from './auth-context';
import { canCreateAdmin, enrollmentPayload, isManager } from './role-permissions';
import FaceEnrollment from './face-enrollment';
import { ErrorBanner, SectionCard } from './components/shared-ui';

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

  const detailsReady = Boolean(values.username.trim() && values.name.trim() && values.email.trim());
  const samplesReady = images.length >= 2;
  const steps = [detailsReady, consent, samplesReady, detailsReady && consent && samplesReady, false];

  const submit = async event => {
    event.preventDefault();
    if (pending.current || reading || !isManager(user)) return;
    if (kind === 'ADMIN' && !canCreateAdmin(user)) { setError('Only SUPER_ADMIN may enroll an ADMIN.'); return; }
    const controller = new AbortController();
    pending.current = controller; setBusy(true); setError('');
    try {
      const payload = enrollmentPayload(values, images, consent);
      const response = await request(kind === 'ADMIN' ? '/admins' : '/users', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
      const account = await response.json();
      if (controller.signal.aborted) return;
      setValues({ username: '', name: '', email: '' }); setImages([]); setConsent(false); setKind('USER');
      onCreated?.(account);
    } catch (err) { if (!controller.signal.aborted) setError(err.message); }
    finally { pending.current = null; if (!controller.signal.aborted) setBusy(false); }
  };

  return <SectionCard id="enrollment" className="enrollment-section" title="Enroll a new identity" description="Capture multiple live samples without retraining the recognition model.">
    <ol className="stepper" aria-label="Enrollment progress">{['Details', 'Consent', 'Samples', 'Review', 'Enroll'].map((label, index) => {
      const complete = steps[index];
      const current = !complete && steps.slice(0, index).every(Boolean);
      return <li className={complete ? 'is-complete' : current ? 'is-current' : ''} key={label}><span>{complete ? <Check size={15} /> : index + 1}</span><small>{label}</small></li>;
    })}</ol>
    <form className="enrollment-form" onSubmit={submit}>
      <fieldset disabled={busy}>
        <div className="form-section"><div className="form-section__heading"><span><UserRoundPlus size={20} /></span><div><h3>1. Account details</h3><p>Use a unique username and an email controlled by this person.</p></div></div>
          <div className="form-grid"><label><span>Username</span><input value={values.username} required maxLength={64} pattern="[a-z][a-z0-9_-]*" placeholder="nguyen_van_a" onChange={event => setValues({ ...values, username: event.target.value.toLowerCase() })} /><small>Lowercase letters, numbers, _ or -</small></label><label><span>Full name</span><input value={values.name} required maxLength={150} placeholder="Nguyen Van A" onChange={event => setValues({ ...values, name: event.target.value })} /></label><label><span>Email</span><input type="email" value={values.email} required maxLength={254} placeholder="name@example.com" onChange={event => setValues({ ...values, email: event.target.value })} /></label>
            {canCreateAdmin(user) ? <label><span>Account role</span><select value={kind} onChange={event => setKind(event.target.value)}><option value="USER">USER</option><option value="ADMIN">ADMIN</option></select><small>Only SUPER_ADMIN can create ADMIN.</small></label> : <label><span>Account role</span><input value="USER" disabled /><small>ADMIN accounts cannot create other admins.</small></label>}
          </div>
        </div>
        <div className={`consent-card ${consent ? 'is-accepted' : ''}`}><span className="consent-card__icon"><Fingerprint size={24} /></span><div><h3>2. Biometric Data Consent</h3><p>VShield uses facial biometric information for authentication and access control. Face images may be processed into embeddings for identity verification and authorized administration.</p><label className="check-label"><input type="checkbox" checked={consent} onChange={event => setConsent(event.target.checked)} required /><span>I understand and agree to the collection and processing of my facial biometric data for authentication purposes.</span></label><small>You may request deletion of biometric data through an administrator when a retention endpoint is available.</small></div></div>
        <FaceEnrollment images={images} onChange={setImages} disabled={busy} verifying={busy} onReadingChange={setReading} />
        <div className="enrollment-review"><ShieldCheck size={22} /><div><h3>4. Review and submit</h3><p>{samplesReady ? `${images.length} samples are queued. Each sample must independently pass server-side face detection and MiniFASNet liveness.` : 'Add at least two distinct face samples to continue.'}</p></div></div>
        <ErrorBanner message={error} />
        <div className="enrollment-submit"><p><strong>{kind}</strong> will be active after all samples pass validation and duplicate-face checks.</p><button className="button button--primary button--large" disabled={busy || reading || !detailsReady || !samplesReady || !consent}>{busy ? <><LoaderCircle className="spin" size={18} />Verifying samples…</> : <><ShieldCheck size={18} />Create {kind}</>}</button></div>
      </fieldset>
    </form>
  </SectionCard>;
}
