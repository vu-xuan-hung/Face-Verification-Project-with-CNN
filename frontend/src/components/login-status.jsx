import { AlertTriangle, CheckCircle2, LoaderCircle, ScanFace, ShieldAlert, UserX, Users, WifiOff } from 'lucide-react';

const statuses = {
  NO_FACE: ['NO FACE DETECTED', 'No face was found. Face the camera directly and improve the lighting.', 'warning', ScanFace],
  MULTIPLE_FACES: ['MULTIPLE FACES', 'Only one person may authenticate at a time.', 'warning', Users],
  SPOOF: ['SPOOF DETECTED', 'The liveness check rejected this presentation. Access remains locked.', 'danger', ShieldAlert],
  PAD_UNCERTAIN: ['LIVENESS UNCERTAIN', 'The liveness score did not meet the safety margin. Please try again.', 'warning', AlertTriangle],
  UNKNOWN: ['UNKNOWN FACE', 'Liveness passed, but this identity is not enrolled in VShield.', 'warning', UserX],
  AMBIGUOUS: ['AMBIGUOUS FACE', 'The identity match was too close to multiple accounts. Access was denied.', 'warning', AlertTriangle],
  USER_DISABLED: ['USER DISABLED', 'This face was recognized, but the account is disabled. Contact an administrator.', 'danger', UserX],
  PAD_UNAVAILABLE: ['MODEL UNAVAILABLE', 'The liveness service is unavailable. VShield fails closed and denies access.', 'danger', WifiOff],
  PAD_ERROR: ['MODEL ERROR', 'The liveness service could not complete verification. Access was denied.', 'danger', WifiOff],
  MODEL_ERROR: ['MODEL ERROR', 'The biometric engine is temporarily unavailable. Access was denied.', 'danger', WifiOff],
  GALLERY_UNAVAILABLE: ['IDENTITY INDEX UNAVAILABLE', 'The enrolled identity index is unavailable. Access was denied.', 'danger', WifiOff],
};

export function loginFeedback(code, success, message, name) {
  if (success) return { title: 'ACCESS GRANTED', message: `Identity confirmed${name ? ` for ${name}` : ''}. Opening your dashboard…`, tone: 'success', Icon: CheckCircle2 };
  const [title, fallback, tone, Icon] = statuses[code] || ['ACCESS DENIED', 'Authentication could not be completed.', 'danger', ShieldAlert];
  return { title, message: message || fallback, tone, Icon };
}

export default function LoginStatus({ loading, feedback }) {
  if (!loading && !feedback) return <div className="login-status login-status--idle"><ScanFace size={22} /><div><strong>Ready for verification</strong><p>Capture a live image or upload an existing image for demonstration.</p></div></div>;
  const Icon = loading ? LoaderCircle : feedback.Icon;
  return <div className={`login-status login-status--${loading ? 'info' : feedback.tone}`} role={loading ? 'status' : 'alert'}>
    <Icon className={loading ? 'spin' : ''} size={22} />
    <div><strong>{loading ? 'CHECKING LIVENESS & IDENTITY' : feedback.title}</strong><p>{loading ? 'Running server-side anti-spoofing before identity matching…' : feedback.message}</p></div>
  </div>;
}
