import { useEffect, useRef } from 'react';
import { AlertCircle, Inbox, LoaderCircle, ShieldCheck, X } from 'lucide-react';

const BADGE_VARIANTS = {
  ACTIVE: 'success', GRANTED: 'success', ACCESS_GRANTED: 'success', REAL: 'success',
  USER: 'info', UNKNOWN_FACE: 'warning', AMBIGUOUS_FACE: 'warning', ADMIN: 'warning',
  SUPER_ADMIN: 'premium', DISABLED: 'neutral', DENIED: 'danger', ACCESS_DENIED: 'danger',
  SPOOF_ATTEMPT: 'danger', USER_DISABLED: 'danger', PAD_ERROR: 'danger', PAD_UNAVAILABLE: 'danger',
};

export function StatusBadge({ value, children, variant }) {
  const text = children ?? value ?? 'Unknown';
  const tone = variant || BADGE_VARIANTS[value] || 'neutral';
  return <span className={`status-badge status-badge--${tone}`}>{text}</span>;
}

export function Avatar({ name, size = 'md' }) {
  const initials = String(name || '?').trim().split(/\s+/).slice(0, 2)
    .map(part => part[0]?.toUpperCase()).join('') || '?';
  return <span className={`avatar avatar--${size}`} aria-hidden="true">{initials}</span>;
}

export function StatCard({ label, value, hint, icon: Icon, tone = 'primary' }) {
  return <article className={`stat-card stat-card--${tone}`}>
    <div className="stat-card__icon"><Icon size={20} /></div>
    <div>
      <p className="stat-card__label">{label}</p>
      <strong className="stat-card__value">{value ?? '—'}</strong>
      {hint && <p className="stat-card__hint">{hint}</p>}
    </div>
  </article>;
}

export function SectionCard({ title, description, action, children, id, className = '' }) {
  return <section id={id} className={`section-card ${className}`.trim()}>
    {(title || description || action) && <header className="section-card__header">
      <div>{title && <h2>{title}</h2>}{description && <p>{description}</p>}</div>
      {action && <div className="section-card__action">{action}</div>}
    </header>}
    {children}
  </section>;
}

export function EmptyState({ title = 'Nothing to show', message, icon: Icon = Inbox }) {
  return <div className="state-view state-view--empty">
    <Icon size={26} />
    <strong>{title}</strong>
    {message && <p>{message}</p>}
  </div>;
}

export function LoadingState({ label = 'Loading…' }) {
  return <div className="state-view" role="status">
    <LoaderCircle className="spin" size={24} /><span>{label}</span>
  </div>;
}

export function ErrorBanner({ message }) {
  if (!message) return null;
  return <div className="alert alert--danger" role="alert">
    <AlertCircle size={18} /><span>{message}</span>
  </div>;
}

export function InfoBanner({ children }) {
  return <div className="alert alert--info"><ShieldCheck size={18} /><span>{children}</span></div>;
}

export function ConfirmDialog({ open, title, message, confirmLabel = 'Confirm', danger = false, busy, onConfirm, onCancel }) {
  const cancelRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    cancelRef.current?.focus();
    const closeOnEscape = event => { if (event.key === 'Escape' && !busy) onCancel(); };
    document.addEventListener('keydown', closeOnEscape);
    return () => document.removeEventListener('keydown', closeOnEscape);
  }, [busy, onCancel, open]);
  if (!open) return null;
  return <div className="dialog-backdrop" role="presentation" onMouseDown={event => {
    if (event.target === event.currentTarget && !busy) onCancel();
  }}>
    <div className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title">
      <button className="icon-btn confirm-dialog__close" onClick={onCancel} disabled={busy} aria-label="Close dialog"><X size={18} /></button>
      <div className={`confirm-dialog__icon ${danger ? 'confirm-dialog__icon--danger' : ''}`}><AlertCircle size={24} /></div>
      <h3 id="confirm-title">{title}</h3>
      <p>{message}</p>
      <div className="confirm-dialog__actions">
        <button ref={cancelRef} className="button button--secondary" onClick={onCancel} disabled={busy}>Cancel</button>
        <button className={`button ${danger ? 'button--danger' : 'button--primary'}`} onClick={onConfirm} disabled={busy}>
          {busy ? 'Working…' : confirmLabel}
        </button>
      </div>
    </div>
  </div>;
}

export function formatDateTime(value) {
  if (!value) return '—';
  const date = new Date(value.includes?.('T') ? value : `${value.replace(' ', 'T')}Z`);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium', timeStyle: 'short',
  }).format(date);
}
