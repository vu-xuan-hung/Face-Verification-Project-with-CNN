import { useCallback, useEffect, useRef, useState } from 'react';
import { Edit3, Fingerprint, Power, RefreshCw, ShieldAlert, Trash2 } from 'lucide-react';
import { useAuth } from './auth-context';
import { canChangeRole, canManageUser, enrollmentNotice, profilePayload } from './role-permissions';
import { Avatar, ConfirmDialog, ErrorBanner, InfoBanner, LoadingState, SectionCard, StatusBadge, formatDateTime } from './components/shared-ui';
import UserEnrollmentForm from './user-enrollment-form';

export default function UserManagement() {
  const { user, request } = useAuth();
  const [users, setUsers] = useState([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [edit, setEdit] = useState(null);
  const [confirm, setConfirm] = useState(null);
  const listRequest = useRef(null);
  const mutation = useRef(null);

  const refresh = useCallback(async () => {
    listRequest.current?.abort();
    const controller = new AbortController();
    listRequest.current = controller; setLoading(true); setError('');
    try {
      const response = await request('/users', { signal: controller.signal });
      const data = await response.json();
      if (!Array.isArray(data)) throw new Error('Server returned an invalid user list.');
      if (!controller.signal.aborted) setUsers(data);
    } catch (err) { if (!controller.signal.aborted) { setUsers([]); setError(err.message); } }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, [request]);

  useEffect(() => { refresh(); return () => { listRequest.current?.abort(); mutation.current?.abort(); }; }, [refresh]);

  const change = async (target, suffix, payload, method = 'PATCH') => {
    if (mutation.current || !canManageUser(user, target) || (suffix === '/role' && !canChangeRole(user, target))) return;
    const controller = new AbortController();
    mutation.current = controller; setBusy(true); setError(''); setNotice(null);
    try {
      await request(`/users/${target.id}${suffix}`, { method, signal: controller.signal,
        ...(payload ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) } : {}) });
      if (controller.signal.aborted) return;
      setEdit(null); setConfirm(null); setNotice({ tone: 'success', text: 'Account updated successfully.' });
      await refresh();
    } catch (err) { if (!controller.signal.aborted) setError(err.message); }
    finally { mutation.current = null; if (!controller.signal.aborted) setBusy(false); }
  };

  const confirmDetails = confirm && {
    status: {
      title: `${confirm.target.status === 'ACTIVE' ? 'Disable' : 'Enable'} ${confirm.target.username}?`,
      message: confirm.target.status === 'ACTIVE' ? 'The account will be blocked from signing in. Biometric data may remain stored.' : 'The account will be allowed to authenticate again.',
      label: confirm.target.status === 'ACTIVE' ? 'Disable account' : 'Enable account',
      danger: confirm.target.status === 'ACTIVE',
      action: () => change(confirm.target, '/status', { status: confirm.target.status === 'ACTIVE' ? 'DISABLED' : 'ACTIVE' }),
    },
    role: {
      title: `Change role for ${confirm.target.username}?`,
      message: `This changes server-side authorization to ${confirm.target.role === 'USER' ? 'ADMIN' : 'USER'} and revokes existing sessions.`,
      label: `Make ${confirm.target.role === 'USER' ? 'ADMIN' : 'USER'}`,
      action: () => change(confirm.target, '/role', { role: confirm.target.role === 'USER' ? 'ADMIN' : 'USER' }),
    },
    delete: {
      title: `Delete ${confirm.target.username}?`,
      message: 'This soft-deletes the account and revokes access. It is not the same as deleting retained biometric templates.',
      label: 'Delete account', danger: true, action: () => change(confirm.target, '', null, 'DELETE'),
    },
  }[confirm.type];

  return <>
    <SectionCard id="users" title="User management" description={`Manage accounts within your ${user.role} scope. Every action is authorized again by the backend.`}
      action={<button className="button button--secondary button--small" disabled={loading || busy} onClick={refresh}><RefreshCw className={loading ? 'spin' : ''} size={16} />Refresh</button>}>
      <InfoBanner><strong>Account disable</strong> blocks login while data may remain. <strong>Biometric deletion</strong> requires a separate retention endpoint, which this backend does not currently expose.</InfoBanner>
      <ErrorBanner message={error} />
      {notice && <div className={`alert alert--${notice.tone}`} role="status">{notice.text}</div>}
      {loading ? <LoadingState label="Loading managed accounts…" /> : <div className="table-wrap"><table className="data-table account-table">
        <thead><tr><th>Account</th><th>Role</th><th>Status</th><th>Biometric</th><th>Created</th><th>Actions</th></tr></thead>
        <tbody>{users.length === 0 ? <tr><td colSpan="6" className="table-empty">No accounts in your management scope.</td></tr> : users.map(target => {
          const manageable = canManageUser(user, target);
          return <tr key={target.id}><td><div className="account-cell"><Avatar name={target.name || target.username} /><span><strong>{target.name || target.username}</strong><small>@{target.username} · {target.email || 'No email'}</small></span></div></td>
            <td><StatusBadge value={target.role} /></td><td><StatusBadge value={target.status} /></td>
            <td><StatusBadge variant="neutral"><Fingerprint size={12} />Not exposed</StatusBadge></td><td className="nowrap">{formatDateTime(target.created_at)}</td>
            <td>{manageable ? <div className="table-actions">
              <button className="icon-btn" title="Edit profile" aria-label={`Edit ${target.username}`} disabled={busy} onClick={() => setEdit({ ...target })}><Edit3 size={16} /></button>
              <button className="icon-btn" title={target.status === 'ACTIVE' ? 'Disable account' : 'Enable account'} aria-label={`${target.status === 'ACTIVE' ? 'Disable' : 'Enable'} ${target.username}`} disabled={busy} onClick={() => setConfirm({ type: 'status', target })}><Power size={16} /></button>
              {canChangeRole(user, target) && <button className="icon-btn" title="Change role" aria-label={`Change role for ${target.username}`} disabled={busy} onClick={() => setConfirm({ type: 'role', target })}><ShieldAlert size={16} /></button>}
              <button className="icon-btn" title="Delete biometric data is unavailable" aria-label="Delete biometric data unavailable" disabled><Fingerprint size={16} /></button>
              <button className="icon-btn icon-btn--danger" title="Delete account" aria-label={`Delete ${target.username}`} disabled={busy} onClick={() => setConfirm({ type: 'delete', target })}><Trash2 size={16} /></button>
            </div> : <span className="muted">Read-only</span>}</td></tr>;
        })}</tbody></table></div>}
      {edit && <form className="inline-edit" onSubmit={event => { event.preventDefault(); change(edit, '', profilePayload(edit)); }}>
        <div><p className="eyebrow">Editing account</p><h3>{edit.username}</h3></div>
        <label><span>Full name</span><input required maxLength={150} value={edit.name || ''} onChange={event => setEdit({ ...edit, name: event.target.value })} /></label>
        <label><span>Email</span><input type="email" maxLength={254} value={edit.email || ''} onChange={event => setEdit({ ...edit, email: event.target.value })} /></label>
        <div className="inline-edit__actions"><button className="button button--primary button--small" disabled={busy}>Save profile</button><button type="button" className="button button--secondary button--small" onClick={() => setEdit(null)}>Cancel</button></div>
      </form>}
    </SectionCard>
    <UserEnrollmentForm onCreated={account => { const pending = account?.vector_sync_status === 'pending'; setNotice({ tone: pending ? 'warning' : 'success', text: enrollmentNotice(account) }); refresh(); }} />
    <ConfirmDialog open={Boolean(confirmDetails)} title={confirmDetails?.title} message={confirmDetails?.message} confirmLabel={confirmDetails?.label}
      danger={confirmDetails?.danger} busy={busy} onCancel={() => setConfirm(null)} onConfirm={confirmDetails?.action} />
  </>;
}
