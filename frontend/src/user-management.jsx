import { useCallback, useEffect, useRef, useState } from 'react';
import { useAuth } from './auth-context';
import { canChangeRole, canManageUser, enrollmentNotice, profilePayload } from './role-permissions';
import UserEnrollmentForm from './user-enrollment-form';

export default function UserManagement() {
  const { user, request } = useAuth();
  const [users, setUsers] = useState([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [warning, setWarning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [edit, setEdit] = useState(null);
  const listRequest = useRef(null);
  const mutation = useRef(null);
  const refresh = useCallback(async () => {
    listRequest.current?.abort();
    const controller = new AbortController();
    listRequest.current = controller;
    setLoading(true);
    try {
      const response = await request('/users', { signal: controller.signal });
      const data = await response.json();
      if (!Array.isArray(data)) throw new Error('Server returned an invalid user list.');
      if (!controller.signal.aborted) setUsers(data);
    } catch (err) { if (!controller.signal.aborted) { setUsers([]); setError(err.message); } }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }, [request]);
  useEffect(() => {
    refresh();
    return () => { listRequest.current?.abort(); mutation.current?.abort(); };
  }, [refresh]);

  const change = async (target, suffix, payload, method = 'PATCH') => {
    if (mutation.current || !canManageUser(user, target)) return;
    if (suffix === '/role' && !canChangeRole(user, target)) return;
    const controller = new AbortController();
    mutation.current = controller;
    setBusy(true); setError(''); setNotice(''); setWarning(false);
    try {
      await request(`/users/${target.id}${suffix}`, { method, signal: controller.signal,
        ...(payload ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) } : {}) });
      if (controller.signal.aborted) return;
      setEdit(null); setNotice('Account updated.');
      await refresh();
    } catch (err) { if (!controller.signal.aborted) setError(err.message); }
    finally { mutation.current = null; if (!controller.signal.aborted) setBusy(false); }
  };
  return <section className="user-management" aria-labelledby="management-title">
    <div className="controls-panel"><h2 id="management-title">User management</h2>
      <button className="secondary-btn small-btn" disabled={loading || busy} onClick={refresh}>Refresh accounts</button></div>
    <p>Signed in as {user.role}. Permissions are checked on the server; SUPER_ADMIN accounts are read-only here.</p>
    {error && <p role="alert" className="result-box error">{error}</p>}
    {notice && <p role={warning ? 'alert' : 'status'} className={`result-box ${warning ? 'warning' : 'success'}`}>{notice}</p>}
    <div className="table-container"><table className="data-table">
      <thead><tr><th>Account</th><th>Role / status</th><th>Actions</th></tr></thead>
      <tbody>{loading ? <tr><td colSpan="3" role="status">Loading accounts…</td></tr> : users.length === 0 ?
        <tr><td colSpan="3">No accounts in your management scope.</td></tr> : users.map(target => <tr key={target.id}>
          <td>{target.name || target.username}<br /><small>{target.username} · {target.email || 'No email'}</small></td>
          <td>{target.role}<br /><small>{target.status}</small></td>
          <td>{canManageUser(user, target) ? <div className="management-actions">
            <button className="secondary-btn small-btn" disabled={busy} onClick={() => setEdit({ ...target })}>Edit</button>
            <button className="secondary-btn small-btn" disabled={busy} onClick={() => {
              const status = target.status === 'ACTIVE' ? 'DISABLED' : 'ACTIVE';
              if (window.confirm(`Set ${target.username} to ${status}? Existing sessions may be revoked.`)) change(target, '/status', { status });
            }}>{target.status === 'ACTIVE' ? 'Disable' : 'Enable'}</button>
            {canChangeRole(user, target) && <button className="secondary-btn small-btn" disabled={busy} onClick={() => {
              const role = target.role === 'USER' ? 'ADMIN' : 'USER';
              if (window.confirm(`Change ${target.username} to ${role}?`)) change(target, '/role', { role });
            }}>Make {target.role === 'USER' ? 'ADMIN' : 'USER'}</button>}
            <button className="logout-btn" disabled={busy} onClick={() => {
              if (window.confirm(`Delete ${target.username}? Access revoked; record soft-deleted. Retained biometric files follow retention policy.`)) change(target, '', null, 'DELETE');
            }}>Delete</button>
          </div> : <span>Read-only</span>}</td>
        </tr>)}</tbody>
    </table></div>
    {edit && <form className="management-form" onSubmit={event => { event.preventDefault(); change(edit, '', profilePayload(edit)); }}>
      <h3>Edit {edit.username}</h3><fieldset disabled={busy}><div className="management-fields">
        <label>Name<input required maxLength={150} className="input-field" value={edit.name || ''} onChange={event => setEdit({ ...edit, name: event.target.value })} /></label>
        <label>Email<input type="email" maxLength={254} className="input-field" value={edit.email || ''} onChange={event => setEdit({ ...edit, email: event.target.value })} /></label>
      </div><div className="management-actions"><button className="primary-btn small-btn">Save profile</button>
        <button type="button" className="secondary-btn small-btn" onClick={() => setEdit(null)}>Cancel</button></div></fieldset>
    </form>}
    <UserEnrollmentForm onCreated={account => {
      setWarning(account?.vector_sync_status === 'pending'); setNotice(enrollmentNotice(account)); refresh();
    }} />
  </section>;
}
