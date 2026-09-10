import { useEffect, useState } from 'react';
import { Activity, CalendarClock, ShieldCheck, ShieldX } from 'lucide-react';
import { useAuth } from '../auth-context';
import AppShell from '../components/app-shell';
import { EmptyState, ErrorBanner, LoadingState, SectionCard, StatCard, StatusBadge, formatDateTime } from '../components/shared-ui';

export default function UserDashboard() {
  const { user, request, logout } = useAuth();
  const [history, setHistory] = useState({ items: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [loggingOut, setLoggingOut] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const response = await request('/access-logs/me?limit=10', { signal: controller.signal });
        const data = await response.json();
        if (!controller.signal.aborted) setHistory({ items: Array.isArray(data.items) ? data.items : [], total: Number(data.total) || 0 });
      } catch (err) { if (!controller.signal.aborted) setError(`Personal history unavailable: ${err.message}`); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    })();
    return () => controller.abort();
  }, [request]);

  const granted = history.items.find(item => item.result === 'GRANTED');
  const recentDenied = history.items.filter(item => item.result === 'DENIED').length;
  const handleLogout = async () => { setLoggingOut(true); await logout(); };
  return <AppShell user={user} title={`Welcome, ${user.name || user.username}`}
    subtitle="Your identity, account status and recent access activity."
    loggingOut={loggingOut} onLogout={handleLogout}>
    <section id="profile" className="profile-grid">
      <SectionCard className="profile-card" title="Verified account" description="Authorization is resolved from your server-side account after biometric identification.">
        <div className="profile-card__hero"><span className="profile-card__shield"><ShieldCheck size={30} /></span><div><h3>{user.name || user.username}</h3><p>@{user.username}</p></div></div>
        <dl className="profile-details">
          <div><dt>Email</dt><dd>{user.email || 'Not provided'}</dd></div>
          <div><dt>Role</dt><dd><StatusBadge value={user.role} /></dd></div>
          <div><dt>Account status</dt><dd><StatusBadge value={user.status} /></dd></div>
        </dl>
      </SectionCard>
      <SectionCard title="How VShield protects you" description="A fail-closed verification sequence runs for every sign-in.">
        <ol className="security-sequence"><li><span>1</span><div><strong>Liveness check</strong><p>Passive anti-spoofing blocks replay and presentation attacks.</p></div></li><li><span>2</span><div><strong>Identity match</strong><p>FaceNet compares your face embedding with enrolled identities.</p></div></li><li><span>3</span><div><strong>Access policy</strong><p>Your active status and role are checked by the backend.</p></div></li></ol>
      </SectionCard>
    </section>
    <div className="stats-grid stats-grid--compact">
      <StatCard label="Total events" value={loading ? '…' : history.total} hint="Your audit history" icon={Activity} />
      <StatCard label="Last granted" value={loading ? '…' : granted ? formatDateTime(granted.timestamp) : 'None yet'} hint="Latest successful access" icon={CalendarClock} tone="success" />
      <StatCard label="Recent denied" value={loading ? '…' : recentDenied} hint="Within latest 10 events" icon={ShieldX} tone="danger" />
    </div>
    <SectionCard title="Recent authentication history" description="Only events tied to your authenticated user ID are shown.">
      <ErrorBanner message={error} />
      {loading ? <LoadingState label="Loading your access history…" /> : history.items.length === 0 ?
        <EmptyState title="No access history yet" message="Your next authentication event will appear here." /> :
        <div className="table-wrap"><table className="data-table"><thead><tr><th>Time</th><th>Event</th><th>Result</th><th>Reason</th></tr></thead><tbody>
          {history.items.map(item => <tr key={item.id}><td>{formatDateTime(item.timestamp)}</td><td><StatusBadge value={item.event_type} /></td><td><StatusBadge value={item.result} /></td><td>{item.reason_code || 'Verified'}</td></tr>)}
        </tbody></table></div>}
    </SectionCard>
  </AppShell>;
}
