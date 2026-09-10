import { useCallback, useEffect, useState } from 'react';
import { Activity, Ban, RefreshCw, ShieldAlert, ShieldCheck, UserCog, Users } from 'lucide-react';
import { useAuth } from '../auth-context';
import AppShell from '../components/app-shell';
import AccessLogsPanel from '../components/access-logs-panel';
import ActivityChart from '../components/activity-chart';
import { ErrorBanner, SectionCard, StatCard } from '../components/shared-ui';
import UserManagement from '../user-management';

const statDefinitions = [
  ['total_users', 'Total users', Users, 'primary'],
  ['total_admins', 'Administrators', UserCog, 'premium'],
  ['recognitions_today', 'Attempts today', Activity, 'info'],
  ['granted_today', 'Granted today', ShieldCheck, 'success'],
  ['denied_today', 'Denied today', Ban, 'danger'],
  ['spoof_attempts_today', 'Spoof attempts', ShieldAlert, 'warning'],
  ['unknown_today', 'Unknown faces', Users, 'neutral'],
];

export default function AdminDashboard() {
  const { user, request, logout } = useAuth();
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [loggingOut, setLoggingOut] = useState(false);

  const fetchStats = useCallback(async signal => {
    setLoading(true); setError('');
    try {
      const response = await request('/dashboard/stats', { signal });
      const data = await response.json();
      if (!signal?.aborted) setStats(data);
    } catch (err) {
      if (!signal?.aborted) { setStats(null); setError(`Dashboard metrics unavailable: ${err.message}`); }
    } finally { if (!signal?.aborted) setLoading(false); }
  }, [request]);

  useEffect(() => {
    const controller = new AbortController();
    fetchStats(controller.signal);
    return () => controller.abort();
  }, [fetchStats]);

  const handleLogout = async () => { setLoggingOut(true); await logout(); };
  return <AppShell user={user} mode="admin" title="Security dashboard"
    subtitle="Monitor biometric access, accounts and presentation attacks from one place."
    loggingOut={loggingOut} onLogout={handleLogout}>
    <section id="overview" className="dashboard-overview">
      <div className="page-actions">
        <div><p className="eyebrow">Live operations overview</p><h2>Today at a glance</h2></div>
        <div>
          <a className="button button--primary button--small" href="#enrollment">Enroll identity</a>
          <button className="button button--secondary button--small" disabled={loading} onClick={() => fetchStats()}><RefreshCw className={loading ? 'spin' : ''} size={16} />Refresh metrics</button>
        </div>
      </div>
      <ErrorBanner message={error} />
      <div className="stats-grid" aria-busy={loading}>
        {statDefinitions.map(([key, label, icon, tone]) => <StatCard key={key} label={label}
          value={loading ? '…' : stats?.[key]} hint={stats ? 'Current system data' : 'Not available'} icon={icon} tone={tone} />)}
      </div>
      <SectionCard title="Access activity" description="Granted and denied decisions recorded during the last seven days.">
        <ActivityChart activity={stats?.recent_activity} />
      </SectionCard>
    </section>
    <UserManagement />
    <AccessLogsPanel request={request} />
  </AppShell>;
}
