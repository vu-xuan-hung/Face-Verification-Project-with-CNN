import { useCallback, useEffect, useRef, useState } from 'react';
import { LogOut, Download, Search } from 'lucide-react';
import { useAuth } from '../auth-context';
import { logsPath } from '../auth-api';
import UserManagement from '../user-management';
import { Link } from 'react-router-dom';

export default function AdminDashboard() {
  const { user, request, logout } = useAuth();
  const [logs, setLogs] = useState([]);
  const [userFilter, setUserFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [loggingOut, setLoggingOut] = useState(false);
  const logsRequest = useRef(null);
  const exportRequest = useRef(null);

  const fetchLogs = useCallback(async (username = '', date = '') => {
    logsRequest.current?.abort();
    const controller = new AbortController();
    logsRequest.current = controller;
    setLoading(true);
    setError('');
    try {
      const response = await request(logsPath('/logs', username, date), { signal: controller.signal });
      const data = await response.json();
      if (!Array.isArray(data)) throw new Error('Server returned invalid log data.');
      if (!controller.signal.aborted) setLogs(data);
    } catch (err) {
      if (!controller.signal.aborted) {
        setLogs([]);
        setError(err.message);
      }
    } finally {
      if (!controller.signal.aborted) setLoading(false);
    }
  }, [request]);

  useEffect(() => {
    fetchLogs();
    return () => {
      logsRequest.current?.abort();
      exportRequest.current?.abort();
    };
  }, [fetchLogs]);

  const handleExport = async () => {
    if (exportRequest.current) return;
    const controller = new AbortController();
    exportRequest.current = controller;
    setExporting(true);
    setError('');
    try {
      const response = await request(logsPath('/logs/export', userFilter, dateFilter), { signal: controller.signal });
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'login-logs.csv';
      document.body.appendChild(link);
      try { link.click(); } finally {
        link.remove();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      if (!controller.signal.aborted) setError(err.message);
    } finally {
      exportRequest.current = null;
      if (!controller.signal.aborted) setExporting(false);
    }
  };

  const handleLogout = async () => {
    setLoggingOut(true);
    await logout();
  };

  return (
    <div className="card-container dashboard-container">
      <div className="header">
        <h1>Admin Dashboard</h1>
        <div className="user-info">
          <span>{user.role}: <strong>{user.username}</strong></span>
          <button className="logout-btn" onClick={handleLogout} disabled={loggingOut}>
            <LogOut size={16} /> {loggingOut ? 'Signing out...' : 'Logout'}
          </button>
        </div>
      </div>
      <Link className="secondary-btn small-btn" to="/user">Personal dashboard</Link>
      <UserManagement />
      <h2>Login logs</h2>
      <div className="controls-panel">
        <div className="filters">
          <input type="text" placeholder="Filter by Username..." aria-label="Filter by username" className="input-field"
            value={userFilter} onChange={e => setUserFilter(e.target.value)} />
          <input type="date" aria-label="Filter by date" className="input-field"
            value={dateFilter} onChange={e => setDateFilter(e.target.value)} />
          <button className="primary-btn small-btn" onClick={() => fetchLogs(userFilter, dateFilter)} disabled={loading || loggingOut}>
            <Search size={16} /> Search
          </button>
        </div>
        <button className="secondary-btn small-btn" onClick={handleExport} disabled={exporting || loggingOut}>
          <Download size={16} /> {exporting ? 'Exporting...' : 'Export CSV'}
        </button>
      </div>
      {error && <div className="result-box error" role="alert">{error}</div>}
      <div className="table-container">
        <table className="data-table">
          <thead><tr><th>Username</th><th>Role</th><th>Time</th></tr></thead>
          <tbody>
            {loading ? <tr><td colSpan="3" role="status">Loading logs...</td></tr> : logs.length === 0 ? (
              <tr><td colSpan="3" style={{ textAlign: 'center', padding: '20px' }}>No records found</td></tr>
            ) : logs.map((log, idx) => (
              <tr key={idx}>
                <td>{log.username}</td>
                <td><span className={`badge ${['ADMIN', 'SUPER_ADMIN'].includes(log.role) ? 'badge-admin' : 'badge-user'}`}>{log.role}</span></td>
                <td>{log.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
