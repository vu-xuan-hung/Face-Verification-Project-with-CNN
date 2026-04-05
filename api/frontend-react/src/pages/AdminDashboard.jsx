import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, Download, Search } from 'lucide-react';

export default function AdminDashboard() {
  const [logs, setLogs] = useState([]);
  const [userFilter, setUserFilter] = useState('');
  const [dateFilter, setDateFilter] = useState('');
  
  const navigate = useNavigate();
  const username = localStorage.getItem('username');
  const role = localStorage.getItem('role');

  useEffect(() => {
    if (role !== 'admin') {
      alert('Access Denied: Admins Only');
      navigate('/login');
    } else {
      fetchLogs();
    }
  }, [role, navigate]);

  const fetchLogs = async (u = userFilter, d = dateFilter) => {
    let url = 'http://localhost:8000/logs?';
    if (u) url += `username=${encodeURIComponent(u)}&`;
    if (d) url += `date=${encodeURIComponent(d)}`;
    
    try {
      const res = await fetch(url);
      const data = await res.json();
      setLogs(data);
    } catch (err) {
      console.error(err);
    }
  };

  const handleExport = () => {
    let url = 'http://localhost:8000/logs/export?';
    if (userFilter) url += `username=${encodeURIComponent(userFilter)}&`;
    if (dateFilter) url += `date=${encodeURIComponent(dateFilter)}`;
    window.open(url, '_blank');
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate('/login');
  };

  return (
    <div className="card-container dashboard-container">
      <div className="header">
        <h1>Admin Dashboard</h1>
        <div className="user-info">
          <span>Admin: <strong>{username}</strong></span>
          <button className="logout-btn" onClick={handleLogout}>
            <LogOut size={16} /> Logout
          </button>
        </div>
      </div>

      <div className="controls-panel">
        <div className="filters">
          <input 
            type="text" 
            placeholder="Filter by Username..." 
            className="input-field" 
            value={userFilter}
            onChange={e => setUserFilter(e.target.value)}
          />
          <input 
            type="date" 
            className="input-field" 
            value={dateFilter}
            onChange={e => setDateFilter(e.target.value)}
          />
          <button className="primary-btn small-btn" onClick={() => fetchLogs(userFilter, dateFilter)}>
            <Search size={16} /> Search
          </button>
        </div>
        <button className="secondary-btn small-btn" onClick={handleExport}>
          <Download size={16} /> Export CSV
        </button>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th>Username</th>
              <th>Role</th>
              <th>Time</th>
            </tr>
          </thead>
          <tbody>
            {logs.length === 0 ? (
              <tr><td colSpan="3" style={{textAlign:'center', padding:'20px'}}>No records found</td></tr>
            ) : (
              logs.map((log, idx) => (
                <tr key={idx}>
                  <td>{log.username}</td>
                  <td>
                    <span className={`badge ${log.role === 'admin' ? 'badge-admin' : 'badge-user'}`}>
                      {log.role}
                    </span>
                  </td>
                  <td>{log.timestamp}</td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
