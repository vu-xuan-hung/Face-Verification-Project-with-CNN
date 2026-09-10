import { useState } from 'react';
import { LogOut } from 'lucide-react';
import { useAuth } from '../auth-context';
import { Link } from 'react-router-dom';
import { isManager } from '../role-permissions';

export default function UserDashboard() {
  const { user, logout } = useAuth();
  const [loggingOut, setLoggingOut] = useState(false);
  const handleLogout = async () => {
    setLoggingOut(true);
    await logout();
  };

  return (
    <div className="card-container login-container" style={{ maxWidth: '500px' }}>
      <h1>Xin chào, <span style={{ color: '#00b4db' }}>{user.username}</span>!</h1>
      <p className="subtitle">Welcome to your personal dashboard</p>
      <div className="result-box success" style={{ margin: '30px 0', textAlign: 'left', background: 'rgba(0,0,0,0.2)', color: 'white' }}>
        <p><strong>Quyền hạn: </strong><span className={`badge badge-${user.role}`}>{user.role}</span></p>
        <p>Phiên đăng nhập và quyền đã được xác minh bởi máy chủ.</p>
      </div>
      <div className="controls">
        {isManager(user) && <Link className="primary-btn" to="/admin">Manage users</Link>}
        <button className="secondary-btn" onClick={handleLogout} disabled={loggingOut} style={{ margin: '0 auto' }}>
          <LogOut size={18} /> {loggingOut ? 'Đang đăng xuất...' : 'Đăng xuất'}
        </button>
      </div>
    </div>
  );
}
