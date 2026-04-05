import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut } from 'lucide-react';

export default function UserDashboard() {
  const navigate = useNavigate();
  const username = localStorage.getItem('username');
  const role = localStorage.getItem('role');
  const timestamp = localStorage.getItem('last_login');

  useEffect(() => {
    if (!username) {
      navigate('/login');
    }
  }, [username, navigate]);

  const handleLogout = () => {
    localStorage.clear();
    navigate('/login');
  };

  return (
    <div className="card-container login-container" style={{maxWidth: '500px'}}>
      <h1>Xin chào, <span style={{color: '#00b4db'}}>{username}</span>! 🎉</h1>
      <p className="subtitle">Welcome to your personal dashboard</p>

      <div className="result-box success" style={{margin: '30px 0', textAlign: 'left', border: '1px solid rgba(255,255,255,0.1)', background: 'rgba(0,0,0,0.2)', color: 'white'}}>
        <p style={{marginBottom: '10px'}}>
          👤 <strong>Quyền hạn (Role): </strong> 
          <span className={`badge ${role === 'admin' ? 'badge-admin' : 'badge-user'}`}>{role}</span>
        </p>
        <p>🕒 <strong>Thời gian đăng nhập: </strong> {timestamp || new Date().toLocaleString()}</p>
      </div>

      <div className="controls">
        <button className="secondary-btn" onClick={handleLogout} style={{margin: '0 auto'}}>
          <LogOut size={18} /> Đăng xuất
        </button>
      </div>
    </div>
  );
}
