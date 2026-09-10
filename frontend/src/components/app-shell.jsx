import { Activity, LayoutDashboard, LogOut, ShieldCheck, Users } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Avatar, StatusBadge } from './shared-ui';

const adminNavigation = [
  { href: '#overview', label: 'Overview', icon: LayoutDashboard },
  { href: '#users', label: 'User management', icon: Users },
  { href: '#logs', label: 'Access logs', icon: Activity },
];

export function Brand({ compact = false }) {
  return <div className={`brand ${compact ? 'brand--compact' : ''}`}>
    <span className="brand__mark"><ShieldCheck size={compact ? 21 : 28} /></span>
    <span><strong>VShield</strong>{!compact && <small>AI Access Control</small>}</span>
  </div>;
}

export default function AppShell({ user, title, subtitle, mode = 'user', loggingOut, onLogout, children }) {
  const admin = mode === 'admin';
  return <div className="app-shell">
    <aside className="sidebar">
      <Brand />
      <nav className="sidebar__nav" aria-label="Dashboard navigation">
        {admin ? adminNavigation.map(({ href, label, icon: Icon }) =>
          <a key={href} href={href}><Icon size={18} /><span>{label}</span></a>) :
          <a href="#profile"><LayoutDashboard size={18} /><span>My dashboard</span></a>}
        {admin ? <Link to="/user"><ShieldCheck size={18} /><span>Personal view</span></Link> :
          ['ADMIN', 'SUPER_ADMIN'].includes(user.role) && <Link to="/admin"><Users size={18} /><span>Admin console</span></Link>}
      </nav>
      <div className="sidebar__security">
        <ShieldCheck size={18} />
        <div><strong>Protected session</strong><small>RBAC enforced by server</small></div>
      </div>
    </aside>
    <main className="app-main">
      <header className="topbar">
        <div><p className="eyebrow">{admin ? 'Security operations' : 'Personal access'}</p><h1>{title}</h1><p>{subtitle}</p></div>
        <div className="topbar__account">
          <Avatar name={user.name || user.username} />
          <div><strong>{user.name || user.username}</strong><StatusBadge value={user.role} /></div>
          <button className="icon-btn" onClick={onLogout} disabled={loggingOut} aria-label="Sign out" title="Sign out">
            <LogOut size={19} />
          </button>
        </div>
      </header>
      <div className="page-content">{children}</div>
    </main>
  </div>;
}
