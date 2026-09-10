import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import { Navigate } from 'react-router-dom';
import { apiRequest, TOKEN_KEY, validateIdentity } from './auth-api';
import { invalidatesSession } from './role-permissions';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const tokenRef = useRef(null);

  const clearSession = useCallback(() => {
    tokenRef.current = null;
    try { sessionStorage.removeItem(TOKEN_KEY); } catch { /* Storage may be disabled. */ }
    setUser(null);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const restore = async () => {
      try {
        const token = sessionStorage.getItem(TOKEN_KEY);
        if (!token) return;
        const response = await apiRequest('/auth/me', { token, signal: controller.signal });
        const identity = validateIdentity(await response.json());
        if (!controller.signal.aborted) {
          tokenRef.current = token;
          setUser(identity);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          clearSession();
          setError(err.status === 401 ? 'Session expired. Please log in again.' : 'Cannot verify session. Please log in again.');
        }
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    };
    restore();
    return () => controller.abort();
  }, [clearSession]);

  const login = useCallback(async (token, signal) => {
    if (typeof token !== 'string' || !token) throw new Error('Server did not issue a session token.');
    const response = await apiRequest('/auth/me', { token, signal });
    const identity = validateIdentity(await response.json());
    if (signal?.aborted) throw new DOMException('Cancelled', 'AbortError');
    sessionStorage.setItem(TOKEN_KEY, token);
    tokenRef.current = token;
    setError('');
    setUser(identity);
    return identity;
  }, []);

  const request = useCallback(async (path, options = {}) => {
    const token = tokenRef.current;
    try {
      return await apiRequest(path, { ...options, token });
    } catch (err) {
      if (token === tokenRef.current && invalidatesSession(err)) {
        clearSession();
        setError('Session expired. Please log in again.');
      }
      throw err;
    }
  }, [clearSession]);

  const logout = useCallback(async () => {
    const token = tokenRef.current;
    setError('');
    clearSession();
    try {
      if (token) await apiRequest('/auth/logout', { method: 'POST', token });
    } catch (err) {
      if (err.status !== 401) setError('Signed out locally, but server revocation could not be confirmed.');
    }
  }, [clearSession]);

  // Recheck revocation/expiry on focus and while a dashboard remains open.
  useEffect(() => {
    if (!user) return undefined;
    const controller = new AbortController();
    const check = async () => {
      const token = tokenRef.current;
      try {
        const response = await request('/auth/me', { signal: controller.signal });
        const identity = validateIdentity(await response.json());
        if (!controller.signal.aborted && token && token === tokenRef.current) setUser(current => (
          current?.username === identity.username && current?.role === identity.role ? current : identity
        ));
      } catch { /* Invalid sessions are cleared by request; network failures can be retried. */ }
    };
    window.addEventListener('focus', check);
    const interval = window.setInterval(check, 60_000);
    return () => {
      controller.abort();
      window.removeEventListener('focus', check);
      window.clearInterval(interval);
    };
  }, [user, request]);

  return <AuthContext.Provider value={{ user, loading, error, login, logout, request }}>{children}</AuthContext.Provider>;
}

export const useAuth = () => useContext(AuthContext);

export function ProtectedRoute({ role, children }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="route-loader" role="status">Verifying secure session…</div>;
  if (!user) return <Navigate to="/login" replace />;
  if (role && !(Array.isArray(role) ? role : [role]).includes(user.role)) return <Navigate to="/user" replace />;
  return children;
}
