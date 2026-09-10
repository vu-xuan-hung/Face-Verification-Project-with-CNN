export const API_URL = (import.meta.env?.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');
export const TOKEN_KEY = 'vshield.access_token';

export async function apiRequest(path, { token, ...options } = {}) {
  const headers = new Headers(options.headers);
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!response.ok) {
    let detail;
    let body;
    try {
      body = await response.json();
      detail = typeof body?.detail === 'string' ? body.detail : (body?.message || (typeof body?.detail === 'object' ? body?.detail?.message : null));
    } catch { /* Non-JSON error response. */ }
    const error = new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}).`);
    error.status = response.status;
    error.body = body;
    error.code = body?.code || (typeof body?.detail === 'object' ? body?.detail?.code : null);
    throw error;
  }
  return response;
}

export function validateIdentity(identity) {
  if (!identity || typeof identity.username !== 'string' || !identity.username.trim()
      || !['USER', 'ADMIN', 'SUPER_ADMIN'].includes(identity.role)
      || !Number.isSafeInteger(identity.id) || identity.id < 1 || identity.status !== 'ACTIVE') {
    throw new Error('Server returned an invalid account.');
  }
  return { id: identity.id, username: identity.username, name: identity.name, email: identity.email,
    role: identity.role, status: identity.status };
}

export function logsPath(path, username, date) {
  const query = new URLSearchParams();
  if (username) query.set('username', username);
  if (date) query.set('date', date);
  return `${path}?${query}`;
}

export function accessLogsPath(path, params = {}) {
  const query = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') {
      query.set(k, v);
    }
  }
  const qs = query.toString();
  return qs ? `${path}?${qs}` : path;
}
