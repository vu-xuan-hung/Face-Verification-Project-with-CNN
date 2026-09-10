import test from 'node:test';
import assert from 'node:assert/strict';
import { API_URL, TOKEN_KEY, apiRequest, logsPath, validateIdentity } from '../src/auth-api.js';

test('API defaults and session key are namespaced', () => {
  assert.equal(API_URL, 'http://localhost:8000');
  assert.equal(TOKEN_KEY, 'vshield.access_token');
});

test('protected requests attach bearer and preserve caller headers/signal', async () => {
  const original = globalThis.fetch;
  const signal = new AbortController().signal;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, `${API_URL}/logs`);
    assert.equal(options.headers.get('Authorization'), 'Bearer server-issued-token');
    assert.equal(options.headers.get('Accept'), 'application/json');
    assert.equal(options.signal, signal);
    assert.equal(options.token, undefined);
    return new Response('[]');
  };
  try {
    const response = await apiRequest('/logs', { token: 'server-issued-token', headers: { Accept: 'application/json' }, signal });
    assert.deepEqual(await response.json(), []);
  } finally { globalThis.fetch = original; }
});

test('unauthorized and forbidden errors preserve status for session invalidation', async () => {
  const original = globalThis.fetch;
  try {
    for (const status of [401, 403]) {
      globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'Denied' }), { status });
      await assert.rejects(apiRequest('/logs'), err => err.status === status && err.message === 'Denied');
    }
    globalThis.fetch = async () => new Response('gateway error', { status: 502 });
    await assert.rejects(apiRequest('/logs'), err => err.status === 502 && err.message.includes('502'));
  } finally { globalThis.fetch = original; }
});

test('identity requires numeric id, an active account and one of three server-issued roles', () => {
  for (const role of ['USER', 'ADMIN', 'SUPER_ADMIN']) {
    const identity = { id: 1, username: 'alice', name: 'Alice', email: 'alice@example.org', role, status: 'ACTIVE' };
    assert.deepEqual(validateIdentity({ ...identity, ignored: true }), identity);
    assert.throws(() => validateIdentity({ ...identity, status: 'DISABLED' }));
    assert.throws(() => validateIdentity({ ...identity, id: '1' }));
  }
  for (const identity of [null, {}, { username: '', role: 'admin' }, { username: 'bob', role: 'owner' }, { username: 5, role: 'user' }]) {
    assert.throws(() => validateIdentity(identity));
  }
});

test('prediction HTTP errors preserve message and prefer textual detail when present', async () => {
  const original = globalThis.fetch;
  try {
    for (const [body, message] of [
      [{ success: false, message: 'PAD model unavailable.' }, 'PAD model unavailable.'],
      [{ detail: 'Invalid image.', message: 'Other error.' }, 'Invalid image.'],
      [{ detail: [{ type: 'validation_error' }], message: 'Invalid input.' }, 'Invalid input.'],
      [{ message: { unexpected: true } }, 'Request failed (503).'],
      [null, 'Request failed (503).'],
    ]) {
      globalThis.fetch = async () => new Response(JSON.stringify(body), { status: 503 });
      await assert.rejects(apiRequest('/predict', { method: 'POST' }), err => err.status === 503 && err.message === message);
    }
  } finally { globalThis.fetch = original; }
});

test('CSV and log query filters are safely encoded', () => {
  const path = logsPath('/logs/export', 'a&role=admin', '2026-09-09');
  const url = new URL(path, API_URL);
  assert.equal(url.pathname, '/logs/export');
  assert.equal(url.searchParams.get('username'), 'a&role=admin');
  assert.equal(url.searchParams.get('date'), '2026-09-09');
  assert.equal(url.searchParams.get('role'), null);
});

test('successful logout supports empty 204 response', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    assert.equal(url, `${API_URL}/auth/logout`);
    assert.equal(options.method, 'POST');
    assert.equal(options.headers.get('Authorization'), 'Bearer token');
    return new Response(null, { status: 204 });
  };
  try { assert.equal((await apiRequest('/auth/logout', { method: 'POST', token: 'token' })).status, 204); }
  finally { globalThis.fetch = original; }
});
