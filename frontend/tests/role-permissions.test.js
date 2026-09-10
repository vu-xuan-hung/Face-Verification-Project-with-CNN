import test from 'node:test';
import assert from 'node:assert/strict';
import { canChangeRole, canCreateAdmin, canManageUser, enrollmentNotice, enrollmentPayload, invalidatesSession, isManager, profilePayload } from '../src/role-permissions.js';

const account = (role, id = 1, status = 'ACTIVE') => ({ id, role, status });
test('USER cannot create/manage accounts; only SUPER_ADMIN creates ADMIN', () => {
  assert.equal(isManager(account('USER')), false);
  assert.equal(canCreateAdmin(account('USER')), false);
  assert.equal(isManager(account('ADMIN')), true);
  assert.equal(canCreateAdmin(account('ADMIN')), false);
  assert.equal(isManager(account('SUPER_ADMIN')), true);
  assert.equal(canCreateAdmin(account('SUPER_ADMIN')), true);
  for (const role of ['USER', 'ADMIN', 'SUPER_ADMIN']) assert.equal(isManager(account(role, 1, 'DISABLED')), false);
});
test('actor/target matrix excludes self, SUPER_ADMIN, deleted and unauthorized targets', () => {
  for (const actor of ['USER', 'ADMIN', 'SUPER_ADMIN']) {
    for (const target of ['USER', 'ADMIN', 'SUPER_ADMIN']) {
      const expected = actor === 'SUPER_ADMIN' && target !== 'SUPER_ADMIN' || actor === 'ADMIN' && target === 'USER';
      assert.equal(canManageUser(account(actor), account(target, 2)), expected, `${actor} -> ${target}`);
      assert.equal(canManageUser(account(actor), account(target, 1)), false);
      assert.equal(canManageUser(account(actor), account(target, 2, 'DELETED')), false);
      assert.equal(canChangeRole(account(actor), account(target, 2)), actor === 'SUPER_ADMIN' && target !== 'SUPER_ADMIN');
    }
  }
  assert.equal(canManageUser(account('ADMIN'), account('USER', 2, 'DISABLED')), true);
  assert.equal(canManageUser(null, account('USER', 2)), false);
});
test('ordinary forbidden CRUD does not invalidate an authenticated session', () => {
  assert.equal(invalidatesSession({ status: 401 }), true);
  for (const status of [403, 409, 422, 503]) assert.equal(invalidatesSession({ status }), false);
});
test('enrollment payload whitelists fields and requires distinct photos and consent', () => {
  const fields = { username: ' alice ', name: ' Alice ', email: ' alice@example.org ', role: 'SUPER_ADMIN', created_by: 9, password_hash: 'bad' };
  const images = ['data:image/jpeg;base64,YQ==', 'data:image/png;base64,Yg=='];
  assert.deepEqual(enrollmentPayload(fields, images, true), { username: 'alice', name: 'Alice', email: 'alice@example.org', images, consent: true });
  assert.throws(() => enrollmentPayload(fields, images, false), /consent/);
  for (const invalid of [[], [images[0]], Array(11).fill(images[0])]) assert.throws(() => enrollmentPayload(fields, invalid, true), /2–10/);
  assert.throws(() => enrollmentPayload(fields, [images[0], images[0]], true), /distinct/);
  assert.throws(() => enrollmentPayload(fields, ['https://example.org/photo', images[0]], true), /Invalid/);
  const huge = `data:image/jpeg;base64,${'a'.repeat(12 * 1024 * 1024)}`;
  assert.throws(() => enrollmentPayload(fields, [huge, images[0]], true), /8 MiB/);
  const total = Array.from({ length: 5 }, (_, i) => `data:image/jpeg;base64,${'a'.repeat(7 * 1024 * 1024)}${i}`);
  assert.throws(() => enrollmentPayload(fields, total, true), /32 MiB/);
  const encodedLimit = Array.from({ length: 4 }, (_, i) => `data:image/jpeg;base64,${'a'.repeat(7_600_000)}${i}`);
  assert.throws(() => enrollmentPayload(fields, encodedLimit, true), /encoded-character/);
});

test('legacy name-only edit omits blank email and never includes privilege fields', () => {
  assert.deepEqual(profilePayload({ name: ' Alice ', email: null, role: 'SUPER_ADMIN' }), { name: 'Alice' });
  assert.deepEqual(profilePayload({ name: 'Alice', email: '  ' }), { name: 'Alice' });
  assert.deepEqual(profilePayload({ name: 'Alice', email: ' a@example.org ' }), { name: 'Alice', email: 'a@example.org' });
});

test('committed enrollment with pending vectors warns instead of claiming login readiness', () => {
  assert.match(enrollmentNotice({ vector_sync_status: 'pending' }), /synchronization is pending/);
  assert.match(enrollmentNotice({ vector_sync_status: 'synced' }), /successfully/);
});
