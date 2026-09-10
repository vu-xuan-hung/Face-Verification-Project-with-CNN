// Presentation only: every operation is authorized again by the backend.
export const ROLES = Object.freeze(['USER', 'ADMIN', 'SUPER_ADMIN']);
export const isManager = user => user?.status === 'ACTIVE' && ['ADMIN', 'SUPER_ADMIN'].includes(user.role);
export const canCreateAdmin = user => isManager(user) && user.role === 'SUPER_ADMIN';
export function canManageUser(actor, target) {
  return isManager(actor) && Boolean(target) && ['USER', 'ADMIN'].includes(target.role)
    && ['ACTIVE', 'DISABLED'].includes(target.status) && actor.id !== target.id
    && (actor.role === 'SUPER_ADMIN' || target.role === 'USER');
}
export const canChangeRole = (actor, target) => canCreateAdmin(actor) && canManageUser(actor, target);
export const invalidatesSession = error => error?.status === 401;

export function enrollmentPayload(values, images, consent) {
  if (!consent) throw new Error('Face enrollment requires explicit consent.');
  if (!Array.isArray(images) || images.length < 2 || images.length > 10) throw new Error('Provide 2–10 face photos.');
  if (new Set(images).size !== images.length) throw new Error('Choose distinct face photos.');
  if (images.some(image => typeof image !== 'string' || !/^data:image\/(jpeg|png|webp);base64,/.test(image)
    || image.length > Math.ceil(8 * 1024 * 1024 / 3) * 4 + 64)) throw new Error('Invalid face photo (maximum 8 MiB each).');
  const payload = { username: values.username.trim(), name: values.name.trim(), email: values.email.trim(), images, consent: true };
  if (new TextEncoder().encode(JSON.stringify(payload)).length > 32 * 1024 * 1024) throw new Error('Photos exceed the 32 MiB request limit.');
  if (images.reduce((total, image) => total + image.length, 0) > 30_000_000) throw new Error('Photos exceed the 30,000,000 encoded-character limit.');
  return payload;
}

export function profilePayload(values) {
  // Legacy accounts may have no email. A blank editor must not send an invalid empty email.
  const payload = { name: values.name.trim() };
  if (values.email?.trim()) payload.email = values.email.trim();
  return payload;
}

export function enrollmentNotice(account) {
  return account?.vector_sync_status === 'pending'
    ? 'Account saved, but face-vector synchronization is pending. Face login may be unavailable until synchronization succeeds.'
    : 'Account enrolled successfully.';
}
