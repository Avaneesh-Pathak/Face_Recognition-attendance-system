/* The server owns the lock. Browser fullscreen is only a display preference. */
window.initializeAttendanceKiosk = function (onChange) {
  const config = document.getElementById('kioskConfig');
  const dialog = document.getElementById('kioskDialog');
  const form = document.getElementById('kioskForm');
  const code = document.getElementById('kioskCode');
  const confirm = document.getElementById('kioskConfirm');
  const error = document.getElementById('kioskError');
  const submit = document.getElementById('kioskSubmit');
  let locked = document.body.dataset.kioskLocked === 'true';
  let submitting = false;
  let revision = 0;

  function applyLock(value) {
    locked = value;
    document.body.dataset.kioskLocked = String(value);
    onChange(value);
  }
  function openDialog() {
    form.reset();
    error.textContent = '';
    document.getElementById('kioskTitle').textContent = locked ? 'Unlock attendance kiosk' : 'Set your kiosk code';
    document.getElementById('kioskDescription').textContent = locked
      ? 'Enter the secret code to leave kiosk mode. Attendance keeps running while this terminal is locked.'
      : 'Choose a secret code for this terminal. You will need it to return to other pages in this browser session.';
    document.getElementById('kioskConfirmGroup').hidden = locked;
    document.getElementById('kioskSetupHelp').hidden = locked;
    confirm.required = !locked;
    code.autocomplete = locked ? 'off' : 'new-password';
    submit.textContent = locked ? 'Unlock terminal' : 'Lock terminal';
    dialog.showModal();
    code.focus();
  }
  document.getElementById('fullscreenBtn')?.addEventListener('click', openDialog);
  document.getElementById('exitFullscreenBtn').addEventListener('click', openDialog);
  document.getElementById('kioskCancel').addEventListener('click', () => { if (!submitting) dialog.close(); });
  dialog.addEventListener('cancel', event => { if (submitting) event.preventDefault(); });
  dialog.addEventListener('close', () => form.reset());
  document.getElementById('kioskExpand').addEventListener('click', async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch (_) { /* The locked viewport also works without native fullscreen support. */ }
  });
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (submitting) return;
    error.textContent = '';
    if (!locked && code.value !== confirm.value) {
      error.textContent = 'The secret codes do not match.';
      confirm.focus();
      return;
    }
    submitting = true;
    revision++;
    submit.disabled = true;
    try {
      const response = await fetch(locked ? config.dataset.unlockUrl : config.dataset.lockUrl, {
        method: 'POST', credentials: 'same-origin', headers: {'X-CSRFToken': form.querySelector('[name=csrfmiddlewaretoken]').value},
        body: new FormData(form),
      });
      if (response.redirected) { location.href = config.dataset.attendanceUrl; return; }
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.message || 'Unable to update the kiosk. Please try again.');
      applyLock(data.locked);
      dialog.close();
      if (!data.locked && document.fullscreenElement) await document.exitFullscreen().catch(() => {});
      (data.locked ? document.getElementById('exitFullscreenBtn') : document.getElementById('fullscreenBtn'))?.focus();
    } catch (err) {
      error.textContent = err.message;
      code.value = '';
      code.focus();
    } finally {
      submitting = false;
      submit.disabled = false;
    }
  });
  // Refresh lock state when returning from another tab or the browser's back cache.
  async function syncLock() {
    if (submitting) return;
    const currentRevision = revision;
    try {
      const response = await fetch(config.dataset.statusUrl, {cache: 'no-store'});
      if (response.redirected) { location.href = config.dataset.attendanceUrl; return; }
      if (response.ok) {
        const data = await response.json();
        if (!submitting && revision === currentRevision) applyLock(data.locked);
      }
    } catch (_) { /* An offline request never clears an existing lock. */ }
  }
  window.addEventListener('pageshow', syncLock);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) syncLock(); });
  applyLock(locked);
};
