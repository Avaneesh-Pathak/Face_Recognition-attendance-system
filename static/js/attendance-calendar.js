(() => {
  'use strict';
  const root = document.getElementById('attendanceCalendar');
  if (!root) return;
  const employee = root.dataset.employee;
  const dialog = document.getElementById('calendarDetail');
  const form = document.getElementById('correctionForm');
  const error = document.getElementById('correctionError');
  let selectedDate = '', loadVersion = 0;
  let previousOverflow = '';
  function setView(view) {
    const list = view === 'list';
    document.getElementById('monthView').hidden = list;
    document.getElementById('listView').hidden = !list;
    for (const [id, active] of [['monthViewButton', !list], ['listViewButton', list]]) {
      const control = document.getElementById(id);
      control.classList.toggle('is-active', active);
      control.setAttribute('aria-pressed', String(active));
    }
    try { localStorage.setItem('attendanceCalendarView', list ? 'list' : 'month'); } catch (_) { /* Storage may be disabled. */ }
  }
  document.getElementById('monthViewButton').addEventListener('click', () => setView('month'));
  document.getElementById('listViewButton').addEventListener('click', () => setView('list'));
  try { setView(localStorage.getItem('attendanceCalendarView') || 'month'); } catch (_) { setView('month'); }
  document.getElementById('monthPicker').addEventListener('change', e => {
    const value = e.target.value;
    if (/^(20\d\d|2100)-(0[1-9]|1[0-2])$/.test(value)) {
      const [year, month] = value.split('-');
      location.href = `/attendance-calendar/${employee}/${year}/${Number(month)}/`;
    }
  });
  document.getElementById('calendarEmployee').addEventListener('change', e => {
    location.href = e.target.dataset.url.replace('999999', e.target.value);
  });
  document.getElementById('closeDetail').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => {
    loadVersion++;
    document.body.style.overflow = previousOverflow;
  });
  const button = (label, handler) => {
    const el = document.createElement('button'); el.type = 'button';
    el.className = 'ac-record-action'; el.textContent = label;
    el.addEventListener('click', handler); return el;
  };
  function edit(record = {}) {
    if (!form) return;
    form.hidden = false;
    document.getElementById('recordId').value = record.id || '';
    document.getElementById('recordType').value = record.attendance_type || 'check_in';
    document.getElementById('recordTime').value = record.local_timestamp || `${selectedDate}T09:00`;
    document.getElementById('recordLocation').value = record.location || '';
    document.getElementById('recordReason').value = '';
    document.getElementById('recordReason').focus();
  }
  document.getElementById('cancelCorrection')?.addEventListener('click', () => {
    form.hidden = true;
    error.textContent = '';
    document.getElementById('addRecord').focus();
  });
  async function mutate(url, data, returnDate = selectedDate) {
    error.textContent = '';
    const response = await fetch(url, {method: 'POST', body: data,
      headers: {'X-CSRFToken': form.querySelector('[name=csrfmiddlewaretoken]').value}});
    const result = await response.json();
    if (!response.ok || !result.success) throw new Error(result.error || 'Unable to save attendance');
    const [year, month] = returnDate.split('-');
    location.href = `/attendance-calendar/${employee}/${year}/${Number(month)}/`;
  }
  async function remove(record, source) {
    const reason = prompt('Explain why this record should be deleted (at least 5 characters):');
    if (!reason) return;
    const data = new FormData(); data.append('reason', reason); source.disabled = true;
    try { await mutate(`/attendance/delete/${record.id}/`, data, selectedDate); }
    catch (e) { error.textContent = e.message; source.disabled = false; }
  }
  root.querySelectorAll('[data-date]').forEach(cell => cell.addEventListener('click', async () => {
    selectedDate = cell.dataset.date; const version = ++loadVersion; error.textContent = '';
    if (form) form.hidden = true;
    document.getElementById('detailTitle').textContent = selectedDate;
    const summary = document.getElementById('detailSummary'); summary.textContent = 'Loading attendance…';
    const rows = document.getElementById('detailLogs'); rows.replaceChildren();
    previousOverflow = document.body.style.overflow;
    dialog.showModal(); document.body.style.overflow = 'hidden';
    try {
      const response = await fetch(`/attendance-day-detail/${employee}/${selectedDate}/`);
      if (!response.ok) throw new Error('Unable to load attendance');
      const data = await response.json(); if (version !== loadVersion) return;
      document.getElementById('detailTitle').textContent = data.date;
      summary.textContent = `${data.status} · ${data.total_hours} completed hours${data.issues.length ? ' · ' + data.issues.join('; ') : ''}`;
      data.logs.forEach(record => {
        const row = document.createElement('tr');
        const labels = ['Event', 'Date & time', 'Location', 'Reason'];
        [record.type, `${record.date} ${record.timestamp}`, record.location || '—', record.notes || '—'].forEach((value, index) => {
          const td = document.createElement('td'); td.dataset.label = labels[index];
          td.textContent = value; row.append(td);
        });
        const actions = document.createElement('td');
        if (data.can_edit && form) {
          actions.append(button('Edit', () => edit(record)));
          const del = button('Delete', () => remove(record, del)); actions.append(del);
        }
        row.append(actions); rows.append(row);
      });
      if (!data.logs.length) {
        const row = document.createElement('tr'), td = document.createElement('td');
        td.colSpan = 5; td.textContent = 'No records for this work date.'; row.append(td); rows.append(row);
      }
    } catch (e) { summary.textContent = e.message; }
  }));
  document.getElementById('addRecord')?.addEventListener('click', () => edit());
  document.getElementById('calendarAddRecord')?.addEventListener('click', () => {
    const dateInput = document.getElementById('addAttendanceDate');
    if (!dateInput.value) return;
    selectedDate = dateInput.value;
    document.getElementById('detailTitle').textContent = selectedDate;
    document.getElementById('detailSummary').textContent = 'Add a past or current attendance record for this date.';
    document.getElementById('detailLogs').replaceChildren();
    error.textContent = '';
    if (!dialog.open) {
      previousOverflow = document.body.style.overflow;
      dialog.showModal();
      document.body.style.overflow = 'hidden';
    }
    edit();
  });
  form?.addEventListener('submit', async e => {
    e.preventDefault(); const submit = form.querySelector('[type=submit]'); submit.disabled = true;
    const id = document.getElementById('recordId').value;
    const data = new FormData(); data.append('employee', employee);
    for (const [name, field] of [['attendance_type', 'recordType'], ['timestamp', 'recordTime'], ['location', 'recordLocation'], ['reason', 'recordReason']]) data.append(name, document.getElementById(field).value);
    const returnDate = document.getElementById('recordTime').value.slice(0, 10) || selectedDate;
    try { await mutate(id ? `/attendance/update/${id}/` : '/attendance/create/', data, returnDate); }
    catch (e) { error.textContent = e.message; submit.disabled = false; }
  });
})();
