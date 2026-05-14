(() => {
  const RECENT_KEY = 'clawed.recentEvents';

  const els = {
    form: document.getElementById('create-form'),
    error: document.getElementById('create-error'),
    recentCard: document.getElementById('recent-card'),
    recent: document.getElementById('recent'),
  };

  function loadRecent() {
    try { return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]'); }
    catch { return []; }
  }

  function saveRecent(entry) {
    const list = loadRecent().filter((e) => e.id !== entry.id);
    list.unshift(entry);
    localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 10)));
  }

  function renderRecent() {
    const list = loadRecent();
    if (!list.length) { els.recentCard.hidden = true; return; }
    els.recentCard.hidden = false;
    els.recent.innerHTML = '';
    for (const e of list) {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = `/e/${e.id}`;
      a.textContent = e.title;
      li.appendChild(a);
      if (e.startsAt) {
        const span = document.createElement('span');
        span.className = 'recent-when';
        span.textContent = new Date(e.startsAt).toLocaleString();
        li.appendChild(span);
      }
      els.recent.appendChild(li);
    }
  }

  els.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    els.error.hidden = true;
    const data = Object.fromEntries(new FormData(els.form).entries());
    if (data.startsAt) {
      const d = new Date(data.startsAt);
      data.startsAt = isNaN(d.getTime()) ? null : d.toISOString();
    } else {
      data.startsAt = null;
    }
    try {
      const res = await fetch('/api/events', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(data),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      saveRecent({ id: body.event.id, title: body.event.title, startsAt: body.event.startsAt });
      window.location.href = `/e/${body.event.id}`;
    } catch (err) {
      els.error.textContent = err.message || 'Could not create event';
      els.error.hidden = false;
    }
  });

  renderRecent();
})();
