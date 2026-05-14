(() => {
  const eventId = window.location.pathname.split('/').filter(Boolean).pop();
  if (!eventId) { window.location.href = '/'; return; }

  const ME_KEY = `clawed.me.${eventId}`;
  const RECENT_KEY = 'clawed.recentEvents';

  const els = {
    title: document.getElementById('event-title'),
    when: document.getElementById('event-when'),
    where: document.getElementById('event-where'),
    host: document.getElementById('event-host'),
    desc: document.getElementById('event-desc'),
    shareUrl: document.getElementById('share-url'),
    shareCopy: document.getElementById('share-copy'),
    rsvpCard: document.getElementById('rsvp-card'),
    rsvpForm: document.getElementById('rsvp-form'),
    rsvpName: document.getElementById('rsvp-name'),
    rsvpError: document.getElementById('rsvp-error'),
    meCard: document.getElementById('me-card'),
    meName: document.getElementById('me-name'),
    meStatus: document.getElementById('me-status'),
    leaveBtn: document.getElementById('leave-btn'),
    toastStack: document.getElementById('toast-stack'),
    rosters: {
      yes: document.getElementById('roster-yes'),
      maybe: document.getElementById('roster-maybe'),
      no: document.getElementById('roster-no'),
    },
    counts: {
      yes: document.getElementById('count-yes'),
      maybe: document.getElementById('count-maybe'),
      no: document.getElementById('count-no'),
    },
    emptyYes: document.getElementById('empty-yes'),
  };

  let event = null;
  let attendees = [];
  let me = loadMe();
  let socket = null;

  function loadMe() {
    try { return JSON.parse(localStorage.getItem(ME_KEY) || 'null'); } catch { return null; }
  }
  function saveMe(a) {
    me = a;
    if (a) localStorage.setItem(ME_KEY, JSON.stringify(a));
    else localStorage.removeItem(ME_KEY);
    renderMe();
    renderRosters();
  }

  function rememberRecent(ev) {
    try {
      const list = JSON.parse(localStorage.getItem(RECENT_KEY) || '[]').filter((e) => e.id !== ev.id);
      list.unshift({ id: ev.id, title: ev.title, startsAt: ev.startsAt });
      localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 10)));
    } catch {}
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
  }

  function renderEvent() {
    document.title = `${event.title} · WhoWantsToClawedRightNow`;
    els.title.textContent = event.title;
    els.when.textContent = event.startsAt
      ? new Date(event.startsAt).toLocaleString(undefined, { dateStyle: 'full', timeStyle: 'short' })
      : 'Date TBD';
    els.where.textContent = event.location ? `📍 ${event.location}` : '';
    els.where.hidden = !event.location;
    els.host.textContent = event.hostName ? `Hosted by ${event.hostName}` : '';
    els.host.hidden = !event.hostName;
    els.desc.textContent = event.description || '';
    els.desc.hidden = !event.description;
    els.shareUrl.value = window.location.href;
  }

  function renderMe() {
    if (me) {
      els.rsvpCard.hidden = true;
      els.meCard.hidden = false;
      els.meName.textContent = me.name;
      els.meStatus.value = me.status;
    } else {
      els.rsvpCard.hidden = false;
      els.meCard.hidden = true;
    }
  }

  function renderRosters() {
    const groups = { yes: [], maybe: [], no: [] };
    for (const a of attendees) (groups[a.status] || groups.yes).push(a);

    for (const status of ['yes', 'maybe', 'no']) {
      els.counts[status].textContent = String(groups[status].length);
      const list = els.rosters[status];
      list.innerHTML = '';
      for (const a of groups[status]) list.appendChild(renderRow(a, status));
    }
    els.emptyYes.hidden = groups.yes.length > 0;
  }

  function renderRow(a, status) {
    const li = document.createElement('li');
    li.className = 'row' + (me && a.id === me.id ? ' is-me' : '');

    const nameEl = document.createElement('span');
    nameEl.className = 'name';
    nameEl.textContent = a.name;
    li.appendChild(nameEl);

    if (me && a.id === me.id) {
      const tag = document.createElement('span');
      tag.className = 'me-tag';
      tag.textContent = 'you';
      li.appendChild(tag);
    }

    if (status === 'yes') {
      const countEl = document.createElement('span');
      countEl.className = 'claw-count';
      countEl.title = 'Clawes received';
      countEl.textContent = a.clawCount > 0 ? `×${a.clawCount}` : '';
      li.appendChild(countEl);

      const btn = document.createElement('button');
      btn.className = 'claw-btn';
      btn.type = 'button';
      btn.textContent = '🦞';
      btn.setAttribute('aria-label', `Claude ${a.name}`);
      if (!me) {
        btn.disabled = true;
        btn.title = 'RSVP first to claude people';
      } else if (a.id === me.id) {
        btn.disabled = true;
        btn.title = "You can't claude yourself";
      } else {
        btn.title = `Claude ${a.name}`;
        btn.addEventListener('click', () => fireClaw(a.id, btn));
      }
      li.appendChild(btn);
    }

    return li;
  }

  function fireClaw(targetId, btn) {
    if (!me || !socket) return;
    btn.classList.remove('fired');
    void btn.offsetWidth;
    btn.classList.add('fired');
    socket.emit('claw', { targetId });
  }

  function showToast({ title, body }) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
      <span class="big">🦞</span>
      <span><strong>${escapeHtml(title)}</strong><br /><span style="color:var(--muted)">${escapeHtml(body)}</span></span>
    `;
    els.toastStack.appendChild(toast);
    setTimeout(() => toast.remove(), 4200);

    document.body.classList.remove('shake');
    void document.body.offsetWidth;
    document.body.classList.add('shake');

    if ('Notification' in window && Notification.permission === 'granted') {
      try { new Notification(title, { body }); } catch {}
    }
  }

  function maybeAskForNotifications() {
    if (!('Notification' in window)) return;
    if (Notification.permission === 'default') {
      try { Notification.requestPermission(); } catch {}
    }
  }

  els.shareCopy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(els.shareUrl.value);
      els.shareCopy.textContent = 'Copied!';
      setTimeout(() => (els.shareCopy.textContent = 'Copy link'), 1400);
    } catch {
      els.shareUrl.select();
    }
  });

  els.rsvpForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    els.rsvpError.hidden = true;
    const fd = new FormData(els.rsvpForm);
    const name = (fd.get('name') || '').toString().trim();
    const status = (fd.get('status') || 'yes').toString();
    if (!name) return;
    try {
      const res = await fetch(`/api/events/${eventId}/rsvp`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ name, status }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      saveMe(body.attendee);
      socket.emit('claim', { attendeeId: body.attendee.id });
      maybeAskForNotifications();
    } catch (err) {
      els.rsvpError.textContent = err.message || 'Could not RSVP';
      els.rsvpError.hidden = false;
    }
  });

  els.meStatus.addEventListener('change', async () => {
    if (!me) return;
    const status = els.meStatus.value;
    try {
      const res = await fetch(`/api/attendees/${me.id}`, {
        method: 'PATCH',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
      saveMe(body.attendee);
    } catch (err) {
      console.error(err);
    }
  });

  els.leaveBtn.addEventListener('click', async () => {
    if (!me) return;
    try {
      await fetch(`/api/attendees/${me.id}`, { method: 'DELETE' });
    } catch {}
    saveMe(null);
  });

  async function loadEvent() {
    const res = await fetch(`/api/events/${eventId}`);
    if (!res.ok) {
      document.body.innerHTML = '<main style="padding:2rem;text-align:center"><h1>Event not found 🦞</h1><p><a href="/">Go home</a></p></main>';
      return;
    }
    const body = await res.json();
    event = body.event;
    attendees = body.attendees;

    if (me) {
      const stillThere = attendees.find((a) => a.id === me.id);
      if (!stillThere) saveMe(null);
    }

    rememberRecent(event);
    renderEvent();
    renderMe();
    renderRosters();

    socket = io();
    socket.on('connect', () => {
      socket.emit('subscribe', { eventId, attendeeId: me ? me.id : null });
    });
    socket.on('roster', (payload) => {
      if (payload.eventId !== eventId) return;
      attendees = payload.attendees;
      if (me) {
        const stillThere = attendees.find((a) => a.id === me.id);
        if (!stillThere) saveMe(null);
        else if (stillThere.status !== me.status) saveMe(stillThere);
      }
      renderRosters();
    });
    socket.on('clawUpdate', ({ id, clawCount }) => {
      const a = attendees.find((x) => x.id === id);
      if (a) { a.clawCount = clawCount; renderRosters(); }
    });
    socket.on('youGotClawed', ({ from }) => {
      showToast({
        title: `${from.name} is trying to claude you!`,
        body: 'Pinch back at WhoWantsToClawedRightNow.com',
      });
    });
  }

  loadEvent();
})();
