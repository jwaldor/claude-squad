const path = require('path');
const http = require('http');
const express = require('express');
const { Server } = require('socket.io');
const { createStore } = require('./db');

const VALID_STATUS = new Set(['yes', 'maybe', 'no']);
const BUILD_ID = process.env.BUILD_ID || String(Date.now());

async function main() {
  const store = await createStore();

  const app = express();
  app.use(express.json({ limit: '64kb' }));

  app.use((req, res, next) => {
    const p = req.path;
    if (p === '/' || p.startsWith('/e/') || p.endsWith('.html')) {
      res.set('Cache-Control', 'no-store, must-revalidate');
    } else if (p.endsWith('.js') || p.endsWith('.css')) {
      res.set('Cache-Control', 'public, max-age=0, must-revalidate');
    }
    next();
  });

  const fs = require('fs');
  const htmlCache = new Map();
  function readHtml(name) {
    if (!htmlCache.has(name)) {
      const raw = fs.readFileSync(path.join(__dirname, 'public', name), 'utf8');
      htmlCache.set(name, raw);
    }
    return htmlCache.get(name).replace(/__BUILD_ID__/g, BUILD_ID);
  }

  app.get(['/', '/e/:id'], (req, res) => {
    const file = req.path === '/' ? 'index.html' : 'event.html';
    res.set('Cache-Control', 'no-store, must-revalidate');
    res.set('Content-Type', 'text/html; charset=utf-8');
    res.send(readHtml(file));
  });

  app.use(express.static(path.join(__dirname, 'public'), {
    setHeaders(res, filePath) {
      if (filePath.endsWith('.html')) {
        res.setHeader('Cache-Control', 'no-store, must-revalidate');
      }
    },
  }));

  const server = http.createServer(app);
  const io = new Server(server, { cors: { origin: '*' } });

  const socketsByAttendee = new Map();

  function addAttendeeSocket(attendeeId, socketId) {
    if (!socketsByAttendee.has(attendeeId)) socketsByAttendee.set(attendeeId, new Set());
    socketsByAttendee.get(attendeeId).add(socketId);
  }

  function removeAttendeeSocket(attendeeId, socketId) {
    const set = socketsByAttendee.get(attendeeId);
    if (!set) return;
    set.delete(socketId);
    if (set.size === 0) socketsByAttendee.delete(attendeeId);
  }

  async function broadcastRoster(eventId) {
    const attendees = await store.listAttendees(eventId);
    io.to(roomFor(eventId)).emit('roster', { eventId, attendees });
  }

  function roomFor(eventId) { return `event:${eventId}`; }

  app.post('/api/events', async (req, res) => {
    try {
      const { title, description, location, startsAt, hostName } = req.body || {};
      const t = (title || '').trim().slice(0, 120);
      if (!t) return res.status(400).json({ error: 'Title required' });
      let starts = null;
      if (startsAt) {
        const d = new Date(startsAt);
        if (isNaN(d.getTime())) return res.status(400).json({ error: 'Invalid startsAt' });
        starts = d.toISOString();
      }
      const event = await store.createEvent({
        title: t,
        description: (description || '').toString().slice(0, 2000),
        location: (location || '').toString().slice(0, 200),
        startsAt: starts,
        hostName: (hostName || '').toString().trim().slice(0, 60),
      });
      res.status(201).json({ event });
    } catch (err) {
      console.error('createEvent', err);
      res.status(500).json({ error: 'Server error' });
    }
  });

  app.get('/api/events/:id', async (req, res) => {
    try {
      const event = await store.getEvent(req.params.id);
      if (!event) return res.status(404).json({ error: 'Not found' });
      const attendees = await store.listAttendees(event.id);
      res.json({ event, attendees });
    } catch (err) {
      console.error('getEvent', err);
      res.status(500).json({ error: 'Server error' });
    }
  });

  app.post('/api/events/:id/rsvp', async (req, res) => {
    try {
      const event = await store.getEvent(req.params.id);
      if (!event) return res.status(404).json({ error: 'Not found' });
      const name = (req.body && req.body.name || '').toString().trim().slice(0, 40);
      const status = (req.body && req.body.status || 'yes').toString();
      if (!name) return res.status(400).json({ error: 'Name required' });
      if (!VALID_STATUS.has(status)) return res.status(400).json({ error: 'Invalid status' });
      const attendee = await store.addAttendee({ eventId: event.id, name, status });
      await broadcastRoster(event.id);
      res.status(201).json({ attendee });
    } catch (err) {
      console.error('rsvp', err);
      res.status(500).json({ error: 'Server error' });
    }
  });

  app.patch('/api/attendees/:id', async (req, res) => {
    try {
      const status = (req.body && req.body.status || '').toString();
      if (!VALID_STATUS.has(status)) return res.status(400).json({ error: 'Invalid status' });
      const attendee = await store.updateAttendeeStatus(req.params.id, status);
      if (!attendee) return res.status(404).json({ error: 'Not found' });
      await broadcastRoster(attendee.eventId);
      res.json({ attendee });
    } catch (err) {
      console.error('updateStatus', err);
      res.status(500).json({ error: 'Server error' });
    }
  });

  app.delete('/api/attendees/:id', async (req, res) => {
    try {
      const attendee = await store.getAttendee(req.params.id);
      if (!attendee) return res.status(404).json({ error: 'Not found' });
      await store.removeAttendee(attendee.id);
      await broadcastRoster(attendee.eventId);
      res.json({ ok: true });
    } catch (err) {
      console.error('removeAttendee', err);
      res.status(500).json({ error: 'Server error' });
    }
  });

  io.on('connection', (socket) => {
    let subscribedEventId = null;
    let myAttendeeId = null;

    socket.on('subscribe', async ({ eventId, attendeeId }, ack) => {
      if (!eventId) { if (ack) ack({ ok: false }); return; }
      if (subscribedEventId) socket.leave(roomFor(subscribedEventId));
      subscribedEventId = eventId;
      socket.join(roomFor(eventId));

      if (attendeeId) {
        const a = await store.getAttendee(attendeeId);
        if (a && a.eventId === eventId) {
          myAttendeeId = a.id;
          addAttendeeSocket(a.id, socket.id);
        }
      }
      const attendees = await store.listAttendees(eventId);
      socket.emit('roster', { eventId, attendees });
      if (ack) ack({ ok: true });
    });

    socket.on('claim', async ({ attendeeId }, ack) => {
      if (!subscribedEventId || !attendeeId) { if (ack) ack({ ok: false }); return; }
      const a = await store.getAttendee(attendeeId);
      if (!a || a.eventId !== subscribedEventId) { if (ack) ack({ ok: false }); return; }
      if (myAttendeeId && myAttendeeId !== a.id) removeAttendeeSocket(myAttendeeId, socket.id);
      myAttendeeId = a.id;
      addAttendeeSocket(a.id, socket.id);
      if (ack) ack({ ok: true });
    });

    socket.on('claw', async ({ targetId }, ack) => {
      if (!subscribedEventId || !myAttendeeId || !targetId) { if (ack) ack({ ok: false }); return; }
      if (targetId === myAttendeeId) { if (ack) ack({ ok: false, error: 'no self-claude' }); return; }
      const sender = await store.getAttendee(myAttendeeId);
      const target = await store.incrementClaw(targetId);
      if (!sender || !target || sender.eventId !== subscribedEventId || target.eventId !== subscribedEventId) {
        if (ack) ack({ ok: false });
        return;
      }
      const targetSockets = socketsByAttendee.get(target.id);
      if (targetSockets) {
        for (const sid of targetSockets) {
          io.to(sid).emit('youGotClawed', {
            eventId: subscribedEventId,
            from: { id: sender.id, name: sender.name },
            at: Date.now(),
          });
        }
      }
      io.to(roomFor(subscribedEventId)).emit('clawUpdate', {
        eventId: subscribedEventId,
        id: target.id,
        clawCount: target.clawCount,
      });
      if (ack) ack({ ok: true });
    });

    socket.on('disconnect', () => {
      if (myAttendeeId) removeAttendeeSocket(myAttendeeId, socket.id);
    });
  });

  const PORT = process.env.PORT || 3000;
  server.listen(PORT, () => {
    console.log(`WhoWantsToClawedRightNow listening on http://localhost:${PORT}`);
  });
}

main().catch((err) => {
  console.error('Fatal startup error', err);
  process.exit(1);
});
