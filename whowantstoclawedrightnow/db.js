const { Pool } = require('pg');

const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  location TEXT NOT NULL DEFAULT '',
  starts_at TIMESTAMPTZ,
  host_name TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attendees (
  id TEXT PRIMARY KEY,
  event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'yes',
  claw_count INTEGER NOT NULL DEFAULT 0,
  joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_attendees_event ON attendees(event_id);
`;

function shortId(len = 7) {
  const alphabet = 'abcdefghjkmnpqrstuvwxyz23456789';
  let out = '';
  for (let i = 0; i < len; i++) out += alphabet[Math.floor(Math.random() * alphabet.length)];
  return out;
}

function attendeeId() {
  return `a_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}

function rowToEvent(r) {
  if (!r) return null;
  return {
    id: r.id,
    title: r.title,
    description: r.description || '',
    location: r.location || '',
    startsAt: r.starts_at ? new Date(r.starts_at).toISOString() : null,
    hostName: r.host_name || '',
    createdAt: new Date(r.created_at).toISOString(),
  };
}

function rowToAttendee(r) {
  if (!r) return null;
  return {
    id: r.id,
    eventId: r.event_id,
    name: r.name,
    status: r.status,
    clawCount: Number(r.claw_count) || 0,
    joinedAt: new Date(r.joined_at).toISOString(),
  };
}

class PgStore {
  constructor(connectionString) {
    this.pool = new Pool({
      connectionString,
      ssl: connectionString && /sslmode=require|railway|render|amazonaws|neon|supabase/i.test(connectionString)
        ? { rejectUnauthorized: false }
        : undefined,
    });
  }

  async migrate() {
    await this.pool.query(SCHEMA_SQL);
  }

  async createEvent({ title, description, location, startsAt, hostName }) {
    let id;
    for (let i = 0; i < 5; i++) {
      id = shortId();
      const dup = await this.pool.query('SELECT 1 FROM events WHERE id = $1', [id]);
      if (dup.rowCount === 0) break;
    }
    const { rows } = await this.pool.query(
      `INSERT INTO events (id, title, description, location, starts_at, host_name)
       VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
      [id, title, description || '', location || '', startsAt || null, hostName || ''],
    );
    return rowToEvent(rows[0]);
  }

  async getEvent(id) {
    const { rows } = await this.pool.query('SELECT * FROM events WHERE id = $1', [id]);
    return rowToEvent(rows[0]);
  }

  async listAttendees(eventId) {
    const { rows } = await this.pool.query(
      'SELECT * FROM attendees WHERE event_id = $1 ORDER BY joined_at ASC',
      [eventId],
    );
    return rows.map(rowToAttendee);
  }

  async addAttendee({ eventId, name, status }) {
    const id = attendeeId();
    const { rows } = await this.pool.query(
      `INSERT INTO attendees (id, event_id, name, status) VALUES ($1, $2, $3, $4) RETURNING *`,
      [id, eventId, name, status || 'yes'],
    );
    return rowToAttendee(rows[0]);
  }

  async updateAttendeeStatus(id, status) {
    const { rows } = await this.pool.query(
      `UPDATE attendees SET status = $2 WHERE id = $1 RETURNING *`,
      [id, status],
    );
    return rowToAttendee(rows[0]);
  }

  async getAttendee(id) {
    const { rows } = await this.pool.query('SELECT * FROM attendees WHERE id = $1', [id]);
    return rowToAttendee(rows[0]);
  }

  async removeAttendee(id) {
    await this.pool.query('DELETE FROM attendees WHERE id = $1', [id]);
  }

  async incrementClaw(targetId) {
    const { rows } = await this.pool.query(
      `UPDATE attendees SET claw_count = claw_count + 1 WHERE id = $1 RETURNING *`,
      [targetId],
    );
    return rowToAttendee(rows[0]);
  }
}

class MemoryStore {
  constructor() {
    this.events = new Map();
    this.attendees = new Map();
  }
  async migrate() {}

  async createEvent({ title, description, location, startsAt, hostName }) {
    let id;
    do { id = shortId(); } while (this.events.has(id));
    const event = {
      id,
      title,
      description: description || '',
      location: location || '',
      startsAt: startsAt ? new Date(startsAt).toISOString() : null,
      hostName: hostName || '',
      createdAt: new Date().toISOString(),
    };
    this.events.set(id, event);
    return event;
  }

  async getEvent(id) { return this.events.get(id) || null; }

  async listAttendees(eventId) {
    return Array.from(this.attendees.values())
      .filter((a) => a.eventId === eventId)
      .sort((a, b) => new Date(a.joinedAt) - new Date(b.joinedAt));
  }

  async addAttendee({ eventId, name, status }) {
    const id = attendeeId();
    const a = {
      id,
      eventId,
      name,
      status: status || 'yes',
      clawCount: 0,
      joinedAt: new Date().toISOString(),
    };
    this.attendees.set(id, a);
    return a;
  }

  async updateAttendeeStatus(id, status) {
    const a = this.attendees.get(id);
    if (!a) return null;
    a.status = status;
    return a;
  }

  async getAttendee(id) { return this.attendees.get(id) || null; }

  async removeAttendee(id) { this.attendees.delete(id); }

  async incrementClaw(targetId) {
    const a = this.attendees.get(targetId);
    if (!a) return null;
    a.clawCount += 1;
    return a;
  }
}

async function createStore() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.warn('[db] DATABASE_URL not set — using in-memory store. Data will not persist across restarts.');
    return new MemoryStore();
  }
  const store = new PgStore(url);
  await store.migrate();
  console.log('[db] Connected to Postgres and ran migrations.');
  return store;
}

module.exports = { createStore };
