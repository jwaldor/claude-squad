const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;
const UPLOAD_PASSWORD = process.env.UPLOAD_PASSWORD || 'sauna-gains-2026';
const DATA_DIR = process.env.DATA_DIR || path.join(__dirname, 'data');
const UPLOAD_DIR = path.join(DATA_DIR, 'uploads');
const SEED_DIR = path.join(__dirname, 'seed-uploads');
const META_FILE = path.join(DATA_DIR, 'photos.json');

const SECTIONS = ['weights', 'vibe-coding', 'sauna', 'dinner'];

const SEED_PHOTOS = [
  {
    id: 'seed-jacob-weights-1',
    filename: '1777071599375-ygtyxt.jpeg',
    section: 'weights',
    caption: '',
    uploader: 'Jacob',
    uploadedAt: '2026-04-24T22:59:59.957Z'
  }
];

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });
if (!fs.existsSync(META_FILE)) fs.writeFileSync(META_FILE, JSON.stringify({ photos: [] }, null, 2));

function seedAndMigrate() {
  let photos;
  try {
    photos = JSON.parse(fs.readFileSync(META_FILE, 'utf8')).photos || [];
  } catch {
    photos = [];
  }

  // migrate any existing climbing photos to weights
  let changed = false;
  photos.forEach(p => {
    if (p.section === 'climbing') { p.section = 'weights'; changed = true; }
  });

  // seed any missing seed photos and copy files into UPLOAD_DIR
  SEED_PHOTOS.forEach(seed => {
    const seedSrc = path.join(SEED_DIR, seed.filename);
    const target = path.join(UPLOAD_DIR, seed.filename);
    if (fs.existsSync(seedSrc) && !fs.existsSync(target)) {
      fs.copyFileSync(seedSrc, target);
    }
    if (!photos.find(p => p.id === seed.id || p.filename === seed.filename)) {
      photos.push({ ...seed, url: `/uploads/${seed.filename}` });
      changed = true;
    }
  });

  if (changed) fs.writeFileSync(META_FILE, JSON.stringify({ photos }, null, 2));
}

seedAndMigrate();

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const safe = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}${ext}`;
    cb(null, safe);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 25 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = /^image\/(jpeg|png|gif|webp|heic|heif)$/.test(file.mimetype);
    cb(ok ? null : new Error('Only image files allowed'), ok);
  }
});

app.use(express.json());
app.use('/uploads', express.static(UPLOAD_DIR));
app.use(express.static(path.join(__dirname, 'public')));

function readPhotos() {
  try {
    return JSON.parse(fs.readFileSync(META_FILE, 'utf8')).photos || [];
  } catch {
    return [];
  }
}

function writePhotos(photos) {
  fs.writeFileSync(META_FILE, JSON.stringify({ photos }, null, 2));
}

app.get('/api/photos', (req, res) => {
  const { section } = req.query;
  const all = readPhotos();
  res.json(section ? all.filter(p => p.section === section) : all);
});

app.post('/api/upload', (req, res) => {
  upload.array('photos', 20)(req, res, (err) => {
    if (err) return res.status(400).json({ error: err.message });

    const password = req.body.password;
    if (password !== UPLOAD_PASSWORD) {
      (req.files || []).forEach(f => fs.unlink(f.path, () => {}));
      return res.status(401).json({ error: 'Wrong password, try again squad member' });
    }

    const section = req.body.section;
    if (!SECTIONS.includes(section)) {
      (req.files || []).forEach(f => fs.unlink(f.path, () => {}));
      return res.status(400).json({ error: 'Invalid section' });
    }

    const caption = (req.body.caption || '').slice(0, 280);
    const uploader = (req.body.uploader || 'Anonymous Squad Member').slice(0, 60);

    const photos = readPhotos();
    const added = (req.files || []).map(f => ({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      filename: f.filename,
      url: `/uploads/${f.filename}`,
      section,
      caption,
      uploader,
      uploadedAt: new Date().toISOString()
    }));

    photos.push(...added);
    writePhotos(photos);

    res.json({ success: true, added });
  });
});

app.get('/health', (req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`Claude Squad website running on port ${PORT}`);
});
