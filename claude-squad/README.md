# Claude Squad

Silly site for the squad that vibe codes on phones (with voice) while lifting heavy things at Vital Climbing Gym, then saunas, then eats dinner.

## Run locally

```bash
npm install
npm start
# open http://localhost:3000
```

## Environment

- `PORT` &mdash; server port (defaults to 3000; Railway sets this automatically).
- `UPLOAD_PASSWORD` &mdash; password required to upload photos. Defaults to `sauna-gains-2026` if not set.

## Deploy on Railway

1. Push this repo to GitHub (or use the Railway CLI).
2. Create a new Railway service from the repo.
3. (Optional) Set `UPLOAD_PASSWORD` in the Railway service variables to override the default.
4. Railway auto-detects Node, runs `npm install`, then `node server.js`.

## Notes

Photo metadata is persisted in `photos.json` and uploaded files in `public/uploads/`. On Railway you'll likely want a volume mounted at the project root (or at `public/uploads/`) so uploads survive redeploys.
