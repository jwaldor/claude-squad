const SECTIONS = ['weights', 'vibe-coding', 'sauna', 'dinner'];
const SECTION_LABELS = {
  'weights': 'LIFTING WEIGHTS',
  'vibe-coding': 'VIBE CODING',
  'sauna': 'SAUNA',
  'dinner': 'DINNER'
};

async function loadGallery(section) {
  const container = document.getElementById(`gallery-${section}`);
  if (!container) return;
  try {
    const res = await fetch(`/api/photos?section=${encodeURIComponent(section)}`);
    const photos = await res.json();
    renderGallery(container, photos, section);
  } catch (e) {
    container.innerHTML = '<div class="empty">couldn\'t load photos. send help.</div>';
  }
}

function renderGallery(container, photos, section) {
  if (!photos.length) {
    container.innerHTML = `<div class="empty">no ${SECTION_LABELS[section].toLowerCase()} pics yet &mdash; be the first!</div>`;
    return;
  }
  photos.sort((a, b) => new Date(b.uploadedAt) - new Date(a.uploadedAt));
  container.innerHTML = photos.map(p => `
    <div class="photo-card">
      <img src="${p.url}" alt="${escapeHtml(p.caption || 'squad photo')}" loading="lazy" data-full="${p.url}">
      <div class="photo-meta">
        <strong>${escapeHtml(p.uploader || 'Anonymous Squad Member')}</strong>
        ${p.caption ? `<em>${escapeHtml(p.caption)}</em>` : ''}
      </div>
    </div>
  `).join('');

  container.querySelectorAll('img').forEach(img => {
    img.addEventListener('click', () => openLightbox(img.dataset.full));
  });
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function openLightbox(url) {
  const box = document.createElement('div');
  box.className = 'lightbox';
  box.innerHTML = `<img src="${url}" alt="">`;
  box.addEventListener('click', () => box.remove());
  document.body.appendChild(box);
}

const modal = document.getElementById('upload-modal');
const modalSection = document.getElementById('modal-section');
const modalTitle = document.getElementById('modal-title');
const modalStatus = document.getElementById('modal-status');
const modalForm = document.getElementById('upload-form');
const modalSubmit = document.getElementById('modal-submit');
const modalClose = document.getElementById('modal-close');

document.querySelectorAll('.upload-trigger').forEach(btn => {
  btn.addEventListener('click', () => {
    const section = btn.dataset.section;
    modalSection.value = section;
    modalTitle.textContent = `UPLOAD ${SECTION_LABELS[section]} PHOTOS`;
    modalStatus.textContent = '';
    modalStatus.className = 'modal-status';
    modalForm.reset();
    modalSection.value = section;
    modal.hidden = false;
  });
});

modalClose.addEventListener('click', () => { modal.hidden = true; });
modal.addEventListener('click', (e) => { if (e.target === modal) modal.hidden = true; });

modalForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  modalSubmit.disabled = true;
  modalStatus.textContent = 'uploading... lifting these pixels...';
  modalStatus.className = 'modal-status';

  const formData = new FormData(modalForm);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) {
      modalStatus.textContent = data.error || 'something broke';
      modalStatus.className = 'modal-status error';
    } else {
      modalStatus.textContent = `success! ${data.added.length} photo(s) added.`;
      modalStatus.className = 'modal-status success';
      const section = modalSection.value;
      await loadGallery(section);
      setTimeout(() => { modal.hidden = true; }, 1200);
    }
  } catch (err) {
    modalStatus.textContent = 'network error. squad is offline.';
    modalStatus.className = 'modal-status error';
  } finally {
    modalSubmit.disabled = false;
  }
});

SECTIONS.forEach(loadGallery);
