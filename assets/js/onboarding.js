(() => {
  'use strict';
  const host = document.querySelector('#onboarding-clips');
  if (!host) return;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const entries = [];
  const visible = new Set();
  const active = new Set();
  function updatePreviews() {
    for (const {video, clip} of entries) {
      if (!reducedMotion && visible.has(video) && active.has(video) && !document.body.classList.contains('gallery-modal-open')) {
        if (!video.childElementCount) {
          for (const [src, type] of [[clip.mp4, 'video/mp4'], [clip.webm, 'video/webm']]) {
            if (!src) continue;
            const source = document.createElement('source');
            source.src = src; source.type = type;
            video.append(source);
          }
        }
        video.play().catch(() => {});
      } else video.pause();
    }
  }
  const observer = !reducedMotion && 'IntersectionObserver' in window
    ? new IntersectionObserver(changes => {
      for (const {target, isIntersecting} of changes) {
        if (isIntersecting) visible.add(target);
        else visible.delete(target);
      }
      updatePreviews();
    }, {threshold: 0.25}) : null;
  // Keep 32 visible posts: 30 onboarding clips plus the two presentation clips.
  // Retain the omitted source assets so this selection can be changed later.
  const omitted = new Set(['hot3d:28', 'tless:28']);
  const clips = [
    ...(window.DOPER_ONBOARDING || []).filter(clip => !omitted.has(`${clip.dataset}:${clip.object_id}`)),
    ...(window.DOPER_PRESENTATION || []),
  ];
  for (const clip of clips) {
    const figure = document.createElement('figure');
    figure.className = 'onboarding-clip';
    const link = document.createElement('a');
    link.className = 'onboarding-open';
    link.href = clip.mp4;
    link.setAttribute('aria-label', `View video: ${clip.caption}`);
    const video = document.createElement('video');
    video.muted = true;
    video.loop = true;
    video.playsInline = true;
    video.preload = 'none';
    const poster = document.createElement('img');
    poster.src = clip.thumbnail || clip.poster;
    poster.alt = '';
    poster.loading = 'lazy';
    poster.decoding = 'async';
    poster.width = poster.height = 320;
    video.width = clip.width;
    video.height = clip.height;
    video.tabIndex = -1;
    video.setAttribute('aria-hidden', 'true');
    video.addEventListener('playing', () => video.classList.add('is-playing'));
    video.addEventListener('pause', () => video.classList.remove('is-playing'));
    video.addEventListener('error', () => video.classList.remove('is-playing'));
    const mark = document.createElement('span');
    mark.className = 'onboarding-video-mark';
    mark.setAttribute('aria-hidden', 'true');
    mark.innerHTML = '<svg viewBox="0 0 24 24" focusable="false"><rect x="2" y="5" width="14" height="14" rx="3"/><path d="m17 9 5-3v12l-5-3z"/></svg>';
    link.append(poster, video, mark);
    figure.append(link);
    host.append(figure);
    entries.push({link, clip, video});
    // A dense grid should not decode 32 movies at once. Preview just the tile
    // being hovered or focused; clicking opens the full playback controls.
    link.addEventListener('pointerenter', () => { active.add(video); updatePreviews(); });
    link.addEventListener('pointerleave', () => { active.delete(video); updatePreviews(); });
    link.addEventListener('focus', () => { active.add(video); updatePreviews(); });
    link.addEventListener('blur', () => { active.delete(video); updatePreviews(); });
    if (observer) observer.observe(video);
  }
  if (window.DOPER_GALLERY) window.DOPER_GALLERY.registerVideos(entries);
  document.addEventListener('doper-gallery-open', updatePreviews);
  document.addEventListener('doper-gallery-close', updatePreviews);
})();
