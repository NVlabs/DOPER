(function () {
  "use strict";

  const links = [...document.querySelectorAll("#static-gallery a.gallery-open")];
  // Without dialog support, the ordinary links still open the original PNGs.
  if (typeof HTMLDialogElement === "undefined") return;
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const imageItems = links.map(link => ({
    link, type: 'image', src: link.href, title: link.dataset.galleryTitle,
    alt: link.querySelector('img').alt,
    description: link.dataset.description,
    facts: [['Dataset', link.dataset.datasetLabel],
      ...(link.dataset.scene ? [['Scene', link.dataset.scene], ['Image', link.dataset.image]] : [['Sample', link.dataset.sample]])],
  }));
  let items = imageItems;

  const dialog = document.createElement("dialog");
  dialog.id = "gallery-lightbox";
  dialog.className = "image-lightbox";
  dialog.setAttribute("aria-labelledby", "gallery-view-title");
  dialog.setAttribute("aria-describedby", "gallery-view-status");
  dialog.innerHTML = `
    <div class="image-lightbox__toolbar">
      <div class="image-lightbox__heading">
        <h2 id="gallery-view-title"></h2>
        <span class="image-lightbox__counter" aria-live="polite"></span>
      </div>
      <div class="image-lightbox__controls">
        <button type="button" data-view="previous" aria-label="Previous image">←</button>
        <button type="button" data-view="next" aria-label="Next image">→</button>
        <button type="button" data-view="zoom" aria-pressed="false">Zoom 2×</button>
        <a class="image-lightbox__original" target="_blank" rel="noopener">Open image ↗</a>
        <button type="button" data-view="close" aria-label="Close image viewer" autofocus>Close ×</button>
      </div>
    </div>
    <div class="image-lightbox__body">
    <div class="image-lightbox__stage">
      <video class="image-lightbox__video" controls playsinline loop preload="auto" hidden></video>
      <div class="image-lightbox__canvas">
        <button type="button" class="image-lightbox__image-button" tabindex="-1" aria-label="Zoom image to 2×" hidden>
          <img class="image-lightbox__image" alt="">
        </button>
      </div>
    </div>
    <aside class="image-lightbox__details" aria-label="Post details">
      <div class="image-lightbox__brand"><span class="image-lightbox__avatar" aria-hidden="true">D</span><div>DOPER<small>Research gallery</small></div></div>
      <p class="image-lightbox__description"></p>
      <dl class="image-lightbox__facts"></dl>
    </aside>
    </div>
    <p class="image-lightbox__status" id="gallery-view-status" role="status"></p>`;
  document.body.append(dialog);

  const title = dialog.querySelector("h2");
  const video = dialog.querySelector('video');
  const canvas = dialog.querySelector('.image-lightbox__canvas');
  const description = dialog.querySelector('.image-lightbox__description');
  const facts = dialog.querySelector('.image-lightbox__facts');
  function stopVideo() {
    video.pause();
    video.replaceChildren();
    video.removeAttribute('src');
    video.load();
    video.hidden = true;
  }
  const counter = dialog.querySelector(".image-lightbox__counter");
  const stage = dialog.querySelector(".image-lightbox__stage");
  const imageButton = dialog.querySelector(".image-lightbox__image-button");
  const image = dialog.querySelector(".image-lightbox__image");
  const original = dialog.querySelector(".image-lightbox__original");
  const status = dialog.querySelector(".image-lightbox__status");
  const zoomButton = dialog.querySelector('[data-view="zoom"]');
  const closeButton = dialog.querySelector('[data-view="close"]');
  let index = 0;
  let zoom = 1;
  let natural = null;
  let opener = null;
  let request = 0;

  function fit(centerX = 0.5, centerY = 0.5) {
    if (!natural) return;
    // Existing scrollbars must not shrink the next fit calculation.
    const scale = Math.min(1, (stage.offsetWidth - 24) / natural.width, (stage.offsetHeight - 24) / natural.height);
    const width = Math.max(1, Math.round(natural.width * scale * zoom));
    const height = Math.max(1, Math.round(natural.height * scale * zoom));
    image.style.width = width + "px";
    image.style.height = height + "px";
    dialog.classList.toggle("is-zoomed", zoom > 1);
    zoomButton.textContent = zoom > 1 ? "Fit image" : "Zoom 2×";
    zoomButton.setAttribute("aria-pressed", String(zoom > 1));
    imageButton.setAttribute("aria-label", zoom > 1 ? "Fit image to viewer" : "Zoom image to 2×");
    stage.scrollLeft = Math.max(0, 12 + width * centerX - stage.clientWidth / 2);
    stage.scrollTop = Math.max(0, 12 + height * centerY - stage.clientHeight / 2);
  }

  function show(next) {
    index = (next + items.length) % items.length;
    const item = items[index];
    const ticket = ++request;
    stopVideo();
    canvas.hidden = item.type === 'video';
    zoomButton.hidden = item.type === 'video';
    description.textContent = item.description;
    facts.replaceChildren();
    for (const [label, value] of item.facts) {
      const term = document.createElement('dt'); term.textContent = label;
      const detail = document.createElement('dd'); detail.textContent = value;
      facts.append(term, detail);
    }
    dialog.querySelector('[data-view="previous"]').setAttribute('aria-label', 'Previous post');
    dialog.querySelector('[data-view="next"]').setAttribute('aria-label', 'Next post');
    closeButton.setAttribute('aria-label', 'Close post viewer');
    // Hiding/disabling the focused zoom control would send focus outside the
    // dialog, breaking subsequent arrow-key navigation while an image loads.
    if (document.activeElement === zoomButton || imageButton.contains(document.activeElement)) {
      closeButton.focus({ preventScroll: true });
    }
    zoom = 1;
    natural = null;
    imageButton.hidden = true;
    zoomButton.disabled = true;
    zoomButton.textContent = "Zoom 2×";
    zoomButton.setAttribute("aria-pressed", "false");
    dialog.classList.remove("is-zoomed");
    title.textContent = item.title;
    counter.textContent = `${index + 1} / ${items.length}`;
    original.href = item.src;
    original.textContent = item.type === 'video' ? 'Open video ↗' : 'Open image ↗';
    if (item.type === 'video') {
      video.hidden = false;
      video.muted = true;
      video.poster = item.poster;
      video.setAttribute('aria-label', item.title);
      for (const [src, type] of [[item.src, 'video/mp4'], [item.webm, 'video/webm']]) {
        if (!src) continue;
        const source = document.createElement('source'); source.src = src; source.type = type;
        video.append(source);
      }
      status.textContent = 'Loading video…';
      dialog.setAttribute('aria-busy', 'true');
      video.onloadeddata = () => {
        if (ticket !== request || !dialog.open) return;
        dialog.setAttribute('aria-busy', 'false');
        status.textContent = 'Use the playback controls. Esc closes.';
      };
      video.onerror = () => {
        if (ticket !== request || !dialog.open) return;
        dialog.setAttribute('aria-busy', 'false');
        status.textContent = 'The video could not be loaded. Try Open video or another post.';
      };
      video.load();
      if (!reducedMotion) video.play().catch(() => {});
      return;
    }
    status.textContent = "Loading full-size image…";
    dialog.setAttribute("aria-busy", "true");
    const pending = new Image();
    pending.onload = () => {
      // Fast next/previous presses or closing must not display an older request.
      if (ticket !== request || !dialog.open) return;
      natural = { width: pending.naturalWidth, height: pending.naturalHeight };
      image.src = pending.src;
      image.alt = item.alt;
      imageButton.hidden = false;
      zoomButton.disabled = false;
      dialog.setAttribute("aria-busy", "false");
      status.textContent = `${natural.width} × ${natural.height} image. Click for 2× zoom; scroll to pan. Arrow keys browse; Esc closes.`;
      fit();
    };
    pending.onerror = () => {
      if (ticket !== request || !dialog.open) return;
      dialog.setAttribute("aria-busy", "false");
      status.textContent = "The full-size image could not be loaded. Try Open image or another image.";
    };
    pending.src = item.src;
  }

  function toggleZoom(event) {
    if (!natural) return;
    let x = 0.5;
    let y = 0.5;
    if (event && event.detail && event.currentTarget === imageButton) {
      const rect = image.getBoundingClientRect();
      x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
      y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
    }
    zoom = zoom === 1 ? 2 : 1;
    fit(x, y);
  }

  function bind(group) {
    group.forEach((item, i) => {
    const link = item.link;
    link.setAttribute("aria-haspopup", "dialog");
    link.setAttribute("aria-controls", dialog.id);
    link.addEventListener("click", (event) => {
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      opener = link;
      items = group;
      dialog.showModal();
      document.body.classList.add("gallery-modal-open");
      document.dispatchEvent(new Event('doper-gallery-open'));
      show(i);
      closeButton.focus({ preventScroll: true });
    });
    });
  }
  bind(imageItems);
  window.DOPER_GALLERY = {
    registerVideos(entries) {
      bind(entries.map(({link, clip}) => ({
        link, type: 'video', src: clip.mp4, webm: clip.webm, poster: clip.poster,
        title: clip.caption,
        description: clip.description || (clip.dataset === 'tless'
          ? 'DOPER poses on T-LESS training views of the object, estimated frame by frame without smoothing. The green outline is the CAD model at the predicted pose.'
          : 'DOPER poses on an onboarding video of the object, estimated frame by frame without smoothing. The green outline is the CAD model at the predicted pose.'),
        facts: clip.facts || [['Dataset', clip.label], ['Object', String(clip.object_id).padStart(2, '0')]],
      })));
    },
  };
  dialog.querySelector('[data-view="previous"]').addEventListener("click", () => show(index - 1));
  dialog.querySelector('[data-view="next"]').addEventListener("click", () => show(index + 1));
  closeButton.addEventListener("click", () => dialog.close());
  zoomButton.addEventListener("click", toggleZoom);
  imageButton.addEventListener("click", toggleZoom);
  dialog.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (event.key === "Tab") {
      const controls = [...dialog.querySelectorAll('.image-lightbox__controls button:not(:disabled):not([hidden]), .image-lightbox__controls a[href], video:not([hidden])')];
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault(); last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault(); first.focus();
      }
      return;
    }
    if (event.target === video) return; // Keep native video seeking/volume shortcuts.
    const next = { ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: items.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    show(next);
  });
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
  dialog.addEventListener("close", () => {
    request += 1;
    stopVideo();
    natural = null;
    imageButton.hidden = true;
    image.removeAttribute("src");
    document.body.classList.remove("gallery-modal-open");
    document.dispatchEvent(new Event('doper-gallery-close'));
    if (opener && opener.isConnected) opener.focus({ preventScroll: true });
  });
  window.addEventListener("resize", () => { if (dialog.open) fit(); });
})();
