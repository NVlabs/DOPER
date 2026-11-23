(function () {
  "use strict";

  const DATA = window.DOPER_DATA;
  const MEDIA = window.DOPER_MEDIA || { clips: [], montages: {} };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (sel, root = document) => root.querySelector(sel);
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null) continue;
      if (key === "class") node.className = value;
      else if (key === "html") node.innerHTML = value;
      else node.setAttribute(key, value);
    }
    for (const child of children) node.append(child);
    return node;
  };

  // Run `start` while `node` is on screen and `stop` when it leaves.
  // Returns a function that stops watching.
  function whileVisible(node, start, stop) {
    if (!("IntersectionObserver" in window)) { start(); return () => {}; }
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) (entry.isIntersecting ? start : stop)();
    }, { threshold: 0.2 });
    observer.observe(node);
    return () => observer.disconnect();
  }

  /* ---------- Navigation ---------- */
  const nav = $(".topnav");
  const navLinks = [...document.querySelectorAll(".topnav .links a")];
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
  if ("IntersectionObserver" in window) {
    const spy = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + entry.target.id));
      }
    }, { rootMargin: "-45% 0px -50% 0px" });
    document.querySelectorAll("main .sec").forEach((s) => spy.observe(s));
  }

  /* ---------- Looping figures ---------- */
  // Each step reveals one region of the paper figure, given as [top, right, bottom, left] insets in %.
  const LOOPS = {
    fig1: {
      caption: "<b>Overview.</b> An approximate CAD model supplies domain-randomized training data. Real observations drive iterative refinement. The updated estimator predicts bounding box, mask, and keypoints for PnP pose recovery.",
      steps: [
        { r: [0, 69, 60, 0], t: "<b>Approximate CAD.</b> The only object model DOPER needs; texture and details may be incomplete." },
        { r: [40, 69, 0, 0], t: "<b>Randomized renders.</b> The CAD model supplies domain-randomized training data with exact keypoint labels." },
        { r: [0, 38, 52, 34], t: "<b>Real observations.</b> Unlabeled real images from the deployment environment." },
        { r: [45, 34, 34, 27], t: "<b>DOPER.</b> One network for detection, segmentation and keypoints." },
        { r: [0, 0, 0, 66], t: "<b>Final output.</b> Bounding box, mask and keypoints, and the 6D pose from PnP." },
        { r: [62, 34, 0, 30], t: "<b>Adaptation loop.</b> Pseudo-labels from real observations update the estimator, round after round.", pulse: true },
      ],
    },
    methodfig: {
      stages: true,
      caption: "<b>Method.</b> Monocular observations provide box, mask, and coarse-orientation targets (top row). Multi-view observations additionally provide keypoint targets and refine object appearance for new training renders (bottom row). Mixed real and synthetic supervision updates the estimator, which labels subsequent observations.",
      steps: [
        { r: [0, 81.2, 31.6, 0], t: "<b>1 · Synthetic initialization.</b> Randomized renders of the approximate CAD model, with exact keypoint labels, train the initial DOPER." },
        { r: [0, 47.9, 31.6, 18.8], t: "<b>2 · Real observations.</b> Monocular images pass an orientation check; calibrated multi-view images get a joint pose fit." },
        { r: [0, 23, 31.6, 52.1], t: "<b>3 · Accepted supervision.</b> Box, mask and coarse orientation from monocular images; keypoint targets from multi-view fits." },
        { r: [0, 0, 31.6, 77], t: "<b>4 · Model adaptation.</b> Mixed real and synthetic training updates the shared features for box, mask, keypoints and rotation." },
        { r: [68.4, 0, 0, 0], t: "<b>Repeat.</b> Exact synthetic supervision stays in every round, and multi-view appearance refinement adds new renders.", pulse: true },
      ],
    },
  };
  const STEP_MS = 1900;
  const HOLD_MS = 4200;

  function setupLoop(id, spec) {
    const fig = document.getElementById(id);
    if (!fig) return;
    const host = $(".fig", fig);
    const cap = $(".cap", fig);
    const stageItems = spec.stages ? [...document.querySelectorAll("#stages li")] : [];
    if (reduceMotion) {
      cap.innerHTML = spec.caption;
      return;
    }
    fig.classList.add("animated");
    const src = $(".base", host).getAttribute("src");
    const layers = spec.steps.map((s) => {
      const img = el("img", { class: "layer" + (s.pulse ? " pulse" : ""), src, alt: "", "aria-hidden": "true" });
      img.style.clipPath = `inset(${s.r[0]}% ${s.r[1]}% ${s.r[2]}% ${s.r[3]}%)`;
      host.append(img);
      return img;
    });
    let timer = null;
    let index = -1;
    let paused = false;
    let visible = false;
    const show = (k) => {
      layers.forEach((layer, i) => layer.classList.toggle("on", i <= k));
      stageItems.forEach((li, i) => li.classList.toggle("on", i === k));
      cap.innerHTML = k >= 0 ? spec.steps[k].t : "";
    };
    const tick = () => {
      index += 1;
      if (index >= layers.length) {
        index = -1;
        show(-1);
        timer = setTimeout(tick, 900);
        return;
      }
      show(index);
      timer = setTimeout(tick, index === layers.length - 1 ? HOLD_MS : STEP_MS);
    };
    show(layers.length - 1);
    cap.innerHTML = spec.caption;
    const start = () => { visible = true; if (!paused && !timer) { index = -1; tick(); } };
    const stop = () => { visible = false; clearTimeout(timer); timer = null; };
    const toggle = $(".loop-toggle", fig);
    toggle.addEventListener("click", () => {
      paused = !paused;
      toggle.textContent = paused ? "Play" : "Pause";
      toggle.setAttribute("aria-label", paused ? "Play animation" : "Pause animation");
      if (paused) {
        clearTimeout(timer); timer = null;
        layers.forEach((layer) => layer.classList.add("on"));
        stageItems.forEach((li) => li.classList.remove("on"));
        cap.innerHTML = spec.caption;
      } else if (visible) {
        index = -1; tick();
      }
    });
    whileVisible(fig, start, stop);
  }
  for (const [id, spec] of Object.entries(LOOPS)) setupLoop(id, spec);

  /* ---------- Ranked tables ---------- */
  // Rank the five best distinct values in a column; ties share a rank, as in the paper.
  function rankMap(values, higherIsBetter) {
    const distinct = [...new Set(values.filter((v) => v !== null && v !== undefined).map(Number))];
    distinct.sort((a, b) => (higherIsBetter ? b - a : a - b));
    const ranks = new Map();
    distinct.slice(0, 5).forEach((v, i) => ranks.set(v, i + 1));
    return ranks;
  }
  const ORDINAL = ["", "1st", "2nd", "3rd", "4th", "5th"];
  function rankCell(text, rank, extra) {
    if (!rank) return el("td", { class: extra.trim() }, text);
    return el("td", { class: "r" + rank + extra, title: ORDINAL[rank] }, text, el("span", { class: "sr" }, ` (${ORDINAL[rank]})`));
  }
  const inputBadge = (input) => {
    const cls = input === "MV RGB" ? "badge mv" : input === "RGB-D" ? "badge d" : "badge";
    return el("span", { class: cls }, input);
  };
  const fmt1 = (v) => (v === null || v === undefined ? "–" : v.toFixed(1));
  const fmt2 = (v) => (v === null || v === undefined ? "–" : v.toFixed(2));

  function renderBop(key) {
    const rows = DATA.bop;
    const ranks = [0, 1, 2].map((c) => rankMap(rows.filter((r) => r[key] && !(key === "mean" && r.meanReference)).map((r) => r[key][c]), true));
    const table = el("table");
    table.append(el("thead", {}, el("tr", { html: "<th scope='col'>Method</th><th scope='col'>Input</th><th scope='col'>AP</th><th scope='col'>MSSD</th><th scope='col'>MSPD</th><th scope='col'>Time (s)</th>" })));
    const body = el("tbody");
    let pendingGroup = null;
    for (const row of rows) {
      if (row.group) { pendingGroup = row; continue; }
      const cells = row[key];
      if (!cells) continue;
      if (pendingGroup) {
        body.append(el("tr", { class: "group" + (pendingGroup.ours ? " ours" : "") }, el("td", { colspan: "6" }, pendingGroup.group)));
        pendingGroup = null;
      }
      const reference = key === "mean" && row.meanReference;
      const name = el("td", {}, row.name);
      if (reference) {
        const dataset = DATA.datasets.find((d) => d.key === reference);
        name.append(el("small", { class: "dataset-note" }, `${dataset.label} only; not a four-dataset mean`));
      }
      const tr = el("tr", { class: row.ours ? "ours" : "" }, name, el("td", {}, inputBadge(row.input)));
      cells.slice(0, 3).forEach((v, c) => {
        const rank = v === null || reference ? 0 : ranks[c].get(v) || 0;
        tr.append(rankCell(fmt1(v), rank, v === null ? " na" : ""));
      });
      tr.append(el("td", { class: "time" + (cells[3] === null ? " na" : "") }, fmt2(cells[3])));
      body.append(tr);
    }
    table.append(body);
    $("#bop-table").replaceChildren(table);
  }

  function legend(node, label) {
    node.replaceChildren(label + " ");
    ["1st", "2nd", "3rd", "4th", "5th"].forEach((t, i) => {
      const sw = el("span", { class: "sw" }, t);
      sw.style.background = `var(--rank${i + 1})`;
      node.append(sw);
    });
  }

  function renderSpeed() {
    const ranked = DATA.speedplus.filter((r) => r.ranked);
    const ranks = ["lightbox", "sunlamp"].map((c) => rankMap(ranked.map((r) => r[c]), false));
    const table = el("table");
    table.append(el("thead", {}, el("tr", { html: "<th scope='col'>Method</th><th scope='col'>Lightbox ↓</th><th scope='col'>Sunlamp ↓</th>" })));
    const body = el("tbody");
    for (const row of DATA.speedplus) {
      if (row.group) { body.append(el("tr", { class: "group" + (row.ours ? " ours" : "") }, el("td", { colspan: "3" }, row.group))); continue; }
      const tr = el("tr", { class: row.ours ? "ours" : "" }, el("td", {}, row.name));
      ["lightbox", "sunlamp"].forEach((c, i) => {
        const rank = row.ranked ? ranks[i].get(Number(row[c])) || 0 : 0;
        tr.append(rankCell(row[c], rank, row.bold ? " bold" : ""));
      });
      body.append(tr);
    }
    table.append(body);
    $("#speed-table").replaceChildren(table);
  }

  function renderShirt() {
    const ranked = DATA.shirt.filter((r) => r.ranked);
    const cols = [["roe1", 0], ["roe1", 1], ["roe2", 0], ["roe2", 1]];
    const ranks = cols.map(([k, i]) => rankMap(ranked.map((r) => r[k][i]), false));
    const table = el("table");
    table.append(el("thead", {},
      el("tr", { html: "<th scope='col'></th><th colspan='2'>ROE1 (hold)</th><th colspan='2'>ROE2 (approach)</th>" }),
      el("tr", { html: "<th scope='col'>Method</th><th scope='col'>R (°)</th><th scope='col'>t (cm)</th><th scope='col'>R (°)</th><th scope='col'>t (cm)</th>" })));
    const body = el("tbody");
    for (const row of DATA.shirt) {
      const tr = el("tr", { class: row.ours ? "ours" : "" }, el("td", {}, row.name));
      cols.forEach(([k, i], c) => {
        const v = row[k][i];
        const rank = row.ranked ? ranks[c].get(Number(v)) || 0 : 0;
        tr.append(rankCell(v, rank, ""));
      });
      body.append(tr);
    }
    table.append(body);
    $("#shirt-table").replaceChildren(table);
    const orbit = el("table");
    orbit.append(el("thead", {},
      el("tr", { html: "<th scope='col'></th><th colspan='2'>ROE1</th><th colspan='2'>ROE2</th>" }),
      el("tr", { html: "<th scope='col'>Method</th><th scope='col'>Axial</th><th scope='col'>Lateral</th><th scope='col'>Axial</th><th scope='col'>Lateral</th>" })));
    const obody = el("tbody");
    for (const row of DATA.shirtOrbit) {
      const tr = el("tr", { class: row.ours ? "ours" : "" }, el("td", {}, row.name));
      [...row.roe1, ...row.roe2].forEach((v) => tr.append(el("td", {}, v)));
      obody.append(tr);
    }
    orbit.append(obody);
    $("#shirt-orbit").replaceChildren(orbit);
  }

  /* ---------- Montages ---------- */
  const FPS = 3;
  function montage(node, frames, caption, opts = {}) {
    node.hidden = false;
    const frame = el("div", { class: "frame" + (opts.fallback ? " fallback" : "") });
    if (opts.ratio) frame.style.aspectRatio = opts.ratio;
    const imgs = frames.map((src, i) => el("img", { src, alt: "", loading: "lazy", decoding: "async", class: i === 0 ? "on" : "" }));
    imgs.forEach((img) => frame.append(img));
    const bar = el("div", { class: "bar" });
    const ticks = frames.map((_, i) => { const t = el("i", { class: i === 0 ? "on" : "" }); bar.append(t); return t; });
    const badge = el("span", { class: "paused" }, reduceMotion ? "Click to advance" : "Paused");
    frame.append(badge);
    frame.setAttribute("tabindex", "0");
    frame.setAttribute("role", "button");
    frame.setAttribute("aria-label", "Show next image");
    node.replaceChildren(frame, bar);
    if (caption) node.append(el("figcaption", { html: caption }));
    let k = 0;
    let timer = null;
    let hovering = false;
    const go = (n) => {
      imgs[k].classList.remove("on"); ticks[k].classList.remove("on");
      k = (n + frames.length) % frames.length;
      imgs[k].classList.add("on"); ticks[k].classList.add("on");
    };
    const delay = 1000 / (opts.fps || FPS);
    const start = () => { if (!reduceMotion && !hovering && !timer && frames.length > 1) timer = setInterval(() => go(k + 1), delay); };
    const stop = () => { clearInterval(timer); timer = null; };
    frame.addEventListener("pointerenter", (e) => { if (e.pointerType !== "touch") { hovering = true; stop(); } });
    frame.addEventListener("pointerleave", (e) => { if (e.pointerType !== "touch") { hovering = false; start(); } });
    frame.addEventListener("click", () => go(k + 1));
    frame.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); stop(); hovering = true; go(k + 1); }
    });
    frame.addEventListener("blur", () => { hovering = false; start(); });
    const unwatch = whileVisible(node, start, stop);
    return () => { unwatch(); stop(); };
  }

  const datasetMontage = (key, label) => {
    const media = MEDIA.montages[key];
    if (media && media.frames.length) {
      return { frames: media.frames, caption: media.caption, opts: { fps: media.fps, ratio: media.ratio } };
    }
    return {
      frames: [0, 1, 2].map((i) => `assets/img/paper-cells/${key}-${i}.webp`),
      caption: `<b>${label}.</b> DOPER predictions from the paper's qualitative figure (cyan CAD overlays; dashed magenta: ground truth).`,
      opts: { fallback: true, fps: 0.7, ratio: key === "hopev2" ? "900 / 506" : "4 / 3" },
    };
  };

  /* ---------- BOP tabs ---------- */
  // Only complete four-dataset means are ranked. Explicit single-dataset
  // references may also appear, labeled separately and excluded from ranking.
  const MEAN = { key: "mean", label: "Mean", images: "Four-dataset means; FoundationPose shows HOPEv2 only" };
  for (const row of DATA.bop) {
    if (row.meanReference) { row.mean = row[row.meanReference]; continue; }
    const per = DATA.datasets.map((d) => row[d.key]);
    if (row.group || per.some((c) => !c)) continue;
    row.mean = [0, 1, 2, 3].map((c) => (per.some((v) => v[c] === null) ? null : per.reduce((s, v) => s + v[c], 0) / per.length));
  }
  const tabs = $("#bop-tabs");
  let stopMontage = () => {};
  let montageKey = null;
  function selectDataset(key) {
    const ds = DATA.datasets.find((d) => d.key === key);
    [...tabs.children].forEach((b) => {
      const on = b.dataset.key === key;
      b.setAttribute("aria-selected", String(on));
      b.setAttribute("tabindex", on ? "0" : "-1");
    });
    renderBop(key);
    // The Mean tab keeps the montage on screen, or shows HANDAL's when nothing is shown yet.
    if (!ds && montageKey) return;
    const show = ds || DATA.datasets[0];
    montageKey = show.key;
    stopMontage();
    const m = datasetMontage(show.key, show.label);
    stopMontage = montage($("#bop-montage"), m.frames, m.caption, m.opts);
  }
  [...DATA.datasets, MEAN].forEach((ds) => {
    const b = el("button", { type: "button", role: "tab", "data-key": ds.key, title: ds.images, "aria-controls": "bop-table" }, ds.label);
    b.addEventListener("click", () => selectDataset(ds.key));
    tabs.append(b);
  });
  tabs.addEventListener("keydown", (e) => {
    const buttons = [...tabs.children];
    const i = buttons.indexOf(document.activeElement);
    const step = { ArrowRight: 1, ArrowLeft: buttons.length - 1 }[e.key];
    let next;
    if (step) next = buttons[(i + step) % buttons.length];
    else if (e.key === "Home") next = buttons[0];
    else if (e.key === "End") next = buttons[buttons.length - 1];
    else return;
    e.preventDefault();
    next.focus(); next.click();
  });
  legend($("#bop-legend"), "Rank per accuracy column:");
  selectDataset("mean");
  renderSpeed();
  renderShirt();

  const overview = $("#overview-video");
  if (overview) {
    overview.muted = true;
    if (!reduceMotion) whileVisible(overview, () => overview.play().catch(() => {}), () => overview.pause());
  }

  if (MEDIA.montages.speedplus && MEDIA.montages.speedplus.frames.length) {
    const m = MEDIA.montages.speedplus;
    const fig = $("#speed-media");
    fig.className = "montage";
    montage(fig, m.frames, m.caption, { fps: m.fps, ratio: m.ratio });
  }
  if (MEDIA.shirt && MEDIA.shirt.length) {
    const host = $("#shirt-videos");
    for (const clip of MEDIA.shirt) {
      const video = el("video", { muted: "", loop: "", playsinline: "", controls: "", preload: "none", poster: clip.poster, "aria-label": clip.alt });
      video.muted = true;
      video.style.setProperty("--ar", clip.ratio);
      video.append(el("source", { src: clip.mp4, type: "video/mp4" }));
      if (clip.webm) video.append(el("source", { src: clip.webm, type: "video/webm" }));
      const figure = el("figure", { class: "clip" }, video);
      if (clip.caption) figure.append(el("figcaption", { html: clip.caption }));
      host.append(figure);
      if (!reduceMotion) whileVisible(video, () => video.play().catch(() => {}), () => video.pause());
    }
    host.hidden = false;
  }

  /* ---------- Live clips ---------- */
  if (MEDIA.clips.length) {
    const host = $("#clips");
    const modeSwitch = $("#clip-mode");
    let clipMode = MEDIA.clips.every((c) => c.smooth) ? "smooth" : "raw";
    const videos = [];
    const setSources = (video, clip) => {
      const src = clipMode === "smooth" ? clip.smooth : clip;
      video.replaceChildren();
      video.poster = src.poster;
      video.append(el("source", { src: src.mp4, type: "video/mp4" }));
      if (src.webm) video.append(el("source", { src: src.webm, type: "video/webm" }));
    };
    for (const clip of MEDIA.clips) {
      const video = el("video", { muted: "", loop: "", playsinline: "", preload: "none", "aria-label": clip.alt });
      video.muted = true;
      if (clip.ratio) video.style.setProperty("--ar", clip.ratio);
      setSources(video, clip);
      videos.push([video, clip]);
      host.append(el("figure", { class: "clip" }, video, el("figcaption", { html: clip.caption })));
      if (!reduceMotion) whileVisible(video, () => video.play().catch(() => video.setAttribute("controls", "")), () => video.pause());
      else video.setAttribute("controls", "");
    }
    const updateMode = () => {
      modeSwitch.querySelectorAll("button").forEach((b) => {
        const on = b.dataset.mode === clipMode;
        b.setAttribute("aria-checked", String(on));
        b.setAttribute("tabindex", on ? "0" : "-1");
      });
      for (const [video, clip] of videos) {
        const time = video.currentTime;
        const playing = !video.paused;
        setSources(video, clip);
        video.load();
        video.addEventListener("loadedmetadata", () => {
          video.currentTime = time;
          if (playing) video.play().catch(() => video.setAttribute("controls", ""));
        }, { once: true });
      }
    };
    if (MEDIA.clips.every((c) => c.smooth)) {
      modeSwitch.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        if (b.dataset.mode !== clipMode) { clipMode = b.dataset.mode; updateMode(); }
      }));
      modeSwitch.addEventListener("keydown", (e) => {
        if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
        e.preventDefault();
        clipMode = clipMode === "raw" ? "smooth" : "raw";
        updateMode();
        $(`#clip-mode [data-mode="${clipMode}"]`).focus();
      });
      modeSwitch.hidden = false;
    }
    $("#live").hidden = false;
  }

  /* ---------- Underspecified CAD comparison ---------- */
  const variantLabel = { notex: "CAD, no texture", removed: "CAD, 20% removed" };
  const gains = new Map(DATA.nurec.map(([obj, notex, removed]) => [obj, { notex, removed }]));
  let cadObject = 20;
  let cadVariant = "notex";
  const compare = $("#compare");
  const range = $("#compare-range");
  const select = $("#cad-object");
  const thumbs = $("#cad-thumbs");
  const objSrc = (obj, variant) => `assets/img/nurec/obj${String(obj).padStart(2, "0")}-${variant}.webp`;
  function setSplit(pct) {
    range.setAttribute("aria-valuetext", `${pct}% ${variantLabel[cadVariant]}, ${100 - pct}% NuRec`);
    $(".before-clip", compare).style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    $(".handle", compare).style.left = pct + "%";
  }
  function updateCad() {
    $(".before", compare).src = objSrc(cadObject, cadVariant);
    $(".after", compare).src = objSrc(cadObject, "nurec");
    $(".before", compare).alt = `HANDAL object ${cadObject}, ${variantLabel[cadVariant]}`;
    $(".after", compare).alt = `HANDAL object ${cadObject}, NuRec reconstruction`;
    $("#cad-left-label").textContent = variantLabel[cadVariant];
    const g = gains.get(cadObject)[cadVariant];
    $("#cad-gain").innerHTML = `Object ${cadObject}: NuRec composites gain <b>+${g.toFixed(1)}</b> AR over ${cadVariant === "notex" ? "untextured" : "20%-removed"} CAD composites`;
    select.value = String(cadObject);
    [...thumbs.children].forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.obj) === cadObject)));
    document.querySelectorAll("#cad-variant button").forEach((b) => {
      const on = b.dataset.variant === cadVariant;
      b.setAttribute("aria-checked", String(on));
      b.setAttribute("tabindex", on ? "0" : "-1");
    });
    setSplit(Number(range.value));
  }
  for (const [obj] of DATA.nurec) {
    select.append(el("option", { value: String(obj) }, `Object ${obj}`));
    const b = el("button", { type: "button", "data-obj": String(obj), "aria-label": `Object ${obj}` },
      el("img", { src: objSrc(obj, "nurec"), alt: "", loading: "lazy" }), el("span", {}, String(obj)));
    b.addEventListener("click", () => { cadObject = obj; updateCad(); });
    thumbs.append(b);
  }
  select.addEventListener("change", () => { cadObject = Number(select.value); updateCad(); });
  document.querySelectorAll("#cad-variant button").forEach((b) => b.addEventListener("click", () => { cadVariant = b.dataset.variant; updateCad(); }));
  $("#cad-variant").addEventListener("keydown", (e) => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)) return;
    e.preventDefault();
    cadVariant = cadVariant === "notex" ? "removed" : "notex";
    updateCad();
    $(`#cad-variant [data-variant="${cadVariant}"]`).focus();
  });
  range.addEventListener("input", () => setSplit(Number(range.value)));
  setSplit(50);
  updateCad();

  /* ---------- Copy buttons ---------- */
  document.querySelectorAll(".code").forEach((block) => {
    const button = el("button", { class: "copy", type: "button" }, "Copy");
    button.addEventListener("click", async () => {
      const text = $("code", block).innerText.replace(/\s+#.*$/gm, "");
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = "Copied";
        button.setAttribute("aria-live", "polite");
      } catch {
        button.textContent = "Select and copy";
      }
      button.classList.add("done");
      setTimeout(() => { button.textContent = "Copy"; button.classList.remove("done"); }, 1600);
    });
    block.append(button);
  });
})();
