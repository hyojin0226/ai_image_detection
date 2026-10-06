// Page chrome only (idle ring graphic, scroll reveal, pointer glow, nav flyout). Loaded before
// TF.js so the page animates without waiting for the model runtime.
(() => {
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── Ring shown in the empty viewfinder ───────────────────────────────
  // A circle of ticks with two waves travelling round it, plus an arc sweeping an inner track.
  // It speeds up while the pointer is over the drop area or a file is being dragged in.
  function startRing(canvas, card) {
    const ctx = canvas.getContext("2d");
    const host = canvas.parentElement;
    const zone = host.closest(".viewfinder");
    const TICKS = 120;
    const CALM = 0.25;
    let size = 0, energy = CALM, hovered = false, phase = 0, last = 0, frame = null;

    function draw() {
      const c = size / 2;
      ctx.clearRect(0, 0, size, size);
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(1.2, size / 190);
      for (let i = 0; i < TICKS; i++) {
        const angle = (i / TICKS) * Math.PI * 2;
        const wave = 0.5 + 0.5 * Math.sin(angle * 2 - phase);
        const ripple = 0.5 + 0.5 * Math.sin(angle * 5 + phase * 0.6 + 1.3);
        const level = Math.pow(wave * 0.75 + ripple * 0.25, 2.2);   // 0 = resting tick, 1 = crest
        const inner = c * 0.66;
        const outer = inner + c * (0.05 + 0.24 * level * (0.6 + 0.4 * energy));
        const cos = Math.cos(angle), sin = Math.sin(angle);
        ctx.strokeStyle = `hsla(211, 100%, ${100 - 30 * level}%, ${0.16 + 0.84 * level})`;
        ctx.beginPath();
        ctx.moveTo(c + cos * inner, c + sin * inner);
        ctx.lineTo(c + cos * outer, c + sin * outer);
        ctx.stroke();
      }
      const track = c * 0.55;
      ctx.lineWidth = Math.max(1, size / 240);
      ctx.strokeStyle = "rgba(255, 255, 255, .12)";
      ctx.beginPath();
      ctx.arc(c, c, track, 0, Math.PI * 2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255, 255, 255, .85)";
      ctx.beginPath();
      ctx.arc(c, c, track, -phase * 0.9, -phase * 0.9 + Math.PI * 0.45);
      ctx.stroke();
    }

    function resize() {
      const next = Math.min(host.clientWidth, host.clientHeight, 340);
      if (next <= 0 || next === size) return;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      size = next;
      canvas.width = canvas.height = Math.round(size * dpr);
      canvas.style.width = canvas.style.height = `${size}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    const shouldRun = () => !reduceMotion && !document.hidden && card.dataset.state === "idle";
    function tick(now) {
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;
      const target = hovered || card.classList.contains("drag") ? 1 : CALM;
      energy += (target - energy) * 0.06;
      phase += dt * (0.9 + energy * 1.8);
      if (size) draw();
      frame = shouldRun() ? requestAnimationFrame(tick) : null;
    }
    function wake() {
      resize();
      if (frame || !shouldRun()) return;
      last = performance.now();
      frame = requestAnimationFrame(tick);
    }

    zone.addEventListener("pointerenter", () => { hovered = true; });
    zone.addEventListener("pointerleave", () => { hovered = false; });
    new ResizeObserver(resize).observe(host);
    new MutationObserver(wake).observe(card, { attributes: true, attributeFilter: ["data-state"] });
    document.addEventListener("visibilitychange", wake);
    wake();
  }
  startRing(document.getElementById("ring"), document.getElementById("card"));

  // ── Scroll reveal + stat count-up on the second screen ───────────────
  function countUp(el) {
    const target = Number(el.dataset.count);
    const start = performance.now();
    const duration = 1400;
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      el.textContent = (target * (1 - Math.pow(1 - t, 4))).toFixed(1);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      if (!reduceMotion) entry.target.querySelectorAll("[data-count]").forEach(countUp);
      revealObserver.unobserve(entry.target);
    });
  }, { threshold: 0.15, rootMargin: "0px 0px -6% 0px" });
  document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

  // ── Backdrop glow that trails the pointer ────────────────────────────
  const glow = document.getElementById("glow-pointer");
  let targetX = 0, targetY = 0, x = 0, y = 0, frame = null;
  function follow() {
    x += (targetX - x) * 0.08;
    y += (targetY - y) * 0.08;
    glow.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    frame = Math.abs(targetX - x) + Math.abs(targetY - y) > 0.5 ? requestAnimationFrame(follow) : null;
  }
  addEventListener("pointermove", (e) => {
    if (e.pointerType !== "mouse") return;
    targetX = e.clientX;
    targetY = e.clientY;
    if (!glow.classList.contains("on")) {
      x = targetX;
      y = targetY;
      glow.classList.add("on");
    }
    if (!frame) frame = requestAnimationFrame(follow);
  }, { passive: true });

  // ── Nav flyout ───────────────────────────────────────────────────────
  const nav = document.getElementById("nav");
  const sheet = document.getElementById("sheet");
  const scrim = document.getElementById("scrim");
  const triggers = [...nav.querySelectorAll("[data-panel-trigger]")];
  const canHover = matchMedia("(hover: hover) and (pointer: fine)").matches;
  let closeTimer = null;

  const isOpen = (name) => nav.classList.contains("sheet-open") && sheet.dataset.open === name;

  function openSheet(name) {
    clearTimeout(closeTimer);
    sheet.dataset.open = name;
    nav.classList.add("sheet-open");
    scrim.classList.add("on");
    triggers.forEach((t) => t.setAttribute("aria-expanded", String(t.dataset.panelTrigger === name)));
  }
  function closeSheet() {
    clearTimeout(closeTimer);
    nav.classList.remove("sheet-open");
    scrim.classList.remove("on");
    triggers.forEach((t) => t.setAttribute("aria-expanded", "false"));
  }

  triggers.forEach((trigger) => {
    const name = trigger.dataset.panelTrigger;
    // With a mouse the panel is already open from hover, so a click only ever opens.
    trigger.addEventListener("click", () => {
      if (isOpen(name) && !(canHover && name !== "all")) closeSheet();
      else openSheet(name);
    });
    if (canHover && name !== "all") trigger.addEventListener("mouseenter", () => openSheet(name));
  });
  if (canHover) {
    nav.addEventListener("mouseleave", () => { closeTimer = setTimeout(closeSheet, 200); });
    nav.addEventListener("mouseenter", () => clearTimeout(closeTimer));
  }
  nav.addEventListener("focusout", (e) => {
    if (e.relatedTarget && !nav.contains(e.relatedTarget)) closeSheet();
  });
  scrim.addEventListener("click", closeSheet);
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !nav.classList.contains("sheet-open")) return;
    const active = triggers.find((t) => t.getAttribute("aria-expanded") === "true");
    closeSheet();
    if (active) active.focus();
  });
  sheet.querySelectorAll('a[href="#detect"]').forEach((link) => link.addEventListener("click", (e) => {
    e.preventDefault();
    closeSheet();
    const dropzone = document.getElementById("dropzone");
    dropzone.scrollIntoView({ behavior: "smooth", block: "center" });
    dropzone.focus({ preventScroll: true });
  }));
})();
