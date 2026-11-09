// Page chrome: hash router, nav/drawer, server-backed login, toast, scroll reveal.
// Loaded before TF.js so the page is usable while the model runtime downloads.
// Shares helpers with app.js through window.Fakit.
(() => {
  const $ = (id) => document.getElementById(id);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ── Toast ────────────────────────────────────────────────────────────
  let toastTimer = null;
  function toast(message, ms = 3200) {
    const el = $("toast");
    el.textContent = message;
    el.hidden = false;
    requestAnimationFrame(() => el.classList.add("on"));
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      el.classList.remove("on");
      setTimeout(() => { el.hidden = true; }, 300);
    }, ms);
  }

  // ── Router (hash based: #/, #/history, #/model …) ────────────────────
  const TITLES = {
    home: "Fakit — AI 얼굴 탐지",
    image: "사진 검사 — Fakit",
    video: "영상 검사 — Fakit",
    history: "기록 — Fakit",
    model: "모델 — Fakit",
    results: "결과 읽는 법 — Fakit",
    privacy: "사진과 기록 관리 — Fakit",
    guide: "가이드 — Fakit",
    about: "소개 — Fakit",
  };
  let current = null;

  function routeName() {
    const fragment = location.hash.replace(/^#\/?/, "").split("?")[0];
    if (fragment && Object.hasOwn(TITLES, fragment)) return fragment;
    const pathname = location.pathname.replace(/\/$/, "");
    return pathname === "/image" ? "image" : pathname === "/video" ? "video" : "home";
  }
  const legacy = location.hash.replace(/^#\/?/, "").split("?")[0];
  if (["image", "video", "text"].includes(legacy) || (location.pathname !== "/image" && /^#(?:image|import_error)=/.test(location.hash))) {
    location.replace((legacy === "video" ? "/video" : "/image") + location.search + (legacy.includes("=") ? location.hash : ""));
  }
  function render() {
    const name = routeName();
    if (name === current) return;
    current = name;
    document.querySelectorAll(".page").forEach((p) => p.classList.toggle("is-active", p.dataset.page === (["image", "video"].includes(name) ? "classifier" : name)));
    document.querySelectorAll("[data-route]").forEach((link) => {
      const on = link.dataset.route === name;
      link.classList.toggle("is-current", on);
      if (on) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    document.title = TITLES[name];
    document.body.dataset.route = name;
    closeDrawer();
    closeUserMenu();
    window.scrollTo({ top: 0, behavior: "instant" });
    document.dispatchEvent(new CustomEvent("fakit:route", { detail: { route: name } }));
  }
  function go(name) {
    location.assign(["image", "video"].includes(name) ? "/" + name : name === "home" ? "/" : "/#/" + name);
  }
  addEventListener("hashchange", render);

  // ── Mobile drawer ────────────────────────────────────────────────────
  const drawer = $("drawer");
  const menuToggle = $("menu-toggle");
  function closeDrawer() {
    drawer.hidden = true;
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "메뉴 열기");
  }
  menuToggle.addEventListener("click", () => {
    const open = drawer.hidden;
    drawer.hidden = !open;
    menuToggle.setAttribute("aria-expanded", String(open));
    menuToggle.setAttribute("aria-label", open ? "메뉴 닫기" : "메뉴 열기");
  });
  drawer.addEventListener("click", (e) => { if (e.target.closest("a")) closeDrawer(); });

  // ── Modals ───────────────────────────────────────────────────────────
  function openModal(dialog) {
    if (!dialog.open) dialog.showModal();
  }
  function closeModal(dialog) {
    if (dialog.open) dialog.close();
  }
  // Click on the backdrop (outside the dialog's box) closes it.
  document.querySelectorAll("dialog.modal").forEach((dialog) => {
    dialog.addEventListener("click", (e) => {
      if (e.target !== dialog) return;
      const r = dialog.getBoundingClientRect();
      const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
      if (!inside) dialog.close();
    });
  });

  // ── Login: only the server can establish an authenticated identity. ──
  let authenticatedUser = null;
  let csrfToken = null;
  let historyAccess;
  let expiryTimer;
  const authChannel = typeof BroadcastChannel === "function" ? new BroadcastChannel("fakit-auth") : null;
  let availableProviders = new Set();
  let authLoaded = false;
  const PROVIDERS = {
    kakao: { label: "카카오", bg: "#fee500", ink: "#191919" },
    google: { label: "Google", bg: "#ffffff", ink: "#1f1f1f", border: "#d9d4ee" },
  };
  const loginModal = $("login-modal");

  function getUser() {
    return authenticatedUser;
  }
  function setUser(user) {
    authenticatedUser = user;
    renderUser();
  }
  function renderUser() {
    const user = getUser();
    $("login-btn").hidden = !!user;
    $("landing-login").textContent = user ? "내 기록" : "로그인";
    $("user-chip").hidden = !user;
    $("dock-login-label").textContent = user ? "내 계정" : "로그인";
    if (user) {
      const p = PROVIDERS[user.provider] || {label:"기존",bg:"#5b2ee0",ink:"#fff"};
      const avatar = $("user-avatar");
      avatar.textContent = user.name.slice(0, 1);
      avatar.style.background = p.bg;
      avatar.style.color = p.ink;
      avatar.style.boxShadow = p.border ? `inset 0 0 0 1px ${p.border}` : "none";
      $("user-name").textContent = user.name;
      $("user-provider").textContent = `${p.label} 계정`;
    }
    document.dispatchEvent(new CustomEvent("fakit:user", { detail: { user } }));
  }
  function openLogin() {
    closeDrawer();
    closeUserMenu();
    if (getUser()) { go("history"); return; }
    renderProviders();
    openModal(loginModal);
  }
  function renderProviders() {
    loginModal.querySelectorAll(".provider").forEach((b) => {
      const enabled = availableProviders.has(b.dataset.provider);
      b.disabled = !enabled;
      b.querySelector("span").textContent = `${PROVIDERS[b.dataset.provider].label}로 시작하기${enabled ? "" : authLoaded ? " · 준비 중" : " · 확인 중"}`;
    });
  }
  async function refreshAuth() {
    try {
      const response = await fetch("/api/auth", { cache: "no-store" });
      if (!response.ok) throw new Error("Auth unavailable");
      const data = await response.json();
      availableProviders = new Set(data.providers.filter(p => p.enabled).map(p => p.id));
      csrfToken = data.csrf;
      clearTimeout(expiryTimer);
      historyAccess = data.user ? data.history : {scope:"guest"};
      if(data.expires) expiryTimer = setTimeout(refreshAuth, Math.max(0,data.expires-Date.now()+100));
      setUser(data.user);
      $("auth-message").textContent = availableProviders.size
        ? "사용할 계정을 선택하세요. 이미지 분석은 로그인 없이도 이용할 수 있어요."
        : "소셜 로그인은 준비 중이에요. 이미지 분석은 로그인 없이 이용할 수 있어요.";
    } catch {
      availableProviders.clear();
      historyAccess = undefined;
      setUser(null);
      $("auth-message").textContent = "로그인 서비스에 연결할 수 없습니다. 이미지 분석은 계속 이용할 수 있어요.";
    } finally { authLoaded = true; renderProviders(); }
  }
  $("login-btn").addEventListener("click", openLogin);
  document.querySelectorAll("[data-open-login]").forEach((b) => b.addEventListener("click", openLogin));
  $("login-close").addEventListener("click", () => closeModal(loginModal));
  $("login-guest").addEventListener("click", () => closeModal(loginModal));
  loginModal.querySelectorAll(".provider").forEach((btn) => {
    btn.addEventListener("click", () => {
      const provider = btn.dataset.provider;
      const p = PROVIDERS[provider];
      loginModal.querySelectorAll(".provider").forEach((b) => { b.disabled = true; });
      btn.querySelector("span").textContent = `${p.label} 연결 중…`;
      location.assign(`/auth/${provider}/login`);
    });
  });

  // User chip popover
  const userBtn = $("user-btn");
  const userMenu = $("user-menu");
  function closeUserMenu() {
    userMenu.hidden = true;
    userBtn.setAttribute("aria-expanded", "false");
  }
  userBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = userMenu.hidden;
    userMenu.hidden = !open;
    userBtn.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#user-chip")) closeUserMenu();
  });
  $("logout-btn").addEventListener("click", async () => {
    $("logout-btn").disabled = true;
    try {
      const response = await fetch("/auth/logout", { method: "POST", headers: { "X-CSRF-Token": csrfToken } });
      if (!response.ok) throw new Error("Logout failed");
      await refreshAuth();
      authChannel?.postMessage("refresh");
      closeUserMenu();
      toast("로그아웃했습니다.");
    } catch { toast("로그아웃하지 못했습니다. 새로고침 후 다시 시도해 주세요."); }
    finally { $("logout-btn").disabled = false; }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    closeDrawer();
    closeUserMenu();
  });

  // ── Scroll reveal + count-up ─────────────────────────────────────────
  function countUp(el) {
    const target = Number(el.dataset.count);
    const start = performance.now();
    const duration = 1300;
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

  window.Fakit = { toast, go, routeName, openModal, closeModal, getUser, getHistoryAccess: () => historyAccess };
  renderUser();
  render();
  refreshAuth();
  if(authChannel) authChannel.onmessage = refreshAuth;
  const authUrl = new URL(location.href);
  if (authUrl.searchParams.has("auth_error")) {
    const messages = { not_configured: "아직 준비 중인 로그인입니다.", invalid_state: "로그인 요청이 만료되었어요. 다시 시도해 주세요.", cancelled: "로그인을 취소했습니다.", failed: "로그인하지 못했습니다. 잠시 후 다시 시도해 주세요." };
    toast(messages[authUrl.searchParams.get("auth_error")] || messages.failed, 6000);
    authUrl.searchParams.delete("auth_error");
    history.replaceState(null, "", authUrl);
  }
  // A page cached by back/forward navigation must wait for a fresh session.
  addEventListener("pagehide", () => {
    clearTimeout(expiryTimer);
    historyAccess = undefined;
    csrfToken = null;
    setUser(null);
  });
  addEventListener("pageshow", e => { if (e.persisted) refreshAuth(); });
})();

// Chrome internal pages cannot be opened by ordinary website links.
document.getElementById('copy-extension-address')?.addEventListener('click', async () => {
  const status = document.getElementById('extension-copy-status');
  try {
    await navigator.clipboard.writeText('chrome://extensions');
    status.textContent = '복사했어요. Chrome의 새 탭 주소창에 붙여넣으세요.';
  } catch {
    status.textContent = '주소를 직접 선택해 복사해 주세요: chrome://extensions';
  }
});

// Bend only the section of each mist stream within reach of the pointer.
(() => {
  const landing = document.querySelector('.classifier-landing');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  const choices = Array.from(landing.querySelectorAll('.classifier-choice'), link => ({
    link, label: link.querySelector('strong'), glow: 0, target: 0
  }));
  let streams, bounds, pointer, frame = 0, previousTime = 0;

  function proximity(distance, radius) {
    const progress = Math.min(1, distance / radius);
    return .5 + .5 * Math.cos(Math.PI * progress);
  }

  function sampleStreams() {
    return Array.from(landing.querySelectorAll('[data-mist-stream]'), stream =>
      Array.from(stream.querySelectorAll('path'), path => {
        const length = path.getTotalLength();
        const points = Array.from({ length: 65 }, (_, index) => {
          const { x, y } = path.getPointAtLength(length * index / 64);
          return { x, y, offset: 0, velocity: 0 };
        });
        return { path, original: path.getAttribute('d'), points, bent: false };
      }));
  }

  function draw({ path, points }) {
    const x = point => (point.x + point.offset).toFixed(2);
    let curve = 'M' + x(points[0]) + ' ' + points[0].y.toFixed(2);
    for (let i = 1; i < points.length - 1; i++) {
      const point = points[i], next = points[i + 1];
      const midX = (point.x + point.offset + next.x + next.offset) / 2;
      curve += 'Q' + x(point) + ' ' + point.y.toFixed(2) + ' ' +
        midX.toFixed(2) + ' ' + ((point.y + next.y) / 2).toFixed(2);
    }
    const end = points.at(-1);
    path.setAttribute('d', curve + 'L' + x(end) + ' ' + end.y.toFixed(2));
  }

  function tick(now) {
    frame = 0;
    const dt = previousTime ? Math.min((now - previousTime) / 1000, .04) : 1 / 60;
    previousTime = now;
    const scaleX = bounds.width / 1000, scaleY = bounds.height / 650;
    const radius = Math.min(225, Math.max(115, bounds.width * .18));
    const spring = 10, decay = Math.exp(-spring * dt);
    let unsettled = false;

    for (const choice of choices) {
      choice.glow += (choice.target - choice.glow) * (1 - Math.exp(-9 * dt));
      choice.link.style.setProperty('--mist-glow', choice.glow.toFixed(3));
      unsettled ||= Math.abs(choice.glow - choice.target) > .003;
    }
    for (const stream of streams) {
      let direction = 0;
      if (pointer) {
        const spine = stream[0].points;
        const closest = spine.reduce((a, b) =>
          Math.abs(a.y * scaleY - pointer.y) < Math.abs(b.y * scaleY - pointer.y) ? a : b);
        direction = closest.x * scaleX < pointer.x ? -1 : 1;
      }
      for (const stroke of stream) {
        let bent = false;
        for (const point of stroke.points) {
          const distance = pointer ? Math.hypot(point.x * scaleX - pointer.x, point.y * scaleY - pointer.y) : radius;
          const target = direction * 42 * proximity(distance, radius) / scaleX;
          const error = point.offset - target;
          const momentum = point.velocity + spring * error;
          point.offset = target + (error + momentum * dt) * decay;
          point.velocity = (point.velocity - spring * momentum * dt) * decay;
          bent ||= Math.abs(point.offset * scaleX) > .08;
          unsettled ||= Math.abs((point.offset - target) * scaleX) > .08 || Math.abs(point.velocity * scaleX) > .5;
        }
        if (bent) draw(stroke);
        else if (stroke.bent) stroke.path.setAttribute('d', stroke.original);
        stroke.bent = bent;
      }
    }
    if (unsettled) schedule();
    else previousTime = 0;
  }

  function schedule() {
    if (!frame && !motion.matches) frame = requestAnimationFrame(tick);
  }

  landing.addEventListener('pointermove', event => {
    if (event.pointerType !== 'mouse' || motion.matches) return;
    bounds = landing.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    streams ??= sampleStreams();
    pointer = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    for (const choice of choices) {
      const label = choice.label.getBoundingClientRect();
      const dx = Math.max(label.left - event.clientX, 0, event.clientX - label.right);
      const dy = Math.max(label.top - event.clientY, 0, event.clientY - label.bottom);
      choice.target = proximity(Math.hypot(dx, dy), 150);
    }
    schedule();
  });
  const releasePointer = () => {
    if (!pointer) return;
    pointer = null;
    for (const choice of choices) choice.target = 0;
    schedule();
  };
  landing.addEventListener('pointerleave', releasePointer);
  addEventListener('resize', releasePointer);
  motion.addEventListener('change', () => {
    if (!motion.matches || !streams) return;
    cancelAnimationFrame(frame);
    frame = previousTime = 0;
    pointer = null;
    for (const choice of choices) {
      choice.glow = choice.target = 0;
      choice.link.style.removeProperty('--mist-glow');
    }
    for (const stroke of streams.flat()) stroke.path.setAttribute('d', stroke.original);
    streams = undefined;
  });
})();
