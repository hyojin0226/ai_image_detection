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
    history: "기록 — Fakit",
    model: "모델 — Fakit",
    results: "결과 읽는 법 — Fakit",
    privacy: "사진과 기록 관리 — Fakit",
    guide: "가이드 — Fakit",
    about: "소개 — Fakit",
  };
  let current = null;

  function routeName() {
    const name = location.hash.replace(/^#\/?/, "").split("?")[0];
    if (name === "text" || name === "video") return "home";
    return TITLES[name] ? name : "home";
  }
  function render() {
    const name = routeName();
    if (name === current) return;
    current = name;
    document.querySelectorAll(".page").forEach((p) => p.classList.toggle("is-active", p.dataset.page === name));
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
    location.hash = name === "home" ? "#/" : `#/${name}`;
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
  let availableProviders = new Set();
  let authLoaded = false;
  const PROVIDERS = {
    kakao: { label: "카카오", bg: "#fee500", ink: "#191919" },
    naver: { label: "네이버", bg: "#03c75a", ink: "#ffffff" },
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
    $("user-chip").hidden = !user;
    $("dock-login-label").textContent = user ? "내 계정" : "로그인";
    if (user) {
      const p = PROVIDERS[user.provider];
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
      setUser(data.user);
      $("auth-message").textContent = availableProviders.size
        ? "사용할 계정을 선택하세요. 이미지 분석은 로그인 없이도 이용할 수 있어요."
        : "소셜 로그인은 준비 중이에요. 이미지 분석은 로그인 없이 이용할 수 있어요.";
    } catch {
      availableProviders.clear();
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

  window.Fakit = { toast, go, routeName, openModal, closeModal, getUser };
  renderUser();
  render();
  refreshAuth();
  const authUrl = new URL(location.href);
  if (authUrl.searchParams.has("auth_error")) {
    const messages = { not_configured: "아직 준비 중인 로그인입니다.", invalid_state: "로그인 요청이 만료되었어요. 다시 시도해 주세요.", cancelled: "로그인을 취소했습니다.", failed: "로그인하지 못했습니다. 잠시 후 다시 시도해 주세요." };
    toast(messages[authUrl.searchParams.get("auth_error")] || messages.failed, 6000);
    authUrl.searchParams.delete("auth_error");
    history.replaceState(null, "", authUrl);
  }
  // Browser back/forward cache must not resurrect an old signed-in display.
  addEventListener("pageshow", e => { if (e.persisted) refreshAuth(); });
})();
