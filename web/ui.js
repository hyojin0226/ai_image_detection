// Page chrome: hash router, nav/drawer, login modal (demo), toast, scroll reveal.
// Loaded before TF.js so the page is usable while the model runtime downloads.
// Shares helpers with app.js through window.Fakit.
// ── Optional Spline scene for the landing hero ──────────────────────────
// Remix https://app.spline.design/community/file/355705e7-5f04-4051-93d5-5007f7034ada in your own
// Spline account, change the text, then Export → Viewer. Paste the two values the export gives you:
//   SPLINE_SCENE      = "https://prod.spline.design/<id>/scene.splinecode"
//   SPLINE_VIEWER_SRC = "https://unpkg.com/@splinetool/viewer@<version>/build/spline-viewer.js"
// Leave both empty to use the built-in lens/scroll scene below.
const SPLINE_SCENE = "";
const SPLINE_VIEWER_SRC = "";

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
    text: "AI 글 탐지 — Fakit",
    video: "AI 영상 탐지 — Fakit",
    guide: "가이드 — Fakit",
    about: "소개 — Fakit",
  };
  let current = null;

  function routeName() {
    const name = location.hash.replace(/^#\/?/, "").split("?")[0];
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

  // ── Login (demo: no real OAuth yet, state lives in this browser only) ──
  const USER_KEY = "fakit.user";
  const PROVIDERS = {
    kakao: { label: "카카오", bg: "#fee500", ink: "#191919" },
    naver: { label: "네이버", bg: "#03c75a", ink: "#ffffff" },
    google: { label: "Google", bg: "#ffffff", ink: "#1f1f1f", border: "#d9d4ee" },
  };
  const loginModal = $("login-modal");

  function getUser() {
    try {
      const u = JSON.parse(localStorage.getItem(USER_KEY));
      return u && PROVIDERS[u.provider] ? u : null;
    } catch { return null; }
  }
  function setUser(user) {
    try {
      if (user) localStorage.setItem(USER_KEY, JSON.stringify(user));
      else localStorage.removeItem(USER_KEY);
    } catch { /* storage blocked: state just won't persist */ }
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
      $("user-provider").textContent = `${p.label} 계정 · 체험 모드`;
    }
    document.dispatchEvent(new CustomEvent("fakit:user", { detail: { user } }));
  }
  function openLogin() {
    closeDrawer();
    closeUserMenu();
    if (getUser()) { go("history"); return; }
    loginModal.querySelectorAll(".provider").forEach((b) => {
      b.disabled = false;
      b.querySelector("span").textContent = `${PROVIDERS[b.dataset.provider].label}로 시작하기`;
    });
    openModal(loginModal);
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
      // Stand-in for the OAuth round trip until the providers are wired up.
      setTimeout(() => {
        setUser({ provider, name: `${p.label} 사용자`, since: Date.now() });
        closeModal(loginModal);
        toast(`${p.label} 계정으로 로그인했습니다 · 체험 모드`);
      }, reduceMotion ? 0 : 700);
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
  $("logout-btn").addEventListener("click", () => {
    setUser(null);
    closeUserMenu();
    toast("로그아웃했습니다.");
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

  // ── Landing scene: blurred display text, a lens that follows the pointer and shows it sharp,
  //    letters that spin in 3D as the page scrolls ────────────────────────────────────────────
  function initScene() {
    const scene = $("scene");
    if (!scene) return;
    scene.querySelector(".scene-cue").addEventListener("click", (e) => {
      e.preventDefault();
      $("composer").scrollIntoView({ behavior: reduceMotion ? "instant" : "smooth", block: "start" });
    });

    if (SPLINE_SCENE && SPLINE_VIEWER_SRC) {
      const script = document.createElement("script");
      script.type = "module";
      script.src = SPLINE_VIEWER_SRC;
      document.head.append(script);
      const viewer = document.createElement("spline-viewer");
      viewer.setAttribute("url", SPLINE_SCENE);
      viewer.setAttribute("loading-anim-type", "spinner-small-dark");
      scene.prepend(viewer);
      scene.classList.add("has-spline");
      return;
    }
    if (reduceMotion) { scene.classList.add("no-lens"); return; }

    // Split both copies into per-letter spans so they can be transformed identically.
    const words = [...scene.querySelectorAll("[data-word]")];
    words.forEach((word) => {
      word.innerHTML = [...word.childNodes].map((node) => {
        if (node.nodeType === Node.ELEMENT_NODE) return `<span class="ch">${node.outerHTML}</span>`;
        return [...node.textContent].map((c) => `<span class="ch">${c === " " ? "&nbsp;" : c}</span>`).join("");
      }).join("");
    });
    const letters = words.map((w) => [...w.querySelectorAll(".ch")]);
    const count = letters[0].length;

    let targetX = 0.5, targetY = 0.46, x = targetX, y = targetY;
    let rot = 0, velocity = 0, lastScroll = scrollY, lastMove = performance.now(), frame = null, visible = true;

    const setTarget = (clientX, clientY) => {
      const r = scene.getBoundingClientRect();
      targetX = Math.min(Math.max((clientX - r.left) / r.width, 0), 1);
      targetY = Math.min(Math.max((clientY - r.top) / r.height, 0), 1);
      lastMove = performance.now();
    };
    scene.addEventListener("pointermove", (e) => setTarget(e.clientX, e.clientY), { passive: true });
    scene.addEventListener("touchmove", (e) => setTarget(e.touches[0].clientX, e.touches[0].clientY), { passive: true });

    function tick(now) {
      // Idle: let the lens wander so the scene stays alive without a pointer.
      if (now - lastMove > 2500) {
        const t = now / 1000;
        targetX = 0.5 + Math.sin(t * 0.45) * 0.3;
        targetY = 0.46 + Math.sin(t * 0.9) * 0.14;
      }
      x += (targetX - x) * 0.12;
      y += (targetY - y) * 0.12;
      scene.style.setProperty("--lens-x", `${(x * 100).toFixed(2)}%`);
      scene.style.setProperty("--lens-y", `${(y * 100).toFixed(2)}%`);
      scene.style.setProperty("--px", `${((x - 0.5) * -18).toFixed(2)}px`);
      scene.style.setProperty("--py", `${((y - 0.5) * -12).toFixed(2)}px`);

      // Scroll: letters flip over as the page moves, and blur more while it is moving fast.
      const s = scrollY;
      velocity += (s - lastScroll - velocity) * 0.2;
      lastScroll = s;
      rot += (s * 0.55 - rot) * 0.1;
      scene.style.setProperty("--boost", `${Math.min(Math.abs(velocity) * 0.5, 22).toFixed(1)}px`);
      for (let i = 0; i < count; i++) {
        const a = rot * (1 + i * 0.1);   // later letters lag a little behind, like a ripple
        const transform = `perspective(700px) rotateX(${a.toFixed(2)}deg)`;
        letters.forEach((list) => { list[i].style.transform = transform; });
      }
      frame = visible && !document.hidden ? requestAnimationFrame(tick) : null;
    }
    function wake() {
      if (frame || !visible || document.hidden) return;
      frame = requestAnimationFrame(tick);
    }
    new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; wake(); }, { threshold: 0 }).observe(scene);
    document.addEventListener("visibilitychange", wake);
    wake();
  }
  initScene();

  window.Fakit = { toast, go, routeName, openModal, closeModal, getUser };
  renderUser();
  render();
})();
