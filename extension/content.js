// 플로팅 버튼 → 페이지 img 수집 → background 로 판정 요청 → 이미지 위에 배지 표시.
// 사이트 클래스명에 의존하지 않고 img 태그와 화면상 크기만 사용한다.
(() => {
  if (window.__aifdLoaded) return;
  window.__aifdLoaded = true;

  const MIN_SIDE = 120; // 화면에 표시된 크기가 이보다 작은 이미지는 무시 (아이콘, 프로필 썸네일)
  const MIN_NATURAL = 64;
  const MAX_IMAGES = 24; // 한 번 스캔할 때 최대 이미지 수
  const NEARBY_MARGIN = 1.0; // 화면 위아래로 화면 높이만큼까지는 미리 스캔

  const resultCache = new Map(); // url → { ok, result | error }
  const entries = new Map(); // img → { url, status, data, nodes }

  // ---------- UI 루트 (Shadow DOM 으로 페이지 CSS 와 분리) ----------
  const host = document.createElement("div");
  host.id = "aifd-host";
  host.style.cssText = "all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483646;";
  const root = host.attachShadow({ mode: "open" });
  const style = document.createElement("link");
  style.rel = "stylesheet";
  style.href = chrome.runtime.getURL("panel.css");
  root.appendChild(style);

  const layer = document.createElement("div");
  layer.className = "aifd-layer";
  root.appendChild(layer);

  const dock = document.createElement("div");
  dock.className = "aifd-dock";
  const scanButton = document.createElement("button");
  scanButton.className = "aifd-scan";
  scanButton.title = "화면의 얼굴 이미지 스캔 (다시 누르면 재스캔)";
  scanButton.textContent = "AI 얼굴 스캔";
  const clearButton = document.createElement("button");
  clearButton.className = "aifd-clear";
  clearButton.title = "배지 지우기";
  clearButton.textContent = "✕";
  clearButton.hidden = true;
  const toast = document.createElement("div");
  toast.className = "aifd-toast";
  toast.hidden = true;
  dock.append(toast, clearButton, scanButton);
  root.appendChild(dock);

  (document.body || document.documentElement).appendChild(host);

  let toastTimer = null;
  function showToast(text, ms = 3500) {
    toast.textContent = text;
    toast.hidden = false;
    clearTimeout(toastTimer);
    if (ms) toastTimer = setTimeout(() => (toast.hidden = true), ms);
  }

  // ---------- 이미지 수집 ----------
  function imageUrl(img) {
    const url = img.currentSrc || img.src;
    if (!url) return null;
    if (/^(https?:|data:image\/|blob:)/i.test(url)) return url;
    return null;
  }

  // (x, y) 지점에서 img 가 다른 요소(모달, 팝업 등)에 가려지지 않았는지.
  // 이미지 위에 겹친 같은 크기 이하의 투명 레이어(인스타 클릭 영역 등)는 가림으로 보지 않는다.
  function pointVisible(img, imgRect, x, y) {
    for (const el of document.elementsFromPoint(x, y)) {
      if (el === host) continue;
      if (el === img || el.contains(img)) return true;
      const r = el.getBoundingClientRect();
      const tol = 8;
      const insideImage =
        r.left >= imgRect.left - tol && r.top >= imgRect.top - tol && r.right <= imgRect.right + tol && r.bottom <= imgRect.bottom + tol;
      if (!insideImage) return false;
    }
    return false;
  }

  // 화면 안에 보이는 이미지 영역 중 가려지지 않은 비율 (3x3 지점 샘플링)
  function visibleFraction(img, rect) {
    const left = Math.max(rect.left, 0);
    const top = Math.max(rect.top, 0);
    const right = Math.min(rect.right, window.innerWidth);
    const bottom = Math.min(rect.bottom, window.innerHeight);
    if (right - left < 4 || bottom - top < 4) return 0;
    let visible = 0;
    for (const fx of [0.2, 0.5, 0.8]) {
      for (const fy of [0.2, 0.5, 0.8]) {
        if (pointVisible(img, rect, left + (right - left) * fx, top + (bottom - top) * fy)) visible++;
      }
    }
    return visible / 9;
  }

  const MIN_VISIBLE_FRACTION = 0.35;

  function collectImages() {
    const viewH = window.innerHeight;
    const viewW = window.innerWidth;
    const candidates = [];
    let occludedInView = false;
    for (const img of document.querySelectorAll("img")) {
      if (!img.complete || img.naturalWidth < MIN_NATURAL || img.naturalHeight < MIN_NATURAL) continue;
      const rect = img.getBoundingClientRect();
      if (rect.width < MIN_SIDE || rect.height < MIN_SIDE) continue;
      const style = getComputedStyle(img);
      if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) continue;
      const url = imageUrl(img);
      if (!url) continue;
      const inView = rect.bottom > 0 && rect.top < viewH && rect.right > 0 && rect.left < viewW;
      const nearby = rect.bottom > -viewH * NEARBY_MARGIN && rect.top < viewH * (1 + NEARBY_MARGIN);
      if (!inView && !nearby) continue;
      if (inView && visibleFraction(img, rect) < MIN_VISIBLE_FRACTION) {
        occludedInView = true; // 모달 뒤 피드 등 가려진 이미지는 건너뛴다
        continue;
      }
      const visibleArea =
        Math.max(0, Math.min(rect.right, viewW) - Math.max(rect.left, 0)) *
        Math.max(0, Math.min(rect.bottom, viewH) - Math.max(rect.top, 0));
      candidates.push({ img, url, inView, visibleArea, distance: Math.abs(rect.top + rect.height / 2 - viewH / 2) });
    }
    // 모달이 열려 있으면 화면 밖 이미지는 모달 뒤 페이지일 가능성이 높으므로 제외
    if (occludedInView) candidates.splice(0, candidates.length, ...candidates.filter((c) => c.inView));
    // 화면에 보이는 것 → 많이 보이는 것 → 화면 중앙에 가까운 것 순
    candidates.sort((a, b) => b.inView - a.inView || b.visibleArea - a.visibleArea || a.distance - b.distance);

    const seen = new Set();
    const picked = [];
    for (const c of candidates) {
      if (seen.has(c.url)) continue;
      seen.add(c.url);
      picked.push(c);
      if (picked.length >= MAX_IMAGES) break;
    }
    return picked;
  }

  // blob: URL 은 이 페이지에서만 읽을 수 있으므로 data URL 로 바꿔서 넘긴다.
  async function blobUrlToDataUrl(url) {
    const blob = await (await fetch(url)).blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async function imageMessage(type, url) {
    const message = { type };
    if (url.startsWith("blob:")) message.dataUrl = await blobUrlToDataUrl(url);
    else if (url.startsWith("data:")) message.dataUrl = url;
    else message.url = url;
    return message;
  }

  async function requestPrediction(url) {
    if (resultCache.has(url)) return resultCache.get(url);
    const response = await chrome.runtime.sendMessage(await imageMessage("predict", url));
    if (response?.ok) resultCache.set(url, response);
    return response || { ok: false, error: "확장 응답 없음" };
  }

  // ---------- 설명 패널 (배지 클릭) ----------
  const panel = document.createElement("aside");
  panel.className = "aifd-panel";
  panel.hidden = true;
  root.appendChild(panel);
  let panelRequest = 0;

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function closePanel() {
    panel.hidden = true;
    panelRequest++;
  }

  function figure(title, src, caption) {
    const box = el("figure", "aifd-figure");
    box.append(el("figcaption", "aifd-figure-title", title));
    if (src) {
      const img = el("img");
      img.src = src;
      img.alt = title;
      box.append(img);
    } else {
      box.append(el("div", "aifd-figure-empty", "준비 중"));
    }
    if (caption) box.append(el("p", "aifd-figure-caption", caption));
    return box;
  }

  function renderPanel(content) {
    const header = el("div", "aifd-panel-header");
    const close = el("button", "aifd-panel-close", "✕");
    close.title = "닫기";
    close.addEventListener("click", closePanel);
    header.append(el("strong", null, "판정 근거"), close);
    panel.replaceChildren(header, ...content);
  }

  async function openPanel(entry, faceIndex) {
    const request = ++panelRequest;
    const face = entry.data.faces[faceIndex];
    const isFake = face.label === "fake";
    panel.hidden = false;
    renderPanel([el("p", "aifd-panel-status", "분석 중…")]);

    let response;
    try {
      response = await chrome.runtime.sendMessage({ ...(await imageMessage("explain", entry.url)), faceIndex });
    } catch (error) {
      response = { ok: false, error: String(error?.message || error) };
    }
    if (request !== panelRequest) return; // 그 사이 닫거나 다른 얼굴을 눌렀음
    if (!response?.ok) {
      renderPanel([el("p", "aifd-panel-status aifd-panel-error", `설명을 불러오지 못했습니다: ${response?.error || "응답 없음"}`)]);
      return;
    }

    const detail = response.result.faces[0];
    const threshold = response.result.threshold_real;
    const verdict = el("div", `aifd-verdict ${isFake ? "aifd-fake" : "aifd-real"}`);
    verdict.append(
      el("span", "aifd-verdict-label", badgeText(detail, threshold)),
      el(
        "span",
        "aifd-verdict-probs",
        `모델 출력 P(real) ${(detail.prob_real * 100).toFixed(1)}% · 기준 ${(threshold * 100).toFixed(0)}% 이상이면 REAL`,
      ),
    );
    // 막대: 기준을 가운데(50%)에 둔 FAKE 확신도
    const meter = el("div", "aifd-meter");
    const fill = el("div", "aifd-meter-fill");
    const fakeScore = isFake ? confidence(detail, threshold) : 1 - confidence(detail, threshold);
    fill.style.width = `${Math.round(fakeScore * 100)}%`;
    const mark = el("div", "aifd-meter-mark");
    mark.style.left = "50%";
    mark.title = "이 선을 넘으면 FAKE";
    meter.append(fill, mark);

    const camWhat = detail.gradcam_target === "fake" ? "FAKE" : "REAL";
    const figures = el("div", "aifd-figures");
    figures.append(
      figure("모델 입력 (얼굴 crop)", detail.crop_image, "모델이 실제로 본 256×256 영역"),
      figure("Grad-CAM", detail.gradcam_image, `${camWhat} 판정에 크게 작용한 영역 · 붉을수록 영향이 큼 (해상도 8×8 기반이라 대략적)`),
      figure("ELA", detail.ela_image, `재압축 오차 · 밝을수록 오차가 큼 (영역 평균 ${detail.ela_stats.mean}, 이미지 평균 ${detail.ela_stats.image_mean})`),
    );

    const notes = el("ul", "aifd-notes");
    for (const text of [
      "확률 추정이며 단정이 아닙니다. 학습에 없던 생성기·보정이 강한 사진은 틀릴 수 있습니다.",
      "Grad-CAM 은 모델이 어디를 보았는지를 보여줄 뿐, 그 부분이 조작되었다는 뜻은 아닙니다.",
      "ELA 는 모델 판정과 별개인 참고 지표입니다. SNS 재압축을 거치면 신호가 약해집니다.",
    ]) notes.append(el("li", null, text));

    renderPanel([verdict, meter, figures, notes]);
  }

  // ---------- 배지 렌더링 ----------
  function clearAll() {
    closePanel();
    for (const entry of entries.values()) entry.nodes.forEach((n) => n.remove());
    entries.clear();
    clearButton.hidden = true;
  }

  function percent(p) {
    return `${Math.round(p * 100)}%`;
  }

  // 배지에 쓰는 판정 확신도. 모델 threshold(예: P(real) 0.05)가 50% 가 되도록 양쪽을 선형으로 늘린다.
  // threshold 가 0.5 가 아니면 원 확률(P(real) 10% 인데 REAL)이 판정과 어긋나 보이기 때문.
  function confidence(face, threshold) {
    const t = Math.min(Math.max(threshold ?? 0.5, 1e-6), 1 - 1e-6);
    const p = face.prob_real;
    return p >= t ? 0.5 + (0.5 * (p - t)) / (1 - t) : 0.5 + (0.5 * (t - p)) / t;
  }

  function badgeText(face, threshold) {
    return `${face.label === "fake" ? "FAKE" : "REAL"} ${percent(confidence(face, threshold))}`;
  }

  function buildNodes(entry) {
    entry.nodes.forEach((n) => n.remove());
    entry.nodes = [];
    const add = (node) => {
      layer.appendChild(node);
      entry.nodes.push(node);
      return node;
    };

    if (entry.status === "loading") {
      const badge = add(document.createElement("div"));
      badge.className = "aifd-badge aifd-loading";
      badge.textContent = "분석 중…";
      badge.dataset.anchor = "image";
      return;
    }
    if (entry.status === "error") {
      const badge = add(document.createElement("div"));
      badge.className = "aifd-badge aifd-error";
      badge.textContent = "오류";
      badge.title = entry.error;
      badge.dataset.anchor = "image";
      return;
    }
    const data = entry.data;
    if (data.status === "no_face" || !data.faces?.length) {
      const badge = add(document.createElement("div"));
      badge.className = "aifd-badge aifd-unknown";
      badge.textContent = "판단 불가";
      badge.title = "얼굴을 찾지 못했습니다";
      badge.dataset.anchor = "image";
      return;
    }
    data.faces.forEach((face, index) => {
      const isFake = face.label === "fake";
      const box = add(document.createElement("div"));
      box.className = `aifd-facebox ${isFake ? "aifd-fake" : "aifd-real"}`;
      box.dataset.face = index;
      const badge = add(document.createElement("div"));
      badge.className = `aifd-badge ${isFake ? "aifd-fake" : "aifd-real"}`;
      badge.textContent = badgeText(face, data.threshold_real);
      badge.title =
        `모델 출력 P(real) ${(face.prob_real * 100).toFixed(1)}%\n` +
        `기준: P(real) ≥ ${(data.threshold_real * 100).toFixed(0)}% 이면 REAL (배지 %는 기준을 50%로 맞춘 확신도)\n클릭하면 판정 근거 보기`;
      badge.dataset.face = index;
      badge.classList.add("aifd-clickable");
      badge.addEventListener("click", () => openPanel(entry, index));
    });
  }

  // object-fit 을 고려해 원본 이미지 좌표(0~1)를 화면 좌표로 바꾼다.
  function contentRect(img, rect) {
    const style = getComputedStyle(img);
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;
    const fit = style.objectFit;
    let w = rect.width;
    let h = rect.height;
    if (fit === "cover" || fit === "contain" || fit === "scale-down" || fit === "none") {
      let scale = 1;
      if (fit === "cover") scale = Math.max(rect.width / nw, rect.height / nh);
      else if (fit === "contain") scale = Math.min(rect.width / nw, rect.height / nh);
      else if (fit === "scale-down") scale = Math.min(1, rect.width / nw, rect.height / nh);
      w = nw * scale;
      h = nh * scale;
    }
    const [px, py] = (style.objectPosition || "50% 50%").split(" ").map((v) => (v.endsWith("%") ? parseFloat(v) / 100 : 0.5));
    return { left: rect.left + (rect.width - w) * px, top: rect.top + (rect.height - h) * py, width: w, height: h };
  }

  function place(node, left, top, clip) {
    // 이미지 영역 밖으로 나가지 않게 고정
    const x = Math.min(Math.max(left, clip.left), Math.max(clip.left, clip.right - node.offsetWidth));
    const y = Math.min(Math.max(top, clip.top), Math.max(clip.top, clip.bottom - node.offsetHeight));
    node.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
  }

  function layout() {
    const viewH = window.innerHeight;
    for (const [img, entry] of entries) {
      const rect = img.isConnected ? img.getBoundingClientRect() : null;
      const visible = rect && rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < viewH;
      // 캐러셀 등으로 이미지가 바뀌었으면 결과를 숨긴다 (재스캔 시 새로 판정)
      const changed = img.isConnected && imageUrl(img) !== entry.url;
      if (!visible || changed) {
        entry.nodes.forEach((n) => (n.style.display = "none"));
        if (!img.isConnected) {
          entry.nodes.forEach((n) => n.remove());
          entries.delete(img);
        }
        continue;
      }
      if (visibleFraction(img, rect) < MIN_VISIBLE_FRACTION) {
        entry.nodes.forEach((n) => (n.style.display = "none"));
        continue;
      }
      const clip = { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom };
      const content = contentRect(img, rect);
      for (const node of entry.nodes) {
        node.style.display = "";
        if (node.dataset.anchor === "image") {
          place(node, rect.left + 6, rect.top + 6, clip);
          continue;
        }
        const face = entry.data.faces[Number(node.dataset.face)];
        const [fx, fy, fw, fh] = face.bbox_norm;
        const left = content.left + fx * content.width;
        const top = content.top + fy * content.height;
        const width = fw * content.width;
        const height = fh * content.height;
        if (node.classList.contains("aifd-facebox")) {
          const l = Math.max(left, clip.left);
          const t = Math.max(top, clip.top);
          node.style.width = `${Math.max(0, Math.min(left + width, clip.right) - l)}px`;
          node.style.height = `${Math.max(0, Math.min(top + height, clip.bottom) - t)}px`;
          node.style.transform = `translate(${Math.round(l)}px, ${Math.round(t)}px)`;
        } else {
          place(node, left, top - node.offsetHeight - 2, clip);
        }
      }
    }
    if (!entries.size) clearButton.hidden = true;
  }

  let layoutQueued = false;
  function queueLayout() {
    if (layoutQueued) return;
    layoutQueued = true;
    requestAnimationFrame(() => {
      layoutQueued = false;
      layout();
    });
  }
  window.addEventListener("scroll", queueLayout, { capture: true, passive: true });
  window.addEventListener("resize", queueLayout, { passive: true });
  setInterval(() => entries.size && queueLayout(), 500); // lazy load 등 레이아웃 변화 대비

  // ---------- 스캔 ----------
  let scanning = false;
  async function scan() {
    if (scanning) return;
    scanning = true;
    scanButton.disabled = true;
    scanButton.textContent = "스캔 중…";
    try {
      const status = await chrome.runtime.sendMessage({ type: "health" });
      if (status?.status !== "ok") {
        showToast("로컬 서버(127.0.0.1:8000)에 연결할 수 없습니다. run_server.bat 실행을 확인하세요.", 6000);
        return;
      }

      clearAll();
      const images = collectImages();
      if (!images.length) {
        showToast("스캔할 이미지가 없습니다.");
        return;
      }
      clearButton.hidden = false;
      for (const { img, url } of images) {
        const entry = { url, status: "loading", data: null, error: null, nodes: [] };
        entries.set(img, entry);
        buildNodes(entry);
      }
      queueLayout();

      let done = 0;
      let fake = 0;
      let serverDown = false;
      await Promise.all(
        images.map(async ({ img, url }) => {
          let response;
          try {
            response = await requestPrediction(url);
          } catch (error) {
            response = { ok: false, error: String(error?.message || error) };
          }
          const entry = entries.get(img);
          if (!entry || entry.url !== url) return; // 그 사이 재스캔/지우기
          if (response.ok) {
            entry.status = "done";
            entry.data = response.result;
            fake += (entry.data.faces || []).filter((f) => f.label === "fake").length;
          } else {
            entry.status = "error";
            entry.error = response.error;
            if (response.error === "SERVER_DOWN") serverDown = true;
          }
          buildNodes(entry);
          done++;
          scanButton.textContent = `스캔 중… ${done}/${images.length}`;
          queueLayout();
        })
      );
      if (serverDown) showToast("분석 중 로컬 서버 연결이 끊겼습니다.", 6000);
      else showToast(`이미지 ${images.length}장 분석 완료 · FAKE 판정 얼굴 ${fake}개`);
    } catch (error) {
      showToast(`스캔 실패: ${error?.message || error}`, 6000);
    } finally {
      scanning = false;
      scanButton.disabled = false;
      scanButton.textContent = "AI 얼굴 스캔";
    }
  }

  scanButton.addEventListener("click", scan);
  clearButton.addEventListener("click", clearAll);
})();
