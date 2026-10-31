(() => {
  const zone = document.getElementById('dropzone');
  const surface = zone.querySelector('.lens-surface');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)');
  let target = [.55, .48], position = [...target], frame = 0;
  function paint() {
    position = position.map((v, i) => reduce.matches ? target[i] : v + (target[i] - v) * .13);
    surface.style.setProperty('--lens-x', `${position[0] * 100}%`);
    surface.style.setProperty('--lens-y', `${position[1] * 100}%`);
    if (Math.abs(position[0] - target[0]) + Math.abs(position[1] - target[1]) > .001) frame = requestAnimationFrame(paint);
    else frame = 0;
  }
  function move(event) {
    if (event.pointerType === 'touch') return;
    const rect = surface.getBoundingClientRect();
    target = [Math.max(.12, Math.min(.88, (event.clientX - rect.left) / rect.width)), Math.max(.22, Math.min(.78, (event.clientY - rect.top) / rect.height))];
    if (!frame) frame = requestAnimationFrame(paint);
  }
  zone.addEventListener('pointermove', move);
  zone.addEventListener('pointerleave', () => { target = [.55,.48]; if (!frame) frame = requestAnimationFrame(paint); });
  document.addEventListener('fakit:inspect', ({ detail }) => {
    const { url, state, label } = detail;
    if (url) {
      zone.classList.add('has-lens-image');
      surface.querySelectorAll('.lens-photo').forEach(img => { img.src = url; img.hidden = false; });
    } else if (state === 'idle') {
      zone.classList.remove('has-lens-image');
      surface.querySelectorAll('.lens-photo').forEach(img => { img.removeAttribute('src'); img.hidden = true; });
    }
    zone.dataset.inspection = state;
    document.getElementById('lens-state').textContent = label;
  });
  const entry = document.getElementById('text-entry');
  const content = document.getElementById('text-content');
  document.getElementById('open-text').addEventListener('click', () => { entry.hidden = false; content.focus(); });
  document.getElementById('close-text').addEventListener('click', () => { entry.hidden = true; document.getElementById('open-text').focus(); });
  content.addEventListener('input', () => { document.getElementById('text-status').textContent = `${content.value.length.toLocaleString()}자 · 글 모델 연결 전`; });
})();
