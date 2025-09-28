(function () {
  // Mount points
  const backEl = document.getElementById('waves-back');
  const midEl  = document.getElementById('waves-mid');
  if (!backEl || !midEl || !window.waves) return;

  // Helpers to derive RGBA from Bootstrap tokens
  const root = document.documentElement;
  const styles = getComputedStyle(root);
  const rgbVar = styles.getPropertyValue('--bs-primary-rgb').trim();
  const hexVar = styles.getPropertyValue('--bs-primary').trim();

  function rgbaFromRgbVar(rgbStr, a) {
    const parts = rgbStr.split(/[ ,]+/).filter(Boolean).map(Number);
    return parts.length === 3 ? `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${a})` : null;
  }
  function rgbaFromHex(h, a) {
    const s = (h || '').replace('#', '').trim();
    if (!s) return null;
    const n = s.length === 3
      ? s.split('').map(c => parseInt(c + c, 16))
      : [parseInt(s.slice(0,2),16), parseInt(s.slice(2,4),16), parseInt(s.slice(4,6),16)];
    if (n.some(Number.isNaN)) return null;
    return `rgba(${n[0]}, ${n[1]}, ${n[2]}, ${a})`;
  }

  // Theme-aware opacities (more dramatic fills)
  const isDark = (root.getAttribute('data-bs-theme') || '').toLowerCase() === 'dark';
  const opacityMul = isDark ? 0.7 : 1.0; // slightly stronger in dark mode

  const fallback = (a) => `rgba(34,197,94,${a})`; // brand green fallback
  function brandRGBA(alpha) {
    return (rgbVar && rgbaFromRgbVar(rgbVar, alpha * opacityMul))
        || (hexVar && rgbaFromHex(hexVar, alpha * opacityMul))
        || fallback(alpha * opacityMul);
  }

  // Motion prefs and tuning
  let prefersReduced = false;
  try { prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch {}

  const isMobile = window.matchMedia('(max-width: 768px)').matches;
  const saveData = !!(navigator.connection && navigator.connection.saveData);

  const baseComplexity   = (isMobile || saveData) ? 4 : 6;
  const baseFlowRateBack = (isMobile || saveData) ? 0.6 : 0.9; // calmer
  const baseFlowRateMid  = (isMobile || saveData) ? 0.9 : 1.2; // calmer

  // Back config (subtle)
  const backCfg = {
    fills: [brandRGBA(0.06), brandRGBA(0.10), brandRGBA(0.14)], // even softer
    flowRate: baseFlowRateBack,
    randomFlowRate: 0.04,
    swayRate: prefersReduced ? 0 : 0.28,   // gentler bob
    randomSwayRate: prefersReduced ? 0 : 0.08,
    swayVelocity: prefersReduced ? 0 : 0.10,
    wavelength: 30, // longer waves
    complexity: baseComplexity,
    curviness: 0.7,
    offset: 0.10,
    randomOffset: 0.10,
  };

  // Mid config (more dynamic)
  const midCfg = {
    fills: [brandRGBA(0.08), brandRGBA(0.12), brandRGBA(0.16)],
    flowRate: baseFlowRateMid,
    randomFlowRate: 0.06,
    swayRate: prefersReduced ? 0 : 0.45,
    randomSwayRate: prefersReduced ? 0 : 0.10,
    swayVelocity: prefersReduced ? 0 : 0.16,
    wavelength: 26,
    complexity: baseComplexity,
    curviness: 0.85,
    offset: 0.14,
    randomOffset: 0.14,
  };

  // Instantiate and mount (synchronous)
  const back = waves(backCfg);
  const mid  = waves(midCfg);

  const backOk = back.mount('#waves-back');
  const midOk  = mid.mount('#waves-mid');

  if (backOk) backEl.classList.add('waves-mounted');
  if (midOk)  midEl.classList.add('waves-mounted');

  // IntersectionObserver to pause/resume when out of view
  const hero = document.querySelector('.hero-group');
  if (!prefersReduced && hero) {
    const io = new IntersectionObserver(entries => {
      for (const e of entries) {
        if (e.isIntersecting) {
          back.play();
          mid.play();
        } else {
          back.stop();
          mid.stop();
        }
      }
    }, { threshold: 0.15 });
    io.observe(hero);
  } else {
    back.stop();
    mid.stop();
  }
})();
