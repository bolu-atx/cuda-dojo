/* CUDA Dojo — interactive widget toolkit (dependency-free).
 *
 * Pages mount widgets declaratively:
 *   <div data-dojo="coalescing" data-stride="1"></div>
 * On load we scan for [data-dojo] and call the matching factory below.
 *
 * Each widget draws into a logical-pixel canvas that CSS scales to width.
 * Keep it self-contained: no external libraries so the docs work offline.
 */
(function () {
  "use strict";

  // ---- tiny DOM + canvas helpers ----------------------------------------
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function shell(root, title, hint) {
    root.classList.add("dojo-widget");
    if (title) root.appendChild(el("p", "dojo-widget__title", title));
    if (hint) root.appendChild(el("p", "dojo-widget__hint", hint));
  }

  function canvas(root, w, h) {
    const c = el("canvas");
    const dpr = window.devicePixelRatio || 1;
    c.width = w * dpr;
    c.height = h * dpr;
    c.style.aspectRatio = w + " / " + h;
    root.appendChild(c);
    const ctx = c.getContext("2d");
    ctx.scale(dpr, dpr);
    return { ctx, w, h };
  }

  function slider(parent, label, min, max, val, step, onChange) {
    const wrap = el("div", "dojo-control");
    const lab = el("label");
    const valSpan = el("span", "dojo-val", String(val));
    lab.textContent = label + ": ";
    lab.appendChild(valSpan);
    const input = el("input");
    input.type = "range";
    input.min = min; input.max = max; input.step = step || 1; input.value = val;
    input.addEventListener("input", function () {
      valSpan.textContent = input.value;
      onChange(parseFloat(input.value));
    });
    wrap.appendChild(lab);
    wrap.appendChild(input);
    parent.appendChild(wrap);
    return input;
  }

  function button(parent, text, onClick) {
    const b = el("button", "dojo-btn", text);
    b.addEventListener("click", onClick);
    parent.appendChild(b);
    return b;
  }

  function controls(root) {
    const c = el("div", "dojo-controls");
    root.appendChild(c);
    return c;
  }

  function readout(root) {
    const r = el("div", "dojo-readout");
    root.appendChild(r);
    return r;
  }

  function css(name, fallback) {
    const v = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
    return v || fallback;
  }
  const C = {
    accent: () => css("--dojo-accent", "#76b900"),
    warn: () => css("--dojo-warn", "#e0633a"),
    cool: () => css("--dojo-cool", "#3a8ee0"),
    ink: () => css("--md-default-fg-color", "#222"),
    faint: () => "rgba(128,128,128,0.25)",
  };
  function textColor() { return css("--md-default-fg-color", "#333"); }

  // =======================================================================
  // 1. thread-index — global index = blockIdx*blockDim + threadIdx
  // =======================================================================
  function threadIndex(root) {
    shell(root, "Thread → global index",
      "Drag the sliders. The global index is the one number that maps a thread to its data.");
    const W = 720, H = 130;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let blockDim = 8, gridDim = 4, blockIdx = 2, threadIdx_ = 3;

    const sB = slider(cs, "blockDim.x", 1, 16, blockDim, 1, v => { blockDim = v; clamp(); draw(); });
    const sG = slider(cs, "gridDim.x", 1, 8, gridDim, 1, v => { gridDim = v; clamp(); draw(); });
    const sBi = slider(cs, "blockIdx.x", 0, 7, blockIdx, 1, v => { blockIdx = v; clamp(); draw(); });
    const sTi = slider(cs, "threadIdx.x", 0, 15, threadIdx_, 1, v => { threadIdx_ = v; clamp(); draw(); });

    function clamp() {
      if (blockIdx > gridDim - 1) { blockIdx = gridDim - 1; sBi.value = blockIdx; }
      if (threadIdx_ > blockDim - 1) { threadIdx_ = blockDim - 1; sTi.value = threadIdx_; }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const total = blockDim * gridDim;
      const gap = 10;
      const usable = W - 20 - gap * (gridDim - 1);
      const cw = usable / total;
      const y = 35, ch = 38;
      let x = 10;
      const sel = blockIdx * blockDim + threadIdx_;
      for (let b = 0; b < gridDim; b++) {
        for (let t = 0; t < blockDim; t++) {
          const gi = b * blockDim + t;
          ctx.fillStyle = gi === sel ? C.accent()
            : (b === blockIdx ? "rgba(118,185,0,0.18)" : C.faint());
          ctx.fillRect(x, y, cw - 1, ch);
          if (cw > 16) {
            ctx.fillStyle = gi === sel ? "#0b1500" : textColor();
            ctx.font = "11px monospace";
            ctx.textAlign = "center";
            ctx.fillText(gi, x + cw / 2, y + ch / 2 + 4);
          }
          x += cw;
        }
        // block label
        ctx.fillStyle = textColor();
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.fillText("block " + b, x - (blockDim * cw) / 2, y - 8);
        x += gap;
      }
      out.innerHTML =
        "global = blockIdx*blockDim + threadIdx = <b>" + blockIdx + "*" + blockDim +
        " + " + threadIdx_ + "</b> = <b>" + sel + "</b> &nbsp;|&nbsp; total threads = " +
        gridDim + "×" + blockDim + " = <b>" + total + "</b>";
    }
    draw();
  }

  // =======================================================================
  // 2. grid-stride — one thread, many elements
  // =======================================================================
  function gridStride(root) {
    shell(root, "Grid-stride loop",
      "Fewer threads than data? Each thread strides across the array by the total thread count, so the work stays balanced and the code is independent of launch size.");
    const W = 720, H = 90;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let n = 32, threads = 8, focus = 0;
    const sN = slider(cs, "N (elements)", 8, 64, n, 1, v => { n = v; draw(); });
    const sT = slider(cs, "total threads", 1, 32, threads, 1, v => { threads = v; if (focus >= threads) focus = 0; sF.max = Math.max(0, threads - 1); draw(); });
    const sF = slider(cs, "highlight thread", 0, threads - 1, focus, 1, v => { focus = v; draw(); });

    function palette(i) {
      const hue = (i * 47) % 360;
      return "hsl(" + hue + ",60%,55%)";
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const cw = (W - 20) / n, y = 30, ch = 34;
      for (let i = 0; i < n; i++) {
        const owner = i % threads;
        const isFocus = owner === focus;
        ctx.fillStyle = isFocus ? palette(owner) : "rgba(128,128,128,0.18)";
        ctx.fillRect(10 + i * cw, y, cw - 2, ch);
        ctx.fillStyle = isFocus ? "#fff" : textColor();
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        if (cw > 12) ctx.fillText(i, 10 + i * cw + cw / 2, y + ch / 2 + 4);
      }
      const handled = [];
      for (let i = focus; i < n; i += threads) handled.push(i);
      out.innerHTML = "thread <b>" + focus + "</b> handles indices [" +
        handled.join(", ") + "] — stride = total threads = <b>" + threads + "</b>";
    }
    draw();
  }

  // =======================================================================
  // 3. simt — warp divergence
  // =======================================================================
  function simt(root) {
    shell(root, "SIMT: a warp is 32 lanes in lockstep",
      "All 32 lanes share one instruction stream. A data-dependent branch forces the warp to execute BOTH sides, masking off the inactive lanes each pass. That waste is 'warp divergence'.");
    const W = 720, H = 150;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let thr = 16;
    slider(cs, "branch: lanes where (laneId < t) take path A", 0, 32, thr, 1, v => { thr = v; draw(); });

    function lanes(yA, label, predicate, color) {
      const cw = (W - 20) / 32;
      ctx.fillStyle = textColor();
      ctx.font = "11px monospace";
      ctx.textAlign = "left";
      ctx.fillText(label, 10, yA - 6);
      let active = 0;
      for (let i = 0; i < 32; i++) {
        const on = predicate(i);
        if (on) active++;
        ctx.fillStyle = on ? color : "rgba(128,128,128,0.15)";
        ctx.fillRect(10 + i * cw, yA, cw - 1.5, 26);
      }
      return active;
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const aA = lanes(30, "pass 1 — path A (if branch):", i => i < thr, C.accent());
      const aB = lanes(95, "pass 2 — path B (else branch):", i => i >= thr, C.cool());
      const diverged = (thr > 0 && thr < 32);
      out.innerHTML = "active lanes: A=<b>" + aA + "</b>, B=<b>" + aB +
        "</b> &nbsp;|&nbsp; " + (diverged
          ? "<span class='warn'>diverged → 2 passes, " + (32 - aA + (32 - aB)) + " lane-slots wasted</span>"
          : "<span class='good'>uniform branch → no divergence</span>");
    }
    draw();
  }

  // =======================================================================
  // 4. mem-latency — the latency cliff
  // =======================================================================
  function memLatency(root) {
    shell(root, "Memory latency cliff",
      "Every tier you fall through costs ~10× more cycles. This is why CUDA performance is mostly about keeping data close — and why you launch thousands of warps to hide the trips you can't avoid.");
    const W = 720, H = 200;
    const { ctx } = canvas(root, W, H);
    const out = readout(root);
    const tiers = [
      { name: "Register", cyc: 1, col: C.accent() },
      { name: "Shared", cyc: 20, col: "#9acd32" },
      { name: "L2 cache", cyc: 200, col: C.cool() },
      { name: "Global (DRAM)", cyc: 500, col: C.warn() },
    ];
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const max = Math.log10(500) + 0.2;
      const barW = (W - 40) / tiers.length;
      tiers.forEach((t, i) => {
        const h = (Math.log10(t.cyc) / max) * 150 + 6;
        const x = 20 + i * barW;
        ctx.fillStyle = t.col;
        ctx.fillRect(x, 170 - h, barW - 18, h);
        ctx.fillStyle = textColor();
        ctx.font = "12px monospace";
        ctx.textAlign = "center";
        ctx.fillText(t.name, x + (barW - 18) / 2, 188);
        ctx.fillText("~" + t.cyc + " cyc", x + (barW - 18) / 2, 170 - h - 6);
      });
      out.innerHTML = "A global-memory miss (~500 cyc) is <b>~500×</b> a register read. " +
        "One stalled warp isn't a problem if 20 others have work — that's latency hiding.";
    }
    draw();
  }

  // =======================================================================
  // 5. coalescing — 32 threads, how many memory transactions?
  // =======================================================================
  function coalescing(root) {
    shell(root, "Memory coalescing",
      "A warp's 32 loads are served in 128-byte transactions. When consecutive threads read consecutive floats (stride 1), the whole warp is one or two transactions. Stride it out and you pay for bytes you never use.");
    const W = 720, H = 170;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let stride = 1;
    slider(cs, "stride (floats between threads)", 1, 8, stride, 1, v => { stride = v; draw(); });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      // memory laid out as floats; 32 floats = 128 B = one segment
      const floatsShown = 32 * Math.min(stride, 4) + 8;
      const fw = (W - 20) / floatsShown;
      const segFloats = 32; // 128 bytes
      const touched = new Set();
      const addr = [];
      for (let t = 0; t < 32; t++) {
        const f = t * stride;
        addr.push(f);
        touched.add(Math.floor(f / segFloats));
      }
      // draw memory cells + segment boundaries
      const y = 70, ch = 26;
      for (let f = 0; f < floatsShown; f++) {
        const seg = Math.floor(f / segFloats);
        ctx.fillStyle = (seg % 2 === 0) ? "rgba(128,128,128,0.10)" : "rgba(128,128,128,0.20)";
        ctx.fillRect(10 + f * fw, y, fw - 0.5, ch);
      }
      // accessed floats
      addr.forEach((f, t) => {
        if (f < floatsShown) {
          ctx.fillStyle = touched.has(Math.floor(f / segFloats)) ? C.accent() : C.warn();
          ctx.fillRect(10 + f * fw, y, Math.max(fw - 0.5, 2), ch);
        }
      });
      // segment labels
      const nSeg = Math.ceil(floatsShown / segFloats);
      for (let s = 0; s < nSeg; s++) {
        ctx.strokeStyle = touched.has(s) ? C.accent() : "rgba(128,128,128,0.4)";
        ctx.lineWidth = touched.has(s) ? 2 : 1;
        ctx.strokeRect(10 + s * segFloats * fw, y - 4, segFloats * fw, ch + 8);
        ctx.fillStyle = textColor();
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.fillText("128B #" + s, 10 + (s + 0.5) * segFloats * fw, y - 10);
      }
      ctx.fillStyle = textColor();
      ctx.font = "11px monospace";
      ctx.textAlign = "left";
      ctx.fillText("32 threads in a warp, each loads 1 float", 10, 120);

      const tx = touched.size;
      const eff = Math.round((32 * 4) / (tx * 128) * 100);
      out.innerHTML = "transactions needed: <b>" + tx + "</b> (×128 B) — bus efficiency <b>" +
        eff + "%</b>. " + (stride === 1
          ? "<span class='good'>Perfectly coalesced.</span>"
          : "<span class='warn'>" + (100 - eff) + "% of fetched bytes are wasted.</span>");
    }
    draw();
  }

  // =======================================================================
  // 6. reduction — tree reduction in shared memory
  // =======================================================================
  function reduction(root) {
    shell(root, "Parallel reduction (tree)",
      "Summing N values serially is N steps. A tree halves the active threads each step and finishes in log₂N. Step through it.");
    const W = 720, H = 220;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    const N = 16;
    let data, step, history;
    function reset() {
      data = Array.from({ length: N }, () => Math.floor(Math.random() * 9) + 1);
      step = 0; history = [data.slice()]; draw();
    }
    function next() {
      const s = N >> (step + 1);
      if (s < 1) return;
      const cur = history[history.length - 1].slice();
      for (let i = 0; i < s; i++) cur[i] = cur[i] + cur[i + s];
      history.push(cur);
      step++; draw();
    }
    button(cs, "Step", next);
    button(cs, "Reset", reset);

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const cw = (W - 20) / N;
      history.forEach((row, r) => {
        const y = 10 + r * 40, ch = 28;
        const activeStride = N >> (r + 1);
        for (let i = 0; i < N; i++) {
          const alive = i < (N >> r);
          if (!alive) continue;
          const willWrite = r < history.length - 1 && i < (N >> (r + 1));
          const willRead = r < history.length - 1 && i >= (N >> (r + 1)) && i < (N >> r);
          ctx.fillStyle = willWrite ? C.accent() : willRead ? C.warn() : "rgba(128,128,128,0.18)";
          ctx.fillRect(10 + i * cw, y, cw - 2, ch);
          ctx.fillStyle = (willWrite || willRead) ? "#0b1500" : textColor();
          ctx.font = "11px monospace";
          ctx.textAlign = "center";
          ctx.fillText(row[i], 10 + i * cw + cw / 2, y + ch / 2 + 4);
        }
      });
      const done = (N >> step) === 1;
      out.innerHTML = "step <b>" + step + "</b> / " + Math.log2(N) +
        (done ? " — <span class='good'>sum = " + history[history.length - 1][0] + "</span>"
              : " — green cells += red cells, then __syncthreads()");
    }
    reset();
  }

  // =======================================================================
  // 7. roofline — memory vs compute bound
  // =======================================================================
  function roofline(root) {
    shell(root, "Roofline model",
      "Where does your kernel hit the ceiling? Below the ridge you're memory-bound (move data faster); above it you're compute-bound (do fewer/cheaper ops). Drag arithmetic intensity.");
    const W = 720, H = 320;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    const peak = 19500;   // GFLOP/s (A100-ish fp32)
    const bw = 1555;      // GB/s
    const ridge = peak / bw; // FLOP/byte
    let ai = 1;
    slider(cs, "arithmetic intensity (FLOP/byte)", 0.1, 100, ai, 0.1, v => { ai = v; draw(); });

    const xMin = 0.05, xMax = 200, yMin = 50, yMax = 30000;
    const padL = 55, padB = 35, padT = 15, padR = 15;
    function X(v) { return padL + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * (W - padL - padR); }
    function Y(v) { return (H - padB) - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)) * (H - padB - padT); }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      // axes
      ctx.strokeStyle = "rgba(128,128,128,0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();
      ctx.fillStyle = textColor();
      ctx.font = "10px monospace";
      [0.1, 1, 10, 100].forEach(v => { ctx.textAlign = "center"; ctx.fillText(v, X(v), H - padB + 14); });
      [100, 1000, 10000].forEach(v => { ctx.textAlign = "right"; ctx.fillText(v, padL - 5, Y(v) + 3); });
      ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
      ctx.fillText("GFLOP/s", 0, 0); ctx.restore();
      ctx.textAlign = "center"; ctx.fillText("arithmetic intensity (FLOP/byte)", W / 2, H - 6);

      // roofline: slanted memory roof then flat compute roof
      ctx.strokeStyle = C.accent(); ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(X(xMin), Y(Math.min(xMin * bw, peak)));
      ctx.lineTo(X(ridge), Y(peak));
      ctx.lineTo(X(xMax), Y(peak));
      ctx.stroke();
      // ridge marker
      ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(128,128,128,0.6)";
      ctx.beginPath(); ctx.moveTo(X(ridge), Y(peak)); ctx.lineTo(X(ridge), H - padB); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = textColor(); ctx.textAlign = "center";
      ctx.fillText("ridge " + ridge.toFixed(1), X(ridge), Y(peak) - 8);

      // kernel point
      const achievable = Math.min(ai * bw, peak);
      ctx.fillStyle = ai < ridge ? C.warn() : C.cool();
      ctx.beginPath(); ctx.arc(X(ai), Y(achievable), 6, 0, 7); ctx.fill();

      const bound = ai < ridge ? "memory-bound" : "compute-bound";
      out.innerHTML = "at AI=<b>" + ai.toFixed(1) + "</b> FLOP/byte → ceiling <b>" +
        Math.round(achievable) + "</b> GFLOP/s — <b>" + bound + "</b> " +
        (ai < ridge ? "(feed it data faster: coalesce, reuse, cache)"
                    : "(it's saturating the math units)");
    }
    draw();
  }

  // =======================================================================
  // 8. streams — overlap copy and compute
  // =======================================================================
  function streams(root) {
    shell(root, "Streams: overlap transfer and compute",
      "Serial: copy, wait, compute, wait, copy. With pinned memory and multiple streams the engines run at once — the copy of chunk 2 hides behind the compute of chunk 1.");
    const W = 720, H = 200;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let chunks = 4;
    slider(cs, "chunks", 1, 8, chunks, 1, v => { chunks = v; draw(); });
    const unit = 26; // ms per stage (illustrative)

    function bar(x, y, w, label, col) {
      ctx.fillStyle = col; ctx.fillRect(x, y, w, 20);
      ctx.fillStyle = "#0b1500"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      if (w > 18) ctx.fillText(label, x + w / 2, y + 14);
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const px = 7; // px per ms
      const serialT = chunks * 3 * unit;
      // serial
      ctx.fillStyle = textColor(); ctx.font = "11px monospace"; ctx.textAlign = "left";
      ctx.fillText("serial:", 10, 24);
      let x = 70;
      for (let i = 0; i < chunks; i++) {
        bar(x, 14, unit * px, "H2D", C.cool()); x += unit * px;
        bar(x, 14, unit * px, "K", C.accent()); x += unit * px;
        bar(x, 14, unit * px, "D2H", C.warn()); x += unit * px;
      }
      // overlapped: 3 lanes (H2D / K / D2H), each chunk shifted by 1 unit
      ctx.fillStyle = textColor(); ctx.fillText("overlapped:", 10, 90);
      const laneY = [78, 104, 130];
      const laneCol = [C.cool(), C.accent(), C.warn()];
      const laneLab = ["H2D", "K", "D2H"];
      let overlapT = 0;
      for (let i = 0; i < chunks; i++) {
        for (let s = 0; s < 3; s++) {
          const start = (i + s) * unit;
          bar(90 + start * px, laneY[s], unit * px, laneLab[s], laneCol[s]);
          overlapT = Math.max(overlapT, start + unit);
        }
      }
      const speedup = (serialT / overlapT).toFixed(2);
      out.innerHTML = "serial ≈ <b>" + serialT + "</b> units, overlapped ≈ <b>" + overlapT +
        "</b> units → <span class='good'>" + speedup + "× faster</span> (more chunks ⇒ closer to the single-engine bound).";
    }
    draw();
  }

  // ---- registry & automount ---------------------------------------------
  const FACTORIES = {
    "thread-index": threadIndex,
    "grid-stride": gridStride,
    "simt": simt,
    "mem-latency": memLatency,
    "coalescing": coalescing,
    "reduction": reduction,
    "roofline": roofline,
    "streams": streams,
  };

  function mountAll() {
    document.querySelectorAll("[data-dojo]").forEach(node => {
      if (node.dataset.dojoMounted) return;
      const f = FACTORIES[node.dataset.dojo];
      if (f) { node.dataset.dojoMounted = "1"; try { f(node); } catch (e) { console.error("dojo widget failed:", node.dataset.dojo, e); } }
    });
  }

  // mkdocs-material uses instant navigation; re-mount on each page swap.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(mountAll);
  } else if (document.readyState !== "loading") {
    mountAll();
  } else {
    document.addEventListener("DOMContentLoaded", mountAll);
  }
})();
