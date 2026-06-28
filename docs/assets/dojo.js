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

  const activeWidgets = [];

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
    const el = document.body || document.documentElement;
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    return v || fallback;
  }
  const C = {
    accent: () => css("--dojo-accent", "#76b900"),
    warn: () => css("--dojo-warn", "#e0633a"),
    cool: () => css("--dojo-cool", "#3a8ee0"),
    ink: () => css("--dojo-canvas-text", "#222"),
    faint: () => css("--dojo-canvas-faint", "rgba(128,128,128,0.25)"),
  };
  function textColor() { return css("--dojo-canvas-text", "#333"); }

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
    return draw;
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
    return draw;
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
    return draw;
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
      { name: "Shared / L1", cyc: 25, col: "#9acd32" },
      { name: "L2 cache", cyc: 250, col: C.cool() },
      { name: "GDDR7 (DRAM)", cyc: 600, col: C.warn() },
    ];
    function draw() {
      ctx.clearRect(0, 0, W, H);
      const max = Math.log10(600) + 0.2;
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
      out.innerHTML = "On an RTX PRO 6000, a GDDR7 miss (~600 cyc) is <b>~600×</b> a register read. " +
        "One stalled warp isn't a problem if 20 others have work — that's latency hiding.";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // 5. coalescing — 32 threads, how many memory transactions?
  // =======================================================================
  function coalescing(root) {
    shell(root, "Memory coalescing",
      "A warp's 32 loads are served by the RTX PRO 6000's memory system in 128-byte transactions. When consecutive threads read consecutive floats (stride 1), the whole warp is one or two transactions. Stride it out and you pay for bytes you never use.");
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
    return draw;
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
    return draw;
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
    const peak = 125000;  // GFLOP/s — RTX PRO 6000 Blackwell fp32 (~125 TFLOP/s)
    const bw = 1792;      // GB/s — 96 GB GDDR7 on a 512-bit bus
    const ridge = peak / bw; // FLOP/byte (~70)
    let ai = 1;
    slider(cs, "arithmetic intensity (FLOP/byte)", 0.1, 100, ai, 0.1, v => { ai = v; draw(); });

    const xMin = 0.05, xMax = 500, yMin = 50, yMax = 200000;
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
      [100, 1000, 10000, 100000].forEach(v => { ctx.textAlign = "right"; ctx.fillText(v, padL - 5, Y(v) + 3); });
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
    return draw;
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

  // =======================================================================
  // 9. thread-map-2d — 2D launch + how x/y mapping decides coalescing
  // =======================================================================
  function threadMap2d(root) {
    shell(root, "2D thread mapping → memory",
      "Lay a dim3 block over a tiny image, then click a pixel. Watch the 32 memory addresses that pixel's warp touches — and how mapping x to the column vs the row decides whether they're contiguous (coalesced) or scattered.");
    const W = 720, H = 250;
    const { ctx } = canvas(root, W, H);
    const cv = root.querySelector("canvas");
    const cs = controls(root);
    const out = readout(root);
    const imgW = 32, imgH = 8;            // a 32×8 image
    let bx = 16, by = 2, swap = false, selX = 5, selY = 1;

    slider(cs, "block.x", 8, 32, bx, 8, v => { bx = v; clampSel(); draw(); });
    slider(cs, "block.y", 1, 8, by, 1, v => { by = v; clampSel(); draw(); });
    const tg = button(cs, "mapping: x → column", () => {
      swap = !swap;
      tg.textContent = swap ? "mapping: x → row (bad)" : "mapping: x → column";
      tg.classList.toggle("dojo-btn--warn", swap);
      draw();
    });

    function clampSel() {
      if (selX >= imgW) selX = imgW - 1;
      if (selY >= imgH) selY = imgH - 1;
    }
    function addr(x, y) { return swap ? x * imgH + y : y * imgW + x; }

    cv.addEventListener("click", e => {
      const gx = 10, gy = 20, cw = (W - 20) / imgW, chh = 120 / imgH;
      const lx = (e.offsetX / cv.clientWidth) * W;
      const ly = (e.offsetY / cv.clientHeight) * H;
      const x = Math.floor((lx - gx) / cw), y = Math.floor((ly - gy) / chh);
      if (x >= 0 && x < imgW && y >= 0 && y < imgH) { selX = x; selY = y; draw(); }
    });

    function warpPixels() {
      // selected pixel's block, then the warp (32 threads, row-major) it sits in
      const blockX = Math.floor(selX / bx), blockY = Math.floor(selY / by);
      const tx = selX % bx, ty = selY % by;
      const t = ty * bx + tx;
      const w0 = Math.floor(t / 32) * 32;
      const px = [];
      for (let L = w0; L < w0 + 32; L++) {
        const lx = L % bx, ly = Math.floor(L / bx);
        if (ly >= by) break;                       // lane past this block
        const x = blockX * bx + lx, y = blockY * by + ly;
        if (x < imgW && y < imgH) px.push({ x, y });
      }
      return px;
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const gx = 10, gy = 20, cw = (W - 20) / imgW, chh = 120 / imgH;
      const warp = warpPixels();
      const warpSet = new Set(warp.map(p => p.y * imgW + p.x));
      // image cells
      for (let y = 0; y < imgH; y++) for (let x = 0; x < imgW; x++) {
        const key = y * imgW + x;
        ctx.fillStyle = (x === selX && y === selY) ? C.accent()
          : warpSet.has(key) ? "rgba(118,185,0,0.30)" : "rgba(128,128,128,0.10)";
        ctx.fillRect(gx + x * cw, gy + y * chh, cw - 1, chh - 1);
      }
      // block boundaries
      ctx.strokeStyle = "rgba(128,128,128,0.7)"; ctx.lineWidth = 1.5;
      for (let x = 0; x <= imgW; x += bx) { ctx.beginPath(); ctx.moveTo(gx + x * cw, gy); ctx.lineTo(gx + x * cw, gy + imgH * chh); ctx.stroke(); }
      for (let y = 0; y <= imgH; y += by) { ctx.beginPath(); ctx.moveTo(gx, gy + y * chh); ctx.lineTo(gx + imgW * cw, gy + y * chh); ctx.stroke(); }
      ctx.fillStyle = textColor(); ctx.font = "11px monospace"; ctx.textAlign = "left";
      ctx.fillText("image " + imgW + "×" + imgH + " — click a pixel", gx, 14);

      // linear memory strip (one warp's addresses), with 128B (=32 float) segments
      const stripY = 165, stripH = 26, addrs = warp.map(p => addr(p.x, p.y));
      const maxAddr = imgW * imgH, sw = (W - 20) / maxAddr;
      const touched = new Set(addrs.map(a => Math.floor(a / 32)));
      for (let a = 0; a < maxAddr; a++) {
        ctx.fillStyle = (Math.floor(a / 32) % 2 === 0) ? "rgba(128,128,128,0.10)" : "rgba(128,128,128,0.18)";
        ctx.fillRect(10 + a * sw, stripY, sw, stripH);
      }
      addrs.forEach(a => { ctx.fillStyle = C.accent(); ctx.fillRect(10 + a * sw, stripY, Math.max(sw, 2), stripH); });
      const nSeg = Math.ceil(maxAddr / 32);
      for (let s = 0; s < nSeg; s++) {
        ctx.strokeStyle = touched.has(s) ? C.accent() : "rgba(128,128,128,0.4)";
        ctx.lineWidth = touched.has(s) ? 2 : 1;
        ctx.strokeRect(10 + s * 32 * sw, stripY - 3, 32 * sw, stripH + 6);
      }
      ctx.fillStyle = textColor(); ctx.font = "11px monospace"; ctx.textAlign = "left";
      ctx.fillText("linear memory — the warp's addresses (idx = " + (swap ? "x*height + y" : "y*width + x") + ")", 10, stripY - 10);

      const tx = touched.size, eff = Math.round((warp.length * 4) / (tx * 128) * 100);
      out.innerHTML = "pixel (<b>" + selX + "," + selY + "</b>) → block(" + Math.floor(selX / bx) + "," +
        Math.floor(selY / by) + ") thread(" + (selX % bx) + "," + (selY % by) + ") → idx <b>" + addr(selX, selY) +
        "</b><br>this warp of " + warp.length + " lanes touches <b>" + tx + "</b> × 128 B segment" + (tx === 1 ? "" : "s") +
        " → bus efficiency <b>" + eff + "%</b> " + (swap
          ? "<span class='warn'>— x→row makes lanes stride by " + imgH + " floats. Scattered = wasted bandwidth.</span>"
          : "<span class='good'>— x→column keeps the warp contiguous. Coalesced.</span>");
    }
    draw();
    return draw;
  }

  // =======================================================================
  // 10. mem-hierarchy — RTX PRO 6000 tiers + the reuse-factor payoff
  // =======================================================================
  function memHierarchy(root) {
    shell(root, "Memory hierarchy — RTX PRO 6000 Blackwell",
      "Each tier down is ~10× slower but much larger. You can't speed up the GDDR7 trip — you can only avoid it. Drag the reuse factor to see what staging data in shared memory buys you.");
    const W = 720, H = 210;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    const tiers = [
      { name: "Registers", lat: 1, size: "256 KB / SM", scope: "per-thread", col: C.accent() },
      { name: "Shared / L1", lat: 25, size: "≤ 128 KB / SM", scope: "per-block", col: "#9acd32" },
      { name: "L2 cache", lat: 250, size: "on-chip, GPU-wide", scope: "whole GPU", col: C.cool() },
      { name: "GDDR7 (DRAM)", lat: 600, size: "96 GB @ 1792 GB/s", scope: "whole GPU", col: C.warn() },
    ];
    let reuse = 1;
    slider(cs, "reuse factor (reads served from shared per DRAM fetch)", 1, 32, reuse, 1, v => { reuse = v; draw(); });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const rowH = 40, x0 = 20, maxBar = W - 250;
      const maxLog = Math.log10(600);
      tiers.forEach((t, i) => {
        const y = 12 + i * rowH;
        const bw = (Math.log10(t.lat) / maxLog) * maxBar + 60;
        ctx.fillStyle = t.col; ctx.fillRect(x0, y, bw, rowH - 10);
        ctx.fillStyle = "#0b1500"; ctx.font = "bold 12px monospace"; ctx.textAlign = "left";
        ctx.fillText(t.name + "  ~" + t.lat + " cyc", x0 + 8, y + 19);
        ctx.fillStyle = textColor(); ctx.font = "11px monospace";
        ctx.fillText(t.size + "  ·  " + t.scope, x0 + bw + 10, y + 19);
      });
      const eff = (1792 * reuse / 1000).toFixed(1);
      out.innerHTML = "Reuse each fetched byte <b>" + reuse + "×</b> from shared (~25 cyc) instead of re-reading GDDR7 (~600 cyc): " +
        "1 DRAM trip + " + (reuse - 1) + " shared hits. The ALUs effectively see <b>" + eff + " TB/s</b> " +
        "(" + reuse + " × 1792 GB/s). " + (reuse === 1
          ? "<span class='warn'>Reuse 1 = vector add: nothing to amortize, pure bandwidth.</span>"
          : "<span class='good'>This is exactly why tiling (GEMM, convolution, transpose) wins.</span>");
    }
    draw();
    return draw;
  }

  // =======================================================================
  // 11. tiled-transpose — coalesced both ways via shared memory
  // =======================================================================
  function tiledTranspose(root) {
    shell(root, "Tiled transpose",
      "A naive transpose is coalesced one way and strided the other. Stage a tile in __shared__: read it coalesced from global, then write it out coalesced — the strided access is confined to fast shared memory. Step through one 8×8 tile.");
    const W = 720, H = 230;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    const N = 8;
    let step = 0;                 // 0..16 : 0-8 load rows, 8-16 store rows
    button(cs, "Step", () => { if (step < 2 * N) { step++; draw(); } });
    button(cs, "Reset", () => { step = 0; draw(); });

    function grid(gx, label, cell, fill, stroke) {
      ctx.fillStyle = textColor(); ctx.font = "11px monospace"; ctx.textAlign = "center";
      ctx.fillText(label, gx + (N * cell) / 2, 24);
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        ctx.fillStyle = fill(r, c);
        ctx.fillRect(gx + c * cell, 34 + r * cell, cell - 1, cell - 1);
      }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.strokeRect(gx - 1, 33, N * cell + 1, N * cell + 1); }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const cell = 18, gA = 30, gB = 280, gC = 530;
      const loadRow = step < N ? step : N;             // rows loaded so far
      const storeRow = step > N ? step - N : 0;        // rows stored so far
      const loading = step < N, storing = step >= N && step < 2 * N;
      const curLoad = step, curStore = step - N;
      const faint = "rgba(128,128,128,0.12)";

      grid(gA, "global IN (read)", cell, (r, c) =>
        loading && r === curLoad ? C.accent() : r < loadRow ? "rgba(118,185,0,0.25)" : faint,
        loading ? C.accent() : null);

      grid(gB, "__shared__ tile", cell, (r, c) => {
        if (loading && r === curLoad) return "rgba(118,185,0,0.25)";
        if (storing && c === curStore) return C.warn();   // transposed read = a column
        return (r < loadRow) ? "rgba(128,128,128,0.22)" : faint;
      }, storing ? C.warn() : null);

      grid(gC, "global OUT (write)", cell, (r, c) =>
        storing && r === curStore ? C.accent() : r < storeRow ? "rgba(118,185,0,0.25)" : faint,
        storing ? C.accent() : null);

      // arrows
      ctx.fillStyle = textColor(); ctx.font = "16px monospace"; ctx.textAlign = "center";
      ctx.fillText("→", (gA + N * cell + gB) / 2, 34 + N * cell / 2);
      ctx.fillText("→", (gB + N * cell + gC) / 2, 34 + N * cell / 2);

      let msg;
      if (step === 0) msg = "Press Step. Phase 1 loads input row by row — each read is <b>coalesced</b> (a contiguous run).";
      else if (loading) msg = "Phase 1 — load row <b>" + curLoad + "</b> of IN into shared. Global read is <span class='good'>coalesced</span>.";
      else if (storing) msg = "Phase 2 — write OUT row <b>" + curStore + "</b> by reading shared <span class='warn'>column " + curStore + "</span> (the transpose). Global write is <span class='good'>coalesced</span>; the strided access stays inside fast shared memory.";
      else msg = "<span class='good'>Done.</span> Both global passes were coalesced — the transpose happened in shared memory.";
      out.innerHTML = "step <b>" + step + "</b> / " + (2 * N) + " — " + msg;
    }
    draw();
    return draw;
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
    "thread-map-2d": threadMap2d,
    "mem-hierarchy": memHierarchy,
    "tiled-transpose": tiledTranspose,
  };

  function mountAll() {
    document.querySelectorAll("[data-dojo]").forEach(node => {
      if (node.dataset.dojoMounted) return;
      const f = FACTORIES[node.dataset.dojo];
      if (f) {
        node.dataset.dojoMounted = "1";
        try {
          const draw = f(node);
          if (typeof draw === "function") {
            activeWidgets.push({ el: node, draw });
          }
        } catch (e) {
          console.error("dojo widget failed:", node.dataset.dojo, e);
        }
      }
    });
  }

  function triggerThemeChangeRedraw() {
    const stillActive = [];
    activeWidgets.forEach(item => {
      if (document.body.contains(item.el)) {
        stillActive.push(item);
        try {
          item.draw();
        } catch (e) {
          console.error("Redraw failed:", e);
        }
      }
    });
    activeWidgets.length = 0;
    activeWidgets.push(...stillActive);
  }

  const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      if (mutation.attributeName === "data-md-color-scheme") {
        triggerThemeChangeRedraw();
      }
    });
  });

  function startObserver() {
    if (document.body) {
      observer.observe(document.body, { attributes: true });
    } else {
      document.addEventListener("DOMContentLoaded", () => {
        observer.observe(document.body, { attributes: true });
      });
    }
  }
  startObserver();

  // mkdocs-material uses instant navigation; re-mount on each page swap.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(mountAll);
  } else if (document.readyState !== "loading") {
    mountAll();
  } else {
    document.addEventListener("DOMContentLoaded", mountAll);
  }
})();
