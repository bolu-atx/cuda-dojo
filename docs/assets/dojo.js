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

  // =======================================================================
  // 12. sm-scheduler — eligible warps and issue slots
  // =======================================================================
  function smScheduler(root) {
    shell(root, "SM scheduler: resident warps are not always ready warps",
      "Predict whether the scheduler has enough eligible warps to fill issue slots. More occupancy helps only when it creates ready work.");
    const W = 720, H = 220;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let resident = 16, latency = 10, registers = 4, ilp = 2;
    slider(cs, "resident warps", 1, 32, resident, 1, v => { resident = v; draw(); });
    slider(cs, "memory wait", 0, 24, latency, 1, v => { latency = v; draw(); });
    slider(cs, "register pressure", 1, 8, registers, 1, v => { registers = v; draw(); });
    slider(cs, "independent work", 1, 8, ilp, 1, v => { ilp = v; draw(); });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const issueSlots = 4;
      const cap = Math.max(1, Math.floor(32 / registers));
      const active = Math.min(resident, cap);
      const readyFrac = Math.max(0.05, Math.min(1, (ilp + 1) / (latency / 3 + 4)));
      const eligible = Math.min(active, Math.max(1, Math.round(active * readyFrac)));
      const issued = Math.min(issueSlots, eligible);
      const stall = issueSlots - issued;

      ctx.fillStyle = textColor();
      ctx.font = "12px monospace";
      ctx.textAlign = "left";
      ctx.fillText("resident warps that fit after register pressure", 20, 22);

      const cols = 16, cell = 18, gap = 4;
      for (let i = 0; i < 32; i++) {
        const x = 20 + (i % cols) * (cell + gap);
        const y = 36 + Math.floor(i / cols) * (cell + gap);
        let fill = "rgba(128,128,128,0.12)";
        if (i < active) fill = i < eligible ? C.accent() : C.warn();
        if (i >= active && i < resident) fill = "rgba(128,128,128,0.28)";
        ctx.fillStyle = fill;
        ctx.fillRect(x, y, cell, cell);
      }

      const sx = 430, sy = 42;
      ctx.fillStyle = textColor();
      ctx.font = "12px monospace";
      ctx.fillText("issue slots this cycle", sx, sy - 14);
      for (let i = 0; i < issueSlots; i++) {
        ctx.fillStyle = i < issued ? C.accent() : "rgba(224,99,58,0.35)";
        ctx.fillRect(sx + i * 52, sy, 42, 42);
        ctx.fillStyle = i < issued ? "#0b1500" : textColor();
        ctx.font = "bold 14px monospace";
        ctx.textAlign = "center";
        ctx.fillText(i < issued ? "ISS" : "IDLE", sx + i * 52 + 21, sy + 27);
      }

      ctx.fillStyle = textColor();
      ctx.textAlign = "left";
      ctx.font = "11px monospace";
      ctx.fillText("green = eligible, orange = resident but waiting, grey = cannot fit", 20, 105);

      const bw = 620, bh = 18, bx = 50, by = 150;
      ctx.fillStyle = "rgba(128,128,128,0.18)";
      ctx.fillRect(bx, by, bw, bh);
      ctx.fillStyle = C.accent();
      ctx.fillRect(bx, by, bw * (issued / issueSlots), bh);
      ctx.fillStyle = C.warn();
      ctx.fillRect(bx + bw * (issued / issueSlots), by, bw * (stall / issueSlots), bh);
      ctx.fillStyle = textColor();
      ctx.textAlign = "center";
      ctx.font = "12px monospace";
      ctx.fillText("scheduler utilization", bx + bw / 2, by - 8);

      const reason = stall === 0 ? "issue slots filled" :
        latency > ilp * 3 ? "memory latency dominates" :
        registers > 5 ? "register pressure limits residency" :
        "not enough ready warps";
      out.innerHTML = "active warps = <b>" + active + "</b>, eligible now = <b>" + eligible +
        "</b>, issued = <b>" + issued + "/" + issueSlots + "</b> — " +
        (stall ? "<span class='warn'>" + reason + "</span>" : "<span class='good'>" + reason + "</span>");
    }
    draw();
    return draw;
  }

  // =======================================================================
  // 13. library-choice — choose the right CUDA tool
  // =======================================================================
  function libraryChoice(root) {
    shell(root, "Library or custom kernel?",
      "Move the workload shape. The lesson is to name the primitive before writing code.");
    const W = 720, H = 230;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let kind = 0, fusion = 0, prototype = 0;
    const kinds = ["reduce/scan", "GEMM", "FFT", "image op", "custom stencil"];
    slider(cs, "workload", 0, kinds.length - 1, kind, 1, v => { kind = v; draw(); });
    slider(cs, "fusion need", 0, 3, fusion, 1, v => { fusion = v; draw(); });
    slider(cs, "prototype speed", 0, 3, prototype, 1, v => { prototype = v; draw(); });

    function pick() {
      if (prototype >= 2 && kind === 0) return ["Thrust", "express it quickly, then replace hot paths with CUB"];
      if (kind === 0) return ["CUB", "standard parallel primitive"];
      if (kind === 1 && fusion >= 2) return ["cuBLASLt / CUTLASS", "GEMM-shaped work with epilogue or custom tiling"];
      if (kind === 1) return ["cuBLAS", "standard dense linear algebra"];
      if (kind === 2) return ["cuFFT", "spectral transform; keep plans outside the hot loop"];
      if (kind === 3 && fusion <= 1) return ["NPP / CV-CUDA", "common production image primitive"];
      if (kind === 3) return ["library + custom glue", "library primitive plus fused boundary/layout code"];
      return fusion >= 2 ? ["custom CUDA", "semantics or fusion are the product"] : ["NPP or custom CUDA", "compare library semantics before coding"];
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const labels = ["Thrust", "CUB", "cuBLAS", "cuFFT", "NPP/CV-CUDA", "CUTLASS", "custom"];
      const chosen = pick();
      const x0 = 24, y0 = 58, w = 92, h = 42, gap = 8;
      ctx.fillStyle = textColor();
      ctx.font = "12px monospace";
      ctx.textAlign = "left";
      ctx.fillText("workload: " + kinds[kind], x0, 24);
      ctx.fillText("decision ladder", x0, 48);
      labels.forEach((label, i) => {
        const x = x0 + i * (w + gap);
        const on = chosen[0].indexOf(label.split("/")[0]) >= 0 || (label === "custom" && chosen[0].indexOf("custom") >= 0);
        ctx.fillStyle = on ? C.accent() : "rgba(128,128,128,0.16)";
        ctx.fillRect(x, y0, w, h);
        ctx.fillStyle = on ? "#0b1500" : textColor();
        ctx.font = "bold 11px monospace";
        ctx.textAlign = "center";
        ctx.fillText(label, x + w / 2, y0 + 26);
        if (i < labels.length - 1) {
          ctx.fillStyle = textColor();
          ctx.fillText("→", x + w + gap / 2, y0 + 26);
        }
      });

      const rx = 70, ry = 140;
      ctx.fillStyle = "rgba(128,128,128,0.12)";
      ctx.fillRect(rx, ry, 580, 52);
      ctx.fillStyle = C.accent();
      ctx.fillRect(rx, ry, Math.min(580, 120 + fusion * 105), 6);
      ctx.fillStyle = textColor();
      ctx.textAlign = "center";
      ctx.font = "12px monospace";
      ctx.fillText("more fusion/custom semantics pushes right; standard primitives push left", rx + 290, ry + 34);

      out.innerHTML = "reach for <b>" + chosen[0] + "</b> — " + chosen[1] +
        ". If you disagree, state the missing semantic before writing a kernel.";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // 14. imaging-pattern — algorithm shape before code
  // =======================================================================
  function imagingPattern(root) {
    shell(root, "Image algorithm shape",
      "Pick the algorithm and kernel size. Predict whether the GPU wants a stencil, separable pass, FFT, or pipeline.");
    const W = 720, H = 235;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let alg = 0, k = 7;
    const algs = ["Sobel", "Gaussian", "Morphology", "FFT conv", "Warp"];
    slider(cs, "algorithm", 0, algs.length - 1, alg, 1, v => { alg = v; draw(); });
    slider(cs, "kernel / footprint", 3, 63, k, 2, v => { k = v; draw(); });

    function recommendation() {
      if (alg === 0) return ["stencil tile", 9, "small fixed neighborhood; coalesce and reuse halo pixels"];
      if (alg === 1) return ["separable passes", 2 * k, "k×k becomes horizontal k plus vertical k"];
      if (alg === 2) return ["window min/max", k * k, "stage the window or use library morphology"];
      if (alg === 3) return ["cuFFT pipeline", Math.round(k * Math.log2(k) * 2), "large kernels can win in frequency space"];
      return ["gather warp", 4, "interpolate cached reads, keep writes contiguous"];
    }

    function drawGrid(x, y, n, highlightFn) {
      const c = 12;
      for (let r = 0; r < n; r++) for (let col = 0; col < n; col++) {
        ctx.fillStyle = highlightFn(r, col) ? C.accent() : "rgba(128,128,128,0.16)";
        ctx.fillRect(x + col * c, y + r * c, c - 1, c - 1);
      }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const rec = recommendation();
      const n = 11, cx = 5, cy = 5;
      ctx.fillStyle = textColor();
      ctx.font = "12px monospace";
      ctx.textAlign = "left";
      ctx.fillText("algorithm: " + algs[alg] + "  |  footprint: " + k + "×" + k, 24, 24);

      drawGrid(40, 50, n, (r, c) => {
        if (alg === 4) return (r === 3 || r === 4) && (c === 6 || c === 7);
        const rad = alg === 0 ? 1 : Math.min(5, Math.floor(k / 12) + 1);
        return Math.abs(r - cy) <= rad && Math.abs(c - cx) <= rad;
      });
      ctx.fillStyle = C.warn();
      ctx.fillRect(40 + cx * 12, 50 + cy * 12, 11, 11);
      ctx.fillStyle = textColor();
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      ctx.fillText("one output pixel asks for this neighborhood", 106, 198);

      const bars = [
        ["direct stencil", k * k],
        ["separable", 2 * k],
        ["FFT-ish", Math.max(20, Math.round(k * Math.log2(k) * 2))],
      ];
      const max = Math.max(...bars.map(b => b[1]));
      bars.forEach((b, i) => {
        const x = 260, y = 62 + i * 44;
        ctx.fillStyle = b[0] === rec[0] || (rec[0] === "stencil tile" && i === 0) || (rec[0] === "window min/max" && i === 0) || (rec[0] === "cuFFT pipeline" && i === 2) || (rec[0] === "separable passes" && i === 1)
          ? C.accent() : "rgba(128,128,128,0.2)";
        ctx.fillRect(x, y, 330 * (b[1] / max), 24);
        ctx.fillStyle = textColor();
        ctx.textAlign = "left";
        ctx.font = "12px monospace";
        ctx.fillText(b[0] + " ~" + b[1] + " work units", x, y - 6);
      });

      out.innerHTML = "shape = <b>" + rec[0] + "</b> — " + rec[2] +
        ". First predict bytes moved and reuse, then write the kernel.";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // execution-model — logical grid/block/thread vs physical SM/warp/lane
  // =======================================================================
  function executionModel(root) {
    shell(root, "Two machines: logical grid → physical SMs",
      "You write the left machine; the runtime schedules it onto the right. Pick a block and trace it: it lands on exactly one SM and splits into warps of 32 lanes. (The block→SM mapping shown is illustrative — the real choice is the runtime's and isn't guaranteed.)");
    const W = 720, H = 280;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let blocks = 8, sms = 4, tpb = 128, sel = 3;

    slider(cs, "blocks in grid", 1, 12, blocks, 1, v => { blocks = v; clampSel(); sSel.max = blocks - 1; draw(); });
    slider(cs, "SMs on the GPU", 1, 6, sms, 1, v => { sms = v; draw(); });
    slider(cs, "threads / block", 32, 256, tpb, 32, v => { tpb = v; draw(); });
    const sSel = slider(cs, "highlight block", 0, blocks - 1, sel, 1, v => { sel = v; draw(); });

    function clampSel() { if (sel > blocks - 1) { sel = blocks - 1; sSel.value = sel; } }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), faint = C.faint(), cool = C.cool();

      ctx.textAlign = "left";
      ctx.font = "bold 12px monospace";
      ctx.fillStyle = ink;
      ctx.fillText("LOGICAL — you write", 10, 16);
      ctx.fillText("PHYSICAL — runtime schedules", 372, 16);

      ctx.strokeStyle = faint;
      ctx.beginPath(); ctx.moveTo(360, 24); ctx.lineTo(360, 200); ctx.stroke();

      // ---- left: the grid of blocks you launched ----
      const cols = 4, rows = Math.ceil(blocks / cols);
      const gx = 10, gy = 30, gw = 330, gh = 165;
      const cw = (gw - (cols - 1) * 8) / cols;
      const chh = Math.min(40, (gh - (rows - 1) * 8) / rows);
      for (let b = 0; b < blocks; b++) {
        const r = Math.floor(b / cols), c = b % cols;
        const x = gx + c * (cw + 8), y = gy + r * (chh + 8);
        ctx.fillStyle = b === sel ? accent : faint;
        ctx.fillRect(x, y, cw, chh);
        ctx.fillStyle = b === sel ? "#0b1500" : ink;
        ctx.font = "10px monospace"; ctx.textAlign = "center";
        ctx.fillText("block " + b, x + cw / 2, y + chh / 2 + 3);
      }

      // ---- right: SM columns, blocks assigned round-robin ----
      const rx = 372, ry = 30, rw = W - rx - 10, rh = 165;
      const smw = (rw - (sms - 1) * 6) / sms;
      for (let s = 0; s < sms; s++) {
        const x = rx + s * (smw + 6);
        ctx.fillStyle = "rgba(128,128,128,0.10)";
        ctx.fillRect(x, ry, smw, rh);
        ctx.strokeStyle = faint; ctx.strokeRect(x, ry, smw, rh);
        ctx.fillStyle = ink; ctx.font = "10px monospace"; ctx.textAlign = "center";
        ctx.fillText("SM " + s, x + smw / 2, ry + rh + 12);
        let slot = 0;
        for (let b = s; b < blocks; b += sms) {
          const by = ry + 6 + slot * 24;
          if (by + 18 > ry + rh) break;
          ctx.fillStyle = b === sel ? accent : "rgba(118,185,0,0.18)";
          ctx.fillRect(x + 4, by, smw - 8, 18);
          ctx.fillStyle = b === sel ? "#0b1500" : ink;
          ctx.font = "9px monospace"; ctx.textAlign = "center";
          ctx.fillText("blk " + b, x + smw / 2, by + 12);
          slot++;
        }
      }

      // ---- bottom: the selected block decomposed into warps ----
      const warps = Math.ceil(tpb / 32);
      const wy = 218, wh = 28, wxx = 10, ww = W - 20;
      ctx.fillStyle = ink; ctx.font = "11px monospace"; ctx.textAlign = "left";
      ctx.fillText("block " + sel + " → " + warps + " warps of 32 lanes:", wxx, wy - 6);
      const segw = (ww - (warps - 1) * 4) / warps;
      for (let w = 0; w < warps; w++) {
        const x = wxx + w * (segw + 4);
        ctx.fillStyle = "rgba(58,142,224,0.30)";
        ctx.fillRect(x, wy, segw, wh);
        ctx.strokeStyle = cool; ctx.strokeRect(x, wy, segw, wh);
        ctx.fillStyle = ink; ctx.font = "10px monospace"; ctx.textAlign = "center";
        if (segw > 28) ctx.fillText("warp " + w, x + segw / 2, wy + wh / 2 + 3);
      }

      out.innerHTML = "block <b>" + sel + "</b> → <b>SM " + (sel % sms) +
        "</b> (one SM, never split) — decomposes into <b>" + warps +
        "</b> warps × 32 = <b>" + (warps * 32) + "</b> lanes. You chose " + blocks +
        " blocks and " + tpb + " threads/block; the runtime chose the SM.";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // block-to-sm — a block lands wholly on one SM; its resources are per-block
  // =======================================================================
  function blockToSm(root) {
    shell(root, "A block never spans two SMs",
      "The runtime drops each block onto exactly one SM for its whole life. That physical fact is why __shared__ and __syncthreads() are per-block — and why there is no __syncblocks(). Pick a block and see where it lands.");
    const W = 720, H = 230;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let blocks = 8, sms = 4, sel = 2;

    slider(cs, "blocks in grid", 1, 16, blocks, 1, v => { blocks = v; clampSel(); draw(); });
    slider(cs, "SMs on the GPU", 1, 6, sms, 1, v => { sms = v; draw(); });
    const sSel = slider(cs, "inspect block", 0, blocks - 1, sel, 1, v => { sel = v; draw(); });

    function clampSel() {
      sSel.max = blocks - 1;
      if (sel > blocks - 1) { sel = blocks - 1; sSel.value = sel; }
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), faint = C.faint(), cool = C.cool();
      const home = sel % sms;   // illustrative round-robin home SM

      // ---- the SM strip; blocks stacked into their home SM ----
      ctx.fillStyle = ink; ctx.font = "bold 12px monospace"; ctx.textAlign = "left";
      ctx.fillText(sms + " SMs — each block occupies exactly one column", 10, 16);
      const ry = 28, rh = 150, rx = 10, rw = W - 20;
      const smw = (rw - (sms - 1) * 8) / sms;
      for (let s = 0; s < sms; s++) {
        const x = rx + s * (smw + 8);
        ctx.fillStyle = s === home ? "rgba(118,185,0,0.10)" : "rgba(128,128,128,0.08)";
        ctx.fillRect(x, ry, smw, rh);
        ctx.strokeStyle = s === home ? accent : faint;
        ctx.lineWidth = s === home ? 2 : 1;
        ctx.strokeRect(x, ry, smw, rh);
        ctx.fillStyle = ink; ctx.font = "11px monospace"; ctx.textAlign = "center";
        ctx.fillText("SM " + s, x + smw / 2, ry + rh + 16);
        let slot = 0;
        for (let b = s; b < blocks; b += sms) {
          const by = ry + 8 + slot * 26;
          if (by + 20 > ry + rh) break;
          ctx.fillStyle = b === sel ? accent : "rgba(118,185,0,0.18)";
          ctx.fillRect(x + 6, by, smw - 12, 20);
          ctx.fillStyle = b === sel ? "#0b1500" : ink;
          ctx.font = "10px monospace";
          ctx.fillText("block " + b + (b === sel ? "  ·shared·" : ""), x + smw / 2, by + 14);
          slot++;
        }
      }

      out.innerHTML = "block <b>" + sel + "</b> → <b>SM " + home +
        "</b> (one SM, never split). Its <b>__shared__</b> is a private allocation on that SM; " +
        "<b>__syncthreads()</b> waits only block " + sel + "'s threads. A different block is a different " +
        "allocation even on the same SM — so crossing blocks needs a <span class='warn'>kernel boundary</span>, " +
        "not a barrier. (Mapping is illustrative; the real choice is the runtime's.)";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // warp-lanes — threadIdx.x → warp id (>>5) and lane id (&31)
  // =======================================================================
  function warpLanes(root) {
    shell(root, "threadIdx.x → warp id and lane id",
      "A block is sliced into warps of 32 consecutive threads. The split is pure bit math: warp = threadIdx.x >> 5, lane = threadIdx.x & 31. Drag the thread and watch which warp and lane it falls in.");
    const W = 720, H = 170;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let dim = 128, tid = 70;

    slider(cs, "blockDim.x", 32, 256, dim, 32, v => { dim = v; if (tid > dim - 1) { tid = dim - 1; sT.value = tid; } sT.max = dim - 1; draw(); });
    const sT = slider(cs, "threadIdx.x", 0, dim - 1, tid, 1, v => { tid = v; draw(); });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), cool = C.cool(), faint = C.faint();
      const warps = Math.ceil(dim / 32);
      const selWarp = tid >> 5, selLane = tid & 31;

      // ---- warp strip: each warp a band of 32 lanes ----
      ctx.fillStyle = ink; ctx.font = "11px monospace"; ctx.textAlign = "left";
      ctx.fillText(dim + " threads → " + warps + " warps of 32 lanes", 10, 16);
      const gx = 10, gy = 28, gw = W - 20, lane = gw / 32, wh = 22, gap = 6;
      for (let w = 0; w < warps; w++) {
        const y = gy + w * (wh + gap);
        if (y + wh > H - 24) break;
        for (let l = 0; l < 32; l++) {
          const here = (w === selWarp && l === selLane);
          ctx.fillStyle = here ? accent : (w === selWarp ? "rgba(58,142,224,0.25)" : faint);
          ctx.fillRect(gx + l * lane, y, lane - 1, wh);
        }
        ctx.strokeStyle = w === selWarp ? cool : faint;
        ctx.lineWidth = w === selWarp ? 2 : 1;
        ctx.strokeRect(gx, y, gw, wh);
        ctx.fillStyle = ink; ctx.font = "9px monospace"; ctx.textAlign = "left";
        ctx.fillText("warp " + w, gx + 3, y - 1);
      }

      const bits = (tid >>> 0).toString(2).padStart(8, "0");
      const hi = bits.slice(0, bits.length - 5), lo = bits.slice(bits.length - 5);
      out.innerHTML = "threadIdx.x = <b>" + tid + "</b> = 0b" + hi + "<u>" + lo + "</u> → " +
        "warp = " + tid + " >> 5 = <b>" + selWarp + "</b>, lane = " + tid + " & 31 = <b>" + selLane + "</b>. " +
        "The high bits pick the warp, the low 5 bits pick the lane — consecutive threads are consecutive lanes, which is exactly why mapping threadIdx.x to the contiguous axis coalesces.";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // sync-mask — toggle lane predicates, watch the __ballot_sync mask change
  // =======================================================================
  function syncMask(root) {
    shell(root, "Masks: the set of participating lanes",
      "A mask is one bit per lane — bit i set means 'lane i is in this _sync op'. Click lanes to toggle their predicate and watch the 32-bit mask __ballot_sync would produce.");
    const W = 720, H = 150;
    const { ctx } = canvas(root, W, H);
    const cv = root.querySelector("canvas");
    const cs = controls(root);
    const out = readout(root);
    const active = Array.from({ length: 32 }, () => true);

    button(cs, "all lanes", () => { active.fill(true); draw(); });
    button(cs, "even lanes", () => { for (let i = 0; i < 32; i++) active[i] = (i % 2 === 0); draw(); });
    button(cs, "lower half", () => { for (let i = 0; i < 32; i++) active[i] = (i < 16); draw(); });

    const gx = 10, gy = 40, lane = (W - 20) / 32, lh = 40;
    cv.addEventListener("click", e => {
      const lx = (e.offsetX / cv.clientWidth) * W;
      const ly = (e.offsetY / cv.clientHeight) * H;
      const i = Math.floor((lx - gx) / lane);
      if (i >= 0 && i < 32 && ly >= gy && ly <= gy + lh) { active[i] = !active[i]; draw(); }
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), faint = C.faint();
      ctx.fillStyle = ink; ctx.font = "11px monospace"; ctx.textAlign = "left";
      ctx.fillText("32 lanes — click to toggle each lane's predicate", 10, 22);

      let mask = 0, count = 0;
      for (let i = 0; i < 32; i++) {
        if (active[i]) { mask |= (1 << i); count++; }
        ctx.fillStyle = active[i] ? accent : faint;
        ctx.fillRect(gx + i * lane, gy, lane - 1.5, lh);
        if (lane > 14) {
          ctx.fillStyle = active[i] ? "#0b1500" : ink;
          ctx.font = "9px monospace"; ctx.textAlign = "center";
          ctx.fillText(i, gx + i * lane + lane / 2, gy + lh / 2 + 3);
        }
      }
      const hex = "0x" + (mask >>> 0).toString(16).padStart(8, "0");
      ctx.fillStyle = ink; ctx.font = "13px monospace"; ctx.textAlign = "left";
      ctx.fillText("mask = " + hex, 10, gy + lh + 28);

      out.innerHTML = "__ballot_sync(0xffffffff, pred) = <b>" + hex + "</b> — <b>" + count +
        "</b> participating lane" + (count === 1 ? "" : "s") + ". " + (count === 32
          ? "<span class='good'>Whole warp converged → 0xffffffff is honest.</span>"
          : "<span class='warn'>Only some lanes — passing 0xffffffff here would name absent lanes and read garbage. Derive the mask before the branch.</span>");
    }
    draw();
    return draw;
  }

  // =======================================================================
  // cross-warp-race — __syncwarp() in warp 0 does nothing for warp 1
  // =======================================================================
  function crossWarpRace(root) {
    shell(root, "Warp-local sync is not block-wide",
      "Warp 0 writes shared memory and calls a barrier; warp 1 reads it. Slide warp 1's head start and toggle the barrier. Predict: which barrier actually makes warp 1 wait for warp 0's write?");
    const W = 720, H = 180;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let skew = 4, blockWide = false;

    slider(cs, "warp 1 head start", 0, 10, skew, 1, v => { skew = v; draw(); });
    const tg = button(cs, "barrier: __syncwarp()", () => {
      blockWide = !blockWide;
      tg.textContent = blockWide ? "barrier: __syncthreads()" : "barrier: __syncwarp()";
      tg.classList.toggle("dojo-btn--warn", false);
      draw();
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), warn = C.warn(), cool = C.cool(), faint = C.faint();
      const px = 42, t0 = 120;
      // timeline: warp 0 writes at writeT; warp 1 reads at readT.
      const writeT = 5;                 // warp 0 reaches the write+barrier here
      const readBase = 3 - skew * 0.4;  // warp 1's natural read time, earlier as skew grows
      // __syncthreads() forces warp 1 to wait until every warp passes the barrier.
      const readT = blockWide ? Math.max(readBase, writeT + 1) : readBase;
      const race = readT < writeT;

      function lane(y, label, col) {
        ctx.strokeStyle = faint; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(t0, y); ctx.lineTo(W - 10, y); ctx.stroke();
        ctx.fillStyle = ink; ctx.font = "11px monospace"; ctx.textAlign = "right";
        ctx.fillText(label, t0 - 8, y + 4);
      }
      lane(60, "warp 0", accent);
      lane(120, "warp 1", cool);

      // warp 0 write/barrier marker
      const wx = t0 + writeT * px;
      ctx.fillStyle = accent; ctx.fillRect(wx - 5, 50, 10, 20);
      ctx.fillStyle = ink; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText("write smem", wx, 44);
      // barrier line
      ctx.strokeStyle = blockWide ? accent : warn;
      ctx.setLineDash([4, 3]); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(wx, 40); ctx.lineTo(wx, blockWide ? 130 : 75); ctx.stroke();
      ctx.setLineDash([]);

      // warp 1 read marker
      const rx = t0 + readT * px;
      ctx.fillStyle = race ? warn : cool; ctx.fillRect(rx - 5, 110, 10, 20);
      ctx.fillStyle = ink; ctx.textAlign = "center";
      ctx.fillText("read smem", rx, 146);

      out.innerHTML = "barrier = <b>" + (blockWide ? "__syncthreads()" : "__syncwarp()") + "</b> — " +
        (blockWide
          ? "<span class='good'>warp 1 is forced past the write before it reads. Safe: the barrier spans the whole block.</span>"
          : (race
            ? "<span class='warn'>RACE: warp 1 read before warp 0 wrote. __syncwarp() in warp 0 never made warp 1 wait — it orders one warp only.</span>"
            : "warp 1 happens to read after the write here, but that's luck, not ordering. __syncwarp() gives warp 1 no guarantee — climb to __syncthreads()."));
    }
    draw();
    return draw;
  }

  // =======================================================================
  // sync-scope — the decision ladder: smallest scope that makes the reader wait
  // =======================================================================
  function syncScope(root) {
    shell(root, "The synchronization ladder",
      "One question picks every primitive: who must see whose writes before continuing? Choose where the producer and consumer sit, and the smallest correct scope lights up. Climb only as high as the answer forces.");
    const W = 720, H = 280;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    const rungs = [
      ["__syncwarp(mask)", "named lanes in one warp"],
      ["__syncthreads()", "all threads in one block"],
      ["tile.sync()", "threads in a programmer-defined tile"],
      ["grid.sync()", "all blocks — cooperative launch only"],
      ["kernel boundary", "all blocks of the previous kernel"],
      ["event + StreamWaitEvent", "one queue waits for a point in another"],
      ["cudaStreamSynchronize", "the host thread waits"],
    ];
    // each scenario maps to the smallest correct rung index
    const scen = [
      ["lanes in one warp", 0],
      ["warps in one block", 1],
      ["a sub-block tile", 2],
      ["blocks, no kernel end", 3],
      ["block → block (relaunch)", 4],
      ["stream → stream", 5],
      ["device → host (CPU reads)", 6],
    ];
    let sel = 1;
    slider(cs, "producer → consumer are…", 0, scen.length - 1, sel, 1, v => { sel = v; draw(); });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), faint = C.faint();
      const need = scen[sel][1];
      ctx.fillStyle = ink; ctx.font = "12px monospace"; ctx.textAlign = "left";
      ctx.fillText("scenario: " + scen[sel][0], 10, 18);

      const x = 14, w = W - 28, y0 = 30, rh = 30, gap = 4;
      rungs.forEach((r, i) => {
        const y = y0 + i * (rh + gap);
        const on = i === need;
        const tooSmall = i < need;
        ctx.fillStyle = on ? accent : (tooSmall ? "rgba(224,99,58,0.18)" : "rgba(128,128,128,0.12)");
        ctx.fillRect(x, y, w, rh);
        ctx.strokeStyle = on ? accent : faint; ctx.lineWidth = on ? 2 : 1;
        ctx.strokeRect(x, y, w, rh);
        ctx.fillStyle = on ? "#0b1500" : ink;
        ctx.font = (on ? "bold " : "") + "12px monospace"; ctx.textAlign = "left";
        ctx.fillText(r[0], x + 10, y + rh / 2 + 4);
        ctx.font = "11px monospace"; ctx.textAlign = "right";
        ctx.fillStyle = on ? "#0b1500" : ink;
        ctx.fillText(r[1], x + w - 10, y + rh / 2 + 4);
      });

      out.innerHTML = "smallest correct scope: <b>" + rungs[need][0] + "</b> — " + rungs[need][1] +
        ". <span class='warn'>Red rungs below are too small to order this handoff</span>; rungs above work but over-serialize. Pick the smallest that makes the right reader wait.";
    }
    draw();
    return draw;
  }

  // =======================================================================
  // stream-event-dependency — two stream queues linked by an event
  // =======================================================================
  function streamEventDependency(root) {
    shell(root, "Ordering two streams with an event",
      "streamB's kernel must not start until streamA's kernel finishes — without stalling the CPU. cudaEventRecord marks a point in A; cudaStreamWaitEvent makes B wait for it. Drag A's length and watch B's start slide.");
    const W = 720, H = 200;
    const { ctx } = canvas(root, W, H);
    const cs = controls(root);
    const out = readout(root);
    let aLen = 6, bQueued = 2, linked = true;

    slider(cs, "kernelA length", 2, 12, aLen, 1, v => { aLen = v; draw(); });
    slider(cs, "B's own work before wait", 0, 6, bQueued, 1, v => { bQueued = v; draw(); });
    const tg = button(cs, "dependency: ON (event)", () => {
      linked = !linked;
      tg.textContent = linked ? "dependency: ON (event)" : "dependency: OFF (race)";
      tg.classList.toggle("dojo-btn--warn", !linked);
      draw();
    });

    function draw() {
      ctx.clearRect(0, 0, W, H);
      const ink = textColor(), accent = C.accent(), cool = C.cool(), warn = C.warn(), faint = C.faint();
      const t0 = 90, px = 44;
      function lane(y, label) {
        ctx.strokeStyle = faint; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(t0, y + 12); ctx.lineTo(W - 10, y + 12); ctx.stroke();
        ctx.fillStyle = ink; ctx.font = "11px monospace"; ctx.textAlign = "right";
        ctx.fillText(label, t0 - 8, y + 16);
      }
      function bar(x, y, w, label, col) {
        ctx.fillStyle = col; ctx.fillRect(x, y, w, 24);
        ctx.fillStyle = "#0b1500"; ctx.font = "10px monospace"; ctx.textAlign = "center";
        if (w > 16) ctx.fillText(label, x + w / 2, y + 16);
      }
      // streamA: kernelA then record(done)
      lane(40, "streamA");
      bar(t0, 40, aLen * px, "kernelA", accent);
      const recX = t0 + aLen * px;
      ctx.fillStyle = warn; ctx.beginPath(); ctx.arc(recX, 52, 6, 0, 7); ctx.fill();
      ctx.fillStyle = ink; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText("record(done)", recX, 34);

      // streamB: own work, then (maybe) wait for the event, then kernelB
      lane(110, "streamB");
      const bWorkEnd = t0 + bQueued * px;
      bar(t0, 110, bQueued * px, "B work", cool);
      const startB = linked ? Math.max(bWorkEnd, recX) : bWorkEnd;
      const waited = linked && startB > bWorkEnd;
      if (waited) {
        ctx.strokeStyle = warn; ctx.setLineDash([3, 3]); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(bWorkEnd, 122); ctx.lineTo(startB, 122); ctx.stroke();
        ctx.setLineDash([]);
        // arrow from record dot down to B's start
        ctx.strokeStyle = warn; ctx.beginPath(); ctx.moveTo(recX, 58); ctx.lineTo(startB, 110); ctx.stroke();
      }
      bar(startB, 110, 4 * px, "kernelB", linked ? cool : warn);

      const safe = startB >= recX;
      out.innerHTML = linked
        ? (waited
          ? "<span class='good'>B waited: kernelB starts at A's end. The event ordered the queues; the CPU never blocked.</span>"
          : "<span class='good'>B's own work already ran past A's end, so the wait is free here — but the event still guarantees the ordering.</span>")
        : (safe
          ? "No event, but B happened to start after A finished — luck, not ordering. Remove the wait and timing alone decides correctness."
          : "<span class='warn'>No event: kernelB starts before kernelA finished. Separate streams are independent queues — without cudaStreamWaitEvent nothing orders them.</span>");
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
    "sm-scheduler": smScheduler,
    "library-choice": libraryChoice,
    "imaging-pattern": imagingPattern,
    "execution-model": executionModel,
    "block-to-sm": blockToSm,
    "warp-lanes": warpLanes,
    "sync-scope": syncScope,
    "sync-mask": syncMask,
    "cross-warp-race": crossWarpRace,
    "stream-event-dependency": streamEventDependency,
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
