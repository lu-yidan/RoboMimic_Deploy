#!/usr/bin/env python3
"""Sensor dashboard: target + ball positions from all sensors in one browser tab.

Subscribes to:
  rt/target_state       — AprilTag target (chest camera)
  rt/cam_ball_state     — camera ball estimate
  rt/lidar_ball_state   — lidar ball estimate
  rt/ball_state         — fused authoritative ball

Serves a single-page web app at http://<host>:<port>/ that shows:
  - Top-down canvas with all four positions drawn simultaneously (no video, no flicker)
  - Side panel with numeric readout per sensor
  - Server-Sent Events push at 20 Hz — browser DOM updates only changed fields

Usage:
    python onboard/perception/debug/sensor_dashboard.py
    python onboard/perception/debug/sensor_dashboard.py --port 8091
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer

sys.path.append(str(Path(__file__).parent.parent.parent.parent.absolute()))

from common.ball_state_dds import (  # noqa: E402
    SOURCE_CAM, SOURCE_LIDAR, SOURCE_NONE,
    BallState, BallStateSubscriber,
)
from common.target_state_dds import (  # noqa: E402
    TARGET_STATE_TOPIC, TARGET_STALE_MS,
    TargetState, TargetStateSubscriber,
)

BALL_STALE_MS = 400


# ── HTML ─────────────────────────────────────────────────────────────────────

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sensor Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg:      #0f172a;
      --panel:   #111827;
      --panel2:  #1e293b;
      --text:    #e5e7eb;
      --muted:   #94a3b8;
      --grid:    #334155;
      --target:  #fbbf24;
      --cam:     #38bdf8;
      --lidar:   #fb923c;
      --fused:   #4ade80;
      --stale:   #475569;
      --axis-x:  #38bdf8;
      --axis-y:  #f472b6;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh;
      background: radial-gradient(circle at top, #1e293b, var(--bg) 55%);
      color: var(--text);
      font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    }
    header {
      padding: 14px 24px 6px;
      display: flex; align-items: baseline; justify-content: space-between;
      flex-wrap: wrap; gap: 10px;
    }
    h1 { margin: 0; font-size: 20px; font-weight: 650; }
    .sub { color: var(--muted); font-size: 12px; }
    #conn { color: var(--muted); font-size: 13px; }
    .layout {
      display: grid;
      grid-template-columns: 1fr 300px;
      gap: 14px; padding: 8px 24px 24px;
    }
    .card {
      background: color-mix(in srgb, var(--panel) 92%, transparent);
      border: 1px solid #334155; border-radius: 16px;
      box-shadow: 0 16px 40px rgba(0,0,0,.24); overflow: hidden;
    }
    .canvasWrap { padding: 12px; }
    canvas {
      width: 100%; height: min(72vh, 700px); display: block;
      background: #020617; border-radius: 10px;
    }
    .side { padding: 14px; display: grid; gap: 10px; align-content: start; }
    .legend { display: grid; gap: 6px; font-size: 13px; }
    .legendRow { display: flex; align-items: center; gap: 8px; color: var(--muted); }
    .dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .sensors { display: grid; gap: 8px; }
    .sbox {
      background: rgba(15,23,42,.74); border: 1px solid #253247;
      border-radius: 12px; padding: 10px 12px;
    }
    .sbox .title {
      font-size: 12px; font-weight: 600; text-transform: uppercase;
      letter-spacing: .05em; display: flex; align-items: center; gap: 6px;
      margin-bottom: 7px;
    }
    .kv {
      display: grid; grid-template-columns: 1fr auto;
      gap: 4px 12px; font-variant-numeric: tabular-nums;
    }
    .kv .k { color: var(--muted); font-size: 12px; }
    .kv .v { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
    .badge {
      font-size: 10px; padding: 1px 5px; border-radius: 4px;
      background: var(--stale); color: #fff; font-weight: 600;
    }
    .badge.valid  { background: #15803d; }
    .badge.coast  { background: #92400e; }
    .badge.stale  { background: #475569; }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; padding: 8px 12px 20px; }
      canvas  { height: 56vh; }
    }
  </style>
</head>
<body>
<header>
  <div><h1>Sensor Dashboard</h1>
    <div class="sub">target + ball — all sensors, pelvis frame: +X fwd, +Y left, +Z up</div>
  </div>
  <span id="conn">connecting…</span>
</header>
<main class="layout">
  <section class="card canvasWrap"><canvas id="c"></canvas></section>
  <aside class="card side">
    <div class="legend">
      <div class="legendRow"><span class="dot" style="background:var(--target)"></span>target (AprilTag)</div>
      <div class="legendRow"><span class="dot" style="background:var(--cam)"></span>cam ball</div>
      <div class="legendRow"><span class="dot" style="background:var(--lidar)"></span>lidar ball</div>
      <div class="legendRow"><span class="dot" style="background:var(--fused)"></span>fused ball</div>
    </div>
    <div class="sensors">
      <div class="sbox" id="box-target">
        <div class="title">
          <span class="dot" style="background:var(--target)"></span>
          Target
          <span class="badge stale" id="badge-target">wait</span>
        </div>
        <div class="kv" id="kv-target"></div>
      </div>
      <div class="sbox" id="box-cam">
        <div class="title">
          <span class="dot" style="background:var(--cam)"></span>
          Cam ball
          <span class="badge stale" id="badge-cam">wait</span>
        </div>
        <div class="kv" id="kv-cam"></div>
      </div>
      <div class="sbox" id="box-lidar">
        <div class="title">
          <span class="dot" style="background:var(--lidar)"></span>
          Lidar ball
          <span class="badge stale" id="badge-lidar">wait</span>
        </div>
        <div class="kv" id="kv-lidar"></div>
      </div>
      <div class="sbox" id="box-fused">
        <div class="title">
          <span class="dot" style="background:var(--fused)"></span>
          Fused ball
          <span class="badge stale" id="badge-fused">wait</span>
        </div>
        <div class="kv" id="kv-fused"></div>
      </div>
    </div>
  </aside>
</main>
<script>
const RANGE_M = __RANGE_M__;
const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d");

function cssVar(n) {
  return getComputedStyle(document.documentElement).getPropertyValue(n).trim();
}

function fmt(v, d = 3) {
  return Number.isFinite(v) ? (v >= 0 ? "+" : "") + v.toFixed(d) + " m" : "--";
}
function fmtPlain(v, d = 2) {
  return Number.isFinite(v) ? v.toFixed(d) + " m" : "--";
}

// ── Canvas ──────────────────────────────────────────────────────────────────

function resize() {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(r.width * dpr), h = Math.round(r.height * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w; canvas.height = h;
  }
}

function drawGrid(cx, cy, scale, w, h) {
  ctx.font = "11px ui-monospace, monospace";
  for (let m = -RANGE_M; m <= RANGE_M + 1e-6; m += 0.5) {
    const major = Math.abs(m % 1) < 1e-6;
    ctx.strokeStyle = cssVar("--grid");
    ctx.lineWidth = major ? 1.2 : 0.8;
    ctx.globalAlpha = major ? 0.5 : 0.15;
    const px = cx - m * scale;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    const py = cy - m * scale;
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
    if (major && Math.abs(m) > 1e-6) {
      ctx.globalAlpha = 0.7;
      ctx.fillStyle = cssVar("--muted");
      ctx.textAlign = "center";
      ctx.fillText((m > 0 ? "+" : "") + m.toFixed(0) + "m", px, cy + 16);
      ctx.textAlign = "left";
      ctx.fillText((m > 0 ? "+" : "") + m.toFixed(0) + "m", cx + 8, py);
    }
  }
  ctx.globalAlpha = 1;
}

function arrow(x1, y1, x2, y2, col, label) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 9 * Math.cos(a - 0.5), y2 - 9 * Math.sin(a - 0.5));
  ctx.lineTo(x2 - 9 * Math.cos(a + 0.5), y2 - 9 * Math.sin(a + 0.5));
  ctx.closePath(); ctx.fill();
  ctx.font = "12px ui-monospace, monospace";
  ctx.fillText(label, x2 + 5, y2 - 5);
}

function robot(cx, cy) {
  ctx.save(); ctx.translate(cx, cy);
  ctx.fillStyle = "#e5e7eb"; ctx.strokeStyle = "#94a3b8"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.roundRect(-16, -22, 32, 44, 7);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#0f172a";
  ctx.beginPath(); ctx.arc(0, -14, 4, 0, Math.PI * 2); ctx.fill();
  ctx.restore();
}

function ballDot(px, py, col, r, alpha, label, glow) {
  ctx.save();
  if (glow) {
    ctx.shadowColor = col; ctx.shadowBlur = 12;
  }
  ctx.globalAlpha = alpha;
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI * 2); ctx.fill();
  if (glow) {
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
    ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.shadowBlur = 0;
  if (label) {
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillStyle = col;
    ctx.fillText(label, px + r + 4, py - r - 2);
  }
  ctx.restore();
}

function targetMark(px, py, col, size, alpha, label) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = col; ctx.lineWidth = 2;
  // Diamond shape
  ctx.beginPath();
  ctx.moveTo(px, py - size);
  ctx.lineTo(px + size, py);
  ctx.lineTo(px, py + size);
  ctx.lineTo(px - size, py);
  ctx.closePath();
  ctx.fillStyle = col + "44";
  ctx.fill(); ctx.stroke();
  // crosshair
  ctx.beginPath(); ctx.moveTo(px - size * 0.5, py); ctx.lineTo(px + size * 0.5, py); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(px, py - size * 0.5); ctx.lineTo(px, py + size * 0.5); ctx.stroke();
  ctx.globalAlpha = 1;
  if (label) {
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillStyle = col;
    ctx.fillText(label, px + size + 4, py - size);
  }
  ctx.restore();
}

// ── State ───────────────────────────────────────────────────────────────────

let latest = null;

function draw() {
  resize();
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr, h = canvas.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const pad = 44;
  const cx = w / 2, cy = h / 2;
  const scale = Math.min((w - 2 * pad), (h - 2 * pad)) / (RANGE_M * 2);

  drawGrid(cx, cy, scale, w, h);
  arrow(cx, cy, cx, cy - scale * 0.8, cssVar("--axis-x"), "+X");
  arrow(cx, cy, cx - scale * 0.8, cy, cssVar("--axis-y"), "+Y");
  robot(cx, cy);

  if (!latest) { requestAnimationFrame(draw); return; }

  // project(x, y) → [canvasX, canvasY]  (X=forward=up, Y=left=left on screen)
  const proj = (x, y) => [cx - y * scale, cy - x * scale];

  const s = latest;
  // fused ball (largest, drawn first so others appear on top)
  if (s.fused && s.fused.has_sample && s.fused.fresh) {
    const [px, py] = proj(s.fused.x, s.fused.y);
    const col = s.fused.valid ? cssVar("--fused") : cssVar("--stale");
    ballDot(px, py, col, s.fused.valid ? 14 : 9, s.fused.valid ? 0.55 : 0.25,
            null, false);
  }
  // lidar ball
  if (s.lidar && s.lidar.has_sample && s.lidar.fresh && s.lidar.valid) {
    const [px, py] = proj(s.lidar.x, s.lidar.y);
    ballDot(px, py, cssVar("--lidar"), 9, 0.85, "L", true);
  }
  // cam ball
  if (s.cam && s.cam.has_sample && s.cam.fresh && s.cam.valid) {
    const [px, py] = proj(s.cam.x, s.cam.y);
    ballDot(px, py, cssVar("--cam"), 9, 0.85, "C", true);
  }
  // target (drawn last = on top)
  if (s.target && s.target.has_sample && s.target.fresh && s.target.valid) {
    const [px, py] = proj(s.target.x, s.target.y);
    targetMark(px, py, cssVar("--target"), 13, 1.0, "T");
  }

  requestAnimationFrame(draw);
}

// ── Panel update ─────────────────────────────────────────────────────────────

function badge(id, state) {
  const el = document.getElementById("badge-" + id);
  if (!el) return;
  if (!state || !state.has_sample) {
    el.textContent = "wait"; el.className = "badge stale"; return;
  }
  if (!state.fresh) { el.textContent = "stale"; el.className = "badge stale"; return; }
  if (state.valid)  { el.textContent = "valid"; el.className = "badge valid"; return; }
  if (state.coast)  { el.textContent = "coast"; el.className = "badge coast"; return; }
  el.textContent = "inval"; el.className = "badge stale";
}

function kvHtml(rows) {
  return rows.map(([k, v]) =>
    `<div class="k">${k}</div><div class="v">${v}</div>`
  ).join("");
}

function fmtAge(ms) {
  if (!Number.isFinite(ms)) return "--";
  return ms < 1000 ? ms.toFixed(0) + " ms" : (ms / 1000).toFixed(1) + " s";
}

function updatePanel(s) {
  if (!s) return;

  // target
  badge("target", s.target);
  const t = s.target;
  document.getElementById("kv-target").innerHTML = kvHtml([
    ["x fwd", t && t.has_sample ? fmt(t.x) : "--"],
    ["y left", t && t.has_sample ? fmt(t.y) : "--"],
    ["z up", t && t.has_sample ? fmt(t.z) : "--"],
    ["dist xy", t && t.has_sample ? fmtPlain(t.dist_xy) : "--"],
    ["conf", t && t.has_sample ? (t.confidence * 100).toFixed(0) + "%" : "--"],
    ["age", t && t.has_sample ? fmtAge(t.age_ms) : "--"],
  ]);

  // cam ball
  badge("cam", s.cam);
  const c = s.cam;
  document.getElementById("kv-cam").innerHTML = kvHtml([
    ["x fwd", c && c.has_sample ? fmt(c.x) : "--"],
    ["y left", c && c.has_sample ? fmt(c.y) : "--"],
    ["z up", c && c.has_sample ? fmt(c.z) : "--"],
    ["dist 3d", c && c.has_sample ? fmtPlain(c.dist_3d) : "--"],
    ["age", c && c.has_sample ? fmtAge(c.age_ms) : "--"],
  ]);

  // lidar ball
  badge("lidar", s.lidar);
  const l = s.lidar;
  document.getElementById("kv-lidar").innerHTML = kvHtml([
    ["x fwd", l && l.has_sample ? fmt(l.x) : "--"],
    ["y left", l && l.has_sample ? fmt(l.y) : "--"],
    ["z up", l && l.has_sample ? fmt(l.z) : "--"],
    ["dist 3d", l && l.has_sample ? fmtPlain(l.dist_3d) : "--"],
    ["age", l && l.has_sample ? fmtAge(l.age_ms) : "--"],
  ]);

  // fused
  badge("fused", s.fused);
  const f = s.fused;
  const src = {0:"none", 1:"cam", 2:"lidar"};
  document.getElementById("kv-fused").innerHTML = kvHtml([
    ["x fwd", f && f.has_sample ? fmt(f.x) : "--"],
    ["y left", f && f.has_sample ? fmt(f.y) : "--"],
    ["z up", f && f.has_sample ? fmt(f.z) : "--"],
    ["dist 3d", f && f.has_sample ? fmtPlain(f.dist_3d) : "--"],
    ["source", f && f.has_sample ? (src[f.source] || "?") : "--"],
    ["age", f && f.has_sample ? fmtAge(f.age_ms) : "--"],
  ]);
}

// ── SSE ──────────────────────────────────────────────────────────────────────

window.addEventListener("resize", resize);
requestAnimationFrame(draw);

const es = new EventSource("/events");
es.onopen  = () => { document.getElementById("conn").textContent = "connected"; };
es.onerror = () => { document.getElementById("conn").textContent = "reconnecting…"; };
es.onmessage = (e) => {
  latest = JSON.parse(e.data);
  updatePanel(latest);
};
</script>
</body>
</html>
"""


# ── Data layer ────────────────────────────────────────────────────────────────

def _ball_snapshot(last, received_at, stale_ms) -> dict:
    if last is None or received_at is None:
        return {"has_sample": False, "fresh": False, "valid": False,
                "coast": False, "age_ms": None,
                "x": 0.0, "y": 0.0, "z": 0.0,
                "dist_xy": 0.0, "dist_3d": 0.0, "source": 0}
    age_ms = (time.monotonic() - received_at) * 1000.0
    fresh = age_ms <= stale_ms
    x, y, z = float(last.x), float(last.y), float(last.z)
    valid = bool(last.valid) and fresh
    coast = (not bool(last.valid)) and bool(last.source) and fresh
    return {
        "has_sample": True, "fresh": bool(fresh), "valid": valid,
        "coast": coast, "age_ms": age_ms,
        "x": x, "y": y, "z": z,
        "dist_xy": math.hypot(x, y),
        "dist_3d": math.sqrt(x*x + y*y + z*z),
        "source": int(last.source),
    }


def _target_snapshot(last, received_at, stale_ms) -> dict:
    if last is None or received_at is None:
        return {"has_sample": False, "fresh": False, "valid": False,
                "age_ms": None, "x": 0.0, "y": 0.0, "z": 0.0,
                "dist_xy": 0.0, "confidence": 0.0}
    age_ms = (time.monotonic() - received_at) * 1000.0
    fresh = age_ms <= stale_ms
    x, y, z = float(last.x), float(last.y), float(last.z)
    return {
        "has_sample": True, "fresh": bool(fresh),
        "valid": bool(last.valid) and fresh,
        "age_ms": age_ms, "x": x, "y": y, "z": z,
        "dist_xy": math.hypot(x, y),
        "confidence": float(last.confidence),
    }


class SensorStore:
    def __init__(self, stale_ms_ball: float, stale_ms_target: float):
        self._lock = threading.Lock()
        self._stale_ball   = stale_ms_ball
        self._stale_target = stale_ms_target
        self._target = None; self._target_at = None
        self._cam    = None; self._cam_at    = None
        self._lidar  = None; self._lidar_at  = None
        self._fused  = None; self._fused_at  = None

    def update_target(self, s: TargetState):
        with self._lock:
            self._target = s; self._target_at = time.monotonic()

    def update_cam(self, s: BallState):
        with self._lock:
            self._cam = s; self._cam_at = time.monotonic()

    def update_lidar(self, s: BallState):
        with self._lock:
            self._lidar = s; self._lidar_at = time.monotonic()

    def update_fused(self, s: BallState):
        with self._lock:
            self._fused = s; self._fused_at = time.monotonic()

    def snapshot(self) -> dict:
        with self._lock:
            t  = _target_snapshot(self._target, self._target_at, self._stale_target)
            c  = _ball_snapshot(self._cam,   self._cam_at,   self._stale_ball)
            l  = _ball_snapshot(self._lidar, self._lidar_at, self._stale_ball)
            f  = _ball_snapshot(self._fused, self._fused_at, self._stale_ball)
        return {"target": t, "cam": c, "lidar": l, "fused": f}


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _make_handler(store: SensorStore, html: bytes, interval_s: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body: bytes, ctype: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(html, "text/html; charset=utf-8"); return
            if self.path == "/state":
                self._send(json.dumps(store.snapshot()).encode(), "application/json"); return
            if self.path == "/health":
                self._send(b"ok\n", "text/plain"); return
            if self.path == "/events":
                self._sse(); return
            self.send_error(HTTPStatus.NOT_FOUND)

        def _sse(self):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    payload = json.dumps(store.snapshot(), separators=(",", ":"))
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(interval_s)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

    return Handler


def _guess_url(host: str, port: int) -> str:
    if host not in ("0.0.0.0", "::"):
        return f"http://{host}:{port}/"
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("192.168.123.1", 1))
        return f"http://{sock.getsockname()[0]}:{port}/"
    except OSError:
        pass
    finally:
        if sock: sock.close()
    try:
        return f"http://{socket.gethostbyname(socket.gethostname())}:{port}/"
    except OSError:
        return f"http://127.0.0.1:{port}/"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sensor dashboard web UI")
    parser.add_argument("--host",        default="0.0.0.0")
    parser.add_argument("--port",        type=int,   default=8091)
    parser.add_argument("--domain-id",   type=int,   default=0)
    parser.add_argument("--target-topic",default="rt/target_state")
    parser.add_argument("--cam-topic",   default="rt/cam_ball_state")
    parser.add_argument("--lidar-topic", default="rt/lidar_ball_state")
    parser.add_argument("--fused-topic", default="rt/ball_state")
    parser.add_argument("--stale-ms",    type=float, default=float(BALL_STALE_MS))
    parser.add_argument("--rate-hz",     type=float, default=20.0)
    parser.add_argument("--range-m",     type=float, default=4.0,
                        help="Canvas half-width in metres (default 4.0)")
    args = parser.parse_args()

    store = SensorStore(
        stale_ms_ball=args.stale_ms,
        stale_ms_target=float(TARGET_STALE_MS),
    )

    target_sub = TargetStateSubscriber(
        domain_id=args.domain_id, callback=store.update_target,
        topic_name=args.target_topic,
    )
    cam_sub = BallStateSubscriber(
        domain_id=args.domain_id, callback=store.update_cam,
        topic_name=args.cam_topic,
    )
    lidar_sub = BallStateSubscriber(
        domain_id=args.domain_id, callback=store.update_lidar,
        topic_name=args.lidar_topic,
    )
    fused_sub = BallStateSubscriber(
        domain_id=args.domain_id, callback=store.update_fused,
        topic_name=args.fused_topic,
    )
    for sub in (target_sub, cam_sub, lidar_sub, fused_sub):
        sub.start()

    html = HTML.replace("__RANGE_M__", str(float(args.range_m))).encode("utf-8")
    interval_s = 1.0 / max(1.0, args.rate_hz)
    handler = _make_handler(store, html, interval_s)

    ThreadingTCPServer.allow_reuse_address = True
    httpd = ThreadingTCPServer((args.host, args.port), handler)
    httpd.daemon_threads = True

    url = _guess_url(args.host, args.port)
    print(f"[dashboard] target  -> {args.target_topic}")
    print(f"[dashboard] cam     -> {args.cam_topic}")
    print(f"[dashboard] lidar   -> {args.lidar_topic}")
    print(f"[dashboard] fused   -> {args.fused_topic}")
    print(f"[dashboard] open    -> {url}")
    print("[dashboard] Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Stopped.")
    finally:
        for sub in (target_sub, cam_sub, lidar_sub, fused_sub):
            sub.stop()
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
