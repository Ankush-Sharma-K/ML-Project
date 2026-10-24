// IDR Navigator — plays a simulated tunnel through the edge engine.
// Live mode: talks to `python -m src.edge_engine.server` (/api/drives, /api/replay).
// Demo mode (no server, e.g. opened as a static file): bundled replays in assets/demo/.
"use strict";

const $ = (id) => document.getElementById(id);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const state = { live: false, drives: [], demo: null, data: null, frame: 0, timer: null, len: 120, route: "correct", layers: [] };

const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([52.4, -1.6], 12);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const marker = L.circleMarker([0, 0], { radius: 7, weight: 3, fillOpacity: 1 });
const sigmaCircle = L.circle([0, 0], { radius: 1, weight: 1, fillOpacity: 0.12 });

function setChip(el, text, cls) { el.textContent = text; el.className = "chip " + cls; }
function fmtClock(s) { const a = Math.max(0, Math.round(s)); return `${Math.floor(a / 60)}:${String(a % 60).padStart(2, "0")}`; }

async function getJSON(url) {
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function init() {
  $("runBtn").disabled = true;
  $("mode").textContent = "Loading drives…";
  try {
    state.drives = await getJSON("/api/drives");
    state.live = true;
    $("mode").textContent = "Live: replays run in the edge engine on this machine (int8 on-device model). First run of an outage takes ~15 s, then it is cached.";
  } catch {
    try {
      state.demo = await getJSON("assets/demo/index.json");
      state.drives = state.demo.drives;
      $("mode").textContent = "Demo mode: bundled replays (start the server for any drive / outage: python -m src.edge_engine.server).";
    } catch {
      $("mode").textContent = "No server and no demo data found.";
      return;
    }
  }
  const sel = $("driveSel");
  sel.innerHTML = state.drives.map((d, i) => `<option value="${i}">${d.drive.replace(".csv", "")} · ${d.split} · ${d.duration_min} min</option>`).join("");
  sel.onchange = fillStarts;
  fillStarts();
  $("runBtn").disabled = false;
  if (new URLSearchParams(location.search).has("run")) $("runBtn").click();  // ?run=1: autoplay (presentations)
}

function fillStarts() {
  const d = state.drives[+$("driveSel").value];
  $("startSel").innerHTML = d.starts.map((s) => `<option value="${s.start}">${s.t_min} min into the drive</option>`).join("");
}

for (const segId of ["lenSeg", "routeSeg"]) {
  $(segId).addEventListener("click", (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    [...$(segId).children].forEach((c) => c.classList.toggle("on", c === b));
    if (segId === "lenSeg") state.len = +b.dataset.v; else state.route = b.dataset.v;
  });
}

$("runBtn").onclick = async () => {
  const d = state.drives[+$("driveSel").value];
  const start = +$("startSel").value;
  stop();
  $("busy").hidden = false;
  $("runBtn").disabled = true;
  try {
    let data;
    if (state.live) {
      data = await getJSON(`/api/replay?drive=${encodeURIComponent(d.drive)}&start=${start}&outage=${state.len}&model=int8&route=${state.route}`);
    } else {
      const key = `${d.drive.replace(".csv", "")}_${start}_${state.len}_int8_${state.route}.json`;
      if (!state.demo.files.includes(key)) throw new Error("This combination is not in the demo bundle — start the server for it.");
      data = await getJSON(`assets/demo/${key}`);
    }
    show(data);
  } catch (e) {
    banner(`⚠ ${e.message}`);
  } finally {
    $("busy").hidden = true;
    $("runBtn").disabled = false;
  }
};

function banner(text) { const b = $("banner"); b.textContent = text; b.hidden = !text; }

function clearLayers() { state.layers.forEach((l) => map.removeLayer(l)); state.layers = []; }
function add(l) { l.addTo(map); state.layers.push(l); return l; }

function show(data) {
  clearLayers();
  state.data = data;
  state.frame = 0;
  add(L.polyline(data.planned_route, { color: css("--route"), weight: 8, opacity: 0.25 }));
  add(L.polyline(data.gps_before, { color: css("--gps"), weight: 4 }));
  add(L.polyline([data.gps_before.at(-1), ...data.gps_hidden.map((p) => [p[1], p[2]]), data.gps_exit],
    { color: css("--gps"), weight: 3, dashArray: "6 8", opacity: 0.8 }));
  state.trail = add(L.layerGroup());
  state.corrected = null;
  marker.addTo(map); sigmaCircle.addTo(map);
  map.fitBounds(L.latLngBounds([...data.gps_before, ...data.gps_hidden.map((p) => [p[1], p[2]]), data.gps_exit]).pad(0.15));
  $("scrub").max = data.live.length - 1; $("scrub").value = 0; $("scrub").disabled = false;
  $("playBtn").disabled = false;
  $("endError").textContent = "";
  banner(`${data.drive.replace(".csv", "")}: GPS is withheld for ${data.outage_s} s — the engine uses phone sensors, the on-device model${data.route_kind === "wrong" ? "; the planned route takes a different road" : ""}.`);
  render(0);
  play();
}

function statusColor(st) { return st === "DR+MAP" ? css("--map") : st === "DR" ? css("--dr") : css("--gnss"); }

function render(i) {
  const d = state.data;
  const p = d.live[i];
  state.trail.clearLayers();
  for (let k = 1; k <= i; k++) {
    const a = d.live[k - 1], b = d.live[k];
    L.polyline([[a.lat, a.lon], [b.lat, b.lon]], { color: statusColor(b.status), weight: 5 }).addTo(state.trail);
  }
  const col = statusColor(p.status);
  marker.setLatLng([p.lat, p.lon]).setStyle({ color: col, fillColor: "#fff" });
  sigmaCircle.setLatLng([p.lat, p.lon]).setRadius(Math.max(p.sigma, 1)).setStyle({ color: col, fillColor: col });
  const st = p.status === "INIT" ? "GNSS" : p.status;
  setChip($("statusChip"), st, "chip-" + st.replace("+", ""));
  $("rSpeed").textContent = (p.speed * 3.6).toFixed(0);
  $("rHeading").textContent = p.heading.toFixed(0);
  $("rSigma").textContent = p.sigma.toFixed(0);
  $("rTime").textContent = p.t > 0 ? Math.min(p.t, d.outage_s).toFixed(0) : "–";
  if (p.route) {
    setChip($("routeChip"), p.route === "OFF_ROUTE" ? "OFF ROUTE" : "ON ROUTE", p.route === "OFF_ROUTE" ? "chip-OFF" : "chip-ON");
    $("routeDist").textContent = p.route_dist != null ? `${p.route_dist.toFixed(0)} m from planned route` : "";
  }
  $("clock").textContent = fmtClock(p.t) + (p.t < 0 ? " before tunnel" : "");
  $("scrub").value = i;
  const last = i === d.live.length - 1;
  if (last && !state.corrected) {
    state.corrected = add(L.polyline(d.track_corrected.map((q) => [q[1], q[2]]), { color: css("--fix"), weight: 4, dashArray: "2 7" }));
    add(L.polyline([d.exit_realtime, d.gps_exit], { color: css("--bad"), weight: 2, dashArray: "3 5" }));
    add(L.circleMarker(d.gps_exit, { radius: 6, color: css("--gps"), fillColor: css("--gps"), fillOpacity: 1 })
      .bindTooltip("GPS back at the tunnel exit"));
    $("endError").textContent = `Error when GPS returned: ${d.end_error_m.toFixed(0)} m`;
    const det = d.route_off_detected_t;
    banner(d.route_kind === "wrong"
      ? (det != null ? `Deviation from the planned route detected ${det.toFixed(0)} s into the tunnel.` : "Deviation was not detected inside the tunnel.")
      : `GPS back after ${d.outage_s} s: error ${d.end_error_m.toFixed(0)} m; tunnel track corrected (dotted).`);
  } else if (!last && state.corrected) {
    map.removeLayer(state.corrected); state.corrected = null;
  }
}

function play() {
  stop();
  $("playBtn").textContent = "❚❚"; $("playBtn").setAttribute("aria-label", "Pause");
  state.timer = setInterval(() => {
    if (state.frame >= state.data.live.length - 1) return stop();
    state.frame += 1; render(state.frame);
  }, 120);
}
function stop() {
  if (state.timer) clearInterval(state.timer);
  state.timer = null;
  $("playBtn").textContent = "▶"; $("playBtn").setAttribute("aria-label", "Play");
}
$("playBtn").onclick = () => (state.timer ? stop() : (state.frame >= state.data.live.length - 1 && (state.frame = 0), play()));
$("scrub").oninput = (e) => { stop(); state.frame = +e.target.value; render(state.frame); };

init();
