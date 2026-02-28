"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Legend,
} from "recharts";

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_BATTERY_API_URL || "http://localhost:8000";
const SSE_URL  = `${API_BASE}/api/battery/stream`;

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function healthColor(soh) {
  if (soh >= 70) return "#16a34a";
  if (soh >= 60) return "#d97706";
  return "#dc2626";
}
function healthLabel(soh) {
  if (soh >= 70) return "GOOD";
  if (soh >= 60) return "DEGRADED";
  return "CRITICAL";
}

// ─── COMPONENTS ───────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, sub, sub2, color = "#0f172a" }) {
  return (
    <div className="card fade-up">
      <div className="stat-label">{label}</div>
      <div style={{ display:"flex", alignItems:"baseline", gap:4, marginTop:4 }}>
        <span style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:42, color, lineHeight:1 }}>
          {value ?? "—"}
        </span>
        {unit && <span style={{ fontSize:14, color:"#94a3b8" }}>{unit}</span>}
      </div>
      {sub  && <div style={{ fontSize:11, color:"#94a3b8", marginTop:4 }}>{sub}</div>}
      {sub2 && <div style={{ fontSize:10, color:"#64748b", marginTop:2 }}>{sub2}</div>}
    </div>
  );
}

function StatusBadge({ label, color }) {
  return (
    <span style={{
      padding:"4px 12px", borderRadius:99, fontSize:10,
      fontWeight:600, letterSpacing:"2px",
      background:color+"22", color,
    }}>{label}</span>
  );
}

function CapTooltip({ active, payload, label, q0 = 50 }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background:"#fff", border:"1px solid #e2e8f0", borderRadius:8,
      padding:"8px 12px", fontSize:10, lineHeight:1.8, minWidth:180,
    }}>
      <div style={{ fontWeight:600, marginBottom:4, color:"#0f172a" }}>Day {label}</div>
      {payload.filter(p => p.value != null).map(p => (
        <div key={p.dataKey} style={{ color:p.color }}>
          {p.name}: {p.value.toFixed(2)} Ah
          <span style={{ color:"#94a3b8" }}> ({(100*p.value/q0).toFixed(1)}% SOH)</span>
        </div>
      ))}
    </div>
  );
}

// ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [data,         setData]         = useState(null);
  const [connected,    setConnected]    = useState(false);
  const [error,        setError]        = useState(null);
  const [animSoh,      setAnimSoh]      = useState(0);
  const [backendStatus, setBackendStatus] = useState("starting"); // "starting" | "ready" | "failed"
  const sourceRef = useRef(null);

  // ── On mount: ping Next.js API route to start battery_api.py if not running ──
  useEffect(() => {
    async function startBackend() {
      try {
        setBackendStatus("starting");
        const res = await fetch("/api/start-battery", { method: "POST" });
        const json = await res.json();
        if (res.ok) {
          setBackendStatus("ready");
        } else {
          setBackendStatus("failed");
          setError(`Backend failed to start: ${json.message ?? "unknown error"}`);
        }
      } catch (e) {
        setBackendStatus("failed");
        setError("Could not reach /api/start-battery — is Next.js running?");
      }
    }
    startBackend();
  }, []);

  const connect = useCallback(() => {
    if (sourceRef.current) sourceRef.current.close();
    const es = new EventSource(SSE_URL);
    sourceRef.current = es;
    es.onopen    = () => { setConnected(true); setError(null); };
    es.onmessage = (e) => { try { setData(JSON.parse(e.data)); } catch {} };
    es.onerror   = () => {
      setConnected(false); setError("Connection lost — retrying…");
      es.close(); setTimeout(connect, 3000);
    };
  }, []);

  // Only start the SSE stream once the backend is confirmed ready
  useEffect(() => {
    if (backendStatus !== "ready") return;
    connect();
    return () => sourceRef.current?.close();
  }, [backendStatus, connect]);

  useEffect(() => {
    if (!data) return;
    const target = data.soh; let cur = animSoh;
    const step = () => {
      cur = cur + (target - cur) * 0.12;
      setAnimSoh(Math.round(cur*10)/10);
      if (Math.abs(cur-target) > 0.05) requestAnimationFrame(step);
      else setAnimSoh(target);
    };
    requestAnimationFrame(step);
  }, [data?.soh]);

  const hc = data ? healthColor(data.soh) : "#16a34a";
  const hl = data ? healthLabel(data.soh) : "—";
  const q0 = data?.q0 ?? 50;

  // ── Backend starting splash ───────────────────────────────────────────────
  if (backendStatus === "starting") {
    return (
      <div style={{
        minHeight:"100vh", background:"#f0f4f8", display:"flex",
        alignItems:"center", justifyContent:"center",
        fontFamily:"'DM Mono','Courier New',monospace",
      }}>
        <div style={{ textAlign:"center" }}>
          <div style={{ fontSize:36, marginBottom:16 }}>⚡</div>
          <div style={{ fontSize:13, letterSpacing:3, color:"#64748b", marginBottom:8 }}>
            STARTING BATTERY BACKEND
          </div>
          <div style={{ fontSize:10, color:"#94a3b8", letterSpacing:2 }}>
            Launching battery_api.py · please wait…
          </div>
          <div style={{ marginTop:24, display:"flex", justifyContent:"center", gap:6 }}>
            {[0,1,2].map(i => (
              <div key={i} style={{
                width:8, height:8, borderRadius:"50%", background:"#22c55e",
                animation:`pulse 1.2s ease-in-out ${i*0.2}s infinite`,
              }}/>
            ))}
          </div>
          <style>{`@keyframes pulse{0%,100%{opacity:0.2}50%{opacity:1}}`}</style>
        </div>
      </div>
    );
  }

  if (backendStatus === "failed") {
    return (
      <div style={{
        minHeight:"100vh", background:"#f0f4f8", display:"flex",
        alignItems:"center", justifyContent:"center",
        fontFamily:"'DM Mono','Courier New',monospace",
      }}>
        <div style={{ textAlign:"center", maxWidth:420 }}>
          <div style={{ fontSize:36, marginBottom:16 }}>⚠️</div>
          <div style={{ fontSize:13, letterSpacing:3, color:"#ef4444", marginBottom:12 }}>
            BACKEND FAILED TO START
          </div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:20, lineHeight:1.7 }}>
            {error ?? "battery_api.py could not be launched. Check that Python is installed and the battery-backend folder is in your project root."}
          </div>
          <button
            onClick={() => { setBackendStatus("starting"); setError(null); }}
            style={{
              padding:"8px 20px", background:"#0f172a", color:"#fff", border:"none",
              borderRadius:8, fontSize:11, cursor:"pointer", letterSpacing:2,
            }}
          >RETRY</button>
        </div>
      </div>
    );
  }

  // ── Combined capacity chart data ──────────────────────────────────────────
  // Past: each entry has { day, q_pred, q_real }
  // Bridge: connects past to future at current day
  // Future: each entry has { day, q_pred_fc, q_real_fc, q_upper, q_lower }
  const combinedData = data ? (() => {
    const past = (data.cap_hist ?? []).map(p => ({
      day:    p.day,
      q_pred: p.Q,       // model-estimated history
      q_real: p.Q_true,  // ground-truth history
    }));

    const bridge = {
      day:      data.day,
      q_pred:   data.q_now,
      q_real:   data.q_true_now,
      q_fc:     data.q_true_now,  // forecast starts from real Q
      q_upper:  data.q_true_now,
      q_lower:  data.q_true_now,
    };

    const fc = data.forecast;
    const future = (fc.days || []).map((d, i) => ({
      day:     d,
      q_fc:    fc.q_real?.[i],   // single forecast anchored at real Q
      q_upper: fc.q_upper?.[i],
      q_lower: fc.q_lower?.[i],
    }));

    return [...past, bridge, ...future];
  })() : [];

  return (
    <div style={{ minHeight:"100vh", background:"#f0f4f8", fontFamily:"'DM Mono','Courier New',monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');
        * { box-sizing:border-box; margin:0; padding:0; }
        .card {
          background:#fff; border:1px solid #e2e8f0; border-radius:14px;
          padding:24px; box-shadow:0 1px 3px rgba(0,0,0,.06),0 4px 16px rgba(0,0,0,.04);
        }
        .stat-label { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:#94a3b8; }
        .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
        @keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }
        .live-dot { width:8px;height:8px;border-radius:50%;background:#16a34a;
          animation:pulse 2s infinite;display:inline-block;margin-right:6px; }
        .dead-dot { width:8px;height:8px;border-radius:50%;background:#f59e0b;
          animation:pulse 1s infinite;display:inline-block;margin-right:6px; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
        .fade-up { animation:fadeUp .4s ease forwards; }
        .health-bar-bg { height:18px;background:#f1f5f9;border-radius:99px;overflow:hidden;
          margin-top:14px;border:1px solid #e2e8f0; }
        .health-bar-fill { height:100%;border-radius:99px;transition:width 1s cubic-bezier(.16,1,.3,1); }
        .seg-bar { display:flex;gap:3px;margin-top:8px; }
        .seg { flex:1;height:5px;border-radius:99px;transition:background .3s; }
        .chart-legend { display:flex;gap:18px;font-size:9px;letter-spacing:1px;
          color:#64748b;margin-bottom:10px;flex-wrap:wrap;align-items:center; }
        .ldot { width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:5px;flex-shrink:0; }
        .ldash { width:18px;height:2px;display:inline-block;margin-right:5px;flex-shrink:0; }
      `}</style>

      {/* ── HEADER ── */}
      <header style={{
        background:"#fff", borderBottom:"1px solid #e2e8f0", padding:"16px 36px",
        display:"flex", alignItems:"center", justifyContent:"space-between",
      }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <div style={{
            width:38, height:38, background:"linear-gradient(135deg,#22c55e,#16a34a)",
            borderRadius:10, display:"flex", alignItems:"center", justifyContent:"center",
            fontSize:20, boxShadow:"0 2px 8px #16a34a33",
          }}>⚡</div>
          <div>
            <div style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:22, letterSpacing:3, color:"#0f172a" }}>
              VOLTWATCH
            </div>
            <div style={{ fontSize:9, letterSpacing:3, color:"#94a3b8" }}>BATTERY TISAC LIVE MONITOR</div>
          </div>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:16, fontSize:11, color:"#64748b" }}>
          {error && <span style={{ color:"#f59e0b" }}><span className="dead-dot"/>{error}</span>}
          {connected && !error && <span><span className="live-dot"/>LIVE STREAM</span>}
          {data && (
            <span style={{ background:"#f8fafc", border:"1px solid #e2e8f0", borderRadius:99, padding:"4px 12px" }}>
              Hour&nbsp;<b>{data.hour}</b> · Day&nbsp;<b>{data.day}</b>
            </span>
          )}
          <button onClick={connect} style={{
            padding:"6px 14px", background:"#0f172a", color:"#fff", border:"none",
            borderRadius:8, fontSize:11, cursor:"pointer", letterSpacing:1,
          }}>RECONNECT</button>
        </div>
      </header>

      <main style={{ padding:"28px 36px", display:"flex", flexDirection:"column", gap:14 }}>

        {/* ── ROW 1: SOH + Stats ── */}
        <div className="grid-2">
          {/* SOH card — shows both predicted and real */}
          <div className="card fade-up">
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
              <div>
                <div className="stat-label">Battery Health (SOH)</div>
                <div style={{ display:"flex", alignItems:"baseline", gap:6, marginTop:4 }}>
                  <span style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:72, color:hc, lineHeight:1 }}>
                    {animSoh.toFixed(1)}
                  </span>
                  <span style={{ fontSize:28, color:"#cbd5e1" }}>%</span>
                  <span style={{ fontSize:13, color:"#3b82f6", marginLeft:8 }}>predicted</span>
                </div>
                <div style={{ fontSize:12, marginTop:6, display:"flex", gap:16 }}>
                  <span style={{ color:"#16a34a" }}>
                    ● Real: <b>{data?.soh_true?.toFixed(1) ?? "—"}%</b>
                    <span style={{ color:"#94a3b8" }}> · {data?.q_true_now ?? "—"} Ah</span>
                  </span>
                  <span style={{ color:"#3b82f6" }}>
                    ● Predicted: <b>{animSoh.toFixed(1)}%</b>
                    <span style={{ color:"#94a3b8" }}> · {data?.q_now ?? "—"} Ah</span>
                  </span>
                </div>
              </div>
              <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:8 }}>
                <StatusBadge label={hl} color={hc}/>
                <StatusBadge
                  label={data?.mode_label?.toUpperCase() ?? "—"}
                  color={data?.mode === 1 ? "#3b82f6" : "#64748b"}
                />
              </div>
            </div>
            <div className="health-bar-bg">
              <div className="health-bar-fill" style={{
                width:`${animSoh}%`, background:`linear-gradient(90deg,${hc}99,${hc})`,
              }}/>
            </div>
            <div className="seg-bar">
              {Array.from({length:25}).map((_,i) => (
                <div key={i} className="seg" style={{ background: i < Math.floor(animSoh/4) ? hc : "#e2e8f0" }}/>
              ))}
            </div>
            <div style={{ display:"flex", justifyContent:"space-between", marginTop:6 }}>
              <span style={{ fontSize:9, color:"#cbd5e1", letterSpacing:2 }}>0% CRITICAL</span>
              <span style={{ fontSize:9, color:"#cbd5e1", letterSpacing:2 }}>100% OPTIMAL</span>
            </div>
          </div>

          {/* Stats: dual EOL/RUL */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
            <StatCard label="Resistance" value={data?.r_now_mohm?.toFixed(3) ?? "—"} unit="mΩ"/>
            <StatCard
              label="Mode"
              value={data?.mode_label ?? "—"}
              sub={`Hour ${data?.hour ?? 0} · Day ${data?.day ?? 0}`}
              color={data?.mode === 1 ? "#3b82f6" : "#16a34a"}
            />
            <StatCard
              label="Predicted EOL"
              value={data?.pred_eol_day_real ?? ">1400"}
              unit={data?.pred_eol_day_real ? "d" : ""}
              sub={data?.pred_rul_days_real != null ? `RUL: ${data.pred_rul_days_real} days` : ""}
              sub2={`95% band: ${data?.eol_band?.[0] ?? "—"}–${data?.eol_band?.[1] ?? "—"} d`}
              color="#16a34a"
            />
          </div>
        </div>

        {/* ── ROW 2: Combined Capacity Chart ── */}
        <div className="card fade-up">
          <div style={{ marginBottom:12 }}>
            <div className="stat-label">Battery Capacity Q — History + Forecast</div>
            <div style={{ fontSize:12, color:"#64748b", marginTop:4 }}>
              EOL = {data?.q_eol ?? 35} Ah (70% SOH) · solid lines = past/present · dashed = future forecast
            </div>
          </div>

          <div className="chart-legend">
            <span style={{ display:"flex", alignItems:"center" }}>
              <span className="ldot" style={{background:"#16a34a"}}/>Real Q (measured)
            </span>
            <span style={{ display:"flex", alignItems:"center" }}>
              <span className="ldot" style={{background:"#3b82f6"}}/>Predicted Q (model)
            </span>
            <span style={{ display:"flex", alignItems:"center" }}>
              <span className="ldash" style={{background:"#16a34a"}}/>Forecast (from real Q)
            </span>
            <span style={{ display:"flex", alignItems:"center" }}>
              <span className="ldot" style={{background:"#e11d48"}}/>EOL threshold
            </span>
          </div>

          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={combinedData} margin={{top:4,right:8,left:-20,bottom:0}}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
              <XAxis dataKey="day" tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false}
                tickFormatter={v=>`d${v}`}/>
              <YAxis tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} domain={["auto","auto"]}/>
              <Tooltip content={<CapTooltip q0={q0}/>}/>

              {/* EOL line */}
              <ReferenceLine y={data?.q_eol ?? 35} stroke="#e11d48" strokeDasharray="5 3"
                label={{value:"EOL 70%", fill:"#e11d48", fontSize:9, position:"insideTopRight"}}/>

              {/* Today */}
              {data?.day != null &&
                <ReferenceLine x={data.day} stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="4 2"
                  label={{value:"now", fill:"#64748b", fontSize:9}}/>}

              {/* Real-anchored EOL vertical */}
              {data?.pred_eol_day_real != null &&
                <ReferenceLine x={data.pred_eol_day_real} stroke="#16a34a" strokeDasharray="3 3"
                  label={{value:"EOL", fill:"#16a34a", fontSize:8, position:"insideTopLeft"}}/>}

              {/* CI band area (use Area trick via hidden Lines + fill) */}
              <Line type="monotone" dataKey="q_upper" stroke="none" dot={false} legendType="none" name="95% upper"/>
              <Line type="monotone" dataKey="q_lower" stroke="none" dot={false} legendType="none" name="95% lower"/>

              {/* HISTORY: real Q (solid green) */}
              <Line type="monotone" dataKey="q_real" stroke="#16a34a" strokeWidth={2.5}
                dot={false} name="Real Q" connectNulls={false}/>

              {/* HISTORY: predicted Q (solid blue) */}
              <Line type="monotone" dataKey="q_pred" stroke="#3b82f6" strokeWidth={2.5}
                dot={false} name="Predicted Q" connectNulls={false}/>

              {/* FUTURE: single forecast anchored at real Q (dashed green) */}
              <Line type="monotone" dataKey="q_fc" stroke="#16a34a" strokeWidth={2}
                strokeDasharray="7 4" dot={false} name="Forecast" connectNulls={false}/>
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ── ROW 3: Resistance + Energy ── */}
        <div className="grid-2">
          <div className="card fade-up">
            <div style={{ marginBottom:16 }}>
              <div className="stat-label">Resistance Tracking (θ₂ = R)</div>
              <div style={{ fontSize:12, color:"#64748b", marginTop:4 }}>Daily estimated resistance [mΩ]</div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={data?.res_hist ?? []} margin={{top:4,right:4,left:-20,bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="day" tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} tickFormatter={v=>`d${v}`}/>
                <YAxis tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} domain={["auto","auto"]}/>
                <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:10}}/>
                <Line type="monotone" dataKey="R" stroke="#ef4444" strokeWidth={2.5} dot={false} name="R [mΩ]"/>
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card fade-up">
            <div style={{ marginBottom:16 }}>
              <div className="stat-label">Streaming Energy y₃ vs Model</div>
              <div style={{ fontSize:12, color:"#64748b", marginTop:4 }}>Last 14 days hourly [Wh]</div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={data?.energy_hist ?? []} margin={{top:4,right:4,left:-20,bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="h" tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false}/>
                <YAxis tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false}/>
                <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:10}}/>
                <Legend wrapperStyle={{fontSize:9}}/>
                <Line type="monotone" dataKey="measured" stroke="#0f172a" strokeWidth={1.5} dot={false} name="Measured y₃"/>
                <Line type="monotone" dataKey="model"    stroke="#ef4444" strokeWidth={2}   dot={false} name="Model y₃"/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── ROW 4: Mode ── */}
        <div className="card fade-up">
          <div style={{ marginBottom:16 }}>
            <div className="stat-label">Detected Mode — 0 = Resting · 1 = Driving</div>
            <div style={{ fontSize:12, color:"#64748b", marginTop:4 }}>Last 14 days hourly</div>
          </div>
          <ResponsiveContainer width="100%" height={100}>
            <AreaChart data={data?.mode_hist ?? []} margin={{top:4,right:4,left:-20,bottom:0}}>
              <defs>
                <linearGradient id="modeGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#22c55e" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
              <XAxis dataKey="h" tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false}/>
              <YAxis tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} domain={[-0.1,1.2]} ticks={[0,1]}/>
              <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:10}}/>
              <Area type="stepAfter" dataKey="mode" stroke="#22c55e" strokeWidth={2}
                fill="url(#modeGrad)" dot={false} name="Detected mode"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* ── FOOTER ── */}
        <div style={{ textAlign:"center", fontSize:10, color:"#cbd5e1", letterSpacing:2, paddingBottom:8 }}>
          BATTERY TISAC · DUAL-MODE STREAMING INFERENCE · VOLTWATCH
          {data && ` · SOH ${data.soh}% (pred) / ${data.soh_true}% (real) · RUL ${data.pred_rul_days ?? ">1400"} DAYS`}
        </div>
      </main>
    </div>
  );
}