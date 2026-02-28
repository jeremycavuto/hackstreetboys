"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Legend,
} from "recharts";

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const API_BASE = process.env.NEXT_PUBLIC_BATTERY_API_URL || "http://localhost:8000";
const SSE_URL  = `${API_BASE}/api/battery/stream`;

// ─── HELPERS ──────────────────────────────────────────────────────────────────
function healthColor(soh) {
  if (soh >= 80) return "#16a34a";
  if (soh >= 60) return "#d97706";
  return "#dc2626";
}
function healthLabel(soh) {
  if (soh >= 80) return "GOOD";
  if (soh >= 60) return "DEGRADED";
  return "CRITICAL";
}
function healthBg(soh) {
  if (soh >= 80) return "#dcfce7";
  if (soh >= 60) return "#fef3c7";
  return "#fee2e2";
}

// ─── COMPONENTS ───────────────────────────────────────────────────────────────
function StatCard({ label, value, unit, sub, color = "#0f172a" }) {
  return (
    <div className="card fade-up">
      <div className="stat-label">{label}</div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4, marginTop: 4 }}>
        <span style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 42, color, lineHeight: 1 }}>
          {value ?? "—"}
        </span>
        {unit && <span style={{ fontSize: 14, color: "#94a3b8" }}>{unit}</span>}
      </div>
      {sub && <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function StatusBadge({ label, color }) {
  return (
    <span style={{
      padding: "4px 12px", borderRadius: 99, fontSize: 10,
      fontWeight: 600, letterSpacing: "2px",
      background: color + "22", color,
    }}>{label}</span>
  );
}

// ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
export default function Dashboard() {
  const [data,       setData]       = useState(null);
  const [connected,  setConnected]  = useState(false);
  const [error,      setError]      = useState(null);
  const [animSoh,    setAnimSoh]    = useState(0);
  const sourceRef = useRef(null);

  // SSE connection
  const connect = useCallback(() => {
    if (sourceRef.current) { sourceRef.current.close(); }
    const es = new EventSource(SSE_URL);
    sourceRef.current = es;

    es.onopen = () => { setConnected(true); setError(null); };
    es.onmessage = (e) => {
      try { setData(JSON.parse(e.data)); }
      catch {}
    };
    es.onerror = () => {
      setConnected(false);
      setError("Connection lost — retrying…");
      es.close();
      setTimeout(connect, 3000);
    };
  }, []);

  useEffect(() => { connect(); return () => sourceRef.current?.close(); }, [connect]);

  // Animate SOH bar
  useEffect(() => {
    if (!data) return;
    const target = data.soh;
    let cur = animSoh;
    const step = () => {
      cur = cur + (target - cur) * 0.12;
      setAnimSoh(Math.round(cur * 10) / 10);
      if (Math.abs(cur - target) > 0.05) requestAnimationFrame(step);
      else setAnimSoh(target);
    };
    requestAnimationFrame(step);
  }, [data?.soh]);

  const hc  = data ? healthColor(data.soh) : "#16a34a";
  const hl  = data ? healthLabel(data.soh) : "—";
  const hbg = data ? healthBg(data.soh)    : "#dcfce7";

  // Forecast chart data
  const forecastData = data ? (() => {
    const fc = data.forecast;
    const seen = data.cap_hist.map(p => ({ day: p.day, q_seen: p.Q }));
    const future = (fc.days || []).map((d, i) => ({
      day: d, q_forecast: fc.q?.[i], q_upper: fc.q_upper?.[i], q_lower: fc.q_lower?.[i],
    }));
    return [...seen, ...future];
  })() : [];

  return (
    <div style={{ minHeight:"100vh", background:"#f0f4f8", fontFamily:"'DM Mono','Courier New',monospace" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        .card {
          background: #fff; border: 1px solid #e2e8f0; border-radius: 14px;
          padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.04);
        }
        .stat-label { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:#94a3b8; }
        .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
        .grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px; }
        .grid-4 { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; }
        @keyframes pulse { 0%,100%{opacity:1}50%{opacity:.3} }
        .live-dot { width:8px;height:8px;border-radius:50%;background:#16a34a;
          animation:pulse 2s infinite;display:inline-block;margin-right:6px; }
        .dead-dot { width:8px;height:8px;border-radius:50%;background:#f59e0b;
          animation:pulse 1s infinite;display:inline-block;margin-right:6px; }
        @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
        .fade-up { animation:fadeUp .4s ease forwards; }
        .health-bar-bg { height:18px; background:#f1f5f9; border-radius:99px; overflow:hidden;
          margin-top:14px; border:1px solid #e2e8f0; }
        .health-bar-fill { height:100%; border-radius:99px; transition:width 1s cubic-bezier(.16,1,.3,1); }
        .seg-bar { display:flex; gap:3px; margin-top:8px; }
        .seg { flex:1; height:5px; border-radius:99px; transition:background .3s; }
      `}</style>

      {/* ── HEADER ────────────────────────────────────────────────────────── */}
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
            <div style={{ fontSize:9, letterSpacing:3, color:"#94a3b8" }}>
              BATTERY TISAC LIVE MONITOR
            </div>
          </div>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:16, fontSize:11, color:"#64748b" }}>
          {error && (
            <span style={{ color:"#f59e0b", fontSize:11 }}>
              <span className="dead-dot" />{error}
            </span>
          )}
          {connected && !error && (
            <span><span className="live-dot" />LIVE STREAM</span>
          )}
          {data && (
            <span style={{ background:"#f8fafc", border:"1px solid #e2e8f0", borderRadius:99, padding:"4px 12px" }}>
              Hour&nbsp;<b>{data.hour}</b> · Day&nbsp;<b>{data.day}</b>
            </span>
          )}
          <button onClick={connect}
            style={{ padding:"6px 14px", background:"#0f172a", color:"#fff", border:"none",
              borderRadius:8, fontSize:11, cursor:"pointer", letterSpacing:1 }}>
            RECONNECT
          </button>
        </div>
      </header>

      <main style={{ padding:"28px 36px", display:"flex", flexDirection:"column", gap:14 }}>

        {/* ── ROW 1: SOH + key stats ───────────────────────────────────── */}
        <div className="grid-2">
          {/* SOH bar card */}
          <div className="card fade-up">
            <div style={{ display:"flex", justifyContent:"space-between", alignItems:"flex-start" }}>
              <div>
                <div className="stat-label">Battery Health (SOH)</div>
                <div style={{ display:"flex", alignItems:"baseline", gap:6, marginTop:4 }}>
                  <span style={{ fontFamily:"'Bebas Neue',sans-serif", fontSize:72, color:hc, lineHeight:1 }}>
                    {animSoh.toFixed(1)}
                  </span>
                  <span style={{ fontSize:28, color:"#cbd5e1" }}>%</span>
                  <span style={{ fontSize:14, color:"#94a3b8", marginLeft:8 }}>
                    Q = {data?.q_now ?? "—"} Ah
                  </span>
                </div>
              </div>
              <div style={{ display:"flex", flexDirection:"column", alignItems:"flex-end", gap:8 }}>
                <StatusBadge label={hl} color={hc} />
                <StatusBadge
                  label={data?.mode_label?.toUpperCase() ?? "—"}
                  color={data?.mode === 1 ? "#3b82f6" : "#64748b"}
                />
              </div>
            </div>
            <div className="health-bar-bg">
              <div className="health-bar-fill" style={{
                width:`${animSoh}%`,
                background:`linear-gradient(90deg,${hc}99,${hc})`,
              }}/>
            </div>
            <div className="seg-bar">
              {Array.from({length:25}).map((_,i) => (
                <div key={i} className="seg" style={{
                  background: i < Math.floor(animSoh/4) ? hc : "#e2e8f0",
                }}/>
              ))}
            </div>
            <div style={{ display:"flex", justifyContent:"space-between", marginTop:6 }}>
              <span style={{ fontSize:9, color:"#cbd5e1", letterSpacing:2 }}>0% CRITICAL</span>
              <span style={{ fontSize:9, color:"#cbd5e1", letterSpacing:2 }}>100% OPTIMAL</span>
            </div>
          </div>

          {/* Stats grid */}
          <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:14 }}>
            <StatCard label="Resistance" value={data?.r_now_mohm?.toFixed(3) ?? "—"} unit="mΩ" />
            <StatCard
              label="Predicted EOL"
              value={data?.pred_eol_day ?? ">1400"}
              unit={data?.pred_eol_day ? "days" : ""}
              sub={data?.pred_rul_days != null ? `RUL: ${data.pred_rul_days} days remaining` : ""}
              color="#0f172a"
            />
            <StatCard
              label="EOL Band (95%)"
              value={data?.eol_band?.[0] != null ? `${data.eol_band[0]}–${data.eol_band[1]}` : "—"}
              unit="days"
            />
            <StatCard
              label="Mode"
              value={data?.mode_label ?? "—"}
              sub={`Hour ${data?.hour ?? 0} · Day ${data?.day ?? 0}`}
              color={data?.mode === 1 ? "#3b82f6" : "#16a34a"}
            />
          </div>
        </div>

        {/* ── ROW 2: Forecast + Capacity ───────────────────────────────── */}
        <div className="grid-2">
          {/* Lifespan Forecast */}
          <div className="card fade-up">
            <div style={{ marginBottom:16 }}>
              <div className="stat-label">Expected Lifespan Forecast</div>
              <div style={{ fontSize:12, color:"#64748b", marginTop:4 }}>
                Capacity Q [Ah] — estimated + 300-day forecast
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={forecastData} margin={{top:4,right:4,left:-20,bottom:0}}>
                <defs>
                  <linearGradient id="seenGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#3b82f6" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="fcGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#0d9488" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#0d9488" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="day" tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false}
                  tickFormatter={v => `d${v}`}/>
                <YAxis tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} domain={["auto","auto"]}/>
                <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:10}}/>
                <ReferenceLine y={data?.q_eol ?? 40} stroke="#e11d48" strokeDasharray="4 2" label={{value:"EOL",fill:"#e11d48",fontSize:9}}/>
                {data?.day != null && <ReferenceLine x={data.day} stroke="#94a3b8" strokeDasharray="4 2"/>}
                {data?.pred_eol_day != null && <ReferenceLine x={data.pred_eol_day} stroke="#3b82f6" strokeDasharray="4 2"/>}
                <Area type="monotone" dataKey="q_seen" stroke="#3b82f6" strokeWidth={2} fill="url(#seenGrad)" dot={false} name="Estimated Q"/>
                <Area type="monotone" dataKey="q_upper" stroke="none" fill="#0d9488" fillOpacity={0.08} dot={false} name="95% CI"/>
                <Area type="monotone" dataKey="q_lower" stroke="none" fill="#fff" fillOpacity={1} dot={false}/>
                <Line type="monotone" dataKey="q_forecast" stroke="#0d9488" strokeWidth={2.5} dot={false} name="Forecast Q"/>
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Capacity tracking */}
          <div className="card fade-up">
            <div style={{ marginBottom:16 }}>
              <div className="stat-label">Capacity Tracking (θ₁ = Q)</div>
              <div style={{ fontSize:12, color:"#64748b", marginTop:4 }}>Daily estimated capacity [Ah]</div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={data?.cap_hist ?? []} margin={{top:4,right:4,left:-20,bottom:0}}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9"/>
                <XAxis dataKey="day" tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} tickFormatter={v=>`d${v}`}/>
                <YAxis tick={{fill:"#94a3b8",fontSize:9}} axisLine={false} tickLine={false} domain={["auto","auto"]}/>
                <Tooltip contentStyle={{background:"#fff",border:"1px solid #e2e8f0",borderRadius:8,fontSize:10}}/>
                <ReferenceLine y={data?.q_eol ?? 40} stroke="#e11d48" strokeDasharray="4 2" label={{value:"EOL",fill:"#e11d48",fontSize:9}}/>
                <ReferenceLine y={data?.q0 ?? 50} stroke="#94a3b8" strokeDasharray="2 2"/>
                <Line type="monotone" dataKey="Q" stroke="#3b82f6" strokeWidth={2.5} dot={false} name="Q [Ah]"/>
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── ROW 3: Resistance + Energy ───────────────────────────────── */}
        <div className="grid-2">
          {/* Resistance tracking */}
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

          {/* Streaming energy */}
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

        {/* ── ROW 4: Mode detection ────────────────────────────────────── */}
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
              <Area type="stepAfter" dataKey="mode" stroke="#22c55e" strokeWidth={2} fill="url(#modeGrad)" dot={false} name="Detected mode"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* ── FOOTER ───────────────────────────────────────────────────── */}
        <div style={{ textAlign:"center", fontSize:10, color:"#cbd5e1", letterSpacing:2, paddingBottom:8 }}>
          BATTERY TISAC · DUAL-MODE STREAMING INFERENCE · VOLTWATCH
          {data && ` · SOH ${data.soh}% · RUL ${data.pred_rul_days ?? ">1400"} DAYS`}
        </div>
      </main>
    </div>
  );
}