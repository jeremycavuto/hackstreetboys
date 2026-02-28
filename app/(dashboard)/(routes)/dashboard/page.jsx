"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { UserButton } from "@clerk/nextjs";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

const CARS = {
  "model-s": { name: "Tesla Model S", year: "2021", nickname: "Daily Driver", health: 78, range: 312, daysLeft: 1247, lastCharge: "2h ago", odometer: "42,381 mi", status: "good",      accentColor: "#22c55e" },
  "model-3": { name: "Tesla Model 3", year: "2023", nickname: "Weekend Car",  health: 94, range: 348, daysLeft: 2891, lastCharge: "1d ago", odometer: "11,204 mi", status: "excellent", accentColor: "#38bdf8" },
  "bolt":    { name: "Chevy Bolt EV", year: "2020", nickname: "Commuter",     health: 61, range: 189, daysLeft: 548,  lastCharge: "3d ago", odometer: "67,920 mi", status: "degraded",  accentColor: "#f59e0b" },
};

const DEGRADATION_DATA = {
  "model-s": [
    {month:"Jan",health:100},{month:"Mar",health:98},{month:"May",health:96},
    {month:"Jul",health:93},{month:"Sep",health:90},{month:"Nov",health:87},
    {month:"Jan",health:84},{month:"Mar",health:81},{month:"Now",health:78},
  ],
  "model-3": [
    {month:"Jan",health:100},{month:"Mar",health:99},{month:"May",health:98},
    {month:"Jul",health:97},{month:"Sep",health:96},{month:"Nov",health:95},
    {month:"Jan",health:95},{month:"Mar",health:94},{month:"Now",health:94},
  ],
  "bolt": [
    {month:"Jan",health:100},{month:"Mar",health:96},{month:"May",health:91},
    {month:"Jul",health:86},{month:"Sep",health:80},{month:"Nov",health:74},
    {month:"Jan",health:68},{month:"Mar",health:64},{month:"Now",health:61},
  ],
};

// Predicted battery decay over the next ~14 months from current health
// Each car degrades at a different rate based on historical data
const PREDICTED_DECAY_DATA = {
  "model-s": [
    {month:"Now",health:78},{month:"Feb",health:77.4},{month:"Mar",health:76.9},
    {month:"Apr",health:76.3},{month:"May",health:75.8},{month:"Jun",health:75.2},
    {month:"Jul",health:74.7},{month:"Aug",health:74.1},{month:"Sep",health:73.6},
    {month:"Oct",health:73.0},{month:"Nov",health:72.5},{month:"Dec",health:71.9},
    {month:"Jan",health:71.4},{month:"Feb",health:70.8},
  ],
  "model-3": [
    {month:"Now",health:94},{month:"Feb",health:93.8},{month:"Mar",health:93.6},
    {month:"Apr",health:93.4},{month:"May",health:93.2},{month:"Jun",health:93.0},
    {month:"Jul",health:92.8},{month:"Aug",health:92.6},{month:"Sep",health:92.4},
    {month:"Oct",health:92.2},{month:"Nov",health:92.0},{month:"Dec",health:91.8},
    {month:"Jan",health:91.6},{month:"Feb",health:91.4},
  ],
  "bolt": [
    {month:"Now",health:61},{month:"Feb",health:59.9},{month:"Mar",health:58.8},
    {month:"Apr",health:57.7},{month:"May",health:56.6},{month:"Jun",health:55.5},
    {month:"Jul",health:54.4},{month:"Aug",health:53.3},{month:"Sep",health:52.2},
    {month:"Oct",health:51.1},{month:"Nov",health:50.0},{month:"Dec",health:48.9},
    {month:"Jan",health:47.8},{month:"Feb",health:46.7},
  ],
};

const STATUS_CONFIG = {
  excellent: { label: "EXCELLENT", color: "#38bdf8" },
  good:      { label: "GOOD",      color: "#22c55e" },
  degraded:  { label: "DEGRADED",  color: "#f59e0b" },
  critical:  { label: "CRITICAL",  color: "#ef4444" },
};

function AnimatedNumber({ target, duration = 1200 }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    setVal(0);
    const start = performance.now();
    const step = (now) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setVal(Math.round(eased * target));
      if (p < 1) requestAnimationFrame(step);
    };
    const raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [target]);
  return <>{val}</>;
}

function CustomTooltip({ active, payload, label, accentColor }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#152335", border: `1px solid ${accentColor}44`,
      borderRadius: 8, padding: "10px 14px",
      fontSize: 11, fontFamily: "'DM Mono', monospace",
      boxShadow: `0 4px 20px rgba(0,0,0,0.6), 0 0 0 1px ${accentColor}22`,
    }}>
      <div style={{ color: "#6a85a0", letterSpacing: 2, marginBottom: 4, fontSize: 9 }}>{label}</div>
      <div style={{ color: accentColor, fontWeight: 500 }}>
        {payload[0].value}{payload[0].name === "health" ? "%" : " cycles"}
      </div>
    </div>
  );
}

function DashboardContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const carId = searchParams.get("car") || "model-s";
  const car = CARS[carId] || CARS["model-s"];
  const degradationData = DEGRADATION_DATA[carId] || DEGRADATION_DATA["model-s"];
  const predictedDecayData = PREDICTED_DECAY_DATA[carId] || PREDICTED_DECAY_DATA["model-s"];
  const status = STATUS_CONFIG[car.status];
  const ac = car.accentColor;
  const hc = car.health > 70 ? "#22c55e" : car.health > 40 ? "#f59e0b" : "#ef4444";

  const [time, setTime] = useState("");
  useEffect(() => {
    const update = () => setTime(new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    update();
    const t = setInterval(update, 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0d1b2a",
      fontFamily: "'DM Mono', monospace",
      color: "#e2e8f0",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --surface: #152335;
          --surface2: #1a2d42;
          --border: #243850;
          --muted: #6a85a0;
          --green: #22c55e;
        }

        .grid-bg {
          position: fixed; inset: 0;
          background-image:
            linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none; z-index: 0;
        }
        .grid-bg::after {
          content: '';
          position: absolute; inset: 0;
          background: radial-gradient(ellipse 80% 50% at 50% 0%, rgba(34,197,94,0.05) 0%, transparent 70%);
        }

        @keyframes fadeUp {
          from { opacity:0; transform:translateY(16px); }
          to { opacity:1; transform:translateY(0); }
        }
        @keyframes pulse {
          0%,100% { opacity:1; transform:scale(1); }
          50% { opacity:0.4; transform:scale(0.85); }
        }
        @keyframes flicker {
          0%,100% { opacity:1; } 92% { opacity:1; } 93% { opacity:0.8; } 94% { opacity:1; }
        }
        @keyframes healthFill {
          from { stroke-dashoffset: var(--circ); }
          to { stroke-dashoffset: var(--offset); }
        }

        .fade-up { animation: fadeUp 0.55s ease forwards; }
        .d1 { animation-delay:0.05s; opacity:0; }
        .d2 { animation-delay:0.12s; opacity:0; }
        .d3 { animation-delay:0.19s; opacity:0; }
        .d4 { animation-delay:0.26s; opacity:0; }
        .d5 { animation-delay:0.33s; opacity:0; }

        .header {
          background: rgba(5,10,18,0.9);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid var(--border);
          padding: 0 48px;
          height: 64px;
          display: flex; align-items: center; justify-content: space-between;
          position: sticky; top: 0; z-index: 50;
        }

        .card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 14px;
          padding: 28px;
          position: relative; overflow: hidden;
        }
        .card::after {
          content: '';
          position: absolute; top: 0; left: 0; right: 0; height: 1px;
          background: linear-gradient(90deg, transparent, var(--card-line, #243850), transparent);
        }

        .stat-label {
          font-size: 9px; letter-spacing: 3px;
          text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
        }

        .live-dot {
          width: 7px; height: 7px; border-radius: 50%;
          background: var(--green); animation: pulse 2s infinite;
          box-shadow: 0 0 8px var(--green); display: inline-block;
        }

        .back-btn {
          display: inline-flex; align-items: center; gap: 6px;
          background: rgba(30,45,61,0.6);
          border: 1px solid var(--border);
          border-radius: 8px; padding: 7px 14px;
          font-family: 'DM Mono', monospace;
          font-size: 9px; letter-spacing: 2px; color: var(--muted);
          cursor: pointer; text-transform: uppercase;
          transition: all 0.15s;
        }
        .back-btn:hover {
          background: rgba(34,197,94,0.08);
          border-color: rgba(34,197,94,0.3);
          color: var(--green);
        }

        .tag {
          display: inline-flex; align-items: center;
          padding: 4px 12px; border-radius: 99px;
          font-size: 9px; letter-spacing: 2px;
          border: 1px solid currentColor;
        }

        .segment-bar { display:flex; gap:3px; margin-top:10px; }
        .segment { flex:1; height:5px; border-radius:99px; }

        .health-track {
          height: 10px; background: #243850;
          border-radius: 99px; overflow: hidden; margin-top: 14px;
          border: 1px solid #2e4560;
        }
        .health-fill {
          height: 100%; border-radius: 99px;
          transition: width 1.5s cubic-bezier(0.16,1,0.3,1);
        }

        .voltwatch-logo {
          font-family: 'Bebas Neue', sans-serif;
          font-size: 22px; letter-spacing: 4px; color: #f0f9ff;
          animation: flicker 8s infinite;
        }
      `}</style>

      <div className="grid-bg" />

      {/* Header */}
      <header className="header" style={{ position: "relative", zIndex: 51 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            width: 36, height: 36,
            background: "linear-gradient(135deg, #22c55e, #16a34a)",
            borderRadius: 10, display: "flex", alignItems: "center",
            justifyContent: "center", fontSize: 18,
            boxShadow: "0 0 16px rgba(34,197,94,0.35)",
          }}>⚡</div>
          <div>
            <div className="voltwatch-logo">VOLTWATCH</div>
            <div style={{ fontSize: 8, letterSpacing: 3, color: "#2a4f70", marginTop: -2 }}>
              EV BATTERY INTELLIGENCE
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{
            fontSize: 11, letterSpacing: 2, color: "#3d6080",
            fontVariantNumeric: "tabular-nums",
          }}>
            {time}
          </div>
          <div style={{
            display: "flex", alignItems: "center", gap: 7,
            background: `${ac}0d`,
            border: `1px solid ${ac}25`,
            borderRadius: 99, padding: "5px 14px",
            fontSize: 9, letterSpacing: 2, color: ac,
          }}>
            <span className="live-dot" style={{ background: ac, boxShadow: `0 0 8px ${ac}` }} />
            LIVE · {car.year.toUpperCase()} {car.name.toUpperCase()}
          </div>
          <button className="back-btn" onClick={() => router.push("/home")}>
            ← GARAGE
          </button>
          <UserButton afterSignOutUrl="/sign-in" />
        </div>
      </header>



      {/* Main */}
      <main style={{ padding: "36px 48px", maxWidth: 1200, margin: "0 auto", position: "relative", zIndex: 1 }}>

        {/* Top row: Health card + Predicted Battery Life */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 16, marginBottom: 16 }}>

          {/* Battery Health */}
          <div className="card fade-up d1" style={{ "--card-line": hc }}>
            {/* Ambient glow */}
            <div style={{
              position: "absolute", top: -60, right: -60,
              width: 200, height: 200,
              background: `radial-gradient(circle, ${hc}15 0%, transparent 70%)`,
              pointerEvents: "none",
            }} />

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 4 }}>
              <div>
                <div className="stat-label">Battery Health</div>
                <div style={{
                  fontFamily: "'Bebas Neue', sans-serif",
                  fontSize: 88, lineHeight: 1,
                  color: hc, letterSpacing: -2,
                  textShadow: `0 0 40px ${hc}44`,
                }}>
                  <AnimatedNumber target={car.health} />
                  <span style={{ fontSize: 36, color: "#243850", marginLeft: 4 }}>%</span>
                </div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div className="tag" style={{ color: status.color, borderColor: `${status.color}40` }}>
                  {status.label}
                </div>
                <div style={{ fontSize: 9, color: "#3d6080", marginTop: 8, letterSpacing: 2 }}>
                  {100 - car.health}% DEGRADED
                </div>
              </div>
            </div>

            <div className="health-track">
              <div className="health-fill" style={{
                width: `${car.health}%`,
                background: `linear-gradient(90deg, ${hc}66, ${hc})`,
                boxShadow: `0 0 12px ${hc}66`,
              }} />
            </div>

            <div className="segment-bar">
              {Array.from({ length: 30 }).map((_, i) => (
                <div key={i} className="segment" style={{
                  background: i < Math.floor(car.health / (100/30)) ? `${hc}cc` : "#1f3248",
                  boxShadow: i < Math.floor(car.health / (100/30)) ? `0 0 4px ${hc}44` : "none",
                }} />
              ))}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
              <span style={{ fontSize: 8, color: "#3d6080", letterSpacing: 2 }}>0 · CRITICAL</span>
              <span style={{ fontSize: 8, color: "#3d6080", letterSpacing: 2 }}>100 · OPTIMAL</span>
            </div>
          </div>

          {/* Predicted Battery Life */}
          <div className="card fade-up d2" style={{
            display: "flex", flexDirection: "column",
            justifyContent: "center", alignItems: "center", textAlign: "center",
            "--card-line": ac,
          }}>
            <div style={{
              position: "absolute", inset: 0,
              background: `radial-gradient(ellipse at 50% 50%, ${ac}08 0%, transparent 70%)`,
              pointerEvents: "none",
            }} />
            <div className="stat-label">Predicted Battery Life</div>
            <div style={{
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: 72, lineHeight: 1,
              color: "#f0f9ff", letterSpacing: -1,
            }}>
              <AnimatedNumber target={car.daysLeft} duration={1400} />
            </div>
            <div style={{ fontSize: 9, color: "#6a85a0", letterSpacing: 3, marginTop: 6 }}>
              DAYS REMAINING
            </div>
            <div style={{ fontSize: 9, color: "#3d6080", letterSpacing: 1, marginTop: 4 }}>
              ≈ {Math.round(car.daysLeft / 365 * 10) / 10} years
            </div>
            <div style={{
              marginTop: 16, padding: "6px 14px",
              background: "rgba(30,45,61,0.6)",
              border: "1px solid #243850",
              borderRadius: 99, fontSize: 9, color: "#6a85a0", letterSpacing: 1,
            }}>
              Based on current degradation rate
            </div>
          </div>
        </div>

        {/* Charts row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

          {/* Degradation */}
          <div className="card fade-up d4" style={{ "--card-line": hc }}>
            <div style={{ marginBottom: 20 }}>
              <div className="stat-label">Battery Degradation</div>
              <div style={{ fontSize: 12, color: "#6a85a0", letterSpacing: 1 }}>Health over vehicle lifetime</div>
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <AreaChart data={degradationData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                <defs>
                  <linearGradient id={`grad-${carId}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={hc} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={hc} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f3248" vertical={false} />
                <XAxis dataKey="month" tick={{ fill: "#3d6080", fontSize: 9, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} />
                <YAxis domain={[55, 100]} tick={{ fill: "#3d6080", fontSize: 9, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip accentColor={hc} />} />
                <Area type="monotone" dataKey="health" stroke={hc} strokeWidth={2}
                  fill={`url(#grad-${carId})`}
                  dot={{ fill: hc, r: 3, strokeWidth: 0 }}
                  filter={`drop-shadow(0 0 4px ${hc}66)`}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Predicted Decay */}
          <div className="card fade-up d5" style={{ "--card-line": ac }}>
            <div style={{ marginBottom: 20 }}>
              <div className="stat-label">Predicted Battery Decay</div>
              <div style={{ fontSize: 12, color: "#6a85a0", letterSpacing: 1 }}>Projected health over next 14 months</div>
            </div>
            <ResponsiveContainer width="100%" height={190}>
              <AreaChart data={predictedDecayData} margin={{ top: 4, right: 4, left: -28, bottom: 0 }}>
                <defs>
                  <linearGradient id={`pred-grad-${carId}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={ac} stopOpacity={0.18} />
                    <stop offset="95%" stopColor={ac} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f3248" vertical={false} />
                <XAxis dataKey="month" tick={{ fill: "#3d6080", fontSize: 9, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} />
                <YAxis
                  domain={[
                    Math.floor(Math.min(...predictedDecayData.map(d => d.health)) - 3),
                    Math.ceil(Math.max(...predictedDecayData.map(d => d.health)) + 1),
                  ]}
                  tick={{ fill: "#3d6080", fontSize: 9, fontFamily: "DM Mono" }}
                  axisLine={false} tickLine={false}
                  tickFormatter={(v) => `${v}%`}
                />
                <Tooltip content={<CustomTooltip accentColor={ac} />} />
                <Area
                  type="monotone" dataKey="health"
                  stroke={ac} strokeWidth={2} strokeDasharray="5 3"
                  fill={`url(#pred-grad-${carId})`}
                  dot={{ fill: ac, r: 3, strokeWidth: 0 }}
                  filter={`drop-shadow(0 0 4px ${ac}66)`}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function Dashboard() {
  return (
    <Suspense fallback={<div style={{ minHeight: "100vh", background: "#0d1b2a" }} />}>
      <DashboardContent />
    </Suspense>
  );
}