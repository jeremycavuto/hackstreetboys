"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { UserButton, useUser } from "@clerk/nextjs";

const CARS = [
  {
    id: "model-s",
    name: "Tesla Model S",
    year: "2021",
    nickname: "Daily Driver",
    battery: "100 kWh",
    health: 78,
    range: 312,
    lastCharge: "2h ago",
    odometer: "42,381 mi",
    status: "good",
    accentColor: "#22c55e",
    tempColor: "#22c55e",
    icon: "S",
  },
  {
    id: "model-3",
    name: "Tesla Model 3",
    year: "2023",
    nickname: "Weekend Car",
    battery: "82 kWh",
    health: 94,
    range: 348,
    lastCharge: "1d ago",
    odometer: "11,204 mi",
    status: "excellent",
    accentColor: "#38bdf8",
    tempColor: "#38bdf8",
    icon: "3",
  },
  {
    id: "bolt",
    name: "Chevy Bolt EV",
    year: "2020",
    nickname: "Commuter",
    battery: "65 kWh",
    health: 61,
    range: 189,
    lastCharge: "3d ago",
    odometer: "67,920 mi",
    status: "degraded",
    accentColor: "#f59e0b",
    tempColor: "#f59e0b",
    icon: "B",
  },
];

const STATUS_CONFIG = {
  excellent: { label: "EXCELLENT", color: "#38bdf8" },
  good:      { label: "GOOD",      color: "#22c55e" },
  degraded:  { label: "DEGRADED",  color: "#f59e0b" },
  critical:  { label: "CRITICAL",  color: "#ef4444" },
};

function RadialProgress({ value, color, size = 80 }) {
  const r = (size / 2) - 8;
  const circ = 2 * Math.PI * r;
  const offset = circ - (value / 100) * circ;
  return (
    <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#1e293b" strokeWidth="6" />
      <circle
        cx={size/2} cy={size/2} r={r}
        fill="none"
        stroke={color}
        strokeWidth="6"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.16,1,0.3,1)", filter: `drop-shadow(0 0 6px ${color}88)` }}
      />
    </svg>
  );
}

export default function HomePage() {
  const { user } = useUser();
  const router = useRouter();
  const firstName = user?.firstName || "Driver";

  const [phase, setPhase] = useState("welcome");
  const [skipSplash, setSkipSplash] = useState(true);
  const [hoveredCar, setHoveredCar] = useState(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const seen = sessionStorage.getItem("voltwatch_welcomed");
    if (seen) {
      setPhase("garage");
      setSkipSplash(true);
    } else {
      setSkipSplash(false);
      setPhase("welcome");
      const slideTimer = setTimeout(() => setPhase("sliding"), 2000);
      const garageTimer = setTimeout(() => {
        setPhase("garage");
        sessionStorage.setItem("voltwatch_welcomed", "true");
      }, 2650);
      return () => { clearTimeout(slideTimer); clearTimeout(garageTimer); };
    }
  }, []);

  // Subtle "live" ticker
  useEffect(() => {
    const t = setInterval(() => setTick(p => p + 1), 2000);
    return () => clearInterval(t);
  }, []);

  const handleSelectCar = (car) => {
    router.push(`/dashboard?car=${car.id}`);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#050a12",
      fontFamily: "'DM Mono', monospace",
      color: "#e2e8f0",
      overflow: "hidden",
      position: "relative",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&family=Space+Grotesk:wght@300;400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --green: #22c55e;
          --green-glow: #22c55e44;
          --surface: #0d1520;
          --surface2: #111c2d;
          --border: #1e2d3d;
          --muted: #4a6080;
        }

        /* Animated grid background */
        .grid-bg {
          position: fixed;
          inset: 0;
          background-image:
            linear-gradient(rgba(34,197,94,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(34,197,94,0.03) 1px, transparent 1px);
          background-size: 40px 40px;
          pointer-events: none;
          z-index: 0;
        }
        .grid-bg::after {
          content: '';
          position: absolute;
          inset: 0;
          background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(34,197,94,0.07) 0%, transparent 70%);
        }

        /* Welcome splash */
        @keyframes splashIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes splashOut {
          from { transform: translateX(0); opacity: 1; }
          to { transform: translateX(-100%); opacity: 0; }
        }
        @keyframes garageIn {
          from { opacity: 0; transform: translateX(40px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(18px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%,100% { opacity:1; transform: scale(1); }
          50% { opacity:0.4; transform: scale(0.85); }
        }
        @keyframes scanline {
          0% { transform: translateY(-100%); }
          100% { transform: translateY(100vh); }
        }
        @keyframes loadBar {
          from { width: 0%; }
          to { width: 100%; }
        }
        @keyframes flicker {
          0%,100% { opacity:1; } 92% { opacity:1; } 93% { opacity:0.8; } 94% { opacity:1; }
        }
        @keyframes floatUp {
          0%,100% { transform: translateY(0px); }
          50% { transform: translateY(-6px); }
        }

        .splash {
          position: fixed; inset: 0; z-index: 200;
          background: #020810;
          display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          animation: splashIn 0.4s ease;
        }
        .splash.out { animation: splashOut 0.65s cubic-bezier(0.76,0,0.24,1) forwards; }

        .splash-scanline {
          position: absolute; left: 0; right: 0; height: 2px;
          background: linear-gradient(90deg, transparent, rgba(34,197,94,0.3), transparent);
          animation: scanline 2s linear infinite;
          pointer-events: none;
        }

        .garage-wrap {
          position: relative; z-index: 1;
          animation: garageIn 0.7s cubic-bezier(0.16,1,0.3,1) forwards;
        }
        .garage-wrap-instant { position: relative; z-index: 1; }

        .fade-up { animation: fadeUp 0.6s ease forwards; }
        .d1 { animation-delay: 0.05s; opacity:0; }
        .d2 { animation-delay: 0.15s; opacity:0; }
        .d3 { animation-delay: 0.22s; opacity:0; }
        .d4 { animation-delay: 0.29s; opacity:0; }
        .d5 { animation-delay: 0.36s; opacity:0; }

        /* Header */
        .header {
          background: rgba(5,10,18,0.85);
          backdrop-filter: blur(20px);
          border-bottom: 1px solid var(--border);
          padding: 16px 48px;
          display: flex; align-items: center; justify-content: space-between;
          position: sticky; top: 0; z-index: 50;
        }

        /* Car cards */
        .car-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 28px;
          cursor: pointer;
          position: relative;
          overflow: hidden;
          transition: border-color 0.25s, box-shadow 0.25s, transform 0.25s;
        }
        .car-card::before {
          content: '';
          position: absolute;
          inset: 0;
          background: radial-gradient(ellipse 120% 80% at 50% -20%, var(--card-glow, transparent), transparent 70%);
          opacity: 0;
          transition: opacity 0.3s;
          pointer-events: none;
        }
        .car-card:hover::before { opacity: 1; }
        .car-card:hover {
          transform: translateY(-5px);
          box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 0 1px var(--card-accent, #22c55e);
        }

        .cta-strip {
          position: absolute; bottom: 0; left: 0; right: 0;
          padding: 12px;
          font-size: 9px; letter-spacing: 3px; text-align: center;
          font-family: 'DM Mono', monospace;
          transform: translateY(100%);
          transition: transform 0.25s cubic-bezier(0.16,1,0.3,1);
          color: #020810;
          font-weight: 600;
        }
        .car-card:hover .cta-strip { transform: translateY(0); }

        .live-dot {
          width: 7px; height: 7px; border-radius: 50%;
          background: var(--green);
          animation: pulse 2s infinite;
          box-shadow: 0 0 8px var(--green);
          display: inline-block;
        }

        .stat-label {
          font-size: 9px; letter-spacing: 3px;
          text-transform: uppercase; color: var(--muted);
          margin-bottom: 3px;
        }

        .health-track {
          height: 3px; background: #1e2d3d;
          border-radius: 99px; overflow: hidden; margin-top: 6px;
        }
        .health-fill {
          height: 100%; border-radius: 99px;
        }

        .tag {
          display: inline-flex; align-items: center;
          padding: 3px 10px; border-radius: 99px;
          font-size: 9px; letter-spacing: 2px;
          border: 1px solid currentColor;
        }

        /* Stat summary cards */
        .stat-card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 12px;
          padding: 20px 24px;
          position: relative; overflow: hidden;
        }
        .stat-card::after {
          content: '';
          position: absolute; top: 0; left: 0; right: 0; height: 1px;
          background: linear-gradient(90deg, transparent, var(--green), transparent);
          opacity: 0.4;
        }

        .voltwatch-logo {
          font-family: 'Bebas Neue', sans-serif;
          font-size: 24px; letter-spacing: 4px;
          color: #f0f9ff;
          animation: flicker 8s infinite;
        }

        .number-big {
          font-family: 'Bebas Neue', sans-serif;
          letter-spacing: -1px; line-height: 1;
        }
      `}</style>

      {/* Grid background */}
      <div className="grid-bg" />

      {/* ── WELCOME SPLASH ── */}
      {!skipSplash && (phase === "welcome" || phase === "sliding") && (
        <div className={`splash ${phase === "sliding" ? "out" : ""}`}>
          <div className="splash-scanline" />

          {/* Ambient glow */}
          <div style={{
            position: "absolute", width: 500, height: 500,
            background: "radial-gradient(circle, rgba(34,197,94,0.12) 0%, transparent 70%)",
            borderRadius: "50%", top: "50%", left: "50%",
            transform: "translate(-50%, -50%)",
            pointerEvents: "none",
          }} />

          {/* Logo mark */}
          <div style={{
            width: 72, height: 72,
            background: "linear-gradient(135deg, #22c55e, #16a34a)",
            borderRadius: "18px",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "34px",
            boxShadow: "0 0 40px rgba(34,197,94,0.4), 0 0 80px rgba(34,197,94,0.15)",
            marginBottom: "32px",
            animation: "floatUp 3s ease-in-out infinite",
          }}>⚡</div>

          {/* WELCOME */}
          <div style={{
            fontSize: "13px", letterSpacing: "8px", color: "#4a6080",
            textTransform: "uppercase", marginBottom: "12px",
          }}>
            VOLTWATCH · EV HEALTH
          </div>

          <div style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: "72px", letterSpacing: "4px",
            color: "#f0f9ff", lineHeight: 1, textAlign: "center",
          }}>
            WELCOME BACK
          </div>

          <div style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: "72px", letterSpacing: "4px",
            background: "linear-gradient(90deg, #22c55e, #4ade80)",
            WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
            lineHeight: 1, textAlign: "center",
          }}>
            {firstName.toUpperCase()}
          </div>

          {/* Loading bar */}
          <div style={{
            width: 160, height: 1, background: "#1e2d3d",
            borderRadius: 99, marginTop: 40, overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              background: "linear-gradient(90deg, #22c55e, #4ade80)",
              boxShadow: "0 0 12px #22c55e",
              animation: "loadBar 1.9s cubic-bezier(0.4,0,0.2,1) forwards",
            }} />
          </div>

          <div style={{
            fontSize: "9px", letterSpacing: "4px", color: "#2a3f55",
            marginTop: 12,
          }}>
            LOADING YOUR GARAGE
          </div>
        </div>
      )}

      {/* ── GARAGE ── */}
      {(phase === "garage" || phase === "sliding") && (
        <div className={phase === "garage" && !skipSplash ? "garage-wrap" : "garage-wrap-instant"}>

          {/* Header */}
          <header className="header">
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{
                width: 36, height: 36,
                background: "linear-gradient(135deg, #22c55e, #16a34a)",
                borderRadius: 10,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 18,
                boxShadow: "0 0 16px rgba(34,197,94,0.35)",
              }}>⚡</div>
              <div>
                <div className="voltwatch-logo">VOLTWATCH</div>
                <div style={{ fontSize: 8, letterSpacing: 3, color: "#2a4a6a", marginTop: -2 }}>
                  EV BATTERY INTELLIGENCE
                </div>
              </div>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
              {/* Live ticker */}
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                background: "rgba(34,197,94,0.06)",
                border: "1px solid rgba(34,197,94,0.15)",
                borderRadius: 99, padding: "5px 14px",
                fontSize: 9, letterSpacing: 2, color: "#22c55e",
              }}>
                <span className="live-dot" />
                LIVE MONITORING
              </div>
              <div style={{
                fontSize: 10, letterSpacing: 2, color: "#4a6080",
                fontVariantNumeric: "tabular-nums",
              }}>
                {new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}
              </div>
              <UserButton afterSignOutUrl="/sign-in" />
            </div>
          </header>

          <main style={{ padding: "48px", maxWidth: 1100, margin: "0 auto" }}>

            {/* Page title */}
            <div className="fade-up d1" style={{ marginBottom: 48 }}>
              <div style={{ fontSize: 9, letterSpacing: 5, color: "#22c55e", textTransform: "uppercase", marginBottom: 10 }}>
                ◈ GARAGE VIEW
              </div>
              <h1 style={{
                fontFamily: "'Bebas Neue', sans-serif",
                fontSize: 52, letterSpacing: 3, color: "#f0f9ff", lineHeight: 1,
              }}>
                {firstName.toUpperCase()}'S FLEET
              </h1>
              <p style={{ fontSize: 11, color: "#4a6080", marginTop: 8, letterSpacing: 1 }}>
                Select a vehicle to open its live battery intelligence dashboard
              </p>
            </div>

            {/* Summary strip */}
            <div className="fade-up d2" style={{
              display: "grid", gridTemplateColumns: "repeat(4,1fr)",
              gap: 12, marginBottom: 40,
            }}>
              {[
                { label: "Fleet Size",        value: CARS.length,    unit: " vehicles" },
                { label: "Avg Health",         value: Math.round(CARS.reduce((a,c)=>a+c.health,0)/CARS.length), unit: "%" },
                { label: "Best Health",        value: Math.max(...CARS.map(c=>c.health)), unit: "%" },
                { label: "Need Attention",     value: CARS.filter(c=>c.health<70).length, unit: " alerts" },
              ].map((s, i) => (
                <div key={i} className="stat-card">
                  <div className="stat-label">{s.label}</div>
                  <div style={{
                    fontFamily: "'Bebas Neue', sans-serif",
                    fontSize: 34, color: "#f0f9ff", letterSpacing: 1, marginTop: 2,
                  }}>
                    {s.value}<span style={{ fontSize: 14, color: "#4a6080", marginLeft: 2 }}>{s.unit}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Car cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 20 }}>
              {CARS.map((car, i) => {
                const status = STATUS_CONFIG[car.status];
                const hc = car.health > 70 ? "#22c55e" : car.health > 40 ? "#f59e0b" : "#ef4444";
                const isHovered = hoveredCar === car.id;

                return (
                  <div
                    key={car.id}
                    className={`car-card fade-up d${i + 3}`}
                    style={{
                      "--card-accent": car.accentColor,
                      "--card-glow": `${car.accentColor}18`,
                    }}
                    onMouseEnter={() => setHoveredCar(car.id)}
                    onMouseLeave={() => setHoveredCar(null)}
                    onClick={() => handleSelectCar(car)}
                  >
                    {/* Top row */}
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                        {/* Icon circle */}
                        <div style={{
                          width: 48, height: 48,
                          background: `${car.accentColor}14`,
                          border: `1px solid ${car.accentColor}30`,
                          borderRadius: 12,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          fontFamily: "'Bebas Neue', sans-serif",
                          fontSize: 22, color: car.accentColor,
                          boxShadow: isHovered ? `0 0 20px ${car.accentColor}30` : "none",
                          transition: "box-shadow 0.3s",
                        }}>
                          {car.icon}
                        </div>
                        <div>
                          <div style={{ fontSize: 9, color: "#4a6080", letterSpacing: 2 }}>{car.year}</div>
                          <div style={{ fontSize: 14, fontWeight: 500, color: "#f0f9ff", letterSpacing: 0.3 }}>{car.name}</div>
                          <div style={{ fontSize: 9, letterSpacing: 2, color: "#4a6080", marginTop: 2 }}>{car.nickname.toUpperCase()}</div>
                        </div>
                      </div>

                      {/* Radial */}
                      <div style={{ position: "relative" }}>
                        <RadialProgress value={car.health} color={hc} size={72} />
                        <div style={{
                          position: "absolute", inset: 0,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          flexDirection: "column",
                        }}>
                          <div style={{
                            fontFamily: "'Bebas Neue', sans-serif",
                            fontSize: 18, color: "#f0f9ff", lineHeight: 1,
                          }}>{car.health}</div>
                          <div style={{ fontSize: 7, color: "#4a6080", letterSpacing: 1 }}>%</div>
                        </div>
                      </div>
                    </div>

                    {/* Health bar */}
                    <div style={{ marginBottom: 20 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                        <span className="stat-label" style={{ marginBottom: 0 }}>Battery Health</span>
                        <span className="tag" style={{ color: status.color, borderColor: `${status.color}40`, fontSize: 8 }}>
                          {status.label}
                        </span>
                      </div>
                      <div className="health-track">
                        <div className="health-fill" style={{
                          width: `${car.health}%`,
                          background: `linear-gradient(90deg, ${hc}88, ${hc})`,
                          boxShadow: `0 0 8px ${hc}66`,
                        }} />
                      </div>
                    </div>

                    {/* Degradation note */}
                    <div style={{
                      paddingTop: 16,
                      borderTop: "1px solid #1e2d3d",
                      paddingBottom: 12,
                      fontSize: 9, color: "#2a4a6a", letterSpacing: 1,
                    }}>
                      {100 - car.health}% CAPACITY LOST SINCE NEW
                    </div>

                    {/* Hover CTA */}
                    <div
                      className="cta-strip"
                      style={{ background: `linear-gradient(90deg, ${car.accentColor}, ${car.accentColor}cc)` }}
                    >
                      OPEN DASHBOARD →
                    </div>
                  </div>
                );
              })}
            </div>
          </main>
        </div>
      )}
    </div>
  );
}