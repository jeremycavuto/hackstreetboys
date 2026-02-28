"use client";

import { useState, useEffect } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { UserButton } from "@clerk/nextjs";

const degradationData = [
  { month: "Jan", health: 100 }, { month: "Mar", health: 98 },
  { month: "May", health: 96 }, { month: "Jul", health: 93 },
  { month: "Sep", health: 90 }, { month: "Nov", health: 87 },
  { month: "Jan", health: 84 }, { month: "Mar", health: 81 },
  { month: "Now", health: 78 },
];

const cycleData = [
  { week: "W1", cycles: 4 }, { week: "W2", cycles: 7 },
  { week: "W3", cycles: 3 }, { week: "W4", cycles: 6 },
  { week: "W5", cycles: 5 }, { week: "W6", cycles: 8 },
  { week: "W7", cycles: 4 }, { week: "W8", cycles: 6 },
];

const BATTERY_HEALTH = 78;
const DAYS_LEFT = 312;

export default function Dashboard() {
  const [animatedHealth, setAnimatedHealth] = useState(0);
  const [animatedDays, setAnimatedDays] = useState(0);

  useEffect(() => {
    const healthTimer = setTimeout(() => {
      let start = 0;
      const step = setInterval(() => {
        start += 2;
        if (start >= BATTERY_HEALTH) { setAnimatedHealth(BATTERY_HEALTH); clearInterval(step); }
        else setAnimatedHealth(start);
      }, 16);
    }, 300);

    const daysTimer = setTimeout(() => {
      let start = 0;
      const step = setInterval(() => {
        start += 5;
        if (start >= DAYS_LEFT) { setAnimatedDays(DAYS_LEFT); clearInterval(step); }
        else setAnimatedDays(start);
      }, 10);
    }, 500);

    return () => { clearTimeout(healthTimer); clearTimeout(daysTimer); };
  }, []);

  const healthColor = animatedHealth > 70 ? "#16a34a" : animatedHealth > 40 ? "#d97706" : "#dc2626";
  const healthBg = animatedHealth > 70 ? "#dcfce7" : animatedHealth > 40 ? "#fef3c7" : "#fee2e2";
  const statusLabel = animatedHealth > 70 ? "GOOD" : animatedHealth > 40 ? "DEGRADED" : "CRITICAL";

  return (
    <div style={{
      minHeight: "100vh",
      background: "#f0f4f8",
      color: "#1a202c",
      fontFamily: "'DM Mono', 'Courier New', monospace",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }

        .card {
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          padding: 28px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
        }

        .stat-label {
          font-size: 10px;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: #94a3b8;
          margin-bottom: 8px;
        }

        .health-bar-bg {
          height: 16px;
          background: #f1f5f9;
          border-radius: 99px;
          overflow: hidden;
          margin-top: 16px;
          border: 1px solid #e2e8f0;
        }

        .health-bar-fill {
          height: 100%;
          border-radius: 99px;
          transition: width 1.5s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .segment-bar {
          display: flex;
          gap: 3px;
          margin-top: 10px;
        }
        .segment {
          flex: 1;
          height: 6px;
          border-radius: 99px;
          transition: background 0.3s;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        .live-dot {
          width: 8px; height: 8px;
          border-radius: 50%;
          background: #16a34a;
          animation: pulse 2s infinite;
          display: inline-block;
          margin-right: 6px;
        }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(16px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .fade-up { animation: fadeUp 0.5s ease forwards; }
        .d1 { animation-delay: 0.05s; opacity: 0; }
        .d2 { animation-delay: 0.15s; opacity: 0; }
        .d3 { animation-delay: 0.25s; opacity: 0; }
        .d4 { animation-delay: 0.35s; opacity: 0; }
      `}</style>

      {/* Header */}
      <header style={{
        background: "#ffffff",
        borderBottom: "1px solid #e2e8f0",
        padding: "18px 40px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <div style={{
            width: 38, height: 38,
            background: "linear-gradient(135deg, #22c55e, #16a34a)",
            borderRadius: "10px",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "20px",
            boxShadow: "0 2px 8px #16a34a33",
          }}>⚡</div>
          <div>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: "22px", letterSpacing: "3px", color: "#0f172a" }}>
              VOLTWATCH
            </div>
            <div style={{ fontSize: "9px", letterSpacing: "3px", color: "#94a3b8", marginTop: "-2px" }}>
              EV BATTERY HEALTH TRACKER
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", color: "#64748b", letterSpacing: "1px" }}>
            <span className="live-dot" />
            LIVE · 2021 MODEL S
          </div>
          <UserButton afterSignOutUrl="/" />
        </div>
      </header>

      {/* Main */}
      <main style={{ padding: "32px 40px", maxWidth: "1100px", margin: "0 auto" }}>

        {/* Health + Days Row */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "16px", marginBottom: "16px" }}>

          {/* Battery Health */}
          <div className="card fade-up d1">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div className="stat-label">Battery Health</div>
                <div style={{
                  fontFamily: "'Bebas Neue', sans-serif",
                  fontSize: "72px",
                  lineHeight: 1,
                  color: healthColor,
                  letterSpacing: "-1px",
                }}>
                  {animatedHealth}<span style={{ fontSize: "32px", color: "#cbd5e1", marginLeft: "2px" }}>%</span>
                </div>
              </div>
              <div style={{
                padding: "6px 14px",
                background: healthBg,
                color: healthColor,
                borderRadius: "99px",
                fontSize: "10px",
                fontWeight: "500",
                letterSpacing: "2px",
                marginTop: "4px",
              }}>
                {statusLabel}
              </div>
            </div>

            <div className="health-bar-bg">
              <div className="health-bar-fill" style={{
                width: `${animatedHealth}%`,
                background: `linear-gradient(90deg, ${healthColor}99, ${healthColor})`,
              }} />
            </div>

            <div className="segment-bar">
              {Array.from({ length: 25 }).map((_, i) => (
                <div key={i} className="segment" style={{
                  background: i < Math.floor(animatedHealth / 4) ? healthColor : "#e2e8f0",
                }} />
              ))}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", marginTop: "8px" }}>
              <span style={{ fontSize: "9px", color: "#cbd5e1", letterSpacing: "2px" }}>0% CRITICAL</span>
              <span style={{ fontSize: "9px", color: "#cbd5e1", letterSpacing: "2px" }}>100% OPTIMAL</span>
            </div>
          </div>

          {/* Days Remaining */}
          <div className="card fade-up d2" style={{ display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center", textAlign: "center" }}>
            <div className="stat-label">Est. Days of Health Remaining</div>
            <div style={{
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: "88px",
              lineHeight: 1,
              color: "#0f172a",
              letterSpacing: "-2px",
              marginTop: "8px",
            }}>
              {animatedDays}
            </div>
            <div style={{
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: "20px",
              color: "#94a3b8",
              letterSpacing: "6px",
              marginTop: "4px",
            }}>
              DAYS
            </div>
            <div style={{ marginTop: "16px", fontSize: "10px", color: "#94a3b8", letterSpacing: "1px" }}>
              ≈ {(DAYS_LEFT / 365).toFixed(1)} years remaining
            </div>
            <div style={{
              marginTop: "10px",
              padding: "5px 14px",
              background: "#f8fafc",
              border: "1px solid #e2e8f0",
              borderRadius: "99px",
              fontSize: "9px",
              letterSpacing: "2px",
              color: "#94a3b8",
            }}>
              REPLACE BY 2028
            </div>
          </div>
        </div>

        {/* Charts Row */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>

          {/* Health Degradation Over Time */}
          <div className="card fade-up d3">
            <div style={{ marginBottom: "20px" }}>
              <div className="stat-label">Health Degradation</div>
              <div style={{ fontSize: "13px", color: "#64748b" }}>Over time (2-year view)</div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <AreaChart data={degradationData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                <defs>
                  <linearGradient id="healthGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#16a34a" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#16a34a" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} domain={[70, 102]} />
                <Tooltip
                  contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px", fontSize: "11px", fontFamily: "DM Mono", boxShadow: "0 4px 12px rgba(0,0,0,0.08)" }}
                  labelStyle={{ color: "#94a3b8" }}
                  itemStyle={{ color: "#16a34a" }}
                />
                <Area type="monotone" dataKey="health" stroke="#16a34a" strokeWidth={2.5} fill="url(#healthGrad)" dot={{ fill: "#16a34a", r: 3, strokeWidth: 0 }} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Charge Cycles per Week */}
          <div className="card fade-up d4">
            <div style={{ marginBottom: "20px" }}>
              <div className="stat-label">Charge Cycles</div>
              <div style={{ fontSize: "13px", color: "#64748b" }}>Per week (last 8 weeks)</div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={cycleData} margin={{ top: 4, right: 4, left: -24, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="week" tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 10, fontFamily: "DM Mono" }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: "8px", fontSize: "11px", fontFamily: "DM Mono", boxShadow: "0 4px 12px rgba(0,0,0,0.08)" }}
                  labelStyle={{ color: "#94a3b8" }}
                  itemStyle={{ color: "#3b82f6" }}
                />
                <Line type="monotone" dataKey="cycles" stroke="#3b82f6" strokeWidth={2.5} dot={{ fill: "#3b82f6", r: 3, strokeWidth: 0 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </main>
    </div>
  );
}