"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { UserButton, useUser } from "@clerk/nextjs";

const CARS = [
  {
    id: "model-s",
    name: "2021 Tesla Model S",
    nickname: "Daily Driver",
    battery: "100 kWh",
    health: 78,
    range: 312,
    lastCharge: "2h ago",
    odometer: "42,381 mi",
    status: "good",
    color: "#dc2626",
    icon: "🚗",
  },
  {
    id: "model-3",
    name: "2023 Tesla Model 3",
    nickname: "Weekend Car",
    battery: "82 kWh",
    health: 94,
    range: 348,
    lastCharge: "1d ago",
    odometer: "11,204 mi",
    status: "excellent",
    color: "#2563eb",
    icon: "🚙",
  },
  {
    id: "bolt",
    name: "2020 Chevy Bolt",
    nickname: "Commuter",
    battery: "65 kWh",
    health: 61,
    range: 189,
    lastCharge: "3d ago",
    odometer: "67,920 mi",
    status: "degraded",
    color: "#d97706",
    icon: "🚘",
  },
];

const STATUS_CONFIG = {
  excellent: { label: "EXCELLENT", color: "#0891b2", bg: "#e0f2fe" },
  good:      { label: "GOOD",      color: "#16a34a", bg: "#dcfce7" },
  degraded:  { label: "DEGRADED",  color: "#d97706", bg: "#fef3c7" },
  critical:  { label: "CRITICAL",  color: "#dc2626", bg: "#fee2e2" },
};

export default function HomePage() {
  const { user } = useUser();
  const router = useRouter();
  const firstName = user?.firstName || "Driver";

  // Phase: "welcome" | "sliding" | "garage"
  const [phase, setPhase] = useState("welcome");
  const [skipSplash, setSkipSplash] = useState(true); // default true to avoid flash

  useEffect(() => {
    // Check sessionStorage — if already seen this session, skip splash
    const seen = sessionStorage.getItem("voltwatch_welcomed");
    if (seen) {
      setPhase("garage");
      setSkipSplash(true);
    } else {
      setSkipSplash(false);
      setPhase("welcome");

      // After 1.8s show welcome, start slide-out
      const slideTimer = setTimeout(() => {
        setPhase("sliding");
      }, 1800);

      // After slide animation (600ms), switch to garage
      const garageTimer = setTimeout(() => {
        setPhase("garage");
        sessionStorage.setItem("voltwatch_welcomed", "true");
      }, 2400);

      return () => {
        clearTimeout(slideTimer);
        clearTimeout(garageTimer);
      };
    }
  }, []);

  const handleSelectCar = (car) => {
    router.push(`/dashboard?car=${car.id}`);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#f0f4f8",
      fontFamily: "'DM Mono', 'Courier New', monospace",
      color: "#0f172a",
      overflow: "hidden",
      position: "relative",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Bebas+Neue&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        @keyframes welcomeFadeIn {
          from { opacity: 0; transform: scale(0.97); }
          to { opacity: 1; transform: scale(1); }
        }
        @keyframes slideOutLeft {
          from { transform: translateX(0); opacity: 1; }
          to { transform: translateX(-100%); opacity: 0; }
        }
        @keyframes slideInRight {
          from { transform: translateX(60px); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }

        .welcome-screen {
          position: fixed;
          inset: 0;
          z-index: 100;
          background: #0f172a;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          gap: 16px;
          animation: welcomeFadeIn 0.5s ease forwards;
        }
        .welcome-screen.sliding {
          animation: slideOutLeft 0.6s cubic-bezier(0.76, 0, 0.24, 1) forwards;
        }

        .garage-content {
          animation: slideInRight 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        .fade-up { animation: fadeUp 0.5s ease forwards; }
        .d1 { animation-delay: 0.05s; opacity: 0; }
        .d2 { animation-delay: 0.12s; opacity: 0; }
        .d3 { animation-delay: 0.19s; opacity: 0; }
        .d4 { animation-delay: 0.26s; opacity: 0; }

        .card {
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 14px;
          padding: 28px;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
        }

        .car-card {
          cursor: pointer;
          transition: all 0.22s ease;
          position: relative;
          overflow: hidden;
        }
        .car-card::after {
          content: "VIEW DASHBOARD →";
          position: absolute;
          bottom: 0; left: 0; right: 0;
          background: linear-gradient(135deg, #22c55e, #16a34a);
          color: white;
          text-align: center;
          font-size: 10px;
          letter-spacing: 2.5px;
          padding: 13px;
          transform: translateY(100%);
          transition: transform 0.22s ease;
          font-family: 'DM Mono', monospace;
        }
        .car-card:hover {
          box-shadow: 0 8px 30px rgba(0,0,0,0.12);
          transform: translateY(-4px);
          border-color: #16a34a;
        }
        .car-card:hover::after {
          transform: translateY(0);
        }

        .stat-label {
          font-size: 9px;
          letter-spacing: 3px;
          text-transform: uppercase;
          color: #94a3b8;
          margin-bottom: 4px;
        }

        .live-dot {
          width: 7px; height: 7px;
          border-radius: 50%;
          background: #16a34a;
          animation: pulse 2s infinite;
          display: inline-block;
        }

        .health-bar-bg {
          height: 5px;
          background: #f1f5f9;
          border-radius: 99px;
          overflow: hidden;
          margin-top: 8px;
        }
        .health-bar-fill {
          height: 100%;
          border-radius: 99px;
        }
      `}</style>

      {/* ── Welcome Splash ── */}
      {(phase === "welcome" || phase === "sliding") && !skipSplash && (
        <div className={`welcome-screen ${phase === "sliding" ? "sliding" : ""}`}>
          {/* Logo */}
          <div style={{
            width: 56, height: 56,
            background: "linear-gradient(135deg, #22c55e, #16a34a)",
            borderRadius: "14px",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "28px",
            boxShadow: "0 4px 24px #16a34a55",
            marginBottom: "8px",
          }}>⚡</div>

          {/* Welcome text */}
          <div style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: "56px",
            letterSpacing: "4px",
            color: "#ffffff",
            lineHeight: 1,
            textAlign: "center",
          }}>
            WELCOME BACK
          </div>
          <div style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: "56px",
            letterSpacing: "4px",
            background: "linear-gradient(135deg, #22c55e, #16a34a)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            lineHeight: 1,
            textAlign: "center",
          }}>
            {firstName.toUpperCase()}
          </div>

          {/* Subtitle */}
          <div style={{
            fontSize: "10px",
            letterSpacing: "4px",
            color: "#475569",
            textTransform: "uppercase",
            marginTop: "8px",
          }}>
            EV Battery Health Tracker
          </div>

          {/* Loading bar */}
          <div style={{
            width: "120px",
            height: "2px",
            background: "#1e293b",
            borderRadius: "99px",
            marginTop: "32px",
            overflow: "hidden",
          }}>
            <div style={{
              height: "100%",
              background: "linear-gradient(90deg, #22c55e, #16a34a)",
              borderRadius: "99px",
              animation: "slideInRight 1.6s ease forwards",
              width: "100%",
              transformOrigin: "left",
            }} />
          </div>
        </div>
      )}

      {/* ── Garage (shown after splash or if already seen) ── */}
      {(phase === "garage" || phase === "sliding") && (
        <div className={phase === "garage" && !skipSplash ? "garage-content" : ""}>

          {/* Header */}
          <header style={{
            background: "#ffffff",
            borderBottom: "1px solid #e2e8f0",
            padding: "18px 40px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            position: "sticky",
            top: 0,
            zIndex: 50,
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
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "#64748b", letterSpacing: "1px" }}>
                <span className="live-dot" />
                <span>GARAGE</span>
              </div>
              <UserButton afterSignOutUrl="/sign-in" />
            </div>
          </header>

          {/* Main */}
          <main style={{ padding: "48px 40px", maxWidth: "1000px", margin: "0 auto" }}>

            {/* Welcome */}
            <div className="fade-up d1" style={{ marginBottom: "48px" }}>
              <div style={{ fontSize: "9px", letterSpacing: "4px", color: "#94a3b8", textTransform: "uppercase", marginBottom: "8px" }}>
                Welcome back
              </div>
              <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: "48px", letterSpacing: "3px", color: "#0f172a", lineHeight: 1 }}>
                {firstName.toUpperCase()}'S GARAGE
              </div>
              <div style={{ fontSize: "11px", color: "#94a3b8", marginTop: "10px", letterSpacing: "1px" }}>
                Choose a vehicle below to open its battery health dashboard
              </div>
            </div>

            {/* Car Cards */}
            <div style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
              gap: "20px",
            }}>
              {CARS.map((car, i) => {
                const status = STATUS_CONFIG[car.status];
                const healthColor = car.health > 70 ? "#16a34a" : car.health > 40 ? "#d97706" : "#dc2626";

                return (
                  <div
                    key={car.id}
                    className={`card car-card fade-up d${i + 2}`}
                    onClick={() => handleSelectCar(car)}
                  >
                    {/* Car icon + name */}
                    <div style={{ display: "flex", alignItems: "center", gap: "14px", marginBottom: "22px" }}>
                      <div style={{
                        width: 50, height: 50,
                        background: `${car.color}12`,
                        border: `1px solid ${car.color}25`,
                        borderRadius: "12px",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: "24px",
                        flexShrink: 0,
                      }}>
                        {car.icon}
                      </div>
                      <div>
                        <div style={{ fontSize: "13px", fontWeight: "500", color: "#0f172a" }}>
                          {car.name}
                        </div>
                        <div style={{ fontSize: "9px", letterSpacing: "2px", color: "#94a3b8", textTransform: "uppercase", marginTop: "3px" }}>
                          {car.nickname}
                        </div>
                      </div>
                    </div>

                    {/* Health */}
                    <div style={{ marginBottom: "20px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <div className="stat-label" style={{ marginBottom: 0 }}>Battery Health</div>
                        <div style={{
                          padding: "3px 10px",
                          background: status.bg,
                          color: status.color,
                          borderRadius: "99px",
                          fontSize: "9px",
                          letterSpacing: "1.5px",
                        }}>
                          {status.label}
                        </div>
                      </div>
                      <div style={{ display: "flex", alignItems: "baseline", gap: "3px", marginTop: "4px" }}>
                        <span style={{
                          fontFamily: "'Bebas Neue', sans-serif",
                          fontSize: "42px",
                          color: healthColor,
                          letterSpacing: "-0.5px",
                          lineHeight: 1,
                        }}>
                          {car.health}
                        </span>
                        <span style={{ fontSize: "16px", color: "#cbd5e1" }}>%</span>
                      </div>
                      <div className="health-bar-bg">
                        <div className="health-bar-fill" style={{
                          width: `${car.health}%`,
                          background: `linear-gradient(90deg, ${healthColor}88, ${healthColor})`,
                        }} />
                      </div>
                    </div>

                    {/* Quick stats */}
                    <div style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr 1fr",
                      gap: "8px",
                      paddingTop: "16px",
                      borderTop: "1px solid #f1f5f9",
                      paddingBottom: "10px",
                    }}>
                      <div>
                        <div className="stat-label">Range</div>
                        <div style={{ fontSize: "12px", color: "#334155", fontWeight: "500" }}>{car.range} mi</div>
                      </div>
                      <div>
                        <div className="stat-label">Charged</div>
                        <div style={{ fontSize: "12px", color: "#334155", fontWeight: "500" }}>{car.lastCharge}</div>
                      </div>
                      <div>
                        <div className="stat-label">Odometer</div>
                        <div style={{ fontSize: "11px", color: "#334155", fontWeight: "500" }}>{car.odometer}</div>
                      </div>
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