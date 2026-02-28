"use client";

import React, { useEffect, useRef, useState } from "react";
import { motion, useAnimationFrame, useMotionValue, useTransform, animate } from "framer-motion";

/* ─── Particle system ─── */
interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  size: number;
  hue: number;
}

function useParticles(count = 28) {
  const [particles, setParticles] = useState<Particle[]>([]);
  const nextId = useRef(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setParticles((prev) => {
        const alive = prev
          .map((p) => ({ ...p, x: p.x + p.vx, y: p.y + p.vy, life: p.life - 1 }))
          .filter((p) => p.life > 0);

        // Emit new particles from exhaust (rear of car ~x=105, y=188)
        const batch: Particle[] = Array.from({ length: 2 }, () => ({
          id: nextId.current++,
          x: 105 + Math.random() * 8 - 4,
          y: 185 + Math.random() * 6 - 3,
          vx: -(1.2 + Math.random() * 2.2),
          vy: (Math.random() - 0.5) * 0.7,
          life: 22 + Math.random() * 18,
          maxLife: 40,
          size: 1.5 + Math.random() * 3,
          hue: 185 + Math.random() * 30,
        }));

        return [...alive, ...batch].slice(-count);
      });
    }, 35);
    return () => clearInterval(interval);
  }, [count]);

  return particles;
}

/* ─── Wheel with spinning spokes ─── */
function PremiumWheel({ cx, cy, r = 26 }: { cx: number; cy: number; r?: number }) {
  const spokes = 6;
  return (
    <g>
      {/* Tire */}
      <circle cx={cx} cy={cy} r={r} fill="rgba(10,14,20,0.9)" stroke="rgba(255,255,255,0.15)" strokeWidth="2" />
      {/* Rim glow */}
      <circle cx={cx} cy={cy} r={r - 4} fill="none" stroke="rgba(34,211,238,0.35)" strokeWidth="1.5" />
      {/* Rotating hub */}
      <motion.g
        animate={{ rotate: 360 }}
        transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
        style={{ transformOrigin: `${cx}px ${cy}px` }}
      >
        {Array.from({ length: spokes }).map((_, i) => {
          const angle = (i / spokes) * Math.PI * 2;
          const x1 = cx + Math.cos(angle) * 6;
          const y1 = cy + Math.sin(angle) * 6;
          const x2 = cx + Math.cos(angle) * (r - 6);
          const y2 = cy + Math.sin(angle) * (r - 6);
          return (
            <line
              key={i}
              x1={x1} y1={y1} x2={x2} y2={y2}
              stroke="rgba(34,211,238,0.75)"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          );
        })}
        <circle cx={cx} cy={cy} r={5} fill="rgba(34,211,238,0.9)" />
        <circle cx={cx} cy={cy} r={2.5} fill="white" />
      </motion.g>
      {/* Tire gloss */}
      <ellipse cx={cx - r * 0.3} cy={cy - r * 0.35} rx={r * 0.22} ry={r * 0.1}
        fill="rgba(255,255,255,0.07)" transform={`rotate(-30, ${cx}, ${cy})`} />
    </g>
  );
}

/* ─── Speed lines ─── */
function SpeedLines() {
  const lines = Array.from({ length: 14 }, (_, i) => ({
    y: 60 + i * 22,
    width: 40 + Math.random() * 80,
    delay: Math.random() * 0.6,
    opacity: 0.04 + Math.random() * 0.08,
    startX: -200 + Math.random() * 100,
  }));

  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {lines.map((l, i) => (
        <motion.div
          key={i}
          className="absolute h-[1px] rounded-full"
          style={{
            top: l.y,
            width: l.width,
            background: `linear-gradient(to right, transparent, rgba(34,211,238,${l.opacity * 3}), transparent)`,
          }}
          animate={{ x: [l.startX, 900], opacity: [0, l.opacity * 2, 0] }}
          transition={{
            duration: 0.55 + Math.random() * 0.3,
            repeat: Infinity,
            delay: l.delay,
            ease: "linear",
          }}
        />
      ))}
    </div>
  );
}

/* ─── HUD Scan line ─── */
function ScanLine() {
  return (
    <motion.div
      className="pointer-events-none absolute inset-x-0 h-[2px]"
      style={{
        background: "linear-gradient(to right, transparent 0%, rgba(34,211,238,0.08) 30%, rgba(34,211,238,0.18) 50%, rgba(34,211,238,0.08) 70%, transparent 100%)",
      }}
      animate={{ top: ["0%", "100%"] }}
      transition={{ duration: 3.5, repeat: Infinity, ease: "linear", repeatDelay: 1.2 }}
    />
  );
}

/* ─── Floating HUD data chip ─── */
function HudChip({
  label,
  value,
  x,
  y,
  delay = 0,
  accent = "cyan",
}: {
  label: string;
  value: string;
  x: string;
  y: string;
  delay?: number;
  accent?: "cyan" | "emerald" | "blue" | "amber";
}) {
  const colors = {
    cyan: { dot: "bg-cyan-400", border: "border-cyan-400/30", text: "text-cyan-300", glow: "shadow-[0_0_12px_rgba(34,211,238,0.25)]" },
    emerald: { dot: "bg-emerald-400", border: "border-emerald-400/30", text: "text-emerald-300", glow: "shadow-[0_0_12px_rgba(52,211,153,0.25)]" },
    blue: { dot: "bg-blue-400", border: "border-blue-400/30", text: "text-blue-300", glow: "shadow-[0_0_12px_rgba(96,165,250,0.25)]" },
    amber: { dot: "bg-amber-400", border: "border-amber-400/30", text: "text-amber-300", glow: "shadow-[0_0_12px_rgba(251,191,36,0.25)]" },
  }[accent];

  return (
    <motion.div
      className={`absolute flex items-center gap-1.5 rounded-lg border ${colors.border} bg-black/60 px-2.5 py-1.5 backdrop-blur-md ${colors.glow}`}
      style={{ left: x, top: y }}
      initial={{ opacity: 0, scale: 0.7, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: "backOut" }}
    >
      <motion.span
        className={`h-1.5 w-1.5 rounded-full ${colors.dot}`}
        animate={{ opacity: [1, 0.3, 1] }}
        transition={{ duration: 1.6, repeat: Infinity, delay: delay * 0.5 }}
      />
      <span className="text-[9px] tracking-[0.16em] text-white/40">{label}</span>
      <span className={`text-[11px] font-semibold ${colors.text}`}>{value}</span>
    </motion.div>
  );
}

/* ─── Connector line from HUD chip to car ─── */
function HudConnector({
  x1, y1, x2, y2, delay = 0
}: { x1: number; y1: number; x2: number; y2: number; delay?: number }) {
  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" style={{ zIndex: 5 }}>
      <motion.line
        x1={x1} y1={y1} x2={x2} y2={y2}
        stroke="rgba(34,211,238,0.25)"
        strokeWidth="1"
        strokeDasharray="3 3"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={{ duration: 0.8, delay, ease: "easeOut" }}
      />
    </svg>
  );
}

/* ─── Road with perspective ─── */
function PerspectiveRoad() {
  return (
    <div className="absolute inset-x-0 bottom-0 h-48">
      {/* Asphalt base */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#080c12] via-[#0a0f18]/80 to-transparent" />

      {/* Road surface with perspective SVG */}
      <svg className="absolute bottom-0 w-full" viewBox="0 0 800 120" preserveAspectRatio="none">
        <defs>
          <linearGradient id="roadGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0d1520" stopOpacity="0" />
            <stop offset="100%" stopColor="#060a10" stopOpacity="1" />
          </linearGradient>
          <linearGradient id="laneGlow" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="rgba(34,211,238,0)" />
            <stop offset="30%" stopColor="rgba(34,211,238,0.6)" />
            <stop offset="50%" stopColor="rgba(34,211,238,0.9)" />
            <stop offset="70%" stopColor="rgba(34,211,238,0.6)" />
            <stop offset="100%" stopColor="rgba(34,211,238,0)" />
          </linearGradient>
        </defs>
        {/* Road fill */}
        <rect x="0" y="20" width="800" height="100" fill="url(#roadGrad)" />
        {/* Edge lines */}
        <line x1="0" y1="34" x2="800" y2="34" stroke="rgba(255,255,255,0.06)" strokeWidth="1.5" />
        <line x1="0" y1="100" x2="800" y2="100" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
        {/* Center divider glow */}
        <rect x="0" y="63" width="800" height="2" fill="url(#laneGlow)" opacity="0.5" />
      </svg>

      {/* Moving lane dashes */}
      {Array.from({ length: 22 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute"
          style={{
            bottom: 42,
            left: `${i * 90}px`,
            width: 52,
            height: 3,
            borderRadius: 2,
            background: "linear-gradient(to right, transparent, rgba(251,191,36,0.85), transparent)",
          }}
          animate={{ x: [0, -2100] }}
          transition={{ duration: 4.8, repeat: Infinity, ease: "linear", delay: i * 0.018 }}
        />
      ))}

      {/* Ground light reflection under car */}
      <motion.div
        className="absolute"
        style={{
          bottom: 8,
          left: "50%",
          transform: "translateX(-50%)",
          width: 280,
          height: 14,
          background: "radial-gradient(ellipse, rgba(34,211,238,0.18) 0%, transparent 70%)",
          filter: "blur(4px)",
        }}
        animate={{ opacity: [0.6, 1, 0.6], scaleX: [0.9, 1.1, 0.9] }}
        transition={{ duration: 2.4, repeat: Infinity, ease: "easeInOut" }}
      />
    </div>
  );
}

/* ─── Volumetric headlight beams ─── */
function HeadlightBeams() {
  return (
    <div className="pointer-events-none absolute inset-0">
      <svg className="absolute inset-0 h-full w-full">
        <defs>
          <radialGradient id="beam1" cx="0%" cy="50%" r="100%">
            <stop offset="0%" stopColor="rgba(147,210,255,0.22)" />
            <stop offset="100%" stopColor="rgba(147,210,255,0)" />
          </radialGradient>
          <radialGradient id="beam2" cx="0%" cy="50%" r="100%">
            <stop offset="0%" stopColor="rgba(34,211,238,0.15)" />
            <stop offset="100%" stopColor="rgba(34,211,238,0)" />
          </radialGradient>
        </defs>
      </svg>
      {/* Left beam */}
      <motion.div
        className="absolute"
        style={{
          right: "18%",
          top: "38%",
          width: 320,
          height: 55,
          background: "linear-gradient(to right, rgba(147,210,255,0.18), transparent)",
          clipPath: "polygon(0 40%, 100% 0%, 100% 100%, 0 60%)",
          filter: "blur(8px)",
          transformOrigin: "left center",
        }}
        animate={{ opacity: [0.7, 1, 0.7], scaleY: [0.9, 1.05, 0.9] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
      />
      {/* Right beam */}
      <motion.div
        className="absolute"
        style={{
          right: "17%",
          top: "42%",
          width: 300,
          height: 45,
          background: "linear-gradient(to right, rgba(34,211,238,0.14), transparent)",
          clipPath: "polygon(0 40%, 100% 0%, 100% 100%, 0 60%)",
          filter: "blur(10px)",
          transformOrigin: "left center",
        }}
        animate={{ opacity: [0.5, 0.85, 0.5], scaleY: [1, 0.95, 1] }}
        transition={{ duration: 3.1, repeat: Infinity, ease: "easeInOut", delay: 0.4 }}
      />
    </div>
  );
}

/* ─── Main component ─── */
export function HeroVisualCard() {
  const particles = useParticles(32);

  return (
    <div
      className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/5 shadow-2xl backdrop-blur-2xl"
      style={{
        background: "linear-gradient(135deg, rgba(5,10,18,0.97) 0%, rgba(8,14,24,0.95) 100%)",
      }}
    >
      {/* Ambient interior glow */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_60%_40%,rgba(34,211,238,0.055),transparent_55%),radial-gradient(ellipse_at_20%_70%,rgba(59,130,246,0.045),transparent_50%)]" />

      {/* Fine grid texture */}
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.025]"
        style={{
          backgroundImage: "linear-gradient(rgba(34,211,238,1) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,1) 1px, transparent 1px)",
          backgroundSize: "32px 32px",
        }}
      />

      {/* ── Header ── */}
      <div className="relative z-10 border-b border-white/[0.07] px-5 py-5">
        <div className="flex items-center gap-2.5">
          <motion.span
            className="h-2 w-2 rounded-full bg-cyan-400"
            animate={{ boxShadow: ["0 0 6px rgba(34,211,238,0.6)", "0 0_18px_rgba(34,211,238,1)", "0 0 6px rgba(34,211,238,0.6)"] }}
            transition={{ duration: 1.8, repeat: Infinity }}
          />
          <span className="text-[10px] tracking-[0.28em] text-white/50">VOLWATCH · LIVE TELEMETRY</span>
          <div className="ml-auto flex items-center gap-1.5">
            <motion.div
              className="h-1 w-1 rounded-full bg-emerald-400"
              animate={{ opacity: [1, 0.2, 1] }}
              transition={{ duration: 0.9, repeat: Infinity }}
            />
            <span className="text-[9px] tracking-widest text-emerald-400/80">ACTIVE</span>
          </div>
        </div>

        <h1 className="mt-4 text-xl font-semibold leading-tight tracking-tight text-white mobile:text-2xl tablet:text-[1.75rem]">
          Predict Battery Health{" "}
          <span className="bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent">
            Before It Becomes a Problem
          </span>
        </h1>
        <p className="mt-2.5 max-w-2xl text-sm leading-relaxed text-white/50 tablet:text-[14px]">
          Mode-aware EV intelligence — SOH tracking, resistance estimation & daily lifespan
          forecasting via time-input Bayesian calibration.
        </p>
      </div>

      {/* ── Animation canvas ── */}
      <div className="relative h-[340px] mobile:h-[390px] tablet:h-[440px] overflow-hidden">

        {/* Scan line */}
        <ScanLine />

        {/* Speed lines */}
        <SpeedLines />

        {/* Headlight beams */}
        <HeadlightBeams />

        {/* Sky atmosphere */}
        <div
          className="absolute inset-x-0 top-0 h-28"
          style={{
            background: "linear-gradient(to bottom, rgba(4,8,16,0.9) 0%, transparent 100%)",
          }}
        />

        {/* Distant city glow on horizon */}
        <motion.div
          className="absolute"
          style={{
            bottom: "36%",
            left: "50%",
            transform: "translateX(-50%)",
            width: 500,
            height: 60,
            background: "radial-gradient(ellipse, rgba(59,130,246,0.12) 0%, transparent 70%)",
            filter: "blur(20px)",
          }}
          animate={{ opacity: [0.5, 0.9, 0.5], scaleX: [0.85, 1.1, 0.85] }}
          transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
        />

        {/* Road */}
        <PerspectiveRoad />

        {/* HUD connector lines (SVG layer, rendered before chips) */}
        <HudConnector x1={155} y1={105} x2={230} y2={155} delay={1.1} />
        <HudConnector x1={565} y1={118} x2={490} y2={160} delay={1.3} />
        <HudConnector x1={570} y1={210} x2={490} y2={195} delay={1.5} />
        <HudConnector x1={148} y1={222} x2={200} y2={198} delay={1.7} />

        {/* ── Car ── */}
        <motion.div
          className="absolute inset-0 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1.0, delay: 0.3 }}
        >
          <motion.div
            className="relative"
            animate={{ y: [0, -4, 0], x: [-6, 6, -6] }}
            transition={{ duration: 5.0, repeat: Infinity, ease: "easeInOut" }}
          >
            {/* Car body glow */}
            <motion.div
              className="absolute"
              style={{
                inset: "-20px",
                background: "radial-gradient(ellipse at center 65%, rgba(34,211,238,0.12) 0%, transparent 60%)",
                filter: "blur(16px)",
              }}
              animate={{ opacity: [0.6, 1, 0.6] }}
              transition={{ duration: 2.5, repeat: Infinity }}
            />

            <svg
              width="600"
              height="250"
              viewBox="0 0 600 250"
              className="w-[95%] max-w-[600px]"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <defs>
                <linearGradient id="bodyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(34,211,238,0.08)" />
                  <stop offset="100%" stopColor="rgba(34,211,238,0.01)" />
                </linearGradient>
                <linearGradient id="roofLine" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="rgba(125,211,252,0)" />
                  <stop offset="20%" stopColor="rgba(125,211,252,0.9)" />
                  <stop offset="80%" stopColor="rgba(125,211,252,0.9)" />
                  <stop offset="100%" stopColor="rgba(125,211,252,0)" />
                </linearGradient>
                <linearGradient id="sidePanel" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgba(255,255,255,0.06)" />
                  <stop offset="60%" stopColor="rgba(255,255,255,0.01)" />
                  <stop offset="100%" stopColor="rgba(255,255,255,0.04)" />
                </linearGradient>
                <filter id="glow">
                  <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id="softGlow">
                  <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                  <feMerge>
                    <feMergeNode in="coloredBlur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              {/* Shadow ellipse */}
              <motion.ellipse
                cx="300" cy="205" rx="175" ry="12"
                fill="rgba(0,0,0,0.55)"
                animate={{ rx: [175, 185, 175], opacity: [0.55, 0.4, 0.55] }}
                transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
              />

              {/* Ground reflection glow */}
              <ellipse cx="300" cy="202" rx="140" ry="7" fill="rgba(34,211,238,0.1)" />

              {/* ── Car body ── */}
              <motion.path
                d="M108 168
                   C130 144,164 126,208 118
                   L256 106
                   C278 100,322 100,346 108
                   L388 122
                   C414 130,436 144,458 168
                   L488 168
                   C498 168,506 176,506 186
                   L506 190
                   C506 200,498 207,488 207
                   L476 207
                   C472 221,456 232,438 232
                   C420 232,404 221,400 207
                   L172 207
                   C168 221,152 232,134 232
                   C116 232,100 221,96 207
                   L84 207
                   C74 207,66 200,66 190
                   L66 186
                   C66 176,74 168,84 168
                   L108 168Z"
                fill="url(#bodyGrad)"
                stroke="none"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
              />

              {/* Body outline with glow */}
              <motion.path
                d="M108 168
                   C130 144,164 126,208 118
                   L256 106
                   C278 100,322 100,346 108
                   L388 122
                   C414 130,436 144,458 168
                   L488 168
                   C498 168,506 176,506 186
                   L506 190
                   C506 200,498 207,488 207
                   L476 207
                   C472 221,456 232,438 232
                   C420 232,404 221,400 207
                   L172 207
                   C168 221,152 232,134 232
                   C116 232,100 221,96 207
                   L84 207
                   C74 207,66 200,66 190
                   L66 186
                   C66 176,74 168,84 168
                   L108 168Z"
                stroke="rgba(125,211,252,0.9)"
                strokeWidth="1.8"
                fill="url(#sidePanel)"
                filter="url(#glow)"
                initial={{ pathLength: 0, opacity: 0 }}
                animate={{ pathLength: 1, opacity: 1 }}
                transition={{ duration: 2.2, ease: "easeOut" }}
              />

              {/* Roof line */}
              <path
                d="M188 122 L258 108 C279 102 321 102 344 108 L378 124"
                stroke="url(#roofLine)"
                strokeWidth="2"
                strokeLinecap="round"
                filter="url(#glow)"
              />

              {/* Side panel accent line */}
              <motion.path
                d="M112 178 L488 178"
                stroke="rgba(34,211,238,0.22)"
                strokeWidth="1"
                strokeLinecap="round"
                initial={{ pathLength: 0 }}
                animate={{ pathLength: 1 }}
                transition={{ duration: 1.4, delay: 0.6 }}
              />

              {/* Windshield */}
              <motion.path
                d="M212 118 L264 108 C280 104 318 104 338 108 L382 122 L370 162 L230 162 Z"
                fill="rgba(147,210,255,0.04)"
                stroke="rgba(147,210,255,0.35)"
                strokeWidth="1.2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8, duration: 0.8 }}
              />

              {/* Windshield interior shimmer */}
              <motion.path
                d="M232 114 L278 106 C285 104 296 104 302 106 L310 108"
                stroke="rgba(255,255,255,0.12)"
                strokeWidth="1.5"
                strokeLinecap="round"
                animate={{ opacity: [0.1, 0.4, 0.1] }}
                transition={{ duration: 3.5, repeat: Infinity, delay: 1 }}
              />

              {/* A-pillar */}
              <path d="M228 162 L212 118" stroke="rgba(255,255,255,0.12)" strokeWidth="2" strokeLinecap="round" />
              {/* B-pillar */}
              <path d="M300 164 L300 108" stroke="rgba(255,255,255,0.08)" strokeWidth="1.5" strokeLinecap="round" />

              {/* ── Headlight (right/front) ── */}
              <g filter="url(#softGlow)">
                <motion.path
                  d="M454 168 L476 170 L476 186 L454 186 Z"
                  fill="rgba(147,210,255,0.15)"
                  stroke="rgba(147,210,255,0.8)"
                  strokeWidth="1.2"
                  animate={{ fill: ["rgba(147,210,255,0.12)", "rgba(147,210,255,0.25)", "rgba(147,210,255,0.12)"] }}
                  transition={{ duration: 1.8, repeat: Infinity }}
                />
                <line x1="476" y1="172" x2="508" y2="170" stroke="rgba(147,210,255,0.9)" strokeWidth="1.5" strokeLinecap="round" />
                <line x1="476" y1="178" x2="510" y2="178" stroke="rgba(147,210,255,0.6)" strokeWidth="1" strokeLinecap="round" />
                <line x1="476" y1="184" x2="508" y2="186" stroke="rgba(147,210,255,0.4)" strokeWidth="0.8" strokeLinecap="round" />
              </g>

              {/* ── Tail light (left/rear) ── */}
              <g filter="url(#glow)">
                <motion.path
                  d="M88 168 L110 170 L110 186 L88 186 Z"
                  fill="rgba(59,130,246,0.15)"
                  stroke="rgba(59,130,246,0.8)"
                  strokeWidth="1.2"
                  animate={{
                    fill: ["rgba(59,130,246,0.1)", "rgba(239,68,68,0.3)", "rgba(59,130,246,0.1)"],
                    stroke: ["rgba(59,130,246,0.8)", "rgba(239,68,68,0.9)", "rgba(59,130,246,0.8)"],
                  }}
                  transition={{ duration: 1.6, repeat: Infinity }}
                />
              </g>

              {/* ── Exhaust port ── */}
              <ellipse cx="90" cy="196" rx="5" ry="3" fill="rgba(0,0,0,0.8)" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />

              {/* DRL strip */}
              <motion.path
                d="M454 166 L476 168"
                stroke="rgba(255,255,255,0.9)"
                strokeWidth="2"
                strokeLinecap="round"
                filter="url(#glow)"
                animate={{ opacity: [0.7, 1, 0.7] }}
                transition={{ duration: 1.2, repeat: Infinity }}
              />

              {/* Charge port indicator */}
              <motion.circle
                cx="286" cy="164" r="3"
                fill="rgba(34,211,238,0.9)"
                filter="url(#glow)"
                animate={{ r: [3, 4, 3], opacity: [0.8, 1, 0.8] }}
                transition={{ duration: 1.4, repeat: Infinity }}
              />

              {/* Wheels */}
              <PremiumWheel cx={134} cy={207} r={26} />
              <PremiumWheel cx={438} cy={207} r={26} />

              {/* Wheel arch accent lines */}
              <path d="M104 182 C104 168 120 160 134 160 C148 160 164 168 164 182"
                stroke="rgba(34,211,238,0.18)" strokeWidth="1.2" fill="none" strokeLinecap="round" />
              <path d="M408 182 C408 168 424 160 438 160 C452 160 468 168 468 182"
                stroke="rgba(34,211,238,0.18)" strokeWidth="1.2" fill="none" strokeLinecap="round" />
            </svg>

            {/* ── Particle exhaust ── */}
            <div className="pointer-events-none absolute inset-0">
              <svg className="absolute inset-0 h-full w-full" viewBox="0 0 600 250">
                {particles.map((p) => {
                  const alpha = (p.life / p.maxLife) * 0.75;
                  return (
                    <circle
                      key={p.id}
                      cx={p.x}
                      cy={p.y}
                      r={p.size * (p.life / p.maxLife)}
                      fill={`hsla(${p.hue}, 85%, 70%, ${alpha})`}
                    />
                  );
                })}
              </svg>
            </div>
          </motion.div>
        </motion.div>

        {/* ── HUD overlay chips ── */}
        <HudChip label="SOH" value="93.2%" x="4%" y="22%" delay={1.0} accent="emerald" />
        <HudChip label="MODE" value="DRIVING" x="64%" y="18%" delay={1.2} accent="cyan" />
        <HudChip label="RUL" value="318 days" x="65%" y="42%" delay={1.4} accent="blue" />
        <HudChip label="TEMP" value="31.4°C" x="4%" y="48%" delay={1.6} accent="amber" />

        {/* Corner bracket decorations */}
        {[
          { style: { top: 8, left: 8 }, rotate: 0 },
          { style: { top: 8, right: 8 }, rotate: 90 },
          { style: { bottom: 8, right: 8 }, rotate: 180 },
          { style: { bottom: 8, left: 8 }, rotate: 270 },
        ].map((corner, i) => (
          <motion.div
            key={i}
            className="pointer-events-none absolute h-5 w-5"
            style={{ ...corner.style }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.35 }}
            transition={{ delay: 0.5 + i * 0.1 }}
          >
            <svg viewBox="0 0 20 20" className="h-full w-full" style={{ transform: `rotate(${corner.rotate}deg)` }}>
              <path d="M0 12 L0 0 L12 0" stroke="rgba(34,211,238,0.8)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
            </svg>
          </motion.div>
        ))}
      </div>
    </div>
  );
}