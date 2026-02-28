"""
Battery TISAC FastAPI Backend — STATELESS VERSION
===================================================
All simulation state is computed deterministically from a single `hour` parameter.
Because the simulation uses a fixed random seed (1337), every call with the same
`hour` value returns identical results. No in-memory SimState, no SSE stream.

The frontend polls /api/battery/snapshot?hour=N once per second, incrementing N
each time. On /api/battery/reset the frontend simply resets its own hour counter
back to 0 — no server state to clear.

SETUP (one-time):
    pip install fastapi uvicorn torch numpy

Copy these four model files into the SAME folder as battery_api.py:
    battery_tisac_drive_model.pt
    battery_tisac_rest_model.pt
    battery_tisac_drive_model_meta.json
    battery_tisac_rest_model_meta.json

RUN:
    python battery_api.py   ->  http://localhost:8000

ENDPOINTS:
    GET  /api/battery/snapshot?hour=N  – Compute and return state at hour N
    GET  /api/battery/snapshot         – Returns state at hour 0 (default)
    GET  /healthz                      – Health check
"""

import json, math
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── paths ──────────────────────────────────────────────────────────────────────
BASE             = Path(__file__).parent
DRIVE_MODEL_FILE = BASE / "battery_tisac_drive_model.pt"
REST_MODEL_FILE  = BASE / "battery_tisac_rest_model.pt"
DEVICE           = "cpu"
HOURS_PER_DAY    = 24
TOTAL_DAYS       = 600
TOTAL_HOURS      = TOTAL_DAYS * HOURS_PER_DAY
SEED             = 1337

# ── model architecture (must match training) ───────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128, 128), out_dim=4):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ThetaNet(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128, 128)):
        super().__init__()
        self.core = MLP(in_dim, hidden=hidden, out_dim=4)
    def forward(self, x):
        o = self.core(x)
        return o[:, 0:1], torch.clamp(o[:, 1:2], -8, 4), o[:, 2:3], torch.clamp(o[:, 3:4], -8, 4)

class YNet(nn.Module):
    def __init__(self, in_dim, hidden=(128, 128, 128), out_dim=3):
        super().__init__()
        self.core = MLP(in_dim, hidden=hidden, out_dim=out_dim)
    def forward(self, x): return self.core(x)

class ModeModel(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.theta_net = ThetaNet(feat_dim)
        self.y_net     = YNet(feat_dim + 2)
    def forward(self, feat):
        mu_q, logv_q, mu_r, logv_r = self.theta_net(feat)
        y_hat = self.y_net(torch.cat([feat, mu_q, mu_r], dim=1))
        return mu_q, logv_q, mu_r, logv_r, y_hat

@dataclass
class NormStats:
    x_mean: list; x_std: list
    q_mean: float; q_std: float
    r_mean: float; r_std: float
    y_mean: list;  y_std: list
    t_max_train_hr: float
    q0: float; r0: float; q_eol: float; mode_name: str

# ── load models ────────────────────────────────────────────────────────────────
def load_model(path: Path):
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ModeModel(feat_dim=len(ckpt["feature_order"])).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm  = NormStats(**ckpt["norm"])
    resid = ckpt.get("resid_stats", {"q_resid_std": 0.3, "r_resid_std": 1e-4, "y_resid_std": [1, 1, 1]})
    return model, norm, resid

print("[BOOT] Loading TISAC models…")
drive_model, drive_norm, drive_resid = load_model(DRIVE_MODEL_FILE)
rest_model,  rest_norm,  rest_resid  = load_model(REST_MODEL_FILE)
print("[BOOT] Models loaded ✓")

# ── pre-compute the ENTIRE simulation trajectory once at startup ───────────────
# Because the seed is fixed, this is fully deterministic.
# We cache it so every request just slices into the pre-built arrays.
print("[BOOT] Pre-computing simulation trajectory…")

_rng  = np.random.default_rng(SEED + 100)
_rng2 = np.random.default_rng(SEED + 999)
np.random.seed(SEED)

# build drive schedule
_mode_sch = np.zeros(TOTAL_HOURS, np.int64)
_cr_sch   = np.zeros(TOTAL_HOURS, np.float32)
for d in range(TOTAL_DAYS):
    b  = d * 24
    h1 = int(np.clip(8  + _rng.integers(-1, 2), 0, 23))
    h2 = int(np.clip(18 + _rng.integers(-1, 2), 0, 23))
    for h in [h1, h2]:
        _mode_sch[b + h] = 1
        _cr_sch[b + h]   = float(np.clip(_rng.normal(1.1, 0.25), 0.4, 2.2))

# build ground truth Q, R, T trajectories
Q0    = drive_norm.q0
R0    = drive_norm.r0
T_amb = 25.0
V_nom = 3.65

_Q     = np.zeros(TOTAL_HOURS)
_R     = np.zeros(TOTAL_HOURS)
_T_arr = np.zeros(TOTAL_HOURS)
_Q[0]  = Q0; _R[0] = R0; _T_arr[0] = T_amb

for k in range(1, TOTAL_HOURS):
    m   = _mode_sch[k - 1]
    x   = float(_cr_sch[k - 1])
    eps = 1 + 0.03 * _rng2.normal()
    if m == 1:
        cd   = max(k * 0.3, 1.0)
        dQ   = (2e-4 * max(x, 1e-6) ** 1.2 / math.sqrt(cd) + 4.5e-4 * max(x, 1e-6) ** 1.5) * Q0 * eps
        dR   = 1.6e-6 * max(x, 1e-6) * (1 + 0.015 * max(_T_arr[k-1] - T_amb, 0)) * eps
        I    = x * max(_Q[k-1], 1e-3)
        T_t  = T_amb + 0.08 * I ** 2 * _R[k-1]
    else:
        ta   = 1 + 0.01 * max(_T_arr[k-1] - T_amb, 0)
        dQ   = 1e-5 * Q0 * ta * eps
        dR   = 4.5e-8 * ta * eps
        T_t  = T_amb + 0.01 * max(_T_arr[k-1] - T_amb, 0)
    _Q[k]     = max(Q0 * 0.35, _Q[k-1] - max(dQ, 0))
    _R[k]     = max(R0, _R[k-1] + max(dR, 0))
    a         = math.exp(-1 / 6)
    _T_arr[k] = a * _T_arr[k-1] + (1 - a) * T_t

_I_dis = _cr_sch * _Q.astype(np.float32)
_V_ocv = V_nom + 0.15 * (_Q / Q0 - 0.5)
_V_avg = (_V_ocv - _I_dis * _R).astype(np.float64)
_E_h   = _V_avg * _I_dis
_E_h   = np.where(_mode_sch == 1, _E_h, 0.0)

_y1 = np.clip(_V_avg + _rng2.normal(0, .007, TOTAL_HOURS), 2.5, 4.4).astype(np.float32)
_y2 = np.clip(_T_arr + _rng2.normal(0, .35,  TOTAL_HOURS), T_amb - 2, 90).astype(np.float32)
_y3 = np.clip(_E_h   + _rng2.normal(0, 3,    TOTAL_HOURS), -5, 500).astype(np.float32)
_y3 = np.where(_mode_sch == 1, _y3, _y3 * 0.2).astype(np.float32)

print(f"[BOOT] Trajectory pre-computed ✓  ({TOTAL_HOURS} hours / {TOTAL_DAYS} days)")

# ── inference helpers ──────────────────────────────────────────────────────────
def predict(model, norm: NormStats, resid, X):
    x_mean = np.array(norm.x_mean, dtype=np.float32)
    x_std  = np.array(norm.x_std,  dtype=np.float32)
    Xn     = ((np.asarray(X, dtype=np.float32) - x_mean) / x_std).astype(np.float32)
    with torch.no_grad():
        mu_q, _, mu_r, _, y_hat_n = model(torch.tensor(Xn, dtype=torch.float32))
    mu_q  = (mu_q.numpy()    * norm.q_std  + norm.q_mean).reshape(-1)
    mu_r  = (mu_r.numpy()    * norm.r_std  + norm.r_mean).reshape(-1)
    y_hat = (y_hat_n.numpy() * np.array(norm.y_std) + np.array(norm.y_mean))
    q_sd  = np.full_like(mu_q, float(resid["q_resid_std"]))
    return mu_q, q_sd, mu_r, y_hat

def make_feat(h_idx, c_rate, cum_d, cum_r, t_max):
    day = h_idx // 24; hod = h_idx % 24; dow = day % 7
    return np.array([
        np.clip(h_idx / max(t_max, 1),    0, 1),
        np.clip(day   / max(t_max/24, 1), 0, 1),
        hod / 23.0,
        math.sin(2*math.pi*hod/24),  math.cos(2*math.pi*hod/24),
        math.sin(2*math.pi*dow/7),   math.cos(2*math.pi*dow/7),
        math.sin(2*math.pi*day/365), math.cos(2*math.pi*day/365),
        np.clip(c_rate / 2.5, 0, 1.5),
        np.clip(cum_d  / max(t_max, 1), 0, 2),
        np.clip(cum_r  / max(t_max, 1), 0, 2),
    ], dtype=np.float32)

def ema(x, a=0.25):
    y = np.array(x, float)
    for i in range(1, len(y)):
        y[i] = a * y[i] + (1 - a) * y[i-1]
    return y

def enforce_physics(q_raw, r_raw, q0, r0, a=0.20):
    if len(q_raw) == 0: return q_raw, r_raw
    qs = ema(q_raw, a); rs = ema(r_raw, a)
    qs[0] = q0; rs[0] = r0
    return np.minimum.accumulate(qs), np.maximum.accumulate(rs)

# ── mode detector (replayed deterministically up to hour k) ───────────────────
def replay_mode_detector(up_to_hour: int) -> list:
    """Returns the mode_detected list for hours 0..up_to_hour-1."""
    state = 0
    e_buf: list = []
    t_buf: list = []
    modes = []
    for k in range(up_to_hour):
        y1 = float(_y1[k]); y2 = float(_y2[k]); y3 = float(_y3[k])
        e_buf.append(y3); t_buf.append(y2)
        if len(e_buf) > 12: e_buf.pop(0); t_buf.pop(0)
        e_now = y3; e_med = float(np.median(e_buf)); tex = y2 - 25.0
        if state == 0:
            if e_now > 20 or e_med > 14 or (tex > 1 and e_now > 5): state = 1
        else:
            if e_now < 8 and e_med < 7 and tex < 0.5: state = 0
        modes.append(state)
    return modes

# ── compute all inferred quantities up to hour k (cached by hour) ─────────────
@lru_cache(maxsize=128)
def compute_state_at_hour(hour: int):
    """
    Replays the simulation from hour 0 to `hour` and returns the full
    state snapshot dict — identical to what the old SimState.step() returned.
    Cached so repeated calls at the same hour are free.
    """
    hour = min(hour, TOTAL_HOURS - 1)

    modes_det = replay_mode_detector(hour + 1)

    cum_d = 0.0; cum_r = 0.0
    q_raw = []; r_raw = []; q_sd_raw = []
    e_meas = []; e_hat_list = []; modes_list = []

    t_max = drive_norm.t_max_train_hr

    for k in range(hour + 1):
        m_det = modes_det[k]
        if m_det == 1: cum_d += 1
        else:          cum_r += 1

        x    = float(_cr_sch[k]) if m_det == 1 else 0.0
        xrow = make_feat(k, x, cum_d, cum_r, t_max)[None, :]

        if m_det == 1:
            mu_q, q_sd, mu_r, y_hat = predict(drive_model, drive_norm, drive_resid, xrow)
        else:
            mu_q, q_sd, mu_r, y_hat = predict(rest_model,  rest_norm,  rest_resid,  xrow)

        q_raw.append(float(mu_q[0]))
        r_raw.append(float(mu_r[0]))
        q_sd_raw.append(float(q_sd[0]))
        e_meas.append(float(_y3[k]))
        e_hat_list.append(float(y_hat[0, 2]))
        modes_list.append(m_det)

    q_disp, r_disp = enforce_physics(np.array(q_raw), np.array(r_raw), Q0, R0)
    q_now     = float(q_disp[-1])
    r_now     = float(r_disp[-1])
    q_true_now = float(_Q[hour])

    soh      = float(np.clip(100 * q_now      / Q0, 0, 110))
    soh_true = float(np.clip(100 * q_true_now / Q0, 0, 110))
    day_now  = hour // 24

    # ── forecast (run every 24h boundary, reuse otherwise) ──────────────────
    fc = run_forecast(hour, cum_d, cum_r, q_disp.tolist(), r_disp.tolist(), q_true_now=q_true_now)

    pred_eol      = fc.get("pred_eol_day")
    pred_eol_real = fc.get("pred_eol_day_real")
    pred_rul      = max(0, pred_eol      - day_now) if pred_eol      is not None else None
    pred_rul_real = max(0, pred_eol_real - day_now) if pred_eol_real is not None else None

    # ── windowed history for charts ──────────────────────────────────────────
    ew = min(len(e_meas),   336)
    dw = min(len(q_disp),   60 * 24)

    cap_hist = []
    for i in range(0, dw, 24):
        hour_idx = hour - dw + i
        true_k   = hour_idx % TOTAL_HOURS
        cap_hist.append({
            "day":    hour_idx // 24,
            "Q":      float(q_disp[-dw:][i]),
            "Q_true": float(_Q[true_k]),
        })

    res_hist   = [{"day": (hour - dw + i) // 24, "R": float(r_disp[-dw:][i]) * 1e3}
                  for i in range(0, dw, 24)]
    e_hist     = [{"h": hour - ew + i, "measured": float(e_meas[-ew:][i]), "model": float(e_hat_list[-ew:][i])}
                  for i in range(ew)]
    m_hist     = [{"h": hour - ew + i, "mode": int(modes_list[-ew:][i])}
                  for i in range(ew)]

    m_det_now = modes_det[hour]

    return {
        "hour":         hour,
        "day":          day_now,
        "soh":          round(soh,       2),
        "q_now":        round(q_now,     3),
        "soh_true":     round(soh_true,  2),
        "q_true_now":   round(q_true_now, 3),
        "r_now_mohm":   round(r_now * 1e3, 3),
        "mode":         m_det_now,
        "mode_label":   "Driving" if m_det_now == 1 else "Resting",
        "q0":           Q0,
        "q_eol":        drive_norm.q_eol,
        "pred_eol_day":       pred_eol,
        "pred_rul_days":      pred_rul,
        "pred_eol_day_real":  pred_eol_real,
        "pred_rul_days_real": pred_rul_real,
        "eol_band":     fc.get("eol_band", [None, None]),
        "forecast": {
            "days":    fc.get("day_f", []),
            "q":       fc.get("q_day_f", []),
            "q_upper": [q + 1.96*sd for q, sd in zip(fc.get("q_day_f", []), fc.get("qsd_day_f", []))],
            "q_lower": [max(0, q - 1.96*sd) for q, sd in zip(fc.get("q_day_f", []), fc.get("qsd_day_f", []))],
            "q_real":  fc.get("q_day_f_real", []),
        },
        "cap_hist":    cap_hist,
        "res_hist":    res_hist,
        "energy_hist": e_hist,
        "mode_hist":   m_hist,
    }

# ── forecast helper ────────────────────────────────────────────────────────────
def run_forecast(cur_h, cum_d, cum_r, q_hist, r_hist, q_true_now=None, days=300):
    t_max  = drive_norm.t_max_train_hr
    q_eol  = drive_norm.q_eol
    q_now  = float(q_hist[-1]) if q_hist else Q0
    r_now  = float(r_hist[-1]) if r_hist else R0
    if q_true_now is None:
        q_true_now = q_now

    cur_row  = make_feat(cur_h, 0.0, float(cum_d), float(cum_r), t_max)[None, :]
    aq_d, _, _, _ = predict(drive_model, drive_norm, drive_resid, cur_row)
    aq_r, _, _, _ = predict(rest_model,  rest_norm,  rest_resid,  cur_row)
    anchor_q      = float((aq_d[0] + aq_r[0]) / 2.0)

    rng   = np.random.default_rng(int(cur_h))
    rows, modes = [], []
    cd, cr2 = float(cum_d), float(cum_r)
    for h in range(cur_h + 1, cur_h + 1 + days * 24):
        hod = h % 24; day = h // 24
        h1  = int(np.clip(8  + ((day*17) % 3) - 1, 0, 23))
        h2  = int(np.clip(18 + ((day*13) % 3) - 1, 0, 23))
        m   = 1 if (hod == h1 or hod == h2) else 0
        x   = float(np.clip(1.1 + 0.25*math.sin(0.31*day) + 0.08*rng.normal(), 0.4, 2.2)) if m == 1 else 0.0
        cd += m; cr2 += (1 - m)
        rows.append(make_feat(h, x, cd, cr2, t_max))
        modes.append(m)

    X_f   = np.asarray(rows, dtype=np.float32)
    modes = np.asarray(modes)
    qf_raw = np.zeros(len(X_f), np.float32)
    rf_raw = np.zeros(len(X_f), np.float32)
    qsdf   = np.zeros(len(X_f), np.float32)

    if np.any(modes == 1):
        i = np.where(modes == 1)[0]
        q, sd, r, _ = predict(drive_model, drive_norm, drive_resid, X_f[i])
        qf_raw[i] = q; rf_raw[i] = r; qsdf[i] = sd
    if np.any(modes == 0):
        i = np.where(modes == 0)[0]
        q, sd, r, _ = predict(rest_model, rest_norm, rest_resid, X_f[i])
        qf_raw[i] = q; rf_raw[i] = r; qsdf[i] = sd

    qsdf  = qsdf * (1.0 + 0.002 * np.arange(len(qsdf)))
    delta = qf_raw.astype(float) - anchor_q

    # forecast from model anchor
    q_anch    = np.clip(q_now + delta, 0.0, Q0)
    q_joined  = np.concatenate([[q_now], q_anch])
    r_joined  = np.concatenate([[r_now], rf_raw.astype(float)])
    q_phys, _ = enforce_physics(q_joined, r_joined, q0=q_now, r0=r_now, a=0.10)
    q_f_phys  = q_phys[1:]

    # forecast from real anchor
    q_anch_real    = np.clip(q_true_now + delta, 0.0, Q0)
    q_joined_real  = np.concatenate([[q_true_now], q_anch_real])
    q_phys_real, _ = enforce_physics(q_joined_real, r_joined, q0=q_true_now, r0=r_now, a=0.10)
    q_f_phys_real  = q_phys_real[1:]

    fut_h   = np.arange(cur_h + 1, cur_h + 1 + days * 24)
    day_arr = fut_h // 24
    d_l, q_l, sd_l, q_real_l = [], [], [], []
    for d in np.unique(day_arr):
        mask = day_arr == d; last = np.where(mask)[0][-1]
        d_l.append(int(d))
        q_l.append(float(q_f_phys[last]))
        sd_l.append(float(qsdf[last]))
        q_real_l.append(float(q_f_phys_real[last]))

    d_arr  = np.array(d_l)
    q_arr  = np.array(q_l)
    qr_arr = np.array(q_real_l)
    sd_arr = np.array(sd_l)
    z      = 1.96

    ei        = np.where(q_arr  <= q_eol)[0]
    ei_r      = np.where(qr_arr <= q_eol)[0]
    elo       = np.where((q_arr + z*sd_arr) <= q_eol)[0]
    ehi       = np.where((q_arr - z*sd_arr) <= q_eol)[0]
    pred_eol      = int(d_arr[ei[0]])   if ei.size   else None
    pred_eol_real = int(d_arr[ei_r[0]]) if ei_r.size else None

    return {
        "day_f":             d_l[:120],
        "q_day_f":           q_l[:120],
        "q_day_f_real":      q_real_l[:120],
        "qsd_day_f":         sd_l[:120],
        "pred_eol_day":      pred_eol,
        "pred_eol_day_real": pred_eol_real,
        "eol_band": [
            int(d_arr[elo[0]]) if elo.size else None,
            int(d_arr[ehi[0]]) if ehi.size else None,
        ],
    }

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="Battery TISAC API — Stateless")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # lock to your Amplify domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/battery/snapshot")
async def snapshot(hour: int = Query(default=0, ge=0, le=TOTAL_HOURS - 1)):
    """
    Returns the full simulation state at the given hour.
    The frontend increments `hour` by 1 each second.
    hour=0 is the start; hour=14399 is the end of the 600-day simulation.
    """
    data = compute_state_at_hour(hour)
    return JSONResponse(data)

@app.get("/healthz")
async def health():
    return {"status": "ok", "total_hours": TOTAL_HOURS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("battery_api:app", host="0.0.0.0", port=8000, reload=False)