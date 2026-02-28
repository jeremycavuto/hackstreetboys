"""
Battery TISAC FastAPI Backend
==============================
Loads pretrained TISAC drive + rest PyTorch models and streams live battery
inference via Server-Sent Events (SSE) to the Next.js frontend.

SETUP (one-time, in the folder where you put this file):
    pip install fastapi uvicorn torch numpy

COPY these four model files into the SAME folder as battery_api.py:
    battery_tisac_drive_model.pt
    battery_tisac_rest_model.pt
    battery_tisac_drive_model_meta.json
    battery_tisac_rest_model_meta.json

RUN:
    python battery_api.py          -> http://localhost:8000

ENDPOINTS:
    GET  /api/battery/stream   – SSE stream (one JSON event per second)
    GET  /api/battery/snapshot – Latest state as plain JSON
    POST /api/battery/reset    – Restart the simulation
    GET  /healthz              – Health check
"""

import asyncio, json, math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── paths ─────────────────────────────────────────────────────────────────────
BASE             = Path(__file__).parent
DRIVE_MODEL_FILE = BASE / "battery_tisac_drive_model.pt"
REST_MODEL_FILE  = BASE / "battery_tisac_rest_model.pt"
DEVICE           = "cpu"
HOURS_PER_DAY    = 24
STREAM_DAYS      = 600
STREAM_INTERVAL  = 1.0   # seconds per SSE event

# ── model architecture (must match training) ──────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(128,128,128), out_dim=4):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ThetaNet(nn.Module):
    def __init__(self, in_dim, hidden=(128,128,128)):
        super().__init__(); self.core = MLP(in_dim, hidden=hidden, out_dim=4)
    def forward(self, x):
        o = self.core(x)
        return o[:,0:1], torch.clamp(o[:,1:2],-8,4), o[:,2:3], torch.clamp(o[:,3:4],-8,4)

class YNet(nn.Module):
    def __init__(self, in_dim, hidden=(128,128,128), out_dim=3):
        super().__init__(); self.core = MLP(in_dim, hidden=hidden, out_dim=out_dim)
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
    x_mean:list; x_std:list
    q_mean:float; q_std:float
    r_mean:float; r_std:float
    y_mean:list;  y_std:list
    t_max_train_hr:float
    q0:float; r0:float; q_eol:float; mode_name:str

# ── load models ───────────────────────────────────────────────────────────────
def load_model(path: Path):
    ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
    model = ModeModel(feat_dim=len(ckpt["feature_order"])).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"]); model.eval()
    norm  = NormStats(**ckpt["norm"])
    resid = ckpt.get("resid_stats", {"q_resid_std":0.3,"r_resid_std":1e-4,"y_resid_std":[1,1,1]})
    return model, norm, resid

print("[BOOT] Loading TISAC models…")
drive_model, drive_norm, drive_resid = load_model(DRIVE_MODEL_FILE)
rest_model,  rest_norm,  rest_resid  = load_model(REST_MODEL_FILE)
print("[BOOT] Models loaded ✓")

# ── inference helpers ─────────────────────────────────────────────────────────
def predict(model, norm: NormStats, resid, X):
    x_mean = np.array(norm.x_mean, dtype=np.float32)
    x_std  = np.array(norm.x_std,  dtype=np.float32)
    Xn = ((np.asarray(X, dtype=np.float32) - x_mean) / x_std).astype(np.float32)
    with torch.no_grad():
        mu_q, _, mu_r, _, y_hat_n = model(torch.tensor(Xn, dtype=torch.float32))
    mu_q  = (mu_q.numpy()    * norm.q_std + norm.q_mean).reshape(-1)
    mu_r  = (mu_r.numpy()    * norm.r_std + norm.r_mean).reshape(-1)
    y_hat = (y_hat_n.numpy() * np.array(norm.y_std) + np.array(norm.y_mean))
    q_sd  = np.full_like(mu_q, float(resid["q_resid_std"]))
    return mu_q, q_sd, mu_r, y_hat

def make_feat(h_idx, c_rate, cum_d, cum_r, t_max):
    day = h_idx // 24; hod = h_idx % 24; dow = day % 7
    return np.array([
        np.clip(h_idx / max(t_max,1), 0, 1),
        np.clip(day   / max(t_max/24,1), 0, 1),
        hod / 23.0,
        math.sin(2*math.pi*hod/24),  math.cos(2*math.pi*hod/24),
        math.sin(2*math.pi*dow/7),   math.cos(2*math.pi*dow/7),
        math.sin(2*math.pi*day/365), math.cos(2*math.pi*day/365),
        np.clip(c_rate/2.5, 0, 1.5),
        np.clip(cum_d/max(t_max,1), 0, 2),
        np.clip(cum_r/max(t_max,1), 0, 2),
    ], dtype=np.float32)

def ema(x, a=0.25):
    y = np.array(x, float)
    for i in range(1, len(y)): y[i] = a*y[i] + (1-a)*y[i-1]
    return y

def enforce_physics(q_raw, r_raw, q0, r0, a=0.20):
    if len(q_raw) == 0: return q_raw, r_raw
    qs = ema(q_raw, a); rs = ema(r_raw, a)
    qs[0]=q0; rs[0]=r0
    return np.minimum.accumulate(qs), np.maximum.accumulate(rs)

# ── mode detector ─────────────────────────────────────────────────────────────
class ModeDetector:
    def __init__(self):
        self.state=0; self.e=[]; self.t=[]
    def update(self, y1, y2, y3):
        self.e.append(float(y3)); self.t.append(float(y2))
        if len(self.e)>12: self.e.pop(0); self.t.pop(0)
        e_now=float(y3); e_med=float(np.median(self.e)); tex=float(y2-25)
        if self.state==0:
            if e_now>20 or e_med>14 or (tex>1 and e_now>5): self.state=1
        else:
            if e_now<8 and e_med<7 and tex<0.5: self.state=0
        return self.state

# ── forecast ──────────────────────────────────────────────────────────────────
def run_forecast(cur_h, cum_d, cum_r, q_hist, r_hist, days=300):
    """
    Anchor-corrected forecast:
      - Ask the model what Q it predicts at the CURRENT hour  => anchor_q_model
      - Ask the model what Q it predicts at each FUTURE hour  => qf_raw
      - Compute delta = qf_raw - anchor_q_model  (relative degradation from now)
      - Final forecast = q_now + delta  (starts from actual current state)
    This ensures EOL moves as the real battery degrades.
    """
    t_max = drive_norm.t_max_train_hr
    q_eol = drive_norm.q_eol
    q_now = float(q_hist[-1]) if len(q_hist) else drive_norm.q0
    r_now = float(r_hist[-1]) if len(r_hist) else drive_norm.r0

    # anchor: model prediction at current hour
    cur_row = make_feat(cur_h, 0.0, float(cum_d), float(cum_r), t_max)[None,:]
    aq_d, _, _, _ = predict(drive_model, drive_norm, drive_resid, cur_row)
    aq_r, _, _, _ = predict(rest_model,  rest_norm,  rest_resid,  cur_row)
    anchor_q = float((aq_d[0] + aq_r[0]) / 2.0)

    # build future feature rows
    rng = np.random.default_rng(int(cur_h))
    rows, modes = [], []
    cd, cr2 = float(cum_d), float(cum_r)
    for h in range(cur_h+1, cur_h+1+days*24):
        hod=h%24; day=h//24
        h1=int(np.clip(8+((day*17)%3)-1,0,23)); h2=int(np.clip(18+((day*13)%3)-1,0,23))
        m=1 if (hod==h1 or hod==h2) else 0
        x=float(np.clip(1.1+0.25*math.sin(0.31*day)+0.08*rng.normal(),0.4,2.2)) if m==1 else 0.0
        cd+=m; cr2+=(1-m)
        rows.append(make_feat(h, x, cd, cr2, t_max)); modes.append(m)
    X_f   = np.asarray(rows, dtype=np.float32)
    modes = np.asarray(modes)
    qf_raw = np.zeros(len(X_f), np.float32)
    rf_raw = np.zeros(len(X_f), np.float32)
    qsdf   = np.zeros(len(X_f), np.float32)

    if np.any(modes==1):
        i=np.where(modes==1)[0]; q,sd,r,_=predict(drive_model,drive_norm,drive_resid,X_f[i])
        qf_raw[i]=q; rf_raw[i]=r; qsdf[i]=sd
    if np.any(modes==0):
        i=np.where(modes==0)[0]; q,sd,r,_=predict(rest_model,rest_norm,rest_resid,X_f[i])
        qf_raw[i]=q; rf_raw[i]=r; qsdf[i]=sd

    # anchor correction: shift model curve to start at real current Q
    delta       = qf_raw.astype(float) - anchor_q
    q_anchored  = np.clip(q_now + delta, 0.0, drive_norm.q0)

    # monotone-decreasing smoothing anchored at current state
    q_joined = np.concatenate([[q_now], q_anchored])
    r_joined = np.concatenate([[r_now], rf_raw.astype(float)])
    q_phys, _ = enforce_physics(q_joined, r_joined, q0=q_now, r0=r_now, a=0.10)
    q_f_phys  = q_phys[1:]

    # uncertainty grows with horizon
    qsdf = qsdf * (1.0 + 0.002*np.arange(len(qsdf)))

    # daily aggregate
    fut_h   = np.arange(cur_h+1, cur_h+1+days*24)
    day_arr = fut_h // 24
    d_l, q_l, sd_l = [], [], []
    for d in np.unique(day_arr):
        mask=day_arr==d; last=np.where(mask)[0][-1]
        d_l.append(int(d)); q_l.append(float(q_f_phys[last])); sd_l.append(float(qsdf[last]))

    d_arr = np.array(d_l); q_arr = np.array(q_l); sd_arr = np.array(sd_l)
    z = 1.96
    ei  = np.where(q_arr <= q_eol)[0]
    elo = np.where((q_arr + z*sd_arr) <= q_eol)[0]
    ehi = np.where((q_arr - z*sd_arr) <= q_eol)[0]
    pred_eol = int(d_arr[ei[0]]) if ei.size else None
    return {
        "day_f":     d_l[:120],
        "q_day_f":   q_l[:120],
        "qsd_day_f": sd_l[:120],
        "pred_eol_day": pred_eol,
        "eol_band": [
            int(d_arr[elo[0]]) if elo.size else None,
            int(d_arr[ehi[0]]) if ehi.size else None,
        ],
    }

# ── simulation state ──────────────────────────────────────────────────────────
class SimState:
    def __init__(self): self.reset()

    def reset(self):
        seed=1337; np.random.seed(seed)
        rng=np.random.default_rng(seed+100)
        total_h=STREAM_DAYS*HOURS_PER_DAY
        # build schedule
        mode=np.zeros(total_h,np.int64); cr=np.zeros(total_h,np.float32)
        for d in range(STREAM_DAYS):
            b=d*24
            h1=int(np.clip(8+rng.integers(-1,2),0,23)); h2=int(np.clip(18+rng.integers(-1,2),0,23))
            for h in [h1,h2]:
                mode[b+h]=1; cr[b+h]=float(np.clip(rng.normal(1.1,0.25),0.4,2.2))
        self.mode_sch=mode; self.cr_sch=cr
        # generate ground truth trajectory
        rng2=np.random.default_rng(seed+999)
        Q0=drive_norm.q0; R0=drive_norm.r0; T_amb=25.0; V_nom=3.65
        Q=np.zeros(total_h); R=np.zeros(total_h); T_arr=np.zeros(total_h)
        Q[0]=Q0; R[0]=R0; T_arr[0]=T_amb
        for k in range(1,total_h):
            m=mode[k-1]; x=float(cr[k-1]); eps=1+0.03*rng2.normal()
            if m==1:
                cd=max(k*0.3,1.0)
                dQ=(2e-4*max(x,1e-6)**1.2/math.sqrt(cd)+4.5e-4*max(x,1e-6)**1.5)*Q0*eps
                dR=1.6e-6*max(x,1e-6)*(1+0.015*max(T_arr[k-1]-T_amb,0))*eps
                I=x*max(Q[k-1],1e-3); T_t=T_amb+0.08*I**2*R[k-1]
            else:
                ta=1+0.01*max(T_arr[k-1]-T_amb,0)
                dQ=1e-5*Q0*ta*eps; dR=4.5e-8*ta*eps; T_t=T_amb+0.01*max(T_arr[k-1]-T_amb,0)
            Q[k]=max(Q0*0.35,Q[k-1]-max(dQ,0)); R[k]=max(R0,R[k-1]+max(dR,0))
            a=math.exp(-1/6); T_arr[k]=a*T_arr[k-1]+(1-a)*T_t
        I_dis=cr*Q.astype(np.float32)
        V_ocv=V_nom+0.15*(Q/Q0-0.5); V_avg=(V_ocv-I_dis*R).astype(np.float64)
        E_h=V_avg*I_dis; E_h=np.where(mode==1,E_h,0.0)
        self.y1=np.clip(V_avg+rng2.normal(0,.007,total_h),2.5,4.4).astype(np.float32)
        self.y2=np.clip(T_arr+rng2.normal(0,.35,total_h),T_amb-2,90).astype(np.float32)
        self.y3=np.clip(E_h+rng2.normal(0,3,total_h),-5,500).astype(np.float32)
        self.y3=np.where(mode==1,self.y3,self.y3*0.2).astype(np.float32)
        self.Q_true=Q.astype(np.float32); self.R_true=R.astype(np.float32)
        self.total_h=total_h
        # runtime state
        self.hour=0; self.cum_d=0.0; self.cum_r=0.0
        self.detector=ModeDetector()
        self.q_raw=[]; self.r_raw=[]; self.qsd=[]; self.e_meas=[]; self.e_hat=[]; self.modes=[]
        self.q_disp=[]; self.r_disp=[]
        self.fc={}; self.last_fc_h=-99; self.last={}

    def step(self):
        k=self.hour % self.total_h
        y1,y2,y3=float(self.y1[k]),float(self.y2[k]),float(self.y3[k])
        m_det=int(self.detector.update(y1,y2,y3))
        if m_det==1: self.cum_d+=1
        else: self.cum_r+=1
        x=float(self.cr_sch[k]) if m_det==1 else 0.0
        t_max=drive_norm.t_max_train_hr
        xrow=make_feat(self.hour,x,self.cum_d,self.cum_r,t_max)[None,:]
        if m_det==1: mu_q,q_sd,mu_r,y_hat=predict(drive_model,drive_norm,drive_resid,xrow)
        else:        mu_q,q_sd,mu_r,y_hat=predict(rest_model, rest_norm, rest_resid, xrow)
        self.q_raw.append(float(mu_q[0])); self.r_raw.append(float(mu_r[0]))
        self.qsd.append(float(q_sd[0])); self.e_meas.append(y3)
        self.e_hat.append(float(y_hat[0,2])); self.modes.append(m_det)
        q_d,r_d=enforce_physics(np.array(self.q_raw),np.array(self.r_raw),drive_norm.q0,drive_norm.r0)
        self.q_disp=q_d.tolist(); self.r_disp=r_d.tolist()
        q_now=float(q_d[-1]); r_now=float(r_d[-1])
        soh=float(np.clip(100*q_now/drive_norm.q0,0,110))
        day_now=self.hour//24
        if self.hour-self.last_fc_h>=24:
            self.fc=run_forecast(self.hour,self.cum_d,self.cum_r,q_d,r_d)
            self.last_fc_h=self.hour
        fc=self.fc; pred_eol=fc.get("pred_eol_day")
        pred_rul=max(0,pred_eol-day_now) if pred_eol is not None else None
        # windowed history for charts (last 14 days / 60 days)
        ew=min(len(self.e_meas),336); dw=min(len(self.q_disp),60*24)
        cap_hist=[{"day":(self.hour-dw+i)//24,"Q":float(self.q_disp[-dw:][i])} for i in range(0,dw,24)]
        res_hist=[{"day":(self.hour-dw+i)//24,"R":float(self.r_disp[-dw:][i])*1e3} for i in range(0,dw,24)]
        e_hist=[{"h":self.hour-ew+i,"measured":float(self.e_meas[-ew:][i]),"model":float(self.e_hat[-ew:][i])} for i in range(ew)]
        m_hist=[{"h":self.hour-ew+i,"mode":int(self.modes[-ew:][i])} for i in range(ew)]
        q_sd_arr=np.array(self.qsd); qsd_day=float(q_sd_arr[-1]) if q_sd_arr.size else 0.0
        event={
            "hour":self.hour,"day":day_now,
            "soh":round(soh,2),"q_now":round(q_now,3),"r_now_mohm":round(r_now*1e3,3),
            "mode":m_det,"mode_label":"Driving" if m_det==1 else "Resting",
            "q0":drive_norm.q0,"q_eol":drive_norm.q_eol,
            "pred_eol_day":pred_eol,"pred_rul_days":pred_rul,
            "eol_band":fc.get("eol_band",[None,None]),
            "forecast":{
                "days":fc.get("day_f",[]),"q":fc.get("q_day_f",[]),
                "q_upper":[q+1.96*sd for q,sd in zip(fc.get("q_day_f",[]),fc.get("qsd_day_f",[]))],
                "q_lower":[max(0,q-1.96*sd) for q,sd in zip(fc.get("q_day_f",[]),fc.get("qsd_day_f",[]))],
            },
            "cap_hist":cap_hist,"res_hist":res_hist,
            "energy_hist":e_hist,"mode_hist":m_hist,
        }
        self.last=event; self.hour+=1; return event

sim=SimState()

# ── app ───────────────────────────────────────────────────────────────────────
app=FastAPI(title="Battery TISAC API")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=True,allow_methods=["*"],allow_headers=["*"])

async def sse_generator():
    while True:
        data=sim.step()
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(STREAM_INTERVAL)

@app.get("/api/battery/stream")
async def stream():
    return StreamingResponse(sse_generator(),media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no","Connection":"keep-alive"})

@app.get("/api/battery/snapshot")
async def snapshot():
    return JSONResponse(sim.last if sim.last else sim.step())

@app.post("/api/battery/reset")
async def reset():
    sim.reset(); return {"status":"reset"}

@app.get("/healthz")
async def health(): return {"status":"ok","hour":sim.hour}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("battery_api:app",host="0.0.0.0",port=8000,reload=False)