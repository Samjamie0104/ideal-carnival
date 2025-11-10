#!/usr/bin/env python3
"""
Adaptive Multi‑Bus Self‑Healing Simulation (PyTorch GPU backend) + Live Dashboard
- Drop‑in GPU refactor of your previous `jetson (1).py`
- Keeps: churn, async loop, CSV logging, mini bar "sparks"
- Replaces: river anomaly model with a tiny PyTorch MLP (CUDA‑accelerated)

Usage examples
--------------
# default 8 buses, 0.5s interval, checkpoints every 60s
python3 jetson_gpu.py --churn 

# faster ticks, mixed precision (if supported), more buses
python3 jetson_gpu.py --buses 16 --interval 0.25 --amp

Note: On Jetson Nano, monitor with `sudo tegrastats` to see GPU utilisation.
"""

import argparse, asyncio, os, time, random, string, signal
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ────────────────────────────────────────────────────────────────────────────────
# UI helpers
BARS = "▁▂▃▄▅▆▇█"   # unicode levels for mini bar charts

def rand_bus_id() -> str:
    return "bus-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Simulation of per‑bus features (same behaviour as your original code)

def simulate_features(bus_id: str):
    seed = sum(ord(c) for c in bus_id) % 9973
    random.seed(seed + int(time.time() // 5))
    base_latency = 5.0 + (hash(bus_id) % 100) / 1000.0
    noise = random.gauss(0.0, 0.8)
    latency = max(0.2, base_latency + noise)
    err = 1 if random.random() < 0.10 else 0
    # rare spikes
    if random.random() < 0.03:
        latency += random.uniform(15, 60)
        err = 1
    return {"latency": latency, "errors": err}

# ────────────────────────────────────────────────────────────────────────────────
# Light online stats (replacement for river.stats.Mean)

class OnlineMean:
    def __init__(self):
        self.n = 0
        self.mu = 0.0
    def update(self, x: float):
        self.n += 1
        self.mu += (x - self.mu) / self.n
    def get(self) -> float:
        return self.mu if self.n else 0.0

# ────────────────────────────────────────────────────────────────────────────────
# PyTorch anomaly model

class AnomalyMLP(nn.Module):
    """Tiny MLP that maps [latency, error] → anomaly score in [0,1]."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

# ────────────────────────────────────────────────────────────────────────────────
# Bus agent using GPU model

@dataclass
class BusAgent:
    bus_id: str
    model_dir: str
    device: torch.device
    use_amp: bool = False

    model: AnomalyMLP = field(default_factory=AnomalyMLP)
    opt: optim.Optimizer = field(init=False)
    loss_fn: nn.Module = field(default_factory=nn.MSELoss)

    score_mean: OnlineMean = field(default_factory=OnlineMean)
    avg_latency: OnlineMean = field(default_factory=OnlineMean)
    avg_errors: OnlineMean = field(default_factory=OnlineMean)

    total_samples: int = 0
    total_anoms: int = 0
    recent: List[int] = field(default_factory=lambda: [0]*30)

    def __post_init__(self):
        self.model = self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-2)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # file path to save per‑bus model
    def model_path(self) -> str:
        return os.path.join(self.model_dir, f"{self.bus_id}.pt")

    def save(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'score_mean': (self.score_mean.n, self.score_mean.mu),
            'avg_latency': (self.avg_latency.n, self.avg_latency.mu),
            'avg_errors': (self.avg_errors.n, self.avg_errors.mu),
            'total_samples': self.total_samples,
            'total_anoms': self.total_anoms,
            'recent': self.recent,
        }, self.model_path())

    def load(self):
        p = self.model_path()
        if not os.path.exists(p):
            return
        ckpt = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
        self.opt.load_state_dict(ckpt['opt'])
        self.score_mean.n, self.score_mean.mu = ckpt['score_mean']
        self.avg_latency.n, self.avg_latency.mu = ckpt['avg_latency']
        self.avg_errors.n, self.avg_errors.mu = ckpt['avg_errors']
        self.total_samples = ckpt['total_samples']
        self.total_anoms = ckpt['total_anoms']
        self.recent = ckpt['recent']

    def step(self):
        f = simulate_features(self.bus_id)

        # Prepare tensors on GPU
        x = torch.tensor([[f["latency"], float(f["errors"])]], dtype=torch.float32, device=self.device)
        # training target heuristic: treat `errors` as anomaly label (0/1)
        y = torch.tensor([[float(f["errors"])]], dtype=torch.float32, device=self.device)

        # Forward / backward (optionally AMP)
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pred = self.model(x)          # shape (1,1)
            loss = self.loss_fn(pred, y)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.opt)
        self.scaler.update()

        score = float(pred.item())
        # adaptive threshold like your original: 2× running mean (fallback 0.6)
        thr = (self.score_mean.get() * 2.0) if self.score_mean.get() > 0 else 0.6
        anom = int(score > thr)

        # update stats
        self.score_mean.update(score)
        self.avg_latency.update(f["latency"])
        self.avg_errors.update(float(f["errors"]))

        self.total_samples += 1
        self.total_anoms += anom
        self.recent.append(anom)
        if len(self.recent) > 30:
            self.recent.pop(0)

        return {"lat": f["latency"], "err": f["errors"], "anom": anom, "score": score, "thr": thr}

    def spark(self) -> str:
        return "".join(BARS[min(len(BARS)-1, int(x*(len(BARS)-1)))] for x in self.recent)

# ────────────────────────────────────────────────────────────────────────────────
# Manager and async loop

class BusManager:
    def __init__(self, n: int, interval: float, checkpoint: int, churn: bool, use_amp: bool, device: torch.device):
        self.interval, self.ckpt, self.churn = interval, checkpoint, churn
        self.device, self.use_amp = device, use_amp
        ensure_dir("logs"); ensure_dir("models")

        # Create agents (unique IDs)
        self.agents: Dict[str, BusAgent] = {}
        for _ in range(n):
            bid = rand_bus_id()
            self.agents[bid] = BusAgent(bid, "models", device=self.device, use_amp=self.use_amp)
            self.agents[bid].load()

        self.shutdown = asyncio.Event()
        self.last_print = time.time()
        self.last_ckpt = time.time()
        self.logfile = f"logs/log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(columns=["timestamp","bus_id","lat","err","anom","score","thr"]).to_csv(self.logfile,index=False)

    async def run(self):
        while not self.shutdown.is_set():
            # occasional churn: add/remove a bus
            if self.churn and random.random() < 0.05:
                self._churn()

            # step all agents (we keep per‑bus optimisation for simplicity)
            recs = []
            for bid, a in self.agents.items():
                r = a.step()
                recs.append({"timestamp": time.time(), "bus_id": bid, **r})

            # append to CSV
            pd.DataFrame(recs).to_csv(self.logfile, mode="a", header=False, index=False)

            # periodic checkpoints
            if time.time() - self.last_ckpt > self.ckpt:
                for a in self.agents.values():
                    a.save()
                self.last_ckpt = time.time()

            # screen refresh (throttled)
            if time.time() - self.last_print > 1.5:
                self._display()
                self.last_print = time.time()

            await asyncio.sleep(self.interval)
        print("\n[manager] shutdown complete")

    def _churn(self):
        if random.random() < 0.5 and len(self.agents) > 2:
            k = random.choice(list(self.agents.keys()))
            self.agents.pop(k, None)
            try:
                os.remove(os.path.join("models", f"{k}.pt"))
            except Exception:
                pass
            print(f"[manager] removed {k}")
        else:
            k = rand_bus_id()
            self.agents[k] = BusAgent(k, "models", device=self.device, use_amp=self.use_amp)
            print(f"[manager] added {k}")

    def _display(self):
        os.system("clear")
        print(f"=== Self‑Healing Live Dashboard @ {datetime.now():%H:%M:%S} ===")
        tot_s = sum(a.total_samples for a in self.agents.values())
        tot_a = sum(a.total_anoms for a in self.agents.values())
        print(f"Device: {self.device}   Buses: {len(self.agents)}   Samples: {tot_s}   Anomalies: {tot_a}")
        print(f"{'Bus ID':<12}{'AvgLat':>8}{'Err%':>8}{'Anom%':>8}{' Thr':>7}{' Score':>8}  Recent Activity")
        print("-"*86)
        for bid, a in sorted(self.agents.items()):
            err = (a.avg_errors.get() or 0.0) * 100.0
            anom = (a.total_anoms/a.total_samples*100.0) if a.total_samples else 0.0
            print(f"{bid:<12}{a.avg_latency.get():>8.2f}{err:>8.2f}{anom:>8.2f}{a.score_mean.get()*2:>7.2f}{'':>1}{'':>1}  {a.spark()}")
        print("\nPress Ctrl+C to stop.")

    def stop(self):
        self.shutdown.set()

# ────────────────────────────────────────────────────────────────────────────────
# Entrypoint

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buses", type=int, default=8, help="number of concurrent bus agents")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between ticks")
    ap.add_argument("--checkpoint", type=int, default=60, help="seconds between model checkpoints")
    ap.add_argument("--churn", action="store_true", help="enable random add/remove of buses")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    ap.add_argument("--amp", action="store_true", help="enable mixed precision (autocast + GradScaler)")
    args = ap.parse_args()

    # choose device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[init] Using device: {device}")
    if device.type == 'cuda':
        print("[init] CUDA available:", torch.cuda.is_available())
        print("[init] GPU name:", torch.cuda.get_device_name(0))

    mgr = BusManager(args.buses, args.interval, args.checkpoint, args.churn, args.amp, device)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, mgr.stop)
        except Exception:
            pass

    await mgr.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user.")


