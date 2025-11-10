#!/usr/bin/env python3
"""
Adaptive Multi-Bus Self-Healing Simulation + Live Dashboard
Press Ctrl+C to stop.
"""

import argparse, asyncio, os, time, random, string, signal, sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
import pandas as pd
from joblib import dump, load
from river import anomaly, stats

BARS = "▁▂▃▄▅▆▇█"   # unicode levels for mini bar charts

def rand_bus_id(): return "bus-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def simulate_features(bus_id):
    seed = sum(ord(c) for c in bus_id) % 9973
    random.seed(seed + int(time.time() // 5))
    base_latency = 5.0 + (hash(bus_id) % 100) / 1000.0
    noise = random.gauss(0.0, 0.8)
    latency = max(0.2, base_latency + noise)
    err = 1 if random.random() < 0.10 else 0
    if random.random() < 0.03:
        latency += random.uniform(15, 60)
        err = 1
    return {"latency": latency, "errors": err}

@dataclass
class BusAgent:
    bus_id: str
    model_dir: str
    model: anomaly.HalfSpaceTrees = field(default_factory=lambda: anomaly.HalfSpaceTrees(seed=random.randint(0, 9999)))
    score_mean: stats.Mean = field(default_factory=stats.Mean)
    avg_latency: stats.Mean = field(default_factory=stats.Mean)
    avg_errors: stats.Mean = field(default_factory=stats.Mean)
    total_samples: int = 0
    total_anoms: int = 0
    recent: list = field(default_factory=lambda: [0]*30)

    def model_path(self): return os.path.join(self.model_dir, f"{self.bus_id}.joblib")

    def step(self):
        f = simulate_features(self.bus_id)
        score = self.model.score_one(f)
        thr = (self.score_mean.get()*2) if self.score_mean.get() else 5
        anom = int(score > thr)
        self.model.learn_one(f)
        self.score_mean.update(score)
        self.avg_latency.update(f["latency"])
        self.avg_errors.update(f["errors"])
        self.total_samples += 1
        self.total_anoms += anom
        self.recent.append(anom)
        if len(self.recent) > 30: self.recent.pop(0)
        return {"lat": f["latency"], "err": f["errors"], "anom": anom}

    def spark(self):
        """Return a small bar graph of recent anomaly pattern."""
        return "".join(BARS[min(len(BARS)-1, int(x*(len(BARS)-1)))] for x in self.recent)

class BusManager:
    def __init__(self, n, interval, checkpoint, churn):
        self.interval, self.ckpt, self.churn = interval, checkpoint, churn
        ensure_dir("logs"); ensure_dir("models")
        self.agents: Dict[str, BusAgent] = {rand_bus_id(): BusAgent(rand_bus_id(), "models") for _ in range(n)}
        self.shutdown = asyncio.Event()
        self.last_print = time.time()
        self.last_ckpt = time.time()
        self.logfile = f"logs/log_{datetime.now():%Y%m%d_%H%M%S}.csv"
        pd.DataFrame(columns=["timestamp","bus_id","lat","err","anom"]).to_csv(self.logfile,index=False)

    async def run(self):
        while not self.shutdown.is_set():
            if self.churn and random.random() < 0.05:
                self._churn()
            recs=[]
            for bid,a in self.agents.items():
                r=a.step()
                recs.append({"timestamp":time.time(),"bus_id":bid,**r})
            pd.DataFrame(recs).to_csv(self.logfile,mode="a",header=False,index=False)

            if time.time()-self.last_ckpt>self.ckpt:
                for a in self.agents.values(): dump(a.model,a.model_path())
                self.last_ckpt=time.time()

            if time.time()-self.last_print>1.5:
                self._display()
                self.last_print=time.time()
            await asyncio.sleep(self.interval)
        print("\n[manager] shutdown complete")

    def _churn(self):
        if random.random()<0.5 and len(self.agents)>2:
            k=random.choice(list(self.agents.keys()))
            self.agents.pop(k,None)
            print(f"[manager] removed {k}")
        else:
            k=rand_bus_id()
            self.agents[k]=BusAgent(k,"models")
            print(f"[manager] added {k}")

    def _display(self):
        os.system("clear")
        print(f"=== Self-Healing Live Dashboard @ {datetime.now():%H:%M:%S} ===")
        tot_s=sum(a.total_samples for a in self.agents.values())
        tot_a=sum(a.total_anoms for a in self.agents.values())
        print(f"Buses: {len(self.agents)}   Samples: {tot_s}   Anomalies: {tot_a}")
        print(f"{'Bus ID':<12}{'AvgLat':>8}{'Err%':>8}{'Anom%':>8}  Recent Activity")
        print("-"*70)
        for bid,a in sorted(self.agents.items()):
            err=(a.avg_errors.get() or 0)*100
            anom=(a.total_anoms/a.total_samples*100) if a.total_samples else 0
            print(f"{bid:<12}{a.avg_latency.get():>8.2f}{err:>8.2f}{anom:>8.2f}  {a.spark()}")
        print("\nPress Ctrl+C to stop.")

    def stop(self): self.shutdown.set()

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--buses",type=int,default=8)
    ap.add_argument("--interval",type=float,default=0.5)
    ap.add_argument("--checkpoint",type=int,default=60)
    ap.add_argument("--churn",action="store_true")
    a=ap.parse_args()
    mgr=BusManager(a.buses,a.interval,a.checkpoint,a.churn)
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGINT,signal.SIGTERM):
        try: loop.add_signal_handler(sig,mgr.stop)
        except Exception: pass
    await mgr.run()

if __name__=="__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: print("\nStopped by user.")

