"""
run_no_rl.py

Baseline: runs the network with SUMO's own fixed-time signal programs
(the agent doesn't choose actions — each traffic light just keeps
advancing through its programmed phase sequence). Logs total waiting
time every step for later comparison against the RL-controlled run.
"""
import os
import sys
import json

sys.path.append(os.path.dirname(__file__))
sys.path.append("/usr/share/sumo/tools")
os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")

import traci
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sumo_rl_env_multi import SumoMultiEnv

SUMO_CFG = os.path.join(os.path.dirname(__file__), "..", "sumo", "koramangala.sumocfg")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MAX_STEPS = 720

env = SumoMultiEnv(SUMO_CFG, gui=False, max_steps=MAX_STEPS)
obs, _ = env.reset()
done = False
step = 0
waiting_times_no_rl = []

while not done:
    # True fixed-time baseline: don't touch the traffic lights at all —
    # let SUMO run each junction's own programmed signal plan untouched.
    # (Re-calling setPhase() with the light's current phase every step
    # would reset SUMO's internal phase timer and can freeze the signal —
    # that would make this baseline look artificially worse than it is.)
    obs, reward, done, truncated, info = env.step(actions=None)
    total_waiting = info["waiting_time"]
    waiting_times_no_rl.append(total_waiting)

    step += 1
    if step % 50 == 0:
        print(f"Step {step} | Waiting time = {total_waiting:.2f}")

env.close()
print("Final total waiting time (no RL):", waiting_times_no_rl[-1])

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "waiting_times_no_rl.json"), "w") as f:
    json.dump(waiting_times_no_rl, f)

plt.figure(figsize=(10, 5))
plt.plot(waiting_times_no_rl, label="Baseline (No RL)", color="red")
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time (s)")
plt.title("SUMO Total Waiting Time — Fixed-Time Signals (No RL)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "waiting_time_no_rl.png"), dpi=150)
print("Saved plot to results/waiting_time_no_rl.png")
