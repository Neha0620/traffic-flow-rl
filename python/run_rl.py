"""
run_rl.py

Loads the trained PPO policy and runs it on the network, logging total
waiting time every step for comparison against the fixed-time baseline
(run_no_rl.py).
"""
import os
import sys
import json

sys.path.append(os.path.dirname(__file__))
sys.path.append("/usr/share/sumo/tools")
os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from sumo_rl_env_multi import SumoMultiEnv

SUMO_CFG = os.path.join(os.path.dirname(__file__), "..", "sumo", "koramangala.sumocfg")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_multi_tls")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MAX_STEPS = 720

model = PPO.load(MODEL_PATH)

env = SumoMultiEnv(SUMO_CFG, gui=False, max_steps=MAX_STEPS)
obs, _ = env.reset()
done = False
step = 0
waiting_times = []
actions_log = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_waiting = info["waiting_time"]
    waiting_times.append(total_waiting)
    actions_log.append(action.tolist())

    step += 1
    if step % 50 == 0:
        print(f"Step {step} | Waiting time = {total_waiting:.2f}")

env.close()
print("Final total waiting time (RL):", waiting_times[-1])

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(os.path.join(RESULTS_DIR, "waiting_times_rl.json"), "w") as f:
    json.dump(waiting_times, f)

plt.figure(figsize=(10, 5))
plt.plot(waiting_times, label="RL Controlled (PPO)", color="blue")
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time (s)")
plt.title("Traffic Waiting Time Over Steps — RL Controlled (PPO)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "waiting_time_rl.png"), dpi=150)
print("Saved plot to results/waiting_time_rl.png")
