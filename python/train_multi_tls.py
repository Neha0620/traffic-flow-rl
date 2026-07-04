"""
train_multi_tls.py

Trains a PPO agent (Stable-Baselines3) to control every traffic light in
the network simultaneously via the SumoMultiEnv MultiDiscrete action
space. Mirrors the training script described in the mini-project report.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append("/usr/share/sumo/tools")
os.environ.setdefault("SUMO_HOME", "/usr/share/sumo")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from sumo_rl_env_multi import SumoMultiEnv

SUMO_CFG = os.path.join(os.path.dirname(__file__), "..", "sumo", "koramangala.sumocfg")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_multi_tls")
MAX_STEPS_PER_EPISODE = 720  # 3600 sim-seconds / 5s per step


def make_env():
    env = SumoMultiEnv(SUMO_CFG, gui=False, max_steps=MAX_STEPS_PER_EPISODE)
    return Monitor(env)


def main(total_timesteps=20000):
    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(MODEL_OUT)
    print(f"Multi-TLS PPO training complete! Saved to {MODEL_OUT}.zip")


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(steps)
