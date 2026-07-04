"""
sumo_rl_env_multi.py

Custom multi-intersection SUMO environment for Reinforcement-Learning-based
traffic signal control, built directly on TraCI (no external RL-for-SUMO
library) — matching the architecture used in the mini-project report:

    State  : per-lane queue length (halting vehicle count) for every
             incoming lane of every controlled traffic light, plus a
             one-hot encoding of each traffic light's current phase.
    Action : MultiDiscrete — one discrete "which phase should be active"
             choice per traffic light, chosen simultaneously every step.
    Reward : reduction in total accumulated waiting time (summed over all
             incoming lanes) since the previous step. Positive reward means
             waiting time went down.

This file fixes several bugs present in the original report listing
(broken quote characters in the SUMO command, `tl_phases` being
overwritten instead of collected into a list, and an action/observation
space that didn't match the state actually being built).
"""
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci


def traci_running():
    """Return True if a TraCI connection is currently open."""
    try:
        return traci._connections and traci._connections.get("default") is not None
    except Exception:
        try:
            traci.getConnection()
            return True
        except Exception:
            return False


class SumoMultiEnv(gym.Env):
    """Simplified multi-intersection SUMO environment for RL traffic control.

    Observation: per-lane queue lengths (all controlled TLS) + one-hot
                 signal phase per TLS.
    Action: MultiDiscrete([n_phases_tls1, n_phases_tls2, ...]) — one phase
            choice per traffic light, applied simultaneously.
    Reward: reduction in total waiting time across all incoming lanes.
    """

    metadata = {"render_modes": []}

    def __init__(self, sumo_cfg, gui=False, max_steps=2000, step_length=5):
        super().__init__()
        self.sumo_cfg = sumo_cfg
        self.gui = gui
        self.max_steps = max_steps
        self.step_length = step_length  # sim-seconds advanced per RL step
        self.step_count = 0
        self.initialized = False

        self.sumo_binary = "sumo-gui" if gui else "sumo"
        self.sumo_cmd = [
            self.sumo_binary,
            "-c", self.sumo_cfg,
            "--start",
            "--quit-on-end",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "300",
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]

        self.prev_waiting = 0.0
        self.tl_ids = []
        self.tl_phases = []   # number of phases for each tl, in order
        self.in_lanes = []

        # Determine the real observation/action space sizes *now* (not on
        # first reset) by briefly connecting to SUMO once at construction
        # time. This matters because vectorised-env wrappers (e.g.
        # Stable-Baselines3's DummyVecEnv) read env.observation_space /
        # env.action_space immediately after __init__, before reset() is
        # ever called.
        self._discover_network_layout()

    # ------------------------------------------------------------------
    def _discover_network_layout(self):
        if traci_running():
            traci.close(False)
        traci.start([
            "sumo", "-c", self.sumo_cfg,
            "--start", "--quit-on-end", "--no-step-log", "true", "--no-warnings", "true",
        ])

        self.tl_ids = list(traci.trafficlight.getIDList())

        self.tl_phases = []
        for tl in self.tl_ids:
            logic = traci.trafficlight.getAllProgramLogics(tl)[0]
            self.tl_phases.append(len(logic.phases))

        incoming = []
        for tl in self.tl_ids:
            controlled = traci.trafficlight.getControlledLinks(tl)
            for lane_list in controlled:
                for link in lane_list:
                    incoming.append(link[0])  # incoming lane id
        self.in_lanes = sorted(set(incoming))

        self.total_phases = sum(self.tl_phases)
        self.obs_size = len(self.in_lanes) + self.total_phases
        self.observation_space = spaces.Box(
            low=0, high=500, shape=(self.obs_size,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(self.tl_phases)
        self.initialized = True

        traci.close(False)

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if traci_running():
            traci.close(False)
        traci.start(self.sumo_cmd)
        self.step_count = 0

        self.prev_waiting = self._compute_total_waiting_time()
        return self._get_state(), {"waiting_time": self.prev_waiting}

    # ------------------------------------------------------------------
    def _get_state(self):
        state = []
        for lane in self.in_lanes:
            state.append(traci.lane.getLastStepHaltingNumber(lane))
        for i, tl in enumerate(self.tl_ids):
            phase = traci.trafficlight.getPhase(tl)
            onehot = np.zeros(self.tl_phases[i], dtype=np.float32)
            onehot[phase] = 1.0
            state.extend(onehot.tolist())
        return np.array(state, dtype=np.float32)

    # ------------------------------------------------------------------
    def step(self, actions=None):
        """If actions is None, no traffic light is touched at all — the
        network just runs its own programmed fixed-time plan untouched.
        This is the correct way to measure a "no RL" / fixed-time
        baseline: repeatedly calling setPhase() with the *same* phase the
        light is already in still resets SUMO's internal phase timer and
        can freeze the signal, which would unfairly make the baseline
        look worse than it really is."""
        self.step_count += 1
        if actions is not None:
            for i, (tl, phase) in enumerate(zip(self.tl_ids, actions)):
                traci.trafficlight.setPhase(tl, int(phase) % self.tl_phases[i])

        for _ in range(self.step_length):
            traci.simulationStep()

        obs = self._get_state()
        current_waiting = self._compute_total_waiting_time()
        reward = self.prev_waiting - current_waiting
        self.prev_waiting = current_waiting

        terminated = self.step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {"waiting_time": current_waiting}

    # ------------------------------------------------------------------
    def _compute_total_waiting_time(self):
        total = 0.0
        for lane in self.in_lanes:
            total += traci.lane.getWaitingTime(lane)
        return total

    def close(self):
        if traci_running():
            traci.close(False)
