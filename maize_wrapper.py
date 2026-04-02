import os
import sys
import glob
import numpy as np
import gymnasium as gym
from collections import deque
from typing import Any
import psutil
import subprocess

# 注入路径
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
PACKAGE_PATH = os.path.join(PROJECT_ROOT, "lib/gym_dssat_pdi_official/gym-dssat-pdi")
if PACKAGE_PATH not in sys.path:
    sys.path.insert(0, PACKAGE_PATH)

from gym_dssat_pdi.envs.dssat_pdi import DssatPdi

# --- 补丁逻辑 (保持原样) ---
def patched_launch_client(self):
    pdi_command = f'/usr/bin/env {self._run_dssat_location} C fileX.MZX {self.experiment_number}'
    full_env = os.environ.copy()
    VENV_LIB = os.path.join(PROJECT_ROOT, "venv/lib/python3.12/site-packages")
    PDI_PY_PATH = "/usr/local/lib/python3/dist-packages"
    full_env["PYTHONPATH"] = f"{PACKAGE_PATH}:{VENV_LIB}:{PDI_PY_PATH}:{full_env.get('PYTHONPATH', '')}"
    full_env["LD_LIBRARY_PATH"] = f"/usr/local/lib:{full_env.get('LD_LIBRARY_PATH', '')}"
    pdi_command = pdi_command.split(' ')
    file_path = self.log_saving_path if self.log_saving_path else os.devnull
    self._f_out = open(file_path, 'a+')
    client_process = subprocess.Popen(pdi_command, stdout=self._f_out, stderr=self._f_out, shell=False, env=full_env, universal_newlines=True, cwd=self._tmp_folder, bufsize=0)
    self._client_process_pid = client_process.pid
DssatPdi._launch_client = patched_launch_client

FLATTEN_DATA_DIR = "/opt/dssat_env/data"
original_make_tmp = DssatPdi._make_tmp_folder
def patched_make_tmp_folder(self):
    original_make_tmp(self)
    files = glob.glob(f"{FLATTEN_DATA_DIR}/*")
    for f in files:
        target = os.path.join(self._tmp_folder, os.path.basename(f))
        if not os.path.exists(target): os.symlink(f, target)
DssatPdi._make_tmp_folder = patched_make_tmp_folder

class MaizeEnvWrapper(gym.Wrapper):
    def __init__(self, run_dssat_location, history_len=14, forecast_len=7):
        # 重要：先初始化缓冲区，因为 DssatPdi.__init__ 会立刻触发一次 reset/get_state
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.history_buffer = deque(maxlen=history_len)
        self._action_map = [0, 20, 40, 60, 80, 100, 140, 200]
        
        print(f"[Wrapper] 开始实例化 DssatPdi...")
        self.env = DssatPdi(run_dssat_location=run_dssat_location,
                           mode='fertilization',
                           cultivar='maize',
                           log_saving_path="maize_debug.log")
        super().__init__(self.env)
        
        self.action_space = gym.spaces.Discrete(len(self._action_map))
        obs_dim = (history_len * 3) + (forecast_len * 4) + 5
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        print(f"[Wrapper] 初始化成功。")

    def reset(self, seed=None, options=None):
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs["options"] = options
        try:
            reset_result = self.env.reset(**reset_kwargs)
        except TypeError:
            reset_result = self.env.reset()
        raw_obs, info = self._normalize_reset_result(reset_result)
        self.history_buffer.clear()
        return self._process_obs(raw_obs), info

    def _normalize_reset_result(self, reset_result: Any):
        if isinstance(reset_result, tuple) and len(reset_result) == 2 and isinstance(reset_result[1], dict):
            return reset_result[0], dict(reset_result[1])
        return reset_result, {}

    def _to_scalar_reward(self, reward: Any) -> float:
        if reward is None:
            return 0.0
        if isinstance(reward, (list, tuple, np.ndarray)):
            return float(np.sum(reward))
        return float(reward)

    def _process_obs(self, raw_obs):
        if not hasattr(self, 'history_buffer'): # 防御性编程
            self.history_buffer = deque(maxlen=self.history_len)
        
        nstres = float(raw_obs.get('nstres', 0)) if raw_obs else 0.0
        swfac = float(raw_obs.get('swfac', 0)) if raw_obs else 0.0
        xlai = float(raw_obs.get('xlai', 0)) if raw_obs else 0.0
        self.history_buffer.append([nstres, swfac, xlai])
        
        hist = list(self.history_buffer)
        while len(hist) < self.history_len: hist.insert(0, [0.0, 0.0, 0.0])
        forecast = np.zeros((self.forecast_len, 4))
        scalars = [
            float(raw_obs.get('istage', 0)) if raw_obs else 0,
            float(raw_obs.get('vstage', 0)) if raw_obs else 0,
            (float(raw_obs.get('dap', 0)) / 200.0) if raw_obs else 0,
            (float(raw_obs.get('cumsumfert', 0)) / 500.0) if raw_obs else 0,
            (float(raw_obs.get('grnwt', 0)) / 10000.0) if raw_obs else 0
        ]
        return np.concatenate([np.array(hist).flatten(), forecast.flatten(), np.array(scalars)]).astype(np.float32)

    def step(self, action_idx):
        action_dict = {'anfer': float(self._action_map[action_idx])}
        step_result = self.env.step(action_dict)
        if len(step_result) == 5:
            raw_obs, reward, terminated, truncated, info = step_result
        else:
            raw_obs, reward, done, info = step_result
            terminated = bool(done)
            truncated = False
        info = dict(info or {})
        info["applied_action"] = dict(action_dict)
        info["action_index"] = int(action_idx)
        info["native_reward"] = self._to_scalar_reward(reward)
        return self._process_obs(raw_obs), info["native_reward"], terminated, truncated, info

    def close(self):
        if hasattr(self, 'env'): self.env.close()
