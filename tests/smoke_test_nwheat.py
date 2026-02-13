import sys
import os
import glob
import numpy as np

# 1. 路径设置
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
PACKAGE_PATH = os.path.join(PROJECT_ROOT, "lib/gym_dssat_pdi_official/gym-dssat-pdi")
if PACKAGE_PATH not in sys.path:
    sys.path.insert(0, PACKAGE_PATH)

# 2. 环境变量
VENV_BIN = os.path.join(PROJECT_ROOT, "venv/bin")
VENV_LIB = os.path.join(PROJECT_ROOT, "venv/lib/python3.12/site-packages")
PDI_PY_PATH = "/usr/local/lib/python3/dist-packages"
os.environ["PATH"] = f"{VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ["PYTHONPATH"] = f"{PACKAGE_PATH}:{VENV_LIB}:{PDI_PY_PATH}:{os.environ.get('PYTHONPATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import gymnasium as gym
from gym_dssat_pdi.envs import DssatPdi

# --- Monkey Patch (数据自动链接) ---
FLATTEN_DATA_DIR = "/opt/dssat_env/data"
original_make_tmp = DssatPdi._make_tmp_folder

def patched_make_tmp_folder(self):
    original_make_tmp(self)
    print(f"[PATCH] 链接小麦 Data 到: {self._tmp_folder}")
    files = glob.glob(f"{FLATTEN_DATA_DIR}/*")
    for f in files:
        target = os.path.join(self._tmp_folder, os.path.basename(f))
        if not os.path.exists(target):
            os.symlink(f, target)

DssatPdi._make_tmp_folder = patched_make_tmp_folder

# --- 运行小麦测试 ---
RUN_DSSAT_PATH = "/opt/dssat_env/inst/run_dssat"

print("--- 启动 Nwheat (小麦) 冒烟测试 ---")
try:
    env = DssatPdi(run_dssat_location=RUN_DSSAT_PATH,
                   mode='fertilization',
                   cultivar='wheat', # 切换为小麦
                   log_saving_path="nwheat_debug.log")
    
    obs = env.reset()
    print(f"小麦 Reset 成功！观测键值: {obs.keys()}")
    
    # 测试一个动作
    action = env.action_space.sample()
    if isinstance(action, dict):
        action = {k: float(v) if isinstance(v, (np.float32, np.float64, np.ndarray)) else v for k, v in action.items()}
    
    print(f"执行小麦 Step，动作: {action}")
    res = env.step(action)
    print(f"小麦 Step 成功！")
    
    env.close()
    print("--- 小麦环境测试通过 ---")

except Exception as e:
    print(f"小麦测试失败: {e}")
    if os.path.exists("nwheat_debug.log"):
        with open("nwheat_debug.log", "r") as f:
            print("\n--- 日志片段 ---")
            print(f.read()[-1000:])
    import traceback
    traceback.print_exc()
