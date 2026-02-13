import sys
import os
import shutil
import glob
import time

# 1. 强行注入路径
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
PACKAGE_PATH = os.path.join(PROJECT_ROOT, "lib/gym_dssat_pdi_official/gym-dssat-pdi")
if PACKAGE_PATH not in sys.path:
    sys.path.insert(0, PACKAGE_PATH)

# 2. 准备打平的数据源
FLATTEN_DATA_DIR = os.path.join(PROJECT_ROOT, "lib/gym_dssat_pdi_official/flatten_data")

# 3. 注入环境变量 (确保 PDI 和 Python 正常)
VENV_BIN = os.path.join(PROJECT_ROOT, "venv/bin")
VENV_LIB = os.path.join(PROJECT_ROOT, "venv/lib/python3.12/site-packages")
PDI_PY_PATH = "/usr/local/lib/python3/dist-packages"
os.environ["PATH"] = f"{VENV_BIN}:{os.environ.get('PATH', '')}"
os.environ["PYTHONPATH"] = f"{PACKAGE_PATH}:{VENV_LIB}:{PDI_PY_PATH}:{os.environ.get('PYTHONPATH', '')}"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import gymnasium as gym
from gym_dssat_pdi.envs import DssatPdi

# --- 核心黑科技：对 DssatPdi 进行补丁 ---
original_make_tmp = DssatPdi._make_tmp_folder

def patched_make_tmp_folder(self):
    # 调用原方法创建临时目录
    original_make_tmp(self)
    print(f"[PATCH] 临时目录已创建: {self._tmp_folder}")
    
    # 暴力链接所有 Data 文件到临时目录
    print(f"[PATCH] 正在链接 Data 文件从: {FLATTEN_DATA_DIR}")
    files = glob.glob(f"{FLATTEN_DATA_DIR}/*")
    for f in files:
        target = os.path.join(self._tmp_folder, os.path.basename(f))
        if not os.path.exists(target):
            os.symlink(f, target)
    print(f"[PATCH] 链接完成，共 {len(files)} 个文件。")

# 应用补丁
DssatPdi._make_tmp_folder = patched_make_tmp_folder
# ---------------------------------------

RUN_DSSAT_PATH = os.path.join(PROJECT_ROOT, "lib/gym_dssat_pdi_official/install_dir/run_dssat")

print(f"--- 启动 Patch 型本地测试 ---")
LOG_FILE = os.path.join(PROJECT_ROOT, "dssat_test_exec.log")

try:
    # 实例化 (这次不传入 auxiliary，因为补丁会自动处理)
    env = DssatPdi(run_dssat_location=RUN_DSSAT_PATH,
                   mode='fertilization',
                   cultivar='maize',
                   log_saving_path=LOG_FILE)
    
    print("环境实例化成功！")
    
    # 执行 Reset
    print("开始 Reset...")
    obs = env.reset()
    print(f"Reset 成功！观测键值: {obs.keys() if isinstance(obs, dict) else type(obs)}")
    
    # 执行 Step
    action = env.action_space.sample()
    # 转换为原生类型以通过官方代码的严格校验
    if isinstance(action, dict):
        action = {k: float(v) if isinstance(v, (np.float32, np.float64, np.ndarray)) else v for k, v in action.items()}
    
    print(f"执行 Step，动作: {action}")
    res = env.step(action)
    print(f"Step 成功！返回内容: {res[2:]}") # 打印 done 等状态
    
    env.close()
    print("--- 测试圆满完成 ---")

except Exception as e:
    print(f"测试失败: {e}")
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            print("\n--- 日志片段 ---")
            print(f.read())
    import traceback
    traceback.print_exc()
