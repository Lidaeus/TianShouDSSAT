import os
import sys
import signal
import numpy as np

# 1. 强制超时设置 (60秒)
def timeout_handler(signum, frame):
    print("\n[TIMEOUT] 脚本执行超过 60 秒，强制退出以防阻塞！")
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)

# 2. 注入路径
PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maize_wrapper import MaizeEnvWrapper

def test_single_env():
    print("--- 启动单环境天授链路测试 (带 60s 强退) ---")
    env = None
    try:
        env = MaizeEnvWrapper(run_dssat_location='/opt/dssat_env/inst/run_dssat')
        
        print("开始首次 Reset...")
        obs, info = env.reset()
        print(f"首次 Reset 成功, Obs shape: {obs.shape}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, rew, done, trunc, info = env.step(action)
            print(f"Step {i}: Action={action}, Reward={rew}, Done={done}")
            if done:
                break
    except Exception as e:
        print(f"测试过程中发生异常: {e}")
    finally:
        if env:
            env.close()
        print("--- 单环境测试清理完成 ---")

if __name__ == "__main__":
    test_single_env()
    # 正常结束则关闭闹钟
    signal.alarm(0)
