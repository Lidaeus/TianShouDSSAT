import os
import sys
import numpy as np

PROJECT_ROOT = "/home/lidaeus/TianShouDSSAT"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from maize_wrapper import MaizeEnvWrapper

def test_wrapper():
    print("--- 开始测试 MaizeEnvWrapper ---")
    run_dssat = "/opt/dssat_env/inst/run_dssat"
    
    # 清理残留进程
    os.system("pkill -9 dscsm048")
    
    try:
        env = MaizeEnvWrapper(run_dssat_location=run_dssat)
        obs, info = env.reset()
        print(f"Reset 成功，Obs shape: {obs.shape}")
        
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            print(f"Step {i+1} 成功，Reward: {reward}, Done: {done}")
            if done:
                break
        
        env.close()
        print("测试完成，环境已关闭。")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_maize_wrapper = test_wrapper()
