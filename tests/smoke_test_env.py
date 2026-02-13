import os
import sys

# 注入 gym -> gymnasium 映射，处理旧代码兼容性
import gymnasium
sys.modules["gym"] = gymnasium

try:
    import gym_dssat_pdi
    print("gym_dssat_pdi package imported successfully!")
    
    import gymnasium as gym
    # 尝试列出已注册的环境，看看 gym-dssat 是否在里面
    envs = [e for e in gym.envs.registration.registry.keys() if "dssat" in e.lower()]
    print(f"Registered DSSAT environments: {envs}")
    
except Exception as e:
    print(f"Error during smoke test: {e}")
    import traceback
    traceback.print_exc()
