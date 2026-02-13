import sys
import os
import shutil
import gymnasium as gym

# 确保 venv/bin 在 PATH 中
venv_bin = os.path.join(os.getcwd(), "venv/bin")
os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

# 设置 DSSAT_HOME
dssat_home = os.path.join(os.getcwd(), "lib/gym_dssat_pdi-stable/dssat-csm-data")
os.environ["DSSAT_HOME"] = dssat_home

print("Importing gym_dssat_pdi...")
import gym_dssat_pdi
print("Imported gym_dssat_pdi successfully.")

def test():
    env = None
    try:
        from gym_dssat_pdi.envs import DssatPdi
        log_file = os.path.join(os.getcwd(), "dssat_debug.log")
        with open(log_file, 'w') as f: f.write("")
        
        # 尝试使用默认的 maize 配置
        print(f"Instantiating DssatPdi for 'maize' (logging to {log_file})...")
        env = DssatPdi(cultivar='maize', log_saving_path=log_file)
        
        print("Attempting env.reset()...")
        obs, info = env.reset()
        print("SUCCESS: Environment reset successfully!")
        print(f"Observation keys count: {len(obs)}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            print("Closing environment...")
            env.close()

if __name__ == "__main__":
    test()
