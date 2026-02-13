import gymnasium as gym
import os
import sys

# Ensure library is in path
sys.path.append(os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))
from gym_dssat_pdi.envs.dssat_pdi import DssatPdi

def verify_tomato_data():
    print("--- Tomato Data Verification Starting ---")
    kwargs = {
        "run_dssat_location": "/opt/dssat_env/inst/run_dssat",
        "cultivar": "tomato",
        "fileX_template_path": "templates/tomato/UFGA0602.jinja2",
        "auxiliary_file_paths": [
            "/opt/dssat_env/data/UFGA.CLI",
            "/opt/dssat_env/data/SOIL.SOL",
            "/opt/dssat_env/data/UFGA0601.WTH"
        ],
        "mode": "all",
    }
    
    env = DssatPdi(**kwargs)
    obs, info = env.reset()
    
    print("Reset Observation Keys: " + str(list(obs.keys())))
    
    # 关键生理指标
    keys_to_check = ['vstage', 'grnwt', 'xlai', 'dap']
    print("\nInitial State (DAP 0):")
    for k in keys_to_check:
        print(f"  {k}: {obs.get(k)}")
        
    # 执行一步 (无水肥操作)
    action = {'anfer': 0.0, 'amir': 0.0}
    obs, reward, terminated, truncated, info = env.step(action)
    
    print("\nState after 1 step (DAP " + str(obs.get('dap')) + "):")
    for k in keys_to_check:
        print(f"  {k}: {obs.get(k)}")
        
    env.close()
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    verify_tomato_data()
