import gymnasium as gym
import os
import sys

# 注入路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib/gym_dssat_pdi_official/gym-dssat-pdi')))

from gym_dssat_pdi.envs.dssat_pdi import DssatPdi

def minimal_test():
    print("--- Minimal Test Starting ---")
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
    print("Environment instantiated successfully")
    obs, info = env.reset()
    print("Reset passed")
    print(f"Observation keys: {obs.keys()}")
    env.close()

if __name__ == "__main__":
    minimal_test()
