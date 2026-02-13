import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../lib/gym_dssat_pdi_official/gym-dssat-pdi')))

from dssat_generic_wrapper import DssatGenericWrapper

def test_maize_generic():
    env_kwargs = {"run_dssat_location": "/opt/dssat_env/inst/run_dssat", "mode": "all"}
    env = DssatGenericWrapper(crop_name="maize", env_kwargs=env_kwargs)
    obs, info = env.reset()
    print(f"Maize Generic Reset Success. Obs shape: {obs.shape}")
    env.close()

if __name__ == "__main__":
    test_maize_generic()
