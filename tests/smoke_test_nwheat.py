import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib/gym_dssat_pdi_official/gym-dssat-pdi")))

from dssat_generic_wrapper import DssatGenericWrapper


def test_wheat_smoke():
    env_kwargs = {"run_dssat_location": "/opt/dssat_env/inst/run_dssat", "mode": "all"}
    env = DssatGenericWrapper(crop_name="wheat", env_kwargs=env_kwargs, history_len=1)
    obs, info = env.reset()
    print(f"Wheat Reset Success. Obs shape: {obs.shape}, info keys: {list(info.keys())}")
    obs, reward, terminated, truncated, info = env.step(0)
    print(
        "Wheat Step Success. "
        f"reward={reward}, terminated={terminated}, truncated={truncated}, obs shape={obs.shape}"
    )
    env.close()


if __name__ == "__main__":
    test_wheat_smoke()
