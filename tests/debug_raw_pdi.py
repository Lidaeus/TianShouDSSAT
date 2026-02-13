import sys
import os
sys.path.append(os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))
from gym_dssat_pdi.envs.dssat_pdi import DssatPdi

def test_raw_pdi():
    print("Initializing DssatPdi...")
    env = DssatPdi(
        run_dssat_location='/opt/dssat_env/inst/run_dssat',
        log_saving_path='logs/debug_pdi/dssat.log',
        cultivar='maize',
        mode='all',
        auxiliary_file_paths=[
            "/opt/dssat_env/data/UFGA.CLI", 
            "/opt/dssat_env/data/SOIL.SOL", 
            "/opt/dssat_env/data/UFGA8201.WTH"
        ],
        pdi_template_path='templates/maize/dssat_pdi.jinja2',
        fileX_template_path='templates/maize/UFGA8201.jinja2'
    )
    print("Resetting...")
    obs, info = env.reset()
    print(f"Reset done. YRDOY: {obs.get('yrdoy')}")
    
    print("Stepping...")
    # Raw env needs dict
    action = {'anfer': 0.0, 'amir': 0.0}
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step done. Terminated: {term}")
    env.close()

if __name__ == "__main__":
    test_raw_pdi()
