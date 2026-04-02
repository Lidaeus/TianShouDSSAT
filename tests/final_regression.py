import os
import sys
import numpy as np

# Ensure library is in path
sys.path.append(os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))
from dssat_generic_wrapper import DssatGenericWrapper

def test_crop(crop_name):
    print("\n>>> Testing " + crop_name.upper() + " environment...")
    try:
        env = DssatGenericWrapper(crop_name=crop_name)
        obs, info = env.reset()
        print("[" + crop_name + "] Reset successful. Obs shape: " + str(obs.shape))
        
        # Take 5 steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print("  Step " + str(i+1) + ": Reward=" + str(reward))
            
        env.close()
        print("[" + crop_name + "] Test PASSED.")
        return True
    except Exception as e:
        print("[" + crop_name + "] Test FAILED: " + str(e))
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Ensure DssatPdi is patched
    os.system("python3 fix_dssat_pdi.py")
    
    results = {}
    results['maize'] = test_crop("maize")
    results['tomato'] = test_crop("tomato")
    results['wheat'] = test_crop("wheat")
    
    print("\n" + "="*30)
    print("FINAL REGRESSION RESULTS:")
    for crop, status_val in results.items():
        status_txt = "OK" if status_val else "FAIL"
        print("  " + crop + ": " + status_txt)
    print("="*30)
    
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
