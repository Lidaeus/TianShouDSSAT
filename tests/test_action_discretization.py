import gymnasium as gym
import pytest
import numpy as np
from dssat_generic_wrapper import DssatGenericWrapper

@pytest.fixture(autouse=True)
def setup_path():
    import sys
    import os
    # Inject library path
    sys.path.append(os.path.abspath("lib/gym_dssat_pdi_official/gym-dssat-pdi"))

def test_action_space_is_discrete():
    """Verify that the wrapper exposes a Discrete action space."""
    env = DssatGenericWrapper(crop_name="maize")
    try:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 36
    finally:
        env.close()

def test_step_accepts_integer():
    """Verify that step() accepts an integer action."""
    env = DssatGenericWrapper(crop_name="maize")
    env.reset()
    try:
        # Action 0 -> Anfer=0, Amir=0
        obs, reward, term, trunc, info = env.step(0)
        assert obs is not None
        assert not term, "Environment terminated immediately on first step!"
        
        # Action 35 -> Max Anfer, Max Amir
        obs, reward, term, trunc, info = env.step(35)
        assert obs is not None
    finally:
        env.close()

def test_map_action_logic():
    """Verify the mapping logic directly (if method is exposed or via subclassing)."""
    env = DssatGenericWrapper(crop_name="maize")
    try:
        # Check if internal helper exists and works
        if hasattr(env, "_map_action"):
            # 0 -> 0, 0
            act_dict = env._map_action(0)
            assert act_dict['anfer'] == 0
            assert act_dict['amir'] == 0
            
            # 6 -> Anfer index 1 (40), Amir index 0 (0) OR vice versa depending on implementation
            # Let's assume Anfer is outer loop (rows), Amir is inner (cols)
            # idx = anfer_idx * len(amir) + amir_idx
            # 6 = 1 * 6 + 0 -> Anfer=40, Amir=0
            
            # 35 -> Anfer index 5 (200), Amir index 5 (50)
            act_dict_max = env._map_action(35)
            # Check values roughly
            assert act_dict_max['anfer'] > 150
            assert act_dict_max['amir'] > 40
    finally:
        env.close()
