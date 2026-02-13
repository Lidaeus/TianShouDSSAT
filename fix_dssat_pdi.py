import os
import textwrap

file_path = 'lib/gym_dssat_pdi_official/gym-dssat-pdi/gym_dssat_pdi/envs/dssat_pdi.py'
backup_path = 'backup_experimental/gym_dssat_pdi-stable/gym-dssat-pdi/gym_dssat_pdi/envs/dssat_pdi.py'

with open(backup_path, 'r') as f:
    content = f.read()

# 1. Imports
content = content.replace('import gym\n', 'import gymnasium as gym\n')
content = content.replace('import gym.spaces as spaces', 'from gymnasium import spaces')
content = content.replace('from gym.utils import seeding', 'from gymnasium.utils import seeding')
content = content.replace('from gym_dssat_pdi.envs import rewards', 'import gym_dssat_pdi.envs.rewards as rewards')
content = content.replace('from gym_dssat_pdi.envs import utils', 'import gym_dssat_pdi.envs.utils as utils')
content = content.replace('.randint(', '.integers(')

# 2. Init
old_init = "    def __init__(self, run_dssat_location='run_dssat', log_saving_path=None, mode='all',\n                 auxiliary_file_paths=None, files_prefix='./', random_weather=True, seed=None, fileX_template_path=None,\n                 experiment_number=None, evaluation=False, cultivar = \"maize\"):"
new_init = "    def __init__(self, run_dssat_location='run_dssat', log_saving_path=None, mode='all',\n                 auxiliary_file_paths=None, files_prefix='./', random_weather=True, seed=None, fileX_template_path=None,\n                 experiment_number=None, evaluation=False, cultivar = \"maize\", pdi_template_path=None):\n        self.closed = False\n        self.done = False\n        self._tmp_folder = None"
content = content.replace(old_init, new_init)

# 3. YAML
old_yaml = "self._pdi_yaml_template = pkgutil.get_data(__name__, f'configs/{cultivar}/dssat_pdi.jinja2').decode('utf-8')"
new_yaml = "if pdi_template_path is not None:\n            with open(pdi_template_path, 'r') as f_pdi:\n                self._pdi_yaml_template = f_pdi.read()\n        else:\n            self._pdi_yaml_template = pkgutil.get_data(__name__, f'configs/{cultivar}/dssat_pdi.jinja2').decode('utf-8')"
content = content.replace(old_yaml, new_yaml)

# 4. Popen
old_popen = """        client_process = subprocess.Popen(pdi_command,
                                          stdout=self._f_out,
                                          stderr=self._f_out,
                                          shell=False,
                                          universal_newlines=True,
                                          cwd=self._tmp_folder,
                                          bufsize=0,
                                          )"""
new_popen = """        env = os.environ.copy()\n        env["PYTHONPATH"] = "/usr/local/lib/python3/dist-packages:/home/lidaeus/TianShouDSSAT/venv/lib/python3.12/site-packages:" + env.get("PYTHONPATH", "")\n        log_f = open(os.path.join(self._tmp_folder, "dssat_raw.log"), "w")\n        client_process = subprocess.Popen(pdi_command,\n                                          stdout=log_f,\n                                          stderr=log_f,\n                                          shell=False,\n                                          universal_newlines=True,\n                                          cwd=self._tmp_folder,\n                                          bufsize=0,\n                                          env=env\n                                          )"""
content = content.replace(old_popen, new_popen)

# 5. Returns & Types
content = content.replace('return self.observation', 'return self.observation, {}')
# 核心修复：确保 step 不返回 None
content = content.replace('return observation, reward, done, context', 'return (observation if observation is not None else self.observation), (reward if reward is not None else 0.0), done, False, (context if context is not None else {})')
content = content.replace('return None, None, self.done, None', 'return self.observation, 0.0, self.done, False, {}')

# 6. NumPy 类型鲁棒性：在 _sanitary_check_action_dict 之前强制转换（或者在里面改）
# 我直接修改 _sanitary_check_action_dict
content = content.replace('assert isinstance(action_dict, dict)', 'for k,v in action_dict.items(): action_dict[k] = float(v)\n        assert isinstance(action_dict, dict)')

# 7. Tomato
content = content.replace('"rice"   : \'IRPI8001\'', '"rice"   : \'IRPI8001\',\n            "tomato" : \'UFGA0602\'')

with open(file_path, 'w') as f:
    f.write(content)
print("DssatPdi.py fixed and ready for final verification.")
