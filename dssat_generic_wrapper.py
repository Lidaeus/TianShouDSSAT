import gymnasium as gym
import numpy as np
from gym_dssat_pdi.envs.dssat_pdi import DssatPdi
from typing import Dict, List, Any, Optional

# 作物特定配置矩阵：避免过度抽象，明确定义差异
CROP_SPECIFIC_CONFIGS = {
    "maize": {
        "cultivar_code": "UFGA8201",
        "obs_keys": [
            "swfac", "vstage", "wtdep", "grnwt", 
            "topwt", "lai", "pcntn", "stresn"
        ],
        "default_template": "templates/maize/UFGA8201.jinja2",
        "is_transplanted": False
    },
    "tomato": {
        "cultivar_code": "UFGA0602",
        "obs_keys": [
            "swfac", "istage", "rtdep", "grnwt", 
            "topwt", "xlai", "nstres"
        ],
        "default_template": "templates/tomato/UFGA0602.jinja2",
        "is_transplanted": True
    }
}

class DssatGenericWrapper(gym.Wrapper):
    """
    Tianshou 2.0 兼容的通用 DSSAT 包装器。
    支持多作物切换，通过配置驱动解决作物间的生理差异。
    """
    def __init__(
        self, 
        crop_name: str,
        env_kwargs: Dict[str, Any] = {},
        history_len: int = 5,
        seed: Optional[int] = None
    ):
        if crop_name not in CROP_SPECIFIC_CONFIGS:
            raise ValueError(f"Unsupported crop: {crop_name}. Available: {list(CROP_SPECIFIC_CONFIGS.keys())}")
        
        self.crop_name = crop_name
        self.config = CROP_SPECIFIC_CONFIGS[crop_name]
        self.history_len = history_len
        
        # 1. 动态注入品种代码（可选，若依然想使用真实作物名）
        # 修正：由于 cultivars_fileX 是实例属性，我们采用“狸猫换太子”策略：
        # 统一使用 'maize' 绕过 DssatPdi 的初始化检查，但通过注入我们自己的模板来实现真实逻辑。
        
        # 2. 准备初始化参数
        final_env_kwargs = {
            "run_dssat_location": "/opt/dssat_env/inst/run_dssat",
            "cultivar": self.crop_name,
            "fileX_template_path": self.config["default_template"],
            "pdi_template_path": f"templates/{self.crop_name}/dssat_pdi.jinja2",
            "seed": seed
        }
        
        # 自动补全资源文件路径
        aux_files = env_kwargs.get("auxiliary_file_paths", [])
        if self.crop_name == "tomato":
            aux_files.extend(["/opt/dssat_env/data/UFGA.CLI", "/opt/dssat_env/data/SOIL.SOL", "/opt/dssat_env/data/UFGA0601.WTH"])
        elif self.crop_name == "maize":
            aux_files.extend(["/opt/dssat_env/data/UFGA.CLI", "/opt/dssat_env/data/SOIL.SOL", "/opt/dssat_env/data/UFGA8201.WTH"])
        
        final_env_kwargs.update(env_kwargs)
        final_env_kwargs["auxiliary_file_paths"] = list(set(aux_files))
        
        # 3. 在调用 super().__init__ 之前初始化缓冲区，防止 reset() 死锁
        self.obs_keys = self.config["obs_keys"]
        self._history_buffer = []
        
        # 4. 实例化底层环境 (直接导入类以规避 gym.make 的循环引用问题)
        from gym_dssat_pdi.envs.dssat_pdi import DssatPdi
        env = DssatPdi(**final_env_kwargs)
        super().__init__(env)
        
        # 5. 重新定义观测空间（包含历史步长）
        num_features = len(self.obs_keys) * self.history_len
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )

    def _extract_features(self, raw_obs: Dict[str, Any]) -> np.ndarray:
        """从 DSSAT 原始字典中提取并归一化特征"""
        vals = []
        for key in self.obs_keys:
            val = raw_obs.get(key, 0.0)
            # 基础归一化逻辑：对于不同作物，此处可进一步细化
            # 例如：sdwt (产量) 在番茄和玉米中的量级可能不同
            if val is None: val = 0.0
            vals.append(float(val))
        return np.array(vals, dtype=np.float32)

    def _update_history(self, feat: np.ndarray) -> np.ndarray:
        """维护滑动窗口特征"""
        if len(self._history_buffer) == 0:
            self._history_buffer = [feat] * self.history_len
        else:
            self._history_buffer.append(feat)
            self._history_buffer.pop(0)
        return np.concatenate(self._history_buffer)

    def reset(self, **kwargs):
        """重置环境并返回初始特征"""
        obs, info = self.env.reset(**kwargs)
        self._history_buffer = [] # 清空缓冲区
        feat = self._extract_features(obs)
        stacked_obs = self._update_history(feat)
        
        # Tianshou 2.0 安全检查
        if info is None: info = {}
        return stacked_obs, info

    def step(self, action):
        """执行动作并返回 5 元组"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 特征处理
        feat = self._extract_features(obs)
        stacked_obs = self._update_history(feat)
        
        # 奖励函数：针对作物特性可以微调
        # 这里使用基础的产量奖励逻辑
        # 番茄和玉米在 'maize' 配置下均映射为 grnwt
        current_yield = obs.get("grnwt", 0.0)
        if current_yield is None: current_yield = 0.0
        
        # Tianshou 2.0 安全补丁
        if info is None: info = {}
        
        # 奖励处理：DssatPdi 在 mode='all' 时可能返回列表 [ferti_reward, irrig_reward]
        if isinstance(reward, list):
            final_reward = float(np.sum(reward))
        elif reward is not None:
            final_reward = float(reward)
        else:
            final_reward = 0.0
            
        return stacked_obs, final_reward, terminated, truncated, info

    def close(self):
        return self.env.close()
