import os
import sys

# 确保能找到项目根目录下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dssat_generic_wrapper import DssatGenericWrapper
import numpy as np

def test_tomato_env():
    print("=== Starting Tomato Environment Smoke Test ===")
    
    # 获取数据目录下的所有文件
    data_dir = "/opt/dssat_env/data/"
    aux_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    
    # 环境参数：使用番茄特有的配置
    env_kwargs = {
        "run_dssat_location": "/opt/dssat_env/inst/run_dssat",
        "mode": "all",
        "log_saving_path": "./logs/tomato_test.log",
        "auxiliary_file_paths": aux_files
    }
    
    print("Pre-initialization check: /opt/dssat_env/inst/run_dssat exists?", os.path.exists("/opt/dssat_env/inst/run_dssat"))
    
    try:
        # 初始化通用包装器
        env = DssatGenericWrapper(
            crop_name="tomato",
            env_kwargs=env_kwargs,
            history_len=5
        )
        
        print(f"Environment initialized successfully for crop: {env.crop_name}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        
        # 重置环境
        obs, info = env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # 运行几个步长验证物理引擎通信
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 检查番茄特有指标
            istage = info.get("istage", "N/A")
            xlai = info.get("xlai", "N/A")
            sdwt = info.get("sdwt", "N/A")
            
            print(f"Step {i+1}: Reward={reward:.4f}, istage={istage}, xlai={xlai}, sdwt={sdwt}")
            
            if terminated or truncated:
                print("Episode finished early.")
                break
        
        env.close()
        print("=== Tomato Environment Smoke Test PASSED ===")
        return True
        
    except Exception as e:
        print(f"=== Tomato Environment Smoke Test FAILED ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tomato_env()
    sys.exit(0 if success else 1)
