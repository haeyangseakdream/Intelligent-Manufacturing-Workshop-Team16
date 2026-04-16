import time
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from rl_env import AdvancedFairTSMCEnv, mask_fn

if __name__ == "__main__":
    print("="*50)
    print(" V4 訓練 (Offline Training)")
    print("="*50)

    # 1. 建立環境與物理遮罩
    base_env = AdvancedFairTSMCEnv()
    env = ActionMasker(base_env, mask_fn)

    # 2. 建立 Maskable PPO 模型
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4
    )

    # 3. 開始訓練 
    start_time = time.time()
    TIMESTEPS = 100000  # 可依實際狀況調整
    print(f" 開始訓練 {TIMESTEPS} 步...")
    model.learn(total_timesteps=TIMESTEPS)
    
    # 4. 儲存大腦
    model_name = "tsmc_advanced_rl_model_v4_ultimate"
    model.save(model_name)
    
    print(f"訓練完成！總耗時: {time.time() - start_time:.1f} 秒")
    print(f"模型已儲存為 {model_name}.zip")
