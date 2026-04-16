import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gurobipy as gp
import time
from rl_env import AdvancedFairTSMCEnv, mask_fn, evaluate_score

# ==========================================
# 載入已訓練的大腦
# ==========================================
if __name__ == "__main__":
    NUM_DRAWS = 100
    print("\n" + "="*50)
    print(f"V4 大腦抽卡，目標: {NUM_DRAWS} 連抽)")
    print("="*50)
    
    base_env = AdvancedFairTSMCEnv()
    env = ActionMasker(base_env, mask_fn)
    
    # 載入訓練好的模型
    model_path = "tsmc_advanced_rl_model_v4_ultimate"
    try:
        model = MaskablePPO.load(model_path)
        print(f"成功載入模型：{model_path}.zip")
    except Exception as e:
        print(f"找不到模型檔案，請先執行 rl_train.py 或確認 .zip 檔存在！錯誤：{e}")
        exit()

    df_demand = pd.read_csv('Shift_Demand.csv')
    date_cols = df_demand['Date'].tolist()[:30]
    shift_map = {0: 'D', 1: 'E', 2: 'N', 3: 'O'}
    group_rev_map = {0: 'D', 1: 'E', 2: 'N'}

    env_gp = gp.Env(empty=True)
    env_gp.setParam('OutputFlag', 0)
    env_gp.start()

    best_score = float('inf')
    best_df = None
    start_time = time.time()

    for draw in range(NUM_DRAWS):
        obs, _ = env.reset()
        result_data = []
        
        while True:
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=np.array([masks]), deterministic=True)
            if not masks[int(action)]: action = np.where(masks)[0][0]
            obs, _, terminated, _, _ = env.step(int(action))
            
            if terminated:
                for i in range(15):
                    row = {'人員': f"engineer_{i+1}", '班別群組': group_rev_map[base_env.default_groups[i]]}
                    for d in range(30): row[date_cols[d]] = shift_map[base_env.schedule[i, d]]
                    result_data.append(row)
                break
                
        df_current = pd.DataFrame(result_data)
        current_score = evaluate_score(df_current, df_demand, env_gp)
        
        if current_score < best_score:
            best_score = current_score
            best_df = df_current.copy()
            
        print(f"抽卡中... 第 {draw+1:03d}/{NUM_DRAWS} 抽 | 本次分數: {current_score:.2f} 分 | 🏆 最強紀錄: {best_score:.2f} 分", end='\r')

    print("\n\n" + "="*50)
    print(f"V4 找到的最低分數是：{best_score:.2f} 分")
    
    output_filename = 'RL_Ultimate_Gacha_Best_V4.csv'
    best_df.to_csv(output_filename, index=False)
    print(f"耗時: {time.time() - start_time:.1f} 秒")
    print(f"📄 V4 班表已匯出至：{output_filename}")
    print("="*50)
    
    env_gp.dispose()
