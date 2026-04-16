import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import gurobipy as gp
from gurobipy import GRB

# ==========================================
# 1. 載入與訓練時一模一樣的環境
# ==========================================
class AdvancedFairTSMCEnv(gym.Env):
    def __init__(self):
        super(AdvancedFairTSMCEnv, self).__init__()
        self.num_engineers = 15
        self.num_days = 30
        self.shifts = ['D', 'E', 'N', 'O']
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-30, high=40, shape=(13,), dtype=np.int32)
        self.current_day = 0
        self.step_in_day = 0 
        self.schedule = np.full((self.num_engineers, self.num_days), 3)
        self.consecutive_work_days = np.zeros(self.num_engineers, dtype=np.int32)
        self.total_work_days = np.zeros(self.num_engineers, dtype=np.int32)
        self.total_off_days = np.zeros(self.num_engineers, dtype=np.int32)
        self.weekend_off_days = np.zeros(self.num_engineers, dtype=np.int32)
        self.default_groups = np.zeros(self.num_engineers, dtype=np.int32)
        self.pref_matrix = np.full((self.num_engineers, self.num_days), -1)
        self.daily_demand_list = []
        self.is_weekend_array = np.zeros(self.num_days, dtype=bool)

        try:
            df_eng = pd.read_csv('Engineer_List.csv')
            df_demand = pd.read_csv('Shift_Demand.csv')
            days = df_demand['Date'].tolist()
            shift_map = {'D': 0, 'E': 1, 'N': 2, 'O': 3}
            group_col_name = df_eng.columns[1]
            for idx, row in df_eng.iterrows():
                if idx >= self.num_engineers: break
                g_str = str(row[group_col_name]).strip()[0].upper()
                self.default_groups[idx] = shift_map.get(g_str, 0)
                for d_idx, day_col in enumerate(days):
                    if d_idx >= self.num_days: break
                    val = row.get(day_col)
                    if pd.notna(val) and str(val).strip().upper() in shift_map:
                        self.pref_matrix[idx, d_idx] = shift_map[str(val).strip().upper()] 

            for d_idx, row in df_demand.iterrows():
                if d_idx >= self.num_days: break
                self.daily_demand_list.append({'D': int(row['Day']), 'E': int(row['Afternoon']), 'N': int(row['Night'])})
                self.is_weekend_array[d_idx] = (str(row['IfWeekend']).strip().upper() == 'Y')
                
            self.current_demand = self.daily_demand_list[0].copy()
        except Exception as e:
            print(f"⚠️ 環境初始化讀檔錯誤：{e}")

    def _set_daily_order(self):
        if self.current_day >= self.num_days: return
        pre_assigned = [i for i in range(self.num_engineers) if self.pref_matrix[i, self.current_day] != -1]
        free = [i for i in range(self.num_engineers) if self.pref_matrix[i, self.current_day] == -1]
        np.random.shuffle(free)
        self.daily_eng_order = np.array(pre_assigned + free)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_day = 0
        self.step_in_day = 0
        self._set_daily_order()
        self.schedule.fill(3)
        self.consecutive_work_days.fill(0)
        self.total_work_days.fill(0)
        self.total_off_days.fill(0)
        self.weekend_off_days.fill(0)
        self.current_demand = self.daily_demand_list[self.current_day].copy()
        return self._get_obs(), {}

    def _get_obs(self):
        safe_day = min(self.current_day, self.num_days - 1)
        current_eng = self.daily_eng_order[self.step_in_day]
        yesterday_shift = self.schedule[current_eng, safe_day - 1] if safe_day > 0 else 3
        before_yesterday_shift = self.schedule[current_eng, safe_day - 2] if safe_day > 1 else 3
        remaining_off_days = 9 - self.total_off_days[current_eng]
        remaining_weekend_off_days = 4 - self.weekend_off_days[current_eng]
        today_is_weekend = int(self.is_weekend_array[safe_day])
        return np.array([
            safe_day, current_eng,
            self.current_demand['D'], self.current_demand['E'], self.current_demand['N'],
            self.consecutive_work_days[current_eng], yesterday_shift, before_yesterday_shift,
            self.default_groups[current_eng], self.total_work_days[current_eng],
            today_is_weekend, remaining_off_days, remaining_weekend_off_days    
        ], dtype=np.int32)

    def valid_action_mask(self):
        current_eng = self.daily_eng_order[self.step_in_day]
        pref_action = self.pref_matrix[current_eng, self.current_day]
        if pref_action != -1:
            valid_actions = np.array([False, False, False, False])
            valid_actions[pref_action] = True  
            return valid_actions

        valid_actions = np.array([True, True, True, True])
        if self.current_demand['D'] <= 0: valid_actions[0] = False
        if self.current_demand['E'] <= 0: valid_actions[1] = False
        if self.current_demand['N'] <= 0: valid_actions[2] = False
        
        remaining_engs = self.num_engineers - self.step_in_day
        remaining_demand = sum(max(0, v) for v in self.current_demand.values())
        if remaining_engs <= remaining_demand:
            valid_actions[3] = False 

        safe_day = min(self.current_day, self.num_days - 1)
        if 0 < safe_day < self.num_days - 1:
            yesterday_shift = self.schedule[current_eng, safe_day - 1]
            before_yesterday_shift = self.schedule[current_eng, safe_day - 2] if safe_day > 1 else 3
            if yesterday_shift == 3 and before_yesterday_shift != 3:
                temp_valid = valid_actions.copy()
                temp_valid[0] = False 
                temp_valid[1] = False 
                temp_valid[2] = False 
                if temp_valid[3] == True: 
                    valid_actions = temp_valid

        if not valid_actions.any():
            if self.current_demand['N'] > 0: valid_actions[2] = True
            elif self.current_demand['E'] > 0: valid_actions[1] = True
            elif self.current_demand['D'] > 0: valid_actions[0] = True
            else: valid_actions[3] = True
        return valid_actions

    def step(self, action):
        current_eng = self.daily_eng_order[self.step_in_day]
        shift_assigned = self.shifts[action]
        self.schedule[current_eng, self.current_day] = action
        is_weekend = self.is_weekend_array[self.current_day]

        if shift_assigned != 'O': 
            self.consecutive_work_days[current_eng] += 1
            self.total_work_days[current_eng] += 1
            self.current_demand[shift_assigned] -= 1
        else: 
            self.consecutive_work_days[current_eng] = 0
            self.total_off_days[current_eng] += 1
            if is_weekend:
                self.weekend_off_days[current_eng] += 1

        self.step_in_day += 1
        if self.step_in_day >= self.num_engineers:
            self.step_in_day = 0
            self.current_day += 1
            self._set_daily_order()
            if self.current_day < self.num_days:
                self.current_demand = self.daily_demand_list[self.current_day].copy()
            
        terminated = self.current_day >= self.num_days
        return self._get_obs(), 0, terminated, False, {}

def mask_fn(env): return env.unwrapped.valid_action_mask()

# ==========================================
# 2. 記憶體內 Gurobi 評分器
# ==========================================
def evaluate_score(df_rl, df_demand, env_gp):
    engineers = df_rl['人員'].tolist()
    default_groups = {row['人員']: row['班別群組'] for _, row in df_rl.iterrows()}
    days = [f"Date_{i+1}" for i in range(30)]
    is_weekend = {f"Date_{i+1}": (str(row['IfWeekend']).strip().upper() == 'Y') for i, row in df_demand.iterrows()}
    weekend_days = [t for t in days if is_weekend[t]]
    shifts = ['D', 'E', 'N', 'O']

    model = gp.Model("Score_Checker", env=env_gp)
    x = model.addVars(engineers, days, shifts, vtype=GRB.BINARY)

    for _, row in df_rl.iterrows():
        eng = row['人員']
        for day in days:
            actual_shift = row[day]
            for s in shifts:
                if s == actual_shift:
                    x[eng, day, s].lb = 1.0; x[eng, day, s].ub = 1.0
                else:
                    x[eng, day, s].lb = 0.0; x[eng, day, s].ub = 0.0

    C6 = model.addVars(engineers, days[:len(days)-5], vtype=GRB.BINARY)
    Q = model.addVars(engineers, days[:len(days)-1], vtype=GRB.BINARY)
    V = model.addVars(engineers, days, vtype=GRB.BINARY)
    ShortO = model.addVars(engineers, vtype=GRB.CONTINUOUS, lb=0)
    ShortW = model.addVars(engineers, vtype=GRB.CONTINUOUS, lb=0)
    Iso = model.addVars(engineers, days[1:len(days)-1], vtype=GRB.BINARY)
    y = model.addVars(engineers, days[:len(days)-1], vtype=GRB.BINARY)
    z = model.addVars(engineers, days[:len(days)-1], vtype=GRB.BINARY)
    LackConsec = model.addVars(engineers, vtype=GRB.CONTINUOUS, lb=0)

    for i in engineers:
        for idx in range(len(days) - 5):
            t_window = days[idx:idx+6]
            model.addConstr(gp.quicksum((1 - x[i, d, 'O']) for d in t_window) - 5 <= C6[i, days[idx]])
        for idx in range(len(days) - 1):
            tc, tn = days[idx], days[idx+1]
            model.addConstr(x[i, tc, 'N'] + x[i, tn, 'D'] - 1 <= Q[i, tc])
            model.addConstr(x[i, tc, 'N'] + x[i, tn, 'E'] - 1 <= Q[i, tc])
            model.addConstr(x[i, tc, 'E'] + x[i, tn, 'D'] - 1 <= Q[i, tc])
            model.addConstr(x[i, tc, 'D'] + x[i, tn, 'N'] - 1 <= Q[i, tc])
            model.addConstr(x[i, tc, 'E'] + x[i, tn, 'N'] - 1 <= Q[i, tc])
        def_s = default_groups[i]
        for t in days:
            model.addConstr(gp.quicksum(x[i, t, s] for s in ['D', 'E', 'N'] if s != def_s) <= V[i, t])

        model.addConstr(gp.quicksum(x[i, t, 'O'] for t in days) + ShortO[i] >= 9)
        model.addConstr(gp.quicksum(x[i, t, 'O'] for t in weekend_days) + ShortW[i] >= 4)
        for idx in range(1, len(days) - 1):
            tp, tc, tn = days[idx-1], days[idx], days[idx+1]
            model.addConstr((1 - x[i, tp, 'O']) + x[i, tc, 'O'] + (1 - x[i, tn, 'O']) - 2 <= Iso[i, tc])
        for idx in range(len(days) - 1):
            tc, tn = days[idx], days[idx+1]
            model.addConstr(y[i, tc] <= x[i, tc, 'O'])
            model.addConstr(y[i, tc] <= x[i, tn, 'O'])
            if idx == 0: model.addConstr(z[i, tc] >= y[i, tc])
            else: model.addConstr(z[i, tc] >= y[i, tc] - y[i, days[idx-1]])
        model.addConstr(gp.quicksum(z[i, t] for t in days[:len(days)-1]) + 2 * LackConsec[i] >= 2)

    model.setObjective(
        1.0 * C6.sum() + 1.0 * Q.sum() + 0.2 * V.sum() + 
        0.1 * ShortO.sum() + 0.1 * ShortW.sum() + 
        0.1 * Iso.sum() + 0.1 * LackConsec.sum(), GRB.MINIMIZE)

    model.optimize()
    score = model.objVal
    model.dispose()
    return score
