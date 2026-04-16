# scheduling_fast_optimal.py
import os
import time
import warnings
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

warnings.simplefilter(action='ignore', category=FutureWarning)

ENGINEER_PATH = 'Engineer_List.csv'
DEMAND_PATH = 'Shift_Demand.csv'
OUTPUT_PATH = 'Scheduling_Output_FastOptimal.csv'
HINT_PATH = 'Scheduling_Output.csv'  # 若存在可當 warm start

NUM_SEARCH_WORKERS = 8
MAX_TIME_SECONDS = 300.0
RANDOM_SEED = 42
USE_HINT = True
USE_SYMMETRY_BREAKING = True
LOG_SEARCH_PROGRESS = True
PRINT_INTERMEDIATE_SOLUTIONS = True


class ObjectiveMonitor(cp_model.CpSolverSolutionCallback):
    def __init__(self, start_time: float):
        super().__init__()
        self.start_time = start_time
        self.solution_count = 0
        self.best_obj = None

    def on_solution_callback(self):
        self.solution_count += 1
        obj = self.ObjectiveValue()
        bound = self.BestObjectiveBound()
        elapsed = time.time() - self.start_time
        if self.best_obj is None or obj < self.best_obj:
            self.best_obj = obj
            print(
                f"🌟 new best #{self.solution_count:03d} | elapsed={elapsed:8.2f}s | "
                f"obj={obj/10.0:.1f} | best_bound={bound/10.0:.1f}"
            )


def get_default_group(raw_value) -> str:
    s = str(raw_value).strip()
    return s[0] if s else ''


def get_day_columns(engineer_df: pd.DataFrame):
    return [c for c in engineer_df.columns if str(c).startswith('Date_')]


def build_model(engineer_df: pd.DataFrame, demand_df: pd.DataFrame):
    day_cols = get_day_columns(engineer_df)
    num_engineers = len(engineer_df)
    num_days = len(day_cols)

    shifts = ['O', 'D', 'E', 'N']
    shift_to_idx = {s: i for i, s in enumerate(shifts)}

    model = cp_model.CpModel()

    # x[e, d, s] = 1 代表員工 e 在第 d 天排班 s
    x = {}
    for e in range(num_engineers):
        for d in range(num_days):
            for s in range(4):
                x[e, d, s] = model.NewBoolVar(f'x_{e}_{d}_{s}')

    # -----------------------------
    # 硬限制
    # -----------------------------
    for e in range(num_engineers):
        for d in range(num_days):
            model.AddExactlyOne(x[e, d, s] for s in range(4))

    for d in range(num_days):
        req_D = int(demand_df.loc[d, 'Day'])
        req_E = int(demand_df.loc[d, 'Afternoon'])
        req_N = int(demand_df.loc[d, 'Night'])
        model.Add(sum(x[e, d, 1] for e in range(num_engineers)) == req_D)
        model.Add(sum(x[e, d, 2] for e in range(num_engineers)) == req_E)
        model.Add(sum(x[e, d, 3] for e in range(num_engineers)) == req_N)

    for e in range(num_engineers):
        for d, col in enumerate(day_cols):
            val = engineer_df.iloc[e][col]
            if pd.notna(val):
                v = str(val).strip()
                if v in shift_to_idx:
                    model.Add(x[e, d, shift_to_idx[v]] == 1)

    # -----------------------------
    # 軟限制
    # 權重縮放: 1.0 -> 10, 0.2 -> 2, 0.1 -> 1
    # -----------------------------
    penalties_10 = []
    penalties_2 = []
    penalties_1 = []

    def off(e, d):
        return x[e, d, 0]

    def work_expr(e, d):
        # 由於每人每天 ExactlyOne，因此 work = 1 - off
        return 1 - x[e, d, 0]

    # 規則 1: 連續上班 6 天以上
    # 少建 is_work 變數，直接用 1 - off 的線性表達式
    for e in range(num_engineers):
        for d in range(num_days - 5):
            p = model.NewBoolVar(f'p_consec6_{e}_{d}')
            model.Add(p >= sum(work_expr(e, d + k) for k in range(6)) - 5)
            penalties_10.append(p)

    # 規則 2: 禁忌接班
    forbidden_pairs = [(3, 1), (3, 2), (2, 1), (1, 3), (2, 3)]
    for e in range(num_engineers):
        for d in range(num_days - 1):
            for s1, s2 in forbidden_pairs:
                p = model.NewBoolVar(f'p_transition_{e}_{d}_{s1}_{s2}')
                model.Add(p >= x[e, d, s1] + x[e, d + 1, s2] - 1)
                penalties_10.append(p)

    # 規則 3: 違反預設群組，排 O 不罰
    for e in range(num_engineers):
        default_grp = get_default_group(engineer_df.iloc[e, 1])
        if default_grp in ['D', 'E', 'N']:
            allowed = shift_to_idx[default_grp]
            for d in range(num_days):
                for s in [1, 2, 3]:
                    if s != allowed:
                        penalties_2.append(x[e, d, s])

    weekend_days = demand_df[demand_df['IfWeekend'].astype(str).str.strip() == 'Y'].index.tolist()

    # 規則 4: 每月休假天數 < 9
    for e in range(num_engineers):
        total_off = sum(off(e, d) for d in range(num_days))
        short_off = model.NewIntVar(0, num_days, f'short_off_{e}')
        model.Add(short_off >= 9 - total_off)
        penalties_1.append(short_off)

    # 規則 5: 每月週末休假天數 < 4
    for e in range(num_engineers):
        total_weekend_off = sum(off(e, d) for d in weekend_days)
        short_weekend_off = model.NewIntVar(0, len(weekend_days), f'short_weekend_off_{e}')
        model.Add(short_weekend_off >= 4 - total_weekend_off)
        penalties_1.append(short_weekend_off)

    # 規則 6: 單獨休假 1 日
    for e in range(num_engineers):
        for d in range(1, num_days - 1):
            p = model.NewBoolVar(f'p_single_off_{e}_{d}')
            model.Add(p >= work_expr(e, d - 1) + off(e, d) + work_expr(e, d + 1) - 2)
            penalties_1.append(p)

    # 規則 7: 連休區段 < 2 次
    # block_start 代表從 d 開始出現一段長度 >= 2 的連休，且 d-1 不是休假
    for e in range(num_engineers):
        block_starts = []
        for d in range(num_days - 1):
            bs = model.NewBoolVar(f'block_start_{e}_{d}')
            if d == 0:
                model.Add(bs <= off(e, 0))
                model.Add(bs <= off(e, 1))
                model.Add(bs >= off(e, 0) + off(e, 1) - 1)
            else:
                model.Add(bs <= work_expr(e, d - 1))
                model.Add(bs <= off(e, d))
                model.Add(bs <= off(e, d + 1))
                model.Add(bs >= work_expr(e, d - 1) + off(e, d) + off(e, d + 1) - 2)
            block_starts.append(bs)

        lack_blocks = model.NewBoolVar(f'lack_blocks_{e}')
        model.Add(sum(block_starts) + 2 * lack_blocks >= 2)
        penalties_1.append(lack_blocks)

    # -----------------------------
    # Symmetry breaking
    # 僅對 signature 完全相同的員工做休假數排序，減少對稱解
    # -----------------------------
    if USE_SYMMETRY_BREAKING:
        signature_to_engineers = defaultdict(list)
        for e in range(num_engineers):
            default_grp = str(engineer_df.iloc[e, 1]).strip()
            preassign = tuple(
                '' if pd.isna(engineer_df.iloc[e][col]) else str(engineer_df.iloc[e][col]).strip()
                for col in day_cols
            )
            signature = (default_grp, preassign)
            signature_to_engineers[signature].append(e)

        off_count = {}
        for e in range(num_engineers):
            oc = model.NewIntVar(0, num_days, f'off_count_{e}')
            model.Add(oc == sum(off(e, d) for d in range(num_days)))
            off_count[e] = oc

        for _, engs in signature_to_engineers.items():
            if len(engs) >= 2:
                engs = sorted(engs)
                for i in range(len(engs) - 1):
                    model.Add(off_count[engs[i]] >= off_count[engs[i + 1]])

    obj = sum(penalties_10) * 10 + sum(penalties_2) * 2 + sum(penalties_1)
    model.Minimize(obj)

    metadata = {
        'num_engineers': num_engineers,
        'num_days': num_days,
        'day_cols': day_cols,
        'shifts': shifts,
        'shift_to_idx': shift_to_idx,
        'x': x,
        'obj': obj,
    }
    return model, metadata


def add_hint_if_available(model: cp_model.CpModel, metadata: dict):
    if not USE_HINT:
        print('ℹ️ 已停用 hint')
        return False

    if not os.path.exists(HINT_PATH):
        print(f'ℹ️ 找不到 hint 檔，略過: {HINT_PATH}')
        return False

    hint_df = pd.read_csv(HINT_PATH)
    num_engineers = metadata['num_engineers']
    day_cols = metadata['day_cols']
    shift_to_idx = metadata['shift_to_idx']
    x = metadata['x']

    if len(hint_df) != num_engineers:
        print(f'ℹ️ hint 檔筆數不符，略過 hint: expected {num_engineers}, got {len(hint_df)}')
        return False

    loaded = 0
    for e in range(num_engineers):
        for d, col in enumerate(day_cols):
            val = hint_df.iloc[e][col]
            if pd.notna(val):
                v = str(val).strip()
                if v in shift_to_idx:
                    for s in range(4):
                        model.AddHint(x[e, d, s], 1 if s == shift_to_idx[v] else 0)
                    loaded += 1

    print(f'✅ 已載入 {HINT_PATH} 作為 hint，共 {loaded} 個日期位置')
    return True


def build_output_dataframe(engineer_df: pd.DataFrame, metadata: dict, solver: cp_model.CpSolver):
    out_df = engineer_df.copy()
    x = metadata['x']
    shifts = metadata['shifts']
    day_cols = metadata['day_cols']
    num_engineers = metadata['num_engineers']
    num_days = metadata['num_days']

    for e in range(num_engineers):
        for d, col in enumerate(day_cols):
            for s in range(4):
                if solver.Value(x[e, d, s]) == 1:
                    out_df.loc[e, col] = shifts[s]
                    break
    return out_df


def evaluate_solution(engineer_df: pd.DataFrame, demand_df: pd.DataFrame, out_df: pd.DataFrame):
    day_cols = get_day_columns(out_df)
    weekend_days = set(demand_df[demand_df['IfWeekend'].astype(str).str.strip() == 'Y'].index.tolist())

    weights = {
        'rule1_consecutive_work': 1.0,
        'rule2_forbidden_transition': 1.0,
        'rule3_group_mismatch': 0.2,
        'rule4_monthly_off_shortage': 0.1,
        'rule5_weekend_off_shortage': 0.1,
        'rule6_single_off': 0.1,
        'rule7_not_enough_off_blocks': 0.1,
    }

    result = {k: 0 for k in weights}

    for e in range(len(out_df)):
        seq = [str(out_df.loc[e, col]).strip() for col in day_cols]
        default_grp = get_default_group(engineer_df.iloc[e, 1])

        run = 0
        for v in seq:
            if v in ['D', 'E', 'N']:
                run += 1
            else:
                if run >= 6:
                    result['rule1_consecutive_work'] += run - 5
                run = 0
        if run >= 6:
            result['rule1_consecutive_work'] += run - 5

        forbidden = {('N', 'D'), ('N', 'E'), ('E', 'D'), ('D', 'N'), ('E', 'N')}
        for d in range(len(seq) - 1):
            if (seq[d], seq[d + 1]) in forbidden:
                result['rule2_forbidden_transition'] += 1

        if default_grp in ['D', 'E', 'N']:
            for v in seq:
                if v in ['D', 'E', 'N'] and v != default_grp:
                    result['rule3_group_mismatch'] += 1

        total_off = sum(1 for v in seq if v == 'O')
        result['rule4_monthly_off_shortage'] += max(0, 9 - total_off)

        weekend_off = sum(1 for d, v in enumerate(seq) if d in weekend_days and v == 'O')
        result['rule5_weekend_off_shortage'] += max(0, 4 - weekend_off)

        for d in range(1, len(seq) - 1):
            if seq[d - 1] != 'O' and seq[d] == 'O' and seq[d + 1] != 'O':
                result['rule6_single_off'] += 1

        off_blocks = 0
        d = 0
        while d < len(seq) - 1:
            if seq[d] == 'O' and seq[d + 1] == 'O' and (d == 0 or seq[d - 1] != 'O'):
                off_blocks += 1
                d += 2
                while d < len(seq) and seq[d] == 'O':
                    d += 1
            else:
                d += 1
        if off_blocks < 2:
            result['rule7_not_enough_off_blocks'] += 1

    weighted_total = sum(result[k] * weights[k] for k in weights)
    return result, weighted_total


def main():
    start_time = time.time()

    engineer_df = pd.read_csv(ENGINEER_PATH)
    demand_df = pd.read_csv(DEMAND_PATH)

    model, metadata = build_model(engineer_df, demand_df)
    hint_loaded = add_hint_if_available(model, metadata)

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = NUM_SEARCH_WORKERS
    solver.parameters.max_time_in_seconds = MAX_TIME_SECONDS
    solver.parameters.random_seed = RANDOM_SEED
    solver.parameters.log_search_progress = LOG_SEARCH_PROGRESS
    solver.parameters.enumerate_all_solutions = False

    print('-' * 70)
    print('開始求未知最佳值')
    print(f'Engineers           : {metadata["num_engineers"]}')
    print(f'Days                : {metadata["num_days"]}')
    print(f'num_search_workers  : {NUM_SEARCH_WORKERS}')
    print(f'max_time_seconds    : {MAX_TIME_SECONDS}')
    print(f'random_seed         : {RANDOM_SEED}')
    print(f'use_hint            : {hint_loaded}')
    print(f'use_symmetry_break  : {USE_SYMMETRY_BREAKING}')
    print('-' * 70)

    callback = ObjectiveMonitor(start_time) if PRINT_INTERMEDIATE_SOLUTIONS else None
    status = solver.Solve(model, callback) if callback else solver.Solve(model)

    elapsed = time.time() - start_time
    print('-' * 70)
    print(f'求解完成，狀態: {solver.StatusName(status)}')
    print(f'總耗時: {elapsed:.2f} 秒')

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out_df = build_output_dataframe(engineer_df, metadata, solver)
        out_df.to_csv(OUTPUT_PATH, index=False)

        breakdown, weighted_total = evaluate_solution(engineer_df, demand_df, out_df)

        print(f'最佳懲罰值 Objective: {solver.ObjectiveValue()/10.0:.1f}')
        print(f'Best bound         : {solver.BestObjectiveBound()/10.0:.1f}')
        print(f'重新計算總懲罰值      : {weighted_total:.1f}')
        print(f'輸出檔案            : {OUTPUT_PATH}')
        print('-' * 70)
        print('各規則懲罰明細')
        print(f"rule1 連續上班過多      : {breakdown['rule1_consecutive_work']} 次  -> {breakdown['rule1_consecutive_work'] * 1.0:.1f}")
        print(f"rule2 禁忌接班          : {breakdown['rule2_forbidden_transition']} 次  -> {breakdown['rule2_forbidden_transition'] * 1.0:.1f}")
        print(f"rule3 違反預設群組      : {breakdown['rule3_group_mismatch']} 次  -> {breakdown['rule3_group_mismatch'] * 0.2:.1f}")
        print(f"rule4 月休不足          : {breakdown['rule4_monthly_off_shortage']} 次  -> {breakdown['rule4_monthly_off_shortage'] * 0.1:.1f}")
        print(f"rule5 週末休不足        : {breakdown['rule5_weekend_off_shortage']} 次  -> {breakdown['rule5_weekend_off_shortage'] * 0.1:.1f}")
        print(f"rule6 單獨休假          : {breakdown['rule6_single_off']} 次  -> {breakdown['rule6_single_off'] * 0.1:.1f}")
        print(f"rule7 連休區段不足      : {breakdown['rule7_not_enough_off_blocks']} 次  -> {breakdown['rule7_not_enough_off_blocks'] * 0.1:.1f}")
        print('-' * 70)
    else:
        print('❌ 找不到可行解，請檢查預排與需求是否衝突。')
        print('-' * 70)


if __name__ == '__main__':
    main()


# scheduling_enumerate_best.py
import os
import time
import warnings

import pandas as pd
from ortools.sat.python import cp_model

warnings.simplefilter(action='ignore', category=FutureWarning)

ENGINEER_PATH = 'Engineer_List.csv'
DEMAND_PATH = 'Shift_Demand.csv'
BEST_SCHEDULE_PATH = 'Scheduling_Output_FastOptimal.csv'
MAX_OPTIONS = 3
NUM_SEARCH_WORKERS = 8
MAX_TIME_SECONDS = 180.0
RANDOM_SEED = 42
LOG_SEARCH_PROGRESS = False


def get_default_group(raw_value) -> str:
    s = str(raw_value).strip()
    return s[0] if s else ''


def get_day_columns(engineer_df: pd.DataFrame):
    return [c for c in engineer_df.columns if str(c).startswith('Date_')]


def build_model(engineer_df: pd.DataFrame, demand_df: pd.DataFrame):
    day_cols = get_day_columns(engineer_df)
    num_engineers = len(engineer_df)
    num_days = len(day_cols)

    shifts = ['O', 'D', 'E', 'N']
    shift_to_idx = {s: i for i, s in enumerate(shifts)}

    model = cp_model.CpModel()
    x = {}
    for e in range(num_engineers):
        for d in range(num_days):
            for s in range(4):
                x[e, d, s] = model.NewBoolVar(f'x_{e}_{d}_{s}')

    def off(e, d):
        return x[e, d, 0]

    def work_expr(e, d):
        return 1 - x[e, d, 0]

    for e in range(num_engineers):
        for d in range(num_days):
            model.AddExactlyOne(x[e, d, s] for s in range(4))

    for d in range(num_days):
        req_D = int(demand_df.loc[d, 'Day'])
        req_E = int(demand_df.loc[d, 'Afternoon'])
        req_N = int(demand_df.loc[d, 'Night'])
        model.Add(sum(x[e, d, 1] for e in range(num_engineers)) == req_D)
        model.Add(sum(x[e, d, 2] for e in range(num_engineers)) == req_E)
        model.Add(sum(x[e, d, 3] for e in range(num_engineers)) == req_N)

    for e in range(num_engineers):
        for d, col in enumerate(day_cols):
            val = engineer_df.iloc[e][col]
            if pd.notna(val):
                v = str(val).strip()
                if v in shift_to_idx:
                    model.Add(x[e, d, shift_to_idx[v]] == 1)

    penalties_10 = []
    penalties_2 = []
    penalties_1 = []

    for e in range(num_engineers):
        for d in range(num_days - 5):
            p = model.NewBoolVar(f'p_consec6_{e}_{d}')
            model.Add(p >= sum(work_expr(e, d + k) for k in range(6)) - 5)
            penalties_10.append(p)

    forbidden_pairs = [(3, 1), (3, 2), (2, 1), (1, 3), (2, 3)]
    for e in range(num_engineers):
        for d in range(num_days - 1):
            for s1, s2 in forbidden_pairs:
                p = model.NewBoolVar(f'p_transition_{e}_{d}_{s1}_{s2}')
                model.Add(p >= x[e, d, s1] + x[e, d + 1, s2] - 1)
                penalties_10.append(p)

    for e in range(num_engineers):
        default_grp = get_default_group(engineer_df.iloc[e, 1])
        if default_grp in ['D', 'E', 'N']:
            allowed = shift_to_idx[default_grp]
            for d in range(num_days):
                for s in [1, 2, 3]:
                    if s != allowed:
                        penalties_2.append(x[e, d, s])

    weekend_days = demand_df[demand_df['IfWeekend'].astype(str).str.strip() == 'Y'].index.tolist()

    for e in range(num_engineers):
        total_off = sum(off(e, d) for d in range(num_days))
        short_off = model.NewIntVar(0, num_days, f'short_off_{e}')
        model.Add(short_off >= 9 - total_off)
        penalties_1.append(short_off)

        total_weekend_off = sum(off(e, d) for d in weekend_days)
        short_weekend_off = model.NewIntVar(0, len(weekend_days), f'short_weekend_off_{e}')
        model.Add(short_weekend_off >= 4 - total_weekend_off)
        penalties_1.append(short_weekend_off)

    for e in range(num_engineers):
        for d in range(1, num_days - 1):
            p = model.NewBoolVar(f'p_single_off_{e}_{d}')
            model.Add(p >= work_expr(e, d - 1) + off(e, d) + work_expr(e, d + 1) - 2)
            penalties_1.append(p)

    for e in range(num_engineers):
        block_starts = []
        for d in range(num_days - 1):
            bs = model.NewBoolVar(f'block_start_{e}_{d}')
            if d == 0:
                model.Add(bs <= off(e, 0))
                model.Add(bs <= off(e, 1))
                model.Add(bs >= off(e, 0) + off(e, 1) - 1)
            else:
                model.Add(bs <= work_expr(e, d - 1))
                model.Add(bs <= off(e, d))
                model.Add(bs <= off(e, d + 1))
                model.Add(bs >= work_expr(e, d - 1) + off(e, d) + off(e, d + 1) - 2)
            block_starts.append(bs)

        lack_blocks = model.NewBoolVar(f'lack_blocks_{e}')
        model.Add(sum(block_starts) + 2 * lack_blocks >= 2)
        penalties_1.append(lack_blocks)

    obj = sum(penalties_10) * 10 + sum(penalties_2) * 2 + sum(penalties_1)
    metadata = {
        'num_engineers': num_engineers,
        'num_days': num_days,
        'day_cols': day_cols,
        'shifts': shifts,
        'shift_to_idx': shift_to_idx,
        'x': x,
        'obj': obj,
    }
    return model, metadata


def compute_objective_from_csv(engineer_df, demand_df, out_df):
    day_cols = get_day_columns(out_df)
    weekend_days = set(demand_df[demand_df['IfWeekend'].astype(str).str.strip() == 'Y'].index.tolist())
    total = 0

    for e in range(len(out_df)):
        seq = [str(out_df.loc[e, col]).strip() for col in day_cols]
        default_grp = get_default_group(engineer_df.iloc[e, 1])

        run = 0
        for v in seq:
            if v in ['D', 'E', 'N']:
                run += 1
            else:
                if run >= 6:
                    total += (run - 5) * 10
                run = 0
        if run >= 6:
            total += (run - 5) * 10

        forbidden = {('N', 'D'), ('N', 'E'), ('E', 'D'), ('D', 'N'), ('E', 'N')}
        for d in range(len(seq) - 1):
            if (seq[d], seq[d + 1]) in forbidden:
                total += 10

        if default_grp in ['D', 'E', 'N']:
            for v in seq:
                if v in ['D', 'E', 'N'] and v != default_grp:
                    total += 2

        total_off = sum(1 for v in seq if v == 'O')
        total += max(0, 9 - total_off)

        weekend_off = sum(1 for d, v in enumerate(seq) if d in weekend_days and v == 'O')
        total += max(0, 4 - weekend_off)

        for d in range(1, len(seq) - 1):
            if seq[d - 1] != 'O' and seq[d] == 'O' and seq[d + 1] != 'O':
                total += 1

        off_blocks = 0
        d = 0
        while d < len(seq) - 1:
            if seq[d] == 'O' and seq[d + 1] == 'O' and (d == 0 or seq[d - 1] != 'O'):
                off_blocks += 1
                d += 2
                while d < len(seq) and seq[d] == 'O':
                    d += 1
            else:
                d += 1
        if off_blocks < 2:
            total += 1

    return total


class BestSolutionEnumerator(cp_model.CpSolverSolutionCallback):
    def __init__(self, x, num_engineers, num_days, shifts, engineer_df, day_cols, limit, start_time):
        super().__init__()
        self.x = x
        self.num_engineers = num_engineers
        self.num_days = num_days
        self.shifts = shifts
        self.engineer_df = engineer_df
        self.day_cols = day_cols
        self.limit = limit
        self.start_time = start_time
        self.count = 0

    def on_solution_callback(self):
        self.count += 1
        elapsed = time.time() - self.start_time
        out_df = self.engineer_df.copy()
        for e in range(self.num_engineers):
            for d, col in enumerate(self.day_cols):
                for s in range(4):
                    if self.Value(self.x[e, d, s]) == 1:
                        out_df.loc[e, col] = self.shifts[s]
                        break

        out_path = f'Scheduling_Output_Option_{self.count}.csv'
        out_df.to_csv(out_path, index=False)
        print(f'🌟 找到 Option_{self.count} | elapsed={elapsed:.2f}s | saved={out_path}')

        if self.count >= self.limit:
            self.StopSearch()


def main():
    if not os.path.exists(BEST_SCHEDULE_PATH):
        raise FileNotFoundError(f'找不到最佳班表檔案: {BEST_SCHEDULE_PATH}')

    engineer_df = pd.read_csv(ENGINEER_PATH)
    demand_df = pd.read_csv(DEMAND_PATH)
    best_df = pd.read_csv(BEST_SCHEDULE_PATH)

    target_obj = compute_objective_from_csv(engineer_df, demand_df, best_df)
    print(f'鎖定最佳 objective = {target_obj} -> penalty = {target_obj / 10.0:.1f}')

    model, metadata = build_model(engineer_df, demand_df)
    model.Add(metadata['obj'] == target_obj)

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1  # enumerate_all_solutions 時建議單執行緒
    solver.parameters.max_time_in_seconds = MAX_TIME_SECONDS
    solver.parameters.random_seed = RANDOM_SEED
    solver.parameters.log_search_progress = LOG_SEARCH_PROGRESS

    start_time = time.time()
    callback = BestSolutionEnumerator(
        metadata['x'],
        metadata['num_engineers'],
        metadata['num_days'],
        metadata['shifts'],
        engineer_df,
        metadata['day_cols'],
        MAX_OPTIONS,
        start_time,
    )

    print('-' * 70)
    print('開始列舉同分最佳解')
    print(f'max_options       : {MAX_OPTIONS}')
    print(f'max_time_seconds  : {MAX_TIME_SECONDS}')
    print('-' * 70)

    status = solver.Solve(model, callback)
    elapsed = time.time() - start_time
    print('-' * 70)
    print(f'列舉完成，狀態: {solver.StatusName(status)}')
    print(f'總耗時: {elapsed:.2f} 秒')
    print(f'找到解數: {callback.count}')
    print('-' * 70)


if __name__ == '__main__':
    main()
