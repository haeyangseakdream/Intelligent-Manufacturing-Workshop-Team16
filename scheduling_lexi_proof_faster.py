import os
import time
import warnings
from collections import defaultdict

import pandas as pd
from ortools.sat.python import cp_model

warnings.simplefilter(action='ignore', category=FutureWarning)

ENGINEER_PATH = 'Engineer_List.csv'
DEMAND_PATH = 'Shift_Demand.csv'
OUTPUT_PATH = 'Scheduling_Output_LexiProofFaster.csv'

# Hint priority: 已知最佳解 > 前次快速最佳解 > 一般可行解
HINT_CANDIDATES = [
    'Scheduling_Output_FastOptimal.csv',
    'Scheduling_Output_Option_1.csv',
    'Scheduling_Output.csv',
]

MAX_TIME_SECONDS_STAGE1 = 90.0
MAX_TIME_SECONDS_STAGE2 = 90.0
MAX_TIME_SECONDS_STAGE3 = 120.0
RANDOM_SEED = 42
NUM_SEARCH_WORKERS = 4  # 建議先試 1 / 4 / 8
USE_HINT = True
USE_SYMMETRY_BREAKING = True
USE_STRONGER_SYMMETRY = True
LOG_SEARCH_PROGRESS = True
PRINT_INTERMEDIATE_SOLUTIONS = True


class ObjectiveMonitor(cp_model.CpSolverSolutionCallback):
    def __init__(self, start_time: float, stage_name: str, scale: float = 1.0):
        super().__init__()
        self.start_time = start_time
        self.stage_name = stage_name
        self.scale = scale
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
                f"🌟 {self.stage_name} new best #{self.solution_count:03d} | "
                f"elapsed={elapsed:8.2f}s | obj={obj / self.scale:.1f} | "
                f"best_bound={bound / self.scale:.1f}"
            )


def get_default_group(raw_value) -> str:
    s = str(raw_value).strip()
    return s[0] if s else ''


def get_day_columns(engineer_df: pd.DataFrame):
    return [c for c in engineer_df.columns if str(c).startswith('Date_')]


def _find_best_hint_path() -> str | None:
    for path in HINT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


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
    # 軟限制細分，方便做 lexicographic 最佳化
    # -----------------------------
    penalties_hi = []   # 權重 10 -> 1.0
    penalties_mid = []  # 權重 2  -> 0.2
    penalties_lo = []   # 權重 1  -> 0.1

    # 規則 1: 連續上班 6 天以上
    for e in range(num_engineers):
        for d in range(num_days - 5):
            p = model.NewBoolVar(f'p_consec6_{e}_{d}')
            work6 = sum(work_expr(e, d + k) for k in range(6))
            model.Add(p >= work6 - 5)
            for k in range(6):
                model.Add(p <= work_expr(e, d + k))
            penalties_hi.append(p)

    # 規則 2: 禁忌接班
    forbidden_pairs = [(3, 1), (3, 2), (2, 1), (1, 3), (2, 3)]
    for e in range(num_engineers):
        for d in range(num_days - 1):
            for s1, s2 in forbidden_pairs:
                p = model.NewBoolVar(f'p_transition_{e}_{d}_{s1}_{s2}')
                model.Add(p >= x[e, d, s1] + x[e, d + 1, s2] - 1)
                model.Add(p <= x[e, d, s1])
                model.Add(p <= x[e, d + 1, s2])
                penalties_hi.append(p)

    # 規則 3: 違反預設群組，排 O 不罰
    for e in range(num_engineers):
        default_grp = get_default_group(engineer_df.iloc[e, 1])
        if default_grp in ['D', 'E', 'N']:
            allowed = shift_to_idx[default_grp]
            for d in range(num_days):
                for s in [1, 2, 3]:
                    if s != allowed:
                        penalties_mid.append(x[e, d, s])

    weekend_days = demand_df[demand_df['IfWeekend'].astype(str).str.strip() == 'Y'].index.tolist()

    # 規則 4: 每月休假天數 < 9
    for e in range(num_engineers):
        total_off = sum(off(e, d) for d in range(num_days))
        short_off = model.NewIntVar(0, num_days, f'short_off_{e}')
        model.Add(short_off >= 9 - total_off)
        penalties_lo.append(short_off)

    # 規則 5: 每月週末休假天數 < 4
    for e in range(num_engineers):
        total_weekend_off = sum(off(e, d) for d in weekend_days)
        short_weekend_off = model.NewIntVar(0, len(weekend_days), f'short_weekend_off_{e}')
        model.Add(short_weekend_off >= 4 - total_weekend_off)
        penalties_lo.append(short_weekend_off)

    # 規則 6: 單獨休假 1 日
    for e in range(num_engineers):
        for d in range(1, num_days - 1):
            p = model.NewBoolVar(f'p_single_off_{e}_{d}')
            prev_work = work_expr(e, d - 1)
            curr_off = off(e, d)
            next_work = work_expr(e, d + 1)
            model.Add(p >= prev_work + curr_off + next_work - 2)
            model.Add(p <= prev_work)
            model.Add(p <= curr_off)
            model.Add(p <= next_work)
            penalties_lo.append(p)

    # 規則 7: 連休區段 < 2 次
    block_starts_by_engineer = {}
    for e in range(num_engineers):
        block_starts = []
        for d in range(num_days - 1):
            bs = model.NewBoolVar(f'block_start_{e}_{d}')
            if d == 0:
                model.Add(bs >= off(e, 0) + off(e, 1) - 1)
                model.Add(bs <= off(e, 0))
                model.Add(bs <= off(e, 1))
            else:
                prev_work = work_expr(e, d - 1)
                model.Add(bs >= prev_work + off(e, d) + off(e, d + 1) - 2)
                model.Add(bs <= prev_work)
                model.Add(bs <= off(e, d))
                model.Add(bs <= off(e, d + 1))
            block_starts.append(bs)

        block_starts_by_engineer[e] = block_starts
        lack_blocks = model.NewBoolVar(f'lack_blocks_{e}')
        model.Add(sum(block_starts) + 2 * lack_blocks >= 2)
        penalties_lo.append(lack_blocks)

    # -----------------------------
    # 強化版 symmetry breaking
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
        weekend_off_count = {}
        first_half_off_count = {}
        for e in range(num_engineers):
            oc = model.NewIntVar(0, num_days, f'off_count_{e}')
            model.Add(oc == sum(off(e, d) for d in range(num_days)))
            off_count[e] = oc

            woc = model.NewIntVar(0, len(weekend_days), f'weekend_off_count_{e}')
            model.Add(woc == sum(off(e, d) for d in weekend_days))
            weekend_off_count[e] = woc

            split = num_days // 2
            foc = model.NewIntVar(0, split, f'first_half_off_count_{e}')
            model.Add(foc == sum(off(e, d) for d in range(split)))
            first_half_off_count[e] = foc

        for _, engs in signature_to_engineers.items():
            if len(engs) >= 2:
                engs = sorted(engs)
                for i in range(len(engs) - 1):
                    a, b = engs[i], engs[i + 1]
                    model.Add(off_count[a] >= off_count[b])
                    if USE_STRONGER_SYMMETRY:
                        model.Add(weekend_off_count[a] >= weekend_off_count[b])
                        model.Add(first_half_off_count[a] >= first_half_off_count[b])

    high_cost = sum(penalties_hi)
    mid_cost = sum(penalties_mid)
    low_cost = sum(penalties_lo)
    total_scaled = 10 * high_cost + 2 * mid_cost + low_cost

    metadata = {
        'num_engineers': num_engineers,
        'num_days': num_days,
        'day_cols': day_cols,
        'shifts': shifts,
        'shift_to_idx': shift_to_idx,
        'x': x,
        'high_cost': high_cost,
        'mid_cost': mid_cost,
        'low_cost': low_cost,
        'total_scaled': total_scaled,
        'forbidden_pairs': forbidden_pairs,
    }
    return model, metadata


def add_hint_if_available(model: cp_model.CpModel, metadata: dict):
    if not USE_HINT:
        print('ℹ️ 已停用 hint')
        return False, None

    hint_path = _find_best_hint_path()
    if hint_path is None:
        print('ℹ️ 找不到任何 hint 檔，略過 warm start')
        return False, None

    hint_df = pd.read_csv(hint_path)
    num_engineers = metadata['num_engineers']
    day_cols = metadata['day_cols']
    shift_to_idx = metadata['shift_to_idx']
    x = metadata['x']

    if len(hint_df) != num_engineers:
        print(f'ℹ️ hint 檔筆數不符，略過 hint: expected {num_engineers}, got {len(hint_df)}')
        return False, None

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

    print(f'✅ 已載入 {hint_path} 作為 hint，共 {loaded} 個日期位置')
    return True, hint_path


def solve_stage(model, objective_expr, stage_name: str, scale: float, max_time_seconds: float):
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = NUM_SEARCH_WORKERS
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.random_seed = RANDOM_SEED
    solver.parameters.log_search_progress = LOG_SEARCH_PROGRESS
    solver.parameters.enumerate_all_solutions = False

    model.Minimize(objective_expr)
    start_time = time.time()
    callback = ObjectiveMonitor(start_time, stage_name, scale) if PRINT_INTERMEDIATE_SOLUTIONS else None
    status = solver.Solve(model, callback) if callback else solver.Solve(model)
    elapsed = time.time() - start_time

    print('-' * 72)
    print(f'{stage_name} 完成 | status={solver.StatusName(status)} | elapsed={elapsed:.2f}s')
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f'{stage_name} objective={solver.ObjectiveValue() / scale:.1f} | bound={solver.BestObjectiveBound() / scale:.1f}')
    print('-' * 72)
    return solver, status, elapsed


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
    overall_start = time.time()

    engineer_df = pd.read_csv(ENGINEER_PATH)
    demand_df = pd.read_csv(DEMAND_PATH)

    model, metadata = build_model(engineer_df, demand_df)
    hint_loaded, hint_path = add_hint_if_available(model, metadata)

    print('=' * 72)
    print('開始三階段 lexicographic 最佳化')
    print(f'Engineers             : {metadata["num_engineers"]}')
    print(f'Days                  : {metadata["num_days"]}')
    print(f'num_search_workers    : {NUM_SEARCH_WORKERS}')
    print(f'use_hint              : {hint_loaded} ({hint_path})')
    print(f'use_symmetry_break    : {USE_SYMMETRY_BREAKING}')
    print(f'use_stronger_symmetry : {USE_STRONGER_SYMMETRY}')
    print('=' * 72)

    # Stage 1: 先壓高權重
    solver1, status1, _ = solve_stage(
        model,
        metadata['high_cost'],
        stage_name='Stage 1 高權重',
        scale=1.0,
        max_time_seconds=MAX_TIME_SECONDS_STAGE1,
    )
    if status1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print('❌ Stage 1 找不到可行解')
        return
    best_high = int(round(solver1.ObjectiveValue()))
    model.Add(metadata['high_cost'] == best_high)

    # Stage 2: 再壓中權重
    solver2, status2, _ = solve_stage(
        model,
        metadata['mid_cost'],
        stage_name='Stage 2 中權重',
        scale=1.0,
        max_time_seconds=MAX_TIME_SECONDS_STAGE2,
    )
    if status2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print('❌ Stage 2 找不到可行解')
        return
    best_mid = int(round(solver2.ObjectiveValue()))
    model.Add(metadata['mid_cost'] == best_mid)

    # Stage 3: 最後壓低權重
    solver3, status3, _ = solve_stage(
        model,
        metadata['low_cost'],
        stage_name='Stage 3 低權重',
        scale=1.0,
        max_time_seconds=MAX_TIME_SECONDS_STAGE3,
    )
    if status3 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print('❌ Stage 3 找不到可行解')
        return

    out_df = build_output_dataframe(engineer_df, metadata, solver3)
    out_df.to_csv(OUTPUT_PATH, index=False)

    breakdown, weighted_total = evaluate_solution(engineer_df, demand_df, out_df)
    overall_elapsed = time.time() - overall_start

    print('=' * 72)
    print(f'三階段完成，最終狀態: {solver3.StatusName(status3)}')
    print(f'總耗時: {overall_elapsed:.2f} 秒')
    print(f'高權重最佳值: {best_high} -> {best_high * 1.0:.1f}')
    print(f'中權重最佳值: {best_mid} -> {best_mid * 0.2:.1f}')
    print(f'低權重最佳值: {solver3.ObjectiveValue()} -> {solver3.ObjectiveValue() * 0.1:.1f}')
    print(f'加權總懲罰值: {weighted_total:.1f}')
    print(f'輸出檔案    : {OUTPUT_PATH}')
    print('-' * 72)
    print('各規則懲罰明細')
    print(f"rule1 連續上班過多      : {breakdown['rule1_consecutive_work']} 次  -> {breakdown['rule1_consecutive_work'] * 1.0:.1f}")
    print(f"rule2 禁忌接班          : {breakdown['rule2_forbidden_transition']} 次  -> {breakdown['rule2_forbidden_transition'] * 1.0:.1f}")
    print(f"rule3 違反預設群組      : {breakdown['rule3_group_mismatch']} 次  -> {breakdown['rule3_group_mismatch'] * 0.2:.1f}")
    print(f"rule4 月休不足          : {breakdown['rule4_monthly_off_shortage']} 次  -> {breakdown['rule4_monthly_off_shortage'] * 0.1:.1f}")
    print(f"rule5 週末休不足        : {breakdown['rule5_weekend_off_shortage']} 次  -> {breakdown['rule5_weekend_off_shortage'] * 0.1:.1f}")
    print(f"rule6 單獨休假          : {breakdown['rule6_single_off']} 次  -> {breakdown['rule6_single_off'] * 0.1:.1f}")
    print(f"rule7 連休區段不足      : {breakdown['rule7_not_enough_off_blocks']} 次  -> {breakdown['rule7_not_enough_off_blocks'] * 0.1:.1f}")
    print('=' * 72)


if __name__ == '__main__':
    main()
