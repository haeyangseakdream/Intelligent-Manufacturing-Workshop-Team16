"""
TSMC IM Workshop 2026 - Intelligent Scheduling Optimizer (Pure GA, no ortools)
"""
import pandas as pd
import numpy as np
import random, copy, time, os
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

START_TIME = time.time()

# ── Parameters ──────────────────────────────────────────────
DAYS        = 30
POP_SIZE    = 500
GENERATIONS = 600
ELITE_SIZE  = 20 #40. 
BASE_MUT    = 0.25
HIGH_MUT    = 0.55
STAGNATE_TH = 150
INJECT_EVERY= 200

W_CONSEC_6       = 1.0
W_SHIFT_ILLEGAL  = 1.0
W_WRONG_GROUP    = 0.2
W_CONSEC_OFF_LT2 = 0.1
W_MONTH_OFF_LT9  = 0.1
W_WEEKEND_LT4    = 0.1
W_SINGLE_OFF     = 0.1
W_DEMAND         = 10.0

SHIFT_MAP   = {"D": "Day", "E": "Afternoon", "N": "Night"}
ALL_SHIFTS  = ["D", "E", "N", "O"]
ENG_GRP_COL = "班別群組(*第一碼為群組代碼第二碼之後為可backup群組)"

# ── Data ────────────────────────────────────────────────────
engineers = pd.read_csv("/Users/yizhen/Desktop/工作坊/Engineer_List.csv", encoding="utf-8-sig")
demand    = pd.read_csv("/Users/yizhen/Desktop/工作坊/Shift_Demand.csv",  encoding="utf-8-sig").reset_index(drop=True)
engineers.columns = engineers.columns.str.strip()
demand.columns    = demand.columns.str.strip()

NUM_ENG     = len(engineers)
WEEKEND_IDX = set(demand[demand["IfWeekend"] == "Y"].index.tolist())

raw        = engineers.iloc[:, 2:2+DAYS].values
FIXED_MASK = np.where(pd.isna(raw) | (raw == ""), None, raw)
PREF_GROUP = [str(engineers.iloc[i][ENG_GRP_COL]).strip()[0] for i in range(NUM_ENG)]

DAY_REQ = demand["Day"].astype(int).values
AFT_REQ = demand["Afternoon"].astype(int).values
NGT_REQ = demand["Night"].astype(int).values

print(f"Loaded {NUM_ENG} engineers | {DAYS} days | {len(WEEKEND_IDX)} weekend days")

# ── Penalty ─────────────────────────────────────────────────
def calc_penalty(individual):
    arr = np.array(individual, dtype=object)
    pen = 0.0
    for i in range(NUM_ENG):
        s, pref = arr[i], PREF_GROUP[i]
        for d in range(DAYS):
            if s[d] != "O" and s[d] != pref:
                pen += W_WRONG_GROUP
        for d in range(1, DAYS):
            p, c = s[d-1], s[d]
            if (p=="N" and c in("D","E")) or (p=="E" and c=="D") or (p in("D","E") and c=="N"):
                pen += W_SHIFT_ILLEGAL
        streak = 0
        for d in range(DAYS):
            if s[d] != "O":
                streak += 1
                if streak >= 6: pen += W_CONSEC_6
            else: streak = 0
        for d in range(1, DAYS-1):
            if s[d]=="O" and s[d-1]!="O" and s[d+1]!="O": pen += W_SINGLE_OFF
        off = int(np.sum(s == "O"))
        if off < 9: pen += W_MONTH_OFF_LT9 * (9 - off)
        wk = sum(1 for d in WEEKEND_IDX if s[d]=="O")
        if wk < 4: pen += W_WEEKEND_LT4 * (4 - wk)
        blocks, run = 0, 0
        for d in range(DAYS):
            if s[d]=="O": run += 1
            else:
                if run >= 2: blocks += 1
                run = 0
        if run >= 2: blocks += 1
        if blocks < 2: pen += W_CONSEC_OFF_LT2
    for d in range(DAYS):
        col = arr[:, d]
        for sc, req in [("D",DAY_REQ[d]),("E",AFT_REQ[d]),("N",NGT_REQ[d])]:
            cnt = int(np.sum(col == sc))
            if cnt < req: pen += W_DEMAND * (req - cnt)
    return pen

def detailed_report(individual):
    arr = np.array(individual, dtype=object)
    b = {k:0.0 for k in ["Consec_Work_6","Illegal_Shift","Wrong_Group","Consec_Off_LT2",
                          "Month_Off_LT9","Weekend_Off_LT4","Single_Off","Demand_Unmet"]}
    for i in range(NUM_ENG):
        s, pref = arr[i], PREF_GROUP[i]
        for d in range(DAYS):
            if s[d]!="O" and s[d]!=pref: b["Wrong_Group"] += W_WRONG_GROUP
        for d in range(1, DAYS):
            p,c = s[d-1],s[d]
            if (p=="N" and c in("D","E")) or (p=="E" and c=="D") or (p in("D","E") and c=="N"):
                b["Illegal_Shift"] += W_SHIFT_ILLEGAL
        streak=0
        for d in range(DAYS):
            if s[d]!="O":
                streak+=1
                if streak>=6: b["Consec_Work_6"]+=W_CONSEC_6
            else: streak=0
        for d in range(1,DAYS-1):
            if s[d]=="O" and s[d-1]!="O" and s[d+1]!="O": b["Single_Off"]+=W_SINGLE_OFF
        off=int(np.sum(s=="O"))
        if off<9: b["Month_Off_LT9"]+=W_MONTH_OFF_LT9*(9-off)
        wk=sum(1 for d in WEEKEND_IDX if s[d]=="O")
        if wk<4: b["Weekend_Off_LT4"]+=W_WEEKEND_LT4*(4-wk)
        blocks,run=0,0
        for d in range(DAYS):
            if s[d]=="O": run+=1
            else:
                if run>=2: blocks+=1
                run=0
        if run>=2: blocks+=1
        if blocks<2: b["Consec_Off_LT2"]+=W_CONSEC_OFF_LT2
    for d in range(DAYS):
        col=arr[:,d]
        for sc,req in [("D",DAY_REQ[d]),("E",AFT_REQ[d]),("N",NGT_REQ[d])]:
            cnt=int(np.sum(col==sc))
            if cnt<req: b["Demand_Unmet"]+=W_DEMAND*(req-cnt)
    print("\n"+"="*48+"\n   PENALTY BREAKDOWN\n"+"="*48)
    total=0.0
    for k,v in b.items():
        print(f"  {k:<24}: {v:>8.2f}"); total+=v
    print("-"*48+f"\n  {'TOTAL':<24}: {total:>8.2f}\n"+"="*48)

# ── Individual Builder ───────────────────────────────────────
def build_individual():
    arr = np.empty((NUM_ENG, DAYS), dtype=object)
    for i in range(NUM_ENG):
        for d in range(DAYS):
            arr[i,d] = FIXED_MASK[i,d]
    for d in range(DAYS):
        free = [i for i in range(NUM_ENG) if arr[i,d] is None]
        random.shuffle(free)
        for sc, req in [("D",DAY_REQ[d]),("E",AFT_REQ[d]),("N",NGT_REQ[d])]:
            have = int(np.sum(arr[:,d]==sc))
            need = max(0, req-have)
            pref_p  = [i for i in free if arr[i,d] is None and PREF_GROUP[i]==sc]
            other_p = [i for i in free if arr[i,d] is None and PREF_GROUP[i]!=sc]
            pool    = pref_p + other_p
            for i in pool[:need]:
                arr[i,d] = sc
                free = [x for x in free if x!=i]
    for i in range(NUM_ENG):
        for d in range(DAYS):
            if arr[i,d] is None: arr[i,d] = PREF_GROUP[i]
    return arr.tolist()

def build_with_off_pattern(period, offset):
    """Build individual with regular 2-day off blocks every `period` days."""
    arr = np.array(build_individual(), dtype=object)
    for i in range(NUM_ENG):
        d = (i*2 + offset) % period
        while d < DAYS - 1:
            if FIXED_MASK[i,d] is None and FIXED_MASK[i,d+1] is None:
                arr[i,d]   = "O"
                arr[i,d+1] = "O"
            d += period
    return arr.tolist()

# ── Repair Operators ─────────────────────────────────────────
def repair_demand(arr):
    for d in range(DAYS):
        for sc, req in [("D",DAY_REQ[d]),("E",AFT_REQ[d]),("N",NGT_REQ[d])]:
            need = req - int(np.sum(arr[:,d]==sc))
            if need <= 0: continue
            cands = [i for i in range(NUM_ENG) if arr[i,d]=="O" and FIXED_MASK[i,d] is None]
            random.shuffle(cands)
            for i in cands[:need]: arr[i,d] = sc

def repair_illegal(arr):
    for i in range(NUM_ENG):
        for d in range(1, DAYS):
            p,c = arr[i,d-1], arr[i,d]
            if ((p=="N" and c in("D","E")) or (p=="E" and c=="D") or
                (p in("D","E") and c=="N")) and FIXED_MASK[i,d] is None:
                arr[i,d] = "O"

def repair_consec_work(arr):
    for i in range(NUM_ENG):
        streak, start = 0, 0
        for d in range(DAYS):
            if arr[i,d] != "O":
                if streak == 0: start = d
                streak += 1
                if streak == 7:
                    mid = start + 3
                    if FIXED_MASK[i,mid] is None: arr[i,mid] = "O"
                    streak = 0
            else: streak = 0

def repair_single_off(arr):
    for _ in range(3):
        changed = False
        for i in range(NUM_ENG):
            for d in range(1, DAYS-1):
                if arr[i,d]!="O" or arr[i,d-1]=="O" or arr[i,d+1]=="O": continue
                if FIXED_MASK[i,d] is not None: continue
                extended = False
                for tgt in [d-1, d+1]:
                    if FIXED_MASK[i,tgt] is not None: continue
                    sh = arr[i,tgt]
                    donors = [j for j in range(NUM_ENG) if j!=i and arr[j,tgt]=="O"
                              and FIXED_MASK[j,tgt] is None]
                    pref_d = [j for j in donors if PREF_GROUP[j]==sh]
                    pool   = pref_d if pref_d else donors
                    if pool:
                        j = random.choice(pool)
                        arr[j,tgt] = sh; arr[i,tgt] = "O"
                        extended = True; changed = True; break
                if not extended:
                    arr[i,d] = PREF_GROUP[i]; changed = True
        if not changed: break

def repair_consec_off(arr):
    for i in range(NUM_ENG):
        def count_blocks():
            bl, r = 0, 0
            for d in range(DAYS):
                if arr[i,d]=="O": r+=1
                else:
                    if r>=2: bl+=1
                    r=0
            if r>=2: bl+=1
            return bl
        if count_blocks() >= 2: continue
        single = [d for d in range(DAYS) if arr[i,d]=="O"
                  and (d==0 or arr[i,d-1]!="O") and (d==DAYS-1 or arr[i,d+1]!="O")]
        for d in single:
            for ext in [d+1, d-1]:
                if not (0<=ext<DAYS): continue
                if FIXED_MASK[i,ext] is not None or arr[i,ext]=="O": continue
                sh = arr[i,ext]
                donors=[j for j in range(NUM_ENG) if j!=i and arr[j,ext]=="O"
                        and FIXED_MASK[j,ext] is None]
                pref_d=[j for j in donors if PREF_GROUP[j]==sh]
                pool = pref_d if pref_d else donors
                if pool:
                    j=random.choice(pool); arr[j,ext]=sh; arr[i,ext]="O"; break
            if count_blocks()>=2: break

def full_repair(individual):
    arr = np.array(individual, dtype=object)
    repair_consec_work(arr)
    repair_illegal(arr)
    repair_single_off(arr)
    repair_consec_off(arr)
    repair_demand(arr)
    return arr.tolist()

# ── Mutation ─────────────────────────────────────────────────
def mutate(individual, rate):
    arr = np.array(individual, dtype=object)
    if random.random() < rate:  # swap day
        d = random.randint(0, DAYS-1)
        free = [i for i in range(NUM_ENG) if FIXED_MASK[i,d] is None]
        if len(free)>=2:
            i1,i2 = random.sample(free,2); arr[i1,d],arr[i2,d]=arr[i2,d],arr[i1,d]
    if random.random() < rate:  # swap block
        span=random.randint(2,5); d0=random.randint(0,DAYS-span)
        cands=[(i,j) for i in range(NUM_ENG) for j in range(i+1,NUM_ENG)
               if all(FIXED_MASK[i,d0+k] is None and FIXED_MASK[j,d0+k] is None for k in range(span))]
        if cands:
            i1,i2=random.choice(cands)
            for k in range(span): arr[i1,d0+k],arr[i2,d0+k]=arr[i2,d0+k],arr[i1,d0+k]
    if random.random() < rate:  # reassign
        i=random.randint(0,NUM_ENG-1); d=random.randint(0,DAYS-1)
        if FIXED_MASK[i,d] is None: arr[i,d]=random.choice(ALL_SHIFTS)
    if random.random() < rate:  # off block
        i=random.randint(0,NUM_ENG-1); d=random.randint(0,DAYS-2)
        if FIXED_MASK[i,d] is None and FIXED_MASK[i,d+1] is None:
            arr[i,d]="O"; arr[i,d+1]="O"
    if random.random() < rate:  # shift rotation
        i=random.randint(0,NUM_ENG-1); n=random.randint(1,5)
        free_d=[d for d in range(DAYS) if FIXED_MASK[i,d] is None]
        if len(free_d)>n:
            vals=[arr[i,d] for d in free_d]; vals=vals[n:]+vals[:n]
            for k,d in enumerate(free_d): arr[i,d]=vals[k]
    return arr.tolist()

def crossover(p1, p2):
    a1,a2 = np.array(p1,dtype=object), np.array(p2,dtype=object)
    mask = np.random.rand(DAYS)<0.5
    return np.where(mask[np.newaxis,:], a1, a2).tolist()

# ── GA Main ──────────────────────────────────────────────────
def run():
    print("\nBuilding initial population...")
    pop = []
    for period in [6,7,8]:
        for off in range(6):
            pop.append(full_repair(build_with_off_pattern(period, off)))
            if len(pop) >= POP_SIZE//2: break
        if len(pop) >= POP_SIZE//2: break
    while len(pop) < POP_SIZE:
        pop.append(full_repair(build_individual()))
    scores = [calc_penalty(ind) for ind in pop]
    best_score = min(scores)
    best_ind   = copy.deepcopy(pop[scores.index(best_score)])
    no_improve = 0
    print(f"Initial best penalty: {best_score:.2f}\n")
    print(f"{'Gen':>5}  {'Best':>8}  {'Avg':>8}  {'Mut':>5}  {'NoImp':>6}")
    print("─"*42)
    for gen in range(GENERATIONS):
        order  = np.argsort(scores)
        pop    = [pop[i] for i in order]
        scores = [scores[i] for i in order]
        cur    = scores[0]
        avg50  = float(np.mean(scores[:50]))
        mut    = BASE_MUT if no_improve < STAGNATE_TH else HIGH_MUT
        if cur < best_score - 1e-9:
            best_score = cur; best_ind = copy.deepcopy(pop[0]); no_improve = 0
        else:
            no_improve += 1
        if gen % 50 == 0 or gen < 3:
            print(f"{gen:>5}  {cur:>8.2f}  {avg50:>8.2f}  {mut:>5.2f}  {no_improve:>6}  ({time.time()-START_TIME:.0f}s)")
        if best_score < 1e-9:
            print("  *** Penalty = 0, perfect schedule! ***"); break
        next_gen = [full_repair(copy.deepcopy(pop[i])) for i in range(ELITE_SIZE)]
        pool_sz  = min(100, len(pop))
        rp       = 0.7 if no_improve < STAGNATE_TH else 0.98
        while len(next_gen) < POP_SIZE:
            child = crossover(random.choice(pop[:pool_sz]), random.choice(pop[:pool_sz]))
            child = mutate(child, mut)
            if random.random() < rp: child = full_repair(child)
            next_gen.append(child)
        pop    = next_gen
        scores = [calc_penalty(ind) for ind in pop]
        if no_improve > 0 and no_improve % INJECT_EVERY == 0:
            n = 80
            print(f"  [Gen {gen}] Injecting {n} fresh individuals")
            fresh = [full_repair(build_individual()) for _ in range(n)]
            fscr  = [calc_penalty(x) for x in fresh]
            worst = np.argsort(scores)[::-1]
            for k in range(n):
                pop[worst[k]] = fresh[k]; scores[worst[k]] = fscr[k]
    return best_ind, best_score

def _find_penalty_cells(schedule):
    """
    回傳所有有懲罰的 (engineer_idx, day_idx, label) tuple。
    label 為懲罰類型簡稱，用於 tooltip / 圖例。
    同一個 cell 可能有多個懲罰，會各自加入清單。
    """
    arr    = np.array(schedule, dtype=object)
    cells  = []   # list of (i, d, label)

    for i in range(NUM_ENG):
        s, pref = arr[i], PREF_GROUP[i]

        # Wrong group
        for d in range(DAYS):
            if s[d] != "O" and s[d] != pref:
                cells.append((i, d, "Wrong\nGroup"))

        # Illegal transitions  → mark the second day (the violating cell)
        for d in range(1, DAYS):
            p, c = s[d-1], s[d]
            if (p=="N" and c in("D","E")) or (p=="E" and c=="D") or (p in("D","E") and c=="N"):
                cells.append((i, d, "Illegal\nShift"))

        # Consecutive work ≥ 6  → mark every extra day
        streak, start = 0, 0
        for d in range(DAYS):
            if s[d] != "O":
                if streak == 0: start = d
                streak += 1
                if streak >= 6:
                    cells.append((i, d, "Consec\nWork"))
            else:
                streak = 0

        # Single isolated off
        for d in range(1, DAYS-1):
            if s[d]=="O" and s[d-1]!="O" and s[d+1]!="O":
                cells.append((i, d, "Single\nOff"))

        # Total off < 9  → mark ALL off cells of that engineer
        off_days = int(np.sum(s == "O"))
        if off_days < 9:
            for d in range(DAYS):
                if s[d] == "O":
                    cells.append((i, d, "Off\nLT9"))

        # Weekend off < 4  → mark weekend off cells
        wk_off = sum(1 for d in WEEKEND_IDX if s[d]=="O")
        if wk_off < 4:
            for d in WEEKEND_IDX:
                if s[d] == "O":
                    cells.append((i, d, "Wknd\nLT4"))

        # Consecutive-off blocks < 2  → mark all off cells
        blocks, run = 0, 0
        for d in range(DAYS):
            if s[d]=="O": run+=1
            else:
                if run>=2: blocks+=1
                run=0
        if run>=2: blocks+=1
        if blocks < 2:
            for d in range(DAYS):
                if s[d]=="O":
                    cells.append((i, d, "ConsOff\nLT2"))

    # Demand unmet  → mark cells of that shift on that day
    for d in range(DAYS):
        col = arr[:, d]
        for sc, req in [("D",DAY_REQ[d]),("E",AFT_REQ[d]),("N",NGT_REQ[d])]:
            cnt = int(np.sum(col == sc))
            if cnt < req:
                for i in range(NUM_ENG):
                    if arr[i, d] == sc:
                        cells.append((i, d, "Demand"))

    return cells


def _draw_base(ax, arr, numeric, title):
    """共用的底圖繪製邏輯。"""
    cmap = ListedColormap(["#4CAF50", "#FFC107", "#3F51B5", "#E0E0E0"])
    im = ax.imshow(numeric, cmap=cmap, vmin=0, vmax=3,
                   aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, DAYS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, NUM_ENG, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for d in WEEKEND_IDX:
        ax.axvspan(d-0.5, d+0.5, color="#FFFDE7", alpha=0.45, zorder=0)

    ax.set_xticks(range(DAYS))
    ax.set_xticklabels(
        [f"{'W' if d in WEEKEND_IDX else ''}{d+1}" for d in range(DAYS)],
        fontsize=8)
    ax.set_yticks(range(NUM_ENG))
    ax.set_yticklabels(engineers["人員"], fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("Day  (W = Weekend)", fontsize=11)
    ax.set_ylabel("Engineer", fontsize=11)

    cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3], pad=0.02, fraction=0.025)
    cbar.ax.set_yticklabels(["D  Day","E  Afternoon","N  Night","O  Off"])
    cbar.ax.tick_params(labelsize=8)

    for i in range(NUM_ENG):
        off_cnt = int(np.sum(arr[i] == "O"))
        ax.text(DAYS+0.4, i, f"Off:{off_cnt}", va="center", fontsize=7, color="#555")

    return im


def plot_heatmap(schedule, save_path=None):
    """圖1：標準熱圖（無標記）。"""
    s_to_n  = {"D":0,"E":1,"N":2,"O":3}
    arr     = np.array(schedule, dtype=object)
    numeric = np.vectorize(s_to_n.get)(arr)

    fig, ax = plt.subplots(figsize=(20, max(6, NUM_ENG*0.55+2)))
    _draw_base(ax, arr, numeric, "IM Workshop 2026 — Scheduling Result")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Heatmap saved: {os.path.basename(save_path)}")
    plt.close()


def plot_heatmap_penalty(schedule, save_path=None):
    """圖2：懲罰標記熱圖，所有有懲罰的格子加上紅框。"""
    s_to_n  = {"D":0,"E":1,"N":2,"O":3}
    arr     = np.array(schedule, dtype=object)
    numeric = np.vectorize(s_to_n.get)(arr)

    fig, ax = plt.subplots(figsize=(20, max(6, NUM_ENG*0.55+2)))
    _draw_base(ax, arr, numeric, "IM Workshop 2026 — Scheduling Result  [Penalty Highlighted]")

    # Collect penalty cells and de-duplicate per (i,d) for box drawing
    raw_cells  = _find_penalty_cells(schedule)
    # Build dict: (i,d) -> set of labels
    cell_labels = {}
    for i, d, lbl in raw_cells:
        cell_labels.setdefault((i,d), set()).add(lbl)

    # Draw red rectangle per violating cell
    from matplotlib.patches import Rectangle
    for (i, d), lbls in cell_labels.items():
        rect = Rectangle(
            (d - 0.48, i - 0.48), 0.96, 0.96,
            linewidth=2.0, edgecolor="red", facecolor="none", zorder=5
        )
        ax.add_patch(rect)

    # ── Legend for penalty types ──────────────────────────────
    legend_labels = sorted({lbl.replace("\n"," ") for _, _, lbl in raw_cells})
    # Count violations per type
    from collections import Counter
    type_count = Counter(lbl.replace("\n"," ") for _, _, lbl in raw_cells)
    legend_handles = [
        plt.Line2D([0],[0], marker="s", color="w",
                   markerfacecolor="none", markeredgecolor="red",
                   markersize=10, markeredgewidth=2,
                   label=f"{lbl}  (×{type_count[lbl]})")
        for lbl in legend_labels
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left",
                  bbox_to_anchor=(1.08, 1.0), fontsize=8,
                  title="Penalty Types", title_fontsize=9,
                  framealpha=0.9)

    # Summary text box
    total_cells = len(cell_labels)
    ax.text(0.01, -0.09,
            f"Total penalised cells: {total_cells}  |  Distinct violations: {len(raw_cells)}",
            transform=ax.transAxes, fontsize=9, color="#c00",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#c00", lw=1))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Penalty heatmap saved: {os.path.basename(save_path)}")
    plt.close()


def save_output(best_ind, best_score):
    out_dir = "/Users/yizhen/Desktop/排班結果報告"
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(best_ind, columns=[f"Date_{i+1}" for i in range(DAYS)])
    df.insert(0, "人員", engineers["人員"])
    df.insert(1, ENG_GRP_COL, engineers[ENG_GRP_COL])
    ps = str(round(best_score,2)).replace(".","_")
    p1 = os.path.join(out_dir, f"Scheduling_Output_Penalty_{ps}.csv")
    p2 = os.path.join(out_dir, "Scheduling_Output.csv")
    df.to_csv(p1, index=False, encoding="utf-8-sig")
    df.to_csv(p2, index=False, encoding="utf-8-sig")
    print(f"\n  Saved: {p1}\n  Saved: {p2}")

    # ── Heatmap 1: standard ──
    img1 = os.path.join(out_dir, f"Heatmap_Penalty_{ps}.png")
    plot_heatmap(best_ind, save_path=img1)

    # ── Heatmap 2: penalty highlighted ──
    img2 = os.path.join(out_dir, f"Heatmap_PenaltyMarked_{ps}.png")
    plot_heatmap_penalty(best_ind, save_path=img2)

    return p1, p2, img1, img2

if __name__ == "__main__":
    print("="*55+"\n  TSMC IM Workshop 2026 — Scheduling Optimizer\n"+"="*55)
    
    # 關鍵：你必須先執行 run()，並把回傳的結果賦值給 best_ind 和 best_score
    best_ind, best_score = run() 
    
    # 現在你就可以看到基因了
    print(f"\nEngineer 0's Gene: {best_ind[0]}")
    
    for i in range(5):  # 看看前五個人
        gene_strip = "".join(best_ind[i][:15]) 
        print(f"Eng_{i:02d} Gene Segment: | {gene_strip} | ...")

    # 接著執行原本的報告與存檔
    print(f"\n{'='*55}\n  FINAL BEST PENALTY : {best_score:.2f}\n  Runtime : {time.time()-START_TIME:.1f}s\n{'='*55}")
    detailed_report(best_ind)
    save_output(best_ind, best_score)
    #print("="*55+"\n  TSMC IM Workshop 2026 — Scheduling Optimizer\n"+"="*55)
    #best_ind, best_score = run()
    #print(f"\n{'='*55}\n  FINAL BEST PENALTY : {best_score:.2f}\n  Runtime : {time.time()-START_TIME:.1f}s\n{'='*55}")
    #detailed_report(best_ind)
    #save_output(best_ind, best_score)