import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib import font_manager as fm

# =========================
# User-editable settings
# =========================
OUTPUT_CSV = 'Scheduling_Output_FastOptimal.csv'
ENGINEER_CSV = 'Engineer_List.csv'
DEMAND_CSV = 'Shift_Demand.csv'
OUT_DIR = 'viz_output'

# Solver summary values from your latest run
SOLVER_STATUS = 'OPTIMAL'
OBJECTIVE_VALUE = 0.6
BEST_BOUND = 0.6
SOLVE_TIME_SECONDS = 59.84

# PDF-like colors
SHIFT_COLORS = {
    'O': '#FFFFFF',   # 白色
    'D': '#E8F5E9',   # 較亮的淺藍
    'E': '#F5E6A1',   # 淺米黃
    'N': '#C9D2F0',   # 淺紫藍
}

ACCENT_RED = '#C00000'
ACCENT_BLUE = '#1F4E79'
ACCENT_GRAY = '#666666'
LIGHT_GRAY = '#D9D9D9'
BG_COLOR = '#FAFAFA'
CARD_BG = '#FFFFFF'

plt.rcParams['figure.dpi'] = 180
plt.rcParams['savefig.dpi'] = 220
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False


# =========================
# Font helper
# =========================
def setup_chinese_font():
    candidates = [
        'Microsoft JhengHei',
        'PingFang TC',
        'Noto Sans CJK TC',
        'Noto Sans CJK SC',
        'Noto Sans TC',
        'SimHei',
        'Arial Unicode MS',
        'Source Han Sans TW',
        'Heiti TC',
    ]
    installed = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            return name
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return 'DejaVu Sans'


# =========================
# CSV helper
# =========================
def read_csv_auto(path):
    encodings = ['utf-8-sig', 'utf-8', 'cp950', 'big5', 'mbcs']
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_error = e
    raise last_error


# =========================
# General helpers
# =========================
def find_engineer_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in {'人員', 'engineer', 'engineers', 'name', '員工'}:
            return c
    return df.columns[0]


def find_default_group_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        cl = str(c).strip().lower()
        if any(k in cl for k in ['預設', 'group', '班別群組', 'default']):
            return c
    return df.columns[1]


def date_cols(df: pd.DataFrame):
    return [c for c in df.columns if str(c).strip().lower().startswith('date_')]


def parse_day_num(col_name: str) -> int:
    s = str(col_name).strip()
    return int(s.split('_')[-1]) if '_' in s else int(s)


def safe_text(x):
    if pd.isna(x):
        return ''
    return str(x).strip()


# =========================
# Penalty recomputation
# Rule order aligned to your latest solver output:
# r1 consecutive work, r2 forbidden transitions, r3 default mismatch,
# r4 month off shortage, r5 weekend off shortage, r6 isolated off, r7 off-block shortage
# =========================
def compute_penalties(schedule_df: pd.DataFrame, engineer_df: pd.DataFrame, demand_df: pd.DataFrame):
    eng_col = find_engineer_col(schedule_df)
    group_col = find_default_group_col(engineer_df)
    dcols = date_cols(schedule_df)
    day_count = len(dcols)

    weekend_days = []
    if 'IfWeekend' in demand_df.columns:
        for i, flag in enumerate(demand_df['IfWeekend'].astype(str).str.strip().tolist()):
            if flag.upper() == 'Y':
                weekend_days.append(i)

    penalties = {f'rule{i}': 0 for i in range(1, 8)}
    details = {'rule3': [], 'rule5': [], 'rule6': [], 'rule7': []}
    forbidden = {('N', 'D'), ('N', 'E'), ('E', 'D'), ('D', 'N'), ('E', 'N')}

    for e in range(len(schedule_df)):
        name = safe_text(schedule_df.iloc[e][eng_col])
        default_group = safe_text(engineer_df.iloc[e][group_col])[:1]
        seq = [safe_text(schedule_df.iloc[e][c]) for c in dcols]

        # rule1: consecutive work > 5
        run = 0
        for s in seq:
            if s in {'D', 'E', 'N'}:
                run += 1
            else:
                if run >= 6:
                    penalties['rule1'] += max(0, run - 5)
                run = 0
        if run >= 6:
            penalties['rule1'] += max(0, run - 5)

        # rule2: forbidden transitions
        for d in range(day_count - 1):
            if (seq[d], seq[d + 1]) in forbidden:
                penalties['rule2'] += 1

        # rule3: default group mismatch, O not penalized
        if default_group in {'D', 'E', 'N'}:
            for d in range(day_count):
                if seq[d] in {'D', 'E', 'N'} and seq[d] != default_group:
                    penalties['rule3'] += 1
                    details['rule3'].append((name, d + 1, default_group, seq[d]))

        # rule4: month off shortage < 9
        total_off = sum(1 for s in seq if s == 'O')
        penalties['rule4'] += max(0, 9 - total_off)

        # rule5: weekend off shortage < 4
        weekend_off = sum(1 for d in weekend_days if d < day_count and seq[d] == 'O')
        lack = max(0, 4 - weekend_off)
        penalties['rule5'] += lack
        if lack > 0:
            details['rule5'].append((name, weekend_off, lack))

        # rule6: isolated off in the middle
        for d in range(1, day_count - 1):
            if seq[d] == 'O' and seq[d - 1] in {'D', 'E', 'N'} and seq[d + 1] in {'D', 'E', 'N'}:
                penalties['rule6'] += 1
                details['rule6'].append((name, d + 1, seq[d - 1], seq[d + 1]))

        # rule7: consecutive off blocks < 2
        off_blocks = 0
        d = 0
        while d < day_count - 1:
            if seq[d] == 'O' and seq[d + 1] == 'O':
                start = d
                off_blocks += 1
                d += 2
                while d < day_count and seq[d] == 'O':
                    d += 1
                details['rule7'].append((name, start + 1, d, 'OO-block'))
            else:
                d += 1
        if off_blocks < 2:
            penalties['rule7'] += 1

    weighted = {
        'rule1': penalties['rule1'] * 1.0,
        'rule2': penalties['rule2'] * 1.0,
        'rule3': penalties['rule3'] * 0.2,
        'rule4': penalties['rule4'] * 0.1,
        'rule5': penalties['rule5'] * 0.1,
        'rule6': penalties['rule6'] * 0.1,
        'rule7': penalties['rule7'] * 0.1,
    }
    return penalties, weighted, details


def build_schedule_matrix(schedule_df: pd.DataFrame):
    eng_col = find_engineer_col(schedule_df)
    dcols = date_cols(schedule_df)
    shift_to_int = {'O': 0, 'D': 1, 'E': 2, 'N': 3}

    mat = np.zeros((len(schedule_df), len(dcols)), dtype=int)
    for i in range(len(schedule_df)):
        for j, c in enumerate(dcols):
            mat[i, j] = shift_to_int.get(safe_text(schedule_df.iloc[i][c]), 0)

    labels = schedule_df[eng_col].astype(str).tolist()
    return mat, labels, dcols


def weekend_off_summary(schedule_df: pd.DataFrame, demand_df: pd.DataFrame):
    eng_col = find_engineer_col(schedule_df)
    dcols = date_cols(schedule_df)

    weekend_days = []
    if 'IfWeekend' in demand_df.columns:
        for i, flag in enumerate(demand_df['IfWeekend'].astype(str).str.strip().tolist()):
            if flag.upper() == 'Y':
                weekend_days.append(i)

    rows = []
    for i in range(len(schedule_df)):
        seq = [safe_text(schedule_df.iloc[i][c]) for c in dcols]
        cnt = sum(1 for d in weekend_days if d < len(seq) and seq[d] == 'O')
        rows.append({
            'Engineer': safe_text(schedule_df.iloc[i][eng_col]),
            'WeekendOffDays': cnt,
            'ShortageTo4': max(0, 4 - cnt),
        })
    return pd.DataFrame(rows).sort_values(['WeekendOffDays', 'Engineer'], ascending=[True, True]).reset_index(drop=True)


# =========================
# Plot 1: Dashboard summary
# =========================
def plot_dashboard(weighted, penalties, out_path, font_used):
    fig = plt.figure(figsize=(14, 8), facecolor=BG_COLOR)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    fig.text(0.05, 0.93, 'Scheduling Optimization Result Dashboard', fontsize=22, fontweight='bold', color=ACCENT_BLUE)
    fig.text(0.05, 0.895, 'IM Workshop Scheduling Result Summary', fontsize=11, color=ACCENT_GRAY)

    cards = [
        ('最佳懲罰值', f'{OBJECTIVE_VALUE:.1f}', '#FFF2F2'),
        ('求解狀態', SOLVER_STATUS, '#EEF6FF'),
        ('求解時間', f'{SOLVE_TIME_SECONDS:.2f} 秒', '#F7F7F7'),
        ('Best Bound', f'{BEST_BOUND:.1f}', '#F6F2FF'),
    ]

    x0, y, w, h, gap = 0.05, 0.73, 0.205, 0.13, 0.025
    for i, (title, value, color) in enumerate(cards):
        xi = x0 + i * (w + gap)
        rect = patches.FancyBboxPatch(
            (xi, y), w, h,
            boxstyle='round,pad=0.012,rounding_size=0.02',
            linewidth=1.0, edgecolor=LIGHT_GRAY, facecolor=color,
            transform=fig.transFigure
        )
        fig.patches.append(rect)
        fig.text(xi + 0.02, y + 0.085, title, fontsize=11, color=ACCENT_GRAY)
        fig.text(xi + 0.02, y + 0.03, value, fontsize=24, fontweight='bold', color=ACCENT_RED if i == 0 else ACCENT_BLUE)

    # Left card: penalties
    left, bottom, width, height = 0.05, 0.12, 0.42, 0.43
    rect = patches.FancyBboxPatch(
        (left, bottom), width, height,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        linewidth=1.0, edgecolor=LIGHT_GRAY, facecolor=CARD_BG,
        transform=fig.transFigure
    )
    fig.patches.append(rect)
    fig.text(left + 0.02, bottom + height - 0.05, '懲罰明細總覽', fontsize=14, fontweight='bold', color=ACCENT_BLUE)

    rows = [
        ('rule1 連續上班過多', penalties['rule1'], weighted['rule1']),
        ('rule2 禁忌接班', penalties['rule2'], weighted['rule2']),
        ('rule3 違反預設群組', penalties['rule3'], weighted['rule3']),
        ('rule4 月休不足', penalties['rule4'], weighted['rule4']),
        ('rule5 週末休不足', penalties['rule5'], weighted['rule5']),
        ('rule6 單獨休假', penalties['rule6'], weighted['rule6']),
        ('rule7 連休區段不足', penalties['rule7'], weighted['rule7']),
    ]

    fig.text(left + 0.03, bottom + height - 0.095, '規則', fontsize=10.5, color=ACCENT_GRAY)
    fig.text(left + 0.31, bottom + height - 0.095, '次數', fontsize=10.5, color=ACCENT_GRAY, ha='right')
    fig.text(left + 0.39, bottom + height - 0.095, '加權值', fontsize=10.5, color=ACCENT_GRAY, ha='right')

    yy = bottom + height - 0.14
    for label, cnt, val in rows:
        fig.text(left + 0.03, yy, label, fontsize=11, color='#333333')
        fig.text(left + 0.31, yy, f'{cnt}', fontsize=11, ha='right', color='#333333')
        fig.text(left + 0.39, yy, f'{val:.1f}', fontsize=11, ha='right', color=ACCENT_RED if val > 0 else ACCENT_GRAY)
        yy -= 0.047

    fig.text(left + 0.03, bottom + 0.04, f'重新計算總懲罰值 = {sum(weighted.values()):.1f}', fontsize=13, fontweight='bold', color=ACCENT_RED)

    # Right card: legend
    left2, width2 = 0.53, 0.42
    rect2 = patches.FancyBboxPatch(
        (left2, bottom), width2, height,
        boxstyle='round,pad=0.012,rounding_size=0.02',
        linewidth=1.0, edgecolor=LIGHT_GRAY, facecolor=CARD_BG,
        transform=fig.transFigure
    )
    fig.patches.append(rect2)
    fig.text(left2 + 0.02, bottom + height - 0.05, '班別色彩對照', fontsize=14, fontweight='bold', color=ACCENT_BLUE)
    fig.text(left2 + 0.02, bottom + height - 0.085, f'字型使用：{font_used}  ｜  顏色對齊 PDF 範例風格', fontsize=10, color=ACCENT_GRAY)

    legend_items = [('O 休假', SHIFT_COLORS['O']), ('D 早班', SHIFT_COLORS['D']), ('E 午班', SHIFT_COLORS['E']), ('N 晚班', SHIFT_COLORS['N'])]
    lx = left2 + 0.04
    ly = bottom + height - 0.16
    for idx, (label, color) in enumerate(legend_items):
        yi = ly - idx * 0.085
        sq = patches.Rectangle((lx, yi), 0.045, 0.05, transform=fig.transFigure, facecolor=color, edgecolor='#666666', linewidth=1.2)
        fig.patches.append(sq)
        fig.text(lx + 0.06, yi + 0.013, label, fontsize=13, color='#333333')

    fig.text(left2 + 0.04, bottom + 0.17, '建議放在成果頁第一張', fontsize=11, color='#444444')
    fig.text(left2 + 0.04, bottom + 0.13, '先交代最優值、求解狀態與時間', fontsize=11, color='#444444')
    fig.text(left2 + 0.04, bottom + 0.09, '再銜接懲罰來源與班表熱圖', fontsize=11, color='#444444')

    plt.savefig(out_path, bbox_inches='tight', facecolor=BG_COLOR)
    plt.close(fig)


# =========================
# Plot 2: Penalty breakdown
# =========================
def plot_penalty_breakdown(weighted, out_path):
    labels = [
        'rule1\n連續上班過多',
        'rule2\n禁忌接班',
        'rule3\n違反預設群組',
        'rule4\n月休不足',
        'rule5\n週末休不足',
        'rule6\n單獨休假',
        'rule7\n連休區段不足',
    ]
    values = [weighted[f'rule{i}'] for i in range(1, 8)]
    colors = [LIGHT_GRAY if v == 0 else ACCENT_RED for v in values]

    fig, ax = plt.subplots(figsize=(12, 6.8), facecolor='white')
    bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.62)
    ax.invert_yaxis()
    ax.set_title('Penalty Breakdown of the Optimal Schedule', fontsize=18, fontweight='bold', color=ACCENT_BLUE, pad=16)
    ax.set_xlabel('Weighted Penalty', fontsize=12)
    ax.set_xlim(0, max(max(values) + 0.08, 0.35))
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)

    for bar, v in zip(bars, values):
        x = bar.get_width()
        ax.text(x + 0.01, bar.get_y() + bar.get_height() / 2, f'{v:.1f}', va='center', fontsize=12, color='#333333', fontweight='bold')

    total = sum(values)
    ax.text(0.99, 0.03, f'Total = {total:.1f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=14, fontweight='bold', color=ACCENT_RED)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =========================
# Plot 3: Schedule heatmap
# =========================
def plot_schedule_heatmap(schedule_df, demand_df, out_path):
    mat, labels, dcols = build_schedule_matrix(schedule_df)
    cmap = ListedColormap([SHIFT_COLORS['O'], SHIFT_COLORS['D'], SHIFT_COLORS['E'], SHIFT_COLORS['N']])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig_w = max(16, len(dcols) * 0.42)
    fig_h = max(7, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.imshow(mat, aspect='auto', cmap=cmap, norm=norm)

    inv_map = {0: 'O', 1: 'D', 2: 'E', 3: 'N'}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, inv_map[mat[i, j]], ha='center', va='center', fontsize=8.3, color='#333333')

    weekend_days = []
    if 'IfWeekend' in demand_df.columns:
        for i, flag in enumerate(demand_df['IfWeekend'].astype(str).str.strip().tolist()):
            if flag.upper() == 'Y':
                weekend_days.append(i)
    for d in weekend_days:
        if d < len(dcols):
            rect = patches.Rectangle((d - 0.5, -0.5), 1, len(labels), linewidth=1.2, edgecolor='#D4A000', facecolor='none', linestyle='--')
            ax.add_patch(rect)

    ax.set_title('Schedule Heatmap', fontsize=18, fontweight='bold', color=ACCENT_BLUE, pad=16)
    ax.set_xticks(np.arange(len(dcols)))
    ax.set_xticklabels([parse_day_num(c) for c in dcols], fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Engineer', fontsize=12)

    ax.set_xticks(np.arange(-.5, len(dcols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
    ax.grid(which='minor', color='#D9D9D9', linestyle='-', linewidth=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    legend_elements = [
        patches.Patch(facecolor=SHIFT_COLORS['O'], edgecolor='#666666', label='O 休假'),
        patches.Patch(facecolor=SHIFT_COLORS['D'], edgecolor='#666666', label='D 早班'),
        patches.Patch(facecolor=SHIFT_COLORS['E'], edgecolor='#666666', label='E 午班'),
        patches.Patch(facecolor=SHIFT_COLORS['N'], edgecolor='#666666', label='N 晚班'),
        Line2D([0], [0], color='#D4A000', lw=1.5, linestyle='--', label='週末欄位'),
    ]
    ax.legend(handles=legend_elements, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# =========================
# Plot 4: Weekend off fairness
# =========================
def plot_weekend_off_distribution(schedule_df, demand_df, out_path):
    wdf = weekend_off_summary(schedule_df, demand_df)
    colors = [ACCENT_RED if x < 4 else '#8FBF8F' for x in wdf['WeekendOffDays']]

    fig_h = max(6, len(wdf) * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_h), facecolor='white')
    bars = ax.barh(wdf['Engineer'], wdf['WeekendOffDays'], color=colors, edgecolor='none', height=0.62)
    ax.axvline(4, color=ACCENT_BLUE, linestyle='--', linewidth=1.8, label='規則門檻 = 4 天')
    ax.set_title('Weekend Off-Day Distribution', fontsize=18, fontweight='bold', color=ACCENT_BLUE, pad=16)
    ax.set_xlabel('Weekend Off Days', fontsize=12)
    ax.set_ylabel('Engineer', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.25)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)

    for bar, shortage in zip(bars, wdf['ShortageTo4']):
        x = bar.get_width()
        txt = f'{x}' if shortage == 0 else f'{x}  缺 {shortage}'
        ax.text(x + 0.06, bar.get_y() + bar.get_height() / 2, txt, va='center', fontsize=10.5, color='#333333')

    shortage_people = int((wdf['WeekendOffDays'] < 4).sum())
    ax.legend(frameon=False, loc='lower right')
    ax.text(0.99, 0.10, f'未達 4 天人數 = {shortage_people}', transform=ax.transAxes, ha='right', va='bottom', fontsize=13, fontweight='bold', color=ACCENT_RED)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    font_used = setup_chinese_font()

    output_path = Path(OUTPUT_CSV)
    engineer_path = Path(ENGINEER_CSV)
    demand_path = Path(DEMAND_CSV)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not output_path.exists():
        raise FileNotFoundError(f'找不到 {OUTPUT_CSV}')
    if not engineer_path.exists():
        raise FileNotFoundError(f'找不到 {ENGINEER_CSV}')
    if not demand_path.exists():
        raise FileNotFoundError(f'找不到 {DEMAND_CSV}')

    schedule_df = read_csv_auto(output_path)
    engineer_df = read_csv_auto(engineer_path)
    demand_df = read_csv_auto(demand_path)

    penalties, weighted, details = compute_penalties(schedule_df, engineer_df, demand_df)

    plot_dashboard(weighted, penalties, out_dir / '01_dashboard_summary.png', font_used)
    plot_penalty_breakdown(weighted, out_dir / '02_penalty_breakdown.png')
    plot_schedule_heatmap(schedule_df, demand_df, out_dir / '03_schedule_heatmap.png')
    plot_weekend_off_distribution(schedule_df, demand_df, out_dir / '04_weekend_off_distribution.png')

    pd.DataFrame(details['rule3'], columns=['Engineer', 'Day', 'DefaultGroup', 'AssignedShift']).to_csv(out_dir / 'detail_rule3_group_mismatch.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(details['rule5'], columns=['Engineer', 'WeekendOffDays', 'ShortageTo4']).to_csv(out_dir / 'detail_rule5_weekend_shortage.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(details['rule6'], columns=['Engineer', 'Day', 'PrevShift', 'NextShift']).to_csv(out_dir / 'detail_rule6_isolated_off.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(details['rule7'], columns=['Engineer', 'StartDay', 'EndDay', 'Type']).to_csv(out_dir / 'detail_rule7_off_blocks.csv', index=False, encoding='utf-8-sig')

    summary = pd.DataFrame([
        ['FontUsed', font_used],
        ['Status', SOLVER_STATUS],
        ['ObjectiveValue', OBJECTIVE_VALUE],
        ['BestBound', BEST_BOUND],
        ['SolveTimeSeconds', SOLVE_TIME_SECONDS],
        ['RecomputedPenalty', round(sum(weighted.values()), 4)],
        ['Rule1Weighted', weighted['rule1']],
        ['Rule2Weighted', weighted['rule2']],
        ['Rule3Weighted', weighted['rule3']],
        ['Rule4Weighted', weighted['rule4']],
        ['Rule5Weighted', weighted['rule5']],
        ['Rule6Weighted', weighted['rule6']],
        ['Rule7Weighted', weighted['rule7']],
    ], columns=['Metric', 'Value'])
    summary.to_csv(out_dir / 'summary_metrics.csv', index=False, encoding='utf-8-sig')

    print('-' * 72)
    print('✅ 視覺化完成')
    print(f'字型: {font_used}')
    print(f'輸出資料夾: {out_dir.resolve()}')
    print(f'重新計算總懲罰值: {sum(weighted.values()):.1f}')
    print('已產出 4 張圖：')
    print('1. 01_dashboard_summary.png')
    print('2. 02_penalty_breakdown.png')
    print('3. 03_schedule_heatmap.png')
    print('4. 04_weekend_off_distribution.png')
    print('-' * 72)


if __name__ == '__main__':
    main()
