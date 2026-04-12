import csv
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np


random.seed(20260410)

# Mac中文顯示
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

ENGINEER_DATA = [
    ("engineer_1", "D"), ("engineer_2", "D"), ("engineer_3", "D"),
    ("engineer_4", "D"), ("engineer_5", "D"), ("engineer_6", "D"),
    ("engineer_7", "D"), ("engineer_8", "E"), ("engineer_9", "E"),
    ("engineer_10", "E"), ("engineer_11", "E"), ("engineer_12", "E"),
    ("engineer_13", "N"), ("engineer_14", "N"), ("engineer_15", "N")
]

ENGINEER_NAMES = [name for name, _ in ENGINEER_DATA]
DEFAULT_GROUPS = [group for _, group in ENGINEER_DATA]

FIXED_SHIFTS = {
    (0, 0): "E", (0, 1): "O", (1, 0): "O", (2, 0): "O",
    (3, 1): "D", (4, 2): "D", (9, 1): "E", (12, 1): "O", (12, 29): "O"
}

WEEKENDS = [0, 1, 7, 8, 14, 15, 21, 22, 28, 29]
WEEKDAYS_CH = ["六", "日", "一", "二", "三", "四", "五"]

COLORS = {
    "Consec6": "#8B0000",
    "Illegal": "#FF0000",
    "WrongGrp": "#00008B",
    "ConsecOff": "#FFA500",
    "MonthOff": "#FFC0CB",
    "WeekendOff": "#FFFF00",
    "IslandOff": "#90EE90",
    "Normal": "#FFFFFF"
}

DARK_COLORS = [COLORS["Consec6"], COLORS["Illegal"], COLORS["WrongGrp"]]

NUM_ENGINEERS = 15
NUM_DAYS = 30

# 規則判斷
def is_illegal_transition(prev_shift, curr_shift):
    return (
        (prev_shift == "N" and curr_shift in ["D", "E"]) or
        (prev_shift == "E" and curr_shift == "D") or
        (prev_shift in ["D", "E"] and curr_shift == "N")
    )


def count_consec_off_blocks(shift_list):
    count = 0
    day = 0

    while day < NUM_DAYS:
        if shift_list[day] == "O":
            run = 0
            while day < NUM_DAYS and shift_list[day] == "O":
                run += 1
                day += 1
            if run >= 2:
                count += 1
        else:
            day += 1

    return count

# 懲罰值
def calc_hs_eng_penalty(shift_list, default_group):
    hard = 0
    soft = 0
    month_off = 0
    weekend_off = 0

    for day in range(NUM_DAYS):
        shift = shift_list[day]

        if shift == "O":
            month_off += 1
            if day in WEEKENDS:
                weekend_off += 1

            prev_is_off = (day == 0 or shift_list[day - 1] == "O")
            next_is_off = (day == NUM_DAYS - 1 or shift_list[day + 1] == "O")
            if not prev_is_off and not next_is_off:
                soft += 1
        else:
            if shift != default_group:
                soft += 2

    if month_off < 9:
        soft += (9 - month_off)

    if weekend_off < 4:
        soft += (4 - weekend_off)

    if count_consec_off_blocks(shift_list) < 2:
        soft += 1

    for day in range(1, NUM_DAYS):
        if is_illegal_transition(shift_list[day - 1], shift_list[day]):
            hard += 1

    consecutive_work = 0
    for day in range(NUM_DAYS):
        if shift_list[day] != "O":
            consecutive_work += 1
        else:
            if consecutive_work >= 6:
                hard += 1
            consecutive_work = 0

    if consecutive_work >= 6:
        hard += 1

    return hard, soft


def total_hs_penalty(schedule):
    total_hard = 0
    total_soft = 0

    for i in range(NUM_ENGINEERS):
        hard, soft = calc_hs_eng_penalty(schedule[i], DEFAULT_GROUPS[i])
        total_hard += hard
        total_soft += soft

    return total_hard, total_soft


def init_engineer_penalties(schedule):
    engineer_penalties = []
    total_hard = 0
    total_soft = 0

    for i in range(NUM_ENGINEERS):
        hard, soft = calc_hs_eng_penalty(schedule[i], DEFAULT_GROUPS[i])
        engineer_penalties.append([hard, soft])
        total_hard += hard
        total_soft += soft

    return engineer_penalties, total_hard, total_soft


def get_final_metrics(schedule):
    stats = {
        "C6": 0,
        "Relief": 0,
        "Grp": 0,
        "COff": 0,
        "MOff": 0,
        "WOff": 0,
        "Iso": 0
    }

    penalty = 0.0
    color_matrix = [[COLORS["Normal"] for _ in range(NUM_DAYS)] for _ in range(NUM_ENGINEERS)]

    for i in range(NUM_ENGINEERS):
        shift_list = schedule[i]
        month_off = shift_list.count("O")
        weekend_off = sum(1 for d in WEEKENDS if shift_list[d] == "O")

        if month_off < 9:
            lack = 9 - month_off
            stats["MOff"] += lack
            penalty += lack * 0.1
            for day in range(NUM_DAYS):
                if shift_list[day] == "O":
                    color_matrix[i][day] = COLORS["MonthOff"]

        if count_consec_off_blocks(shift_list) < 2:
            stats["COff"] += 1
            penalty += 0.1
            for day in range(NUM_DAYS):
                if shift_list[day] == "O":
                    color_matrix[i][day] = COLORS["ConsecOff"]

        if weekend_off < 4:
            lack = 4 - weekend_off
            stats["WOff"] += lack
            penalty += lack * 0.1
            for day in WEEKENDS:
                if shift_list[day] != "O":
                    color_matrix[i][day] = COLORS["WeekendOff"]

        for day in range(NUM_DAYS):
            shift = shift_list[day]

            if shift == "O":
                prev_is_off = (day == 0 or shift_list[day - 1] == "O")
                next_is_off = (day == NUM_DAYS - 1 or shift_list[day + 1] == "O")

                if not prev_is_off and not next_is_off:
                    stats["Iso"] += 1
                    penalty += 0.1
                    color_matrix[i][day] = COLORS["IslandOff"]
            elif shift != DEFAULT_GROUPS[i]:
                stats["Grp"] += 1
                penalty += 0.2
                color_matrix[i][day] = COLORS["WrongGrp"]

        for day in range(1, NUM_DAYS):
            if is_illegal_transition(shift_list[day - 1], shift_list[day]):
                stats["Relief"] += 1
                penalty += 1.0
                color_matrix[i][day] = COLORS["Illegal"]

        consecutive_work = 0
        block_start = 0

        for day in range(NUM_DAYS):
            if shift_list[day] != "O":
                if consecutive_work == 0:
                    block_start = day
                consecutive_work += 1
            else:
                if consecutive_work >= 6:
                    stats["C6"] += 1
                    penalty += 1.0
                    for work_day in range(block_start, day):
                        color_matrix[i][work_day] = COLORS["Consec6"]
                consecutive_work = 0

        if consecutive_work >= 6:
            stats["C6"] += 1
            penalty += 1.0
            for work_day in range(block_start, NUM_DAYS):
                color_matrix[i][work_day] = COLORS["Consec6"]

    return round(penalty, 2), stats, color_matrix


# 初始解班表
def smart_initialize():
    schedule = [[None for _ in range(NUM_DAYS)] for _ in range(NUM_ENGINEERS)]

    for (eng_idx, day), shift in FIXED_SHIFTS.items():
        schedule[eng_idx][day] = shift

    for day in range(NUM_DAYS):
        required_d = 4 if day in WEEKENDS else 5
        needed = {
            "N": 2,
            "E": 3,
            "D": required_d,
            "O": NUM_ENGINEERS - required_d - 5
        }

        for eng_idx in range(NUM_ENGINEERS):
            if schedule[eng_idx][day] is not None:
                needed[schedule[eng_idx][day]] -= 1

        unassigned = [i for i in range(NUM_ENGINEERS) if schedule[i][day] is None]

        for shift_type in ["N", "E", "D", "O"]:
            while needed[shift_type] > 0:
                matched = [i for i in unassigned if DEFAULT_GROUPS[i] == shift_type]
                chosen = random.choice(matched) if matched else random.choice(unassigned)
                schedule[chosen][day] = shift_type
                unassigned.remove(chosen)
                needed[shift_type] -= 1

    return schedule


def generate_neighbor(schedule):
    for _ in range(30):
        group = random.choice(["D", "E", "N"])
        group_engineers = [i for i in range(NUM_ENGINEERS) if DEFAULT_GROUPS[i] == group]
        rand = random.random()

        if rand < 0.5:
            e1, e2 = random.sample(group_engineers, 2)
            day1 = random.randint(0, NUM_DAYS - 1)
            s1, s2 = schedule[e1][day1], schedule[e2][day1]

            if s1 != s2 and (e1, day1) not in FIXED_SHIFTS and (e2, day1) not in FIXED_SHIFTS:
                valid_days = [
                    d for d in range(NUM_DAYS)
                    if schedule[e1][d] == s2
                    and schedule[e2][d] == s1
                    and (e1, d) not in FIXED_SHIFTS
                    and (e2, d) not in FIXED_SHIFTS
                ]
                if valid_days:
                    return ("teleport", e1, e2, day1, random.choice(valid_days))

        elif rand < 0.8:
            e1, e2 = random.sample(group_engineers, 2)
            span = random.choice([1, 1, 1, 2, 2, 3, 4, 5])
            day = random.randint(0, NUM_DAYS - span)
            return ("2way", e1, e2, day, span)

        else:
            if len(group_engineers) >= 3:
                e1, e2, e3 = random.sample(group_engineers, 3)
                span = random.choice([1, 1, 2])
                day = random.randint(0, NUM_DAYS - span)
                return ("3way", e1, e2, e3, day, span)

    return ("2way", 0, 1, 0, 1)


def get_affected_engineers(op):
    if op[0] == "2way":
        _, e1, e2, _, _ = op
        return [e1, e2]

    if op[0] == "3way":
        _, e1, e2, e3, _, _ = op
        return [e1, e2, e3]

    if op[0] == "teleport":
        _, e1, e2, _, _ = op
        return [e1, e2]

    return []


def apply_op(schedule, op):
    op_type = op[0]

    if op_type == "2way":
        _, e1, e2, day, span = op

        if any((e, day + k) in FIXED_SHIFTS for e in (e1, e2) for k in range(span)):
            return False

        schedule[e1][day:day + span], schedule[e2][day:day + span] = (
            schedule[e2][day:day + span][:],
            schedule[e1][day:day + span][:]
        )
        return True

    if op_type == "3way":
        _, e1, e2, e3, day, span = op

        if any((e, day + k) in FIXED_SHIFTS for e in (e1, e2, e3) for k in range(span)):
            return False

        schedule[e1][day:day + span], schedule[e2][day:day + span], schedule[e3][day:day + span] = (
            schedule[e3][day:day + span][:],
            schedule[e1][day:day + span][:],
            schedule[e2][day:day + span][:]
        )
        return True

    if op_type == "teleport":
        _, e1, e2, day1, day2 = op
        schedule[e1][day1], schedule[e2][day1] = schedule[e2][day1], schedule[e1][day1]
        schedule[e1][day2], schedule[e2][day2] = schedule[e2][day2], schedule[e1][day2]
        return True

    return False


def undo_op(schedule, op):
    op_type = op[0]

    if op_type == "2way":
        _, e1, e2, day, span = op
        schedule[e1][day:day + span], schedule[e2][day:day + span] = (
            schedule[e2][day:day + span][:],
            schedule[e1][day:day + span][:]
        )

    elif op_type == "3way":
        _, e1, e2, e3, day, span = op
        schedule[e1][day:day + span], schedule[e2][day:day + span], schedule[e3][day:day + span] = (
            schedule[e2][day:day + span][:],
            schedule[e3][day:day + span][:],
            schedule[e1][day:day + span][:]
        )

    elif op_type == "teleport":
        _, e1, e2, day1, day2 = op
        schedule[e1][day1], schedule[e2][day1] = schedule[e2][day1], schedule[e1][day1]
        schedule[e1][day2], schedule[e2][day2] = schedule[e2][day2], schedule[e1][day2]


def lexicographic_sa(schedule, max_iter=180000, initial_temp=3.0):
    engineer_penalties, current_h, current_s = init_engineer_penalties(schedule)
    best_schedule = [row[:] for row in schedule]
    best_h, best_s = current_h, current_s

    temp = initial_temp
    alpha = 0.99997
    no_improve = 0

    for _ in range(max_iter):
        op = generate_neighbor(schedule)

        if not apply_op(schedule, op):
            continue

        affected = get_affected_engineers(op)

        old_h_sum = 0
        old_s_sum = 0
        for i in affected:
            old_h_sum += engineer_penalties[i][0]
            old_s_sum += engineer_penalties[i][1]

        new_values = []
        new_h_sum = 0
        new_s_sum = 0
        for i in affected:
            hard, soft = calc_hs_eng_penalty(schedule[i], DEFAULT_GROUPS[i])
            new_values.append((i, hard, soft))
            new_h_sum += hard
            new_s_sum += soft

        new_h = current_h - old_h_sum + new_h_sum
        new_s = current_s - old_s_sum + new_s_sum

        accept = False

        if new_h < current_h:
            accept = True
        elif new_h == current_h:
            if new_s < current_s:
                accept = True
            else:
                diff = new_s - current_s
                if random.random() < math.exp(-diff / temp):
                    accept = True
        else:
            diff = new_h - current_h
            if random.random() < math.exp(-(diff * 20) / temp):
                accept = True

        if accept:
            current_h, current_s = new_h, new_s

            for i, hard, soft in new_values:
                engineer_penalties[i] = [hard, soft]

            if (current_h < best_h) or (current_h == best_h and current_s < best_s):
                best_h, best_s = current_h, current_s
                best_schedule = [row[:] for row in schedule]
                no_improve = 0
            else:
                no_improve += 1
        else:
            undo_op(schedule, op)
            no_improve += 1

        temp = max(0.0005, temp * alpha)

        if no_improve > 15000:
            temp = random.uniform(1.0, 3.5)
            no_improve = 0
            schedule = [row[:] for row in best_schedule]
            engineer_penalties, current_h, current_s = init_engineer_penalties(schedule)

        if best_h == 0 and best_s <= 6:
            break

    return best_schedule, best_h, best_s

def solve():
    best_overall_schedule = None
    best_overall_score = (999, 999)

    print("Generating schedule... Please wait.")

    for restart in range(30):
        schedule = smart_initialize()
        best_schedule, best_h, best_s = lexicographic_sa(schedule, max_iter=180000, initial_temp=3.0)

        real_score = best_h * 1.0 + best_s * 0.1
        print(f"  > 第{restart + 1}輪結束: 懲罰值為{round(real_score, 2)}")

        if (best_h < best_overall_score[0]) or (
            best_h == best_overall_score[0] and best_s < best_overall_score[1]
        ):
            best_overall_score = (best_h, best_s)
            best_overall_schedule = [row[:] for row in best_schedule]

            if best_h == 0 and best_s <= 6:
                break

    return best_overall_schedule


# 主程式
if __name__ == "__main__":
    start_time = time.time()

    final_schedule = solve()
    final_penalty, final_stats, color_matrix = get_final_metrics(final_schedule)
    duration = round(time.time() - start_time, 1)

    print("\n=== 懲罰值細項 ===")
    for i in range(NUM_ENGINEERS):
        hard, soft = calc_hs_eng_penalty(final_schedule[i], DEFAULT_GROUPS[i])
        score = hard * 10 + soft
        if score > 0:
            print(f"{ENGINEER_NAMES[i]} ({DEFAULT_GROUPS[i]}): {score / 10.0} 分")

    # 圖
    fig = plt.figure(figsize=(28, 14))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.5, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    rgb_img = np.zeros((NUM_ENGINEERS, NUM_DAYS, 3))
    for i in range(NUM_ENGINEERS):
        for day in range(NUM_DAYS):
            rgb_img[i, day] = mc.to_rgb(color_matrix[i][day])

    ax1.imshow(rgb_img, aspect="auto")

    for i in range(NUM_ENGINEERS):
        for day in range(NUM_DAYS):
            bg = color_matrix[i][day]
            txt_color = "white" if bg in DARK_COLORS else "black"
            is_fixed = (i, day) in FIXED_SHIFTS
            font_weight = "bold" if is_fixed or bg != COLORS["Normal"] else "normal"

            ax1.text(
                day,
                i,
                final_schedule[i][day],
                ha="center",
                va="center",
                color=txt_color,
                fontweight=font_weight,
                fontsize=12
            )

    ax1.set_xticks(np.arange(NUM_DAYS))
    x_labels = [
        f"D{day + 1}\n({WEEKDAYS_CH[day % 7]})\n\nD:{4 if day in WEEKENDS else 5}\nE:3\nN:2"
        for day in range(NUM_DAYS)
    ]
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.xaxis.tick_top()

    ax1.set_yticks(np.arange(NUM_ENGINEERS))
    ax1.set_yticklabels([f"{ENGINEER_NAMES[i]} ({DEFAULT_GROUPS[i]})" for i in range(NUM_ENGINEERS)])

    ax1.set_title(f"班表 (懲罰值: {final_penalty})", pad=100, fontsize=24, fontweight="bold")
    ax1.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)
    ax1.set_xticks(np.arange(-0.5, NUM_DAYS, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, NUM_ENGINEERS, 1), minor=True)

    ax2.axis("off")

    rows = [
        ["演算法資訊", ""],
        ["執行時間", f"{duration}s"],
        ["最佳化狀態", "Lex SA"],
        ["懲罰值", f"{final_penalty}"],
        ["", ""],
        ["排班規則檢查", "違反次數"],
        ["+1連續上六天班", f"{int(final_stats['C6'])}"],
        ["+1晚接早午,午接早", f"{int(final_stats['Relief'])}"],
        ["+0.2違反預設班別", f"{int(final_stats['Grp'])}"],
        ["+0.1月連休 < 2 次", f"{int(final_stats['COff'])}"],
        ["+0.1月休 < 9 天", f"{int(final_stats['MOff'])}"],
        ["+0.1週末休 < 4 天", f"{int(final_stats['WOff'])}"],
        ["+0.1僅排休 1 日", f"{int(final_stats['Iso'])}"]
    ]

    table = ax2.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        colWidths=[0.6, 0.3],
        bbox=[0.05, 0.1, 0.9, 0.8]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(15)

    color_map = {
        6: COLORS["Consec6"],
        7: COLORS["Illegal"],
        8: COLORS["WrongGrp"],
        9: COLORS["ConsecOff"],
        10: COLORS["MonthOff"],
        11: COLORS["WeekendOff"],
        12: COLORS["IslandOff"]
    }

    cells = table.get_celld()
    for row in range(13):
        if row in [0, 5]:
            cells[(row, 0)].set_facecolor("#F0F0F0")
            cells[(row, 1)].set_facecolor("#F0F0F0")
            cells[(row, 0)].get_text().set_weight("bold")
            cells[(row, 1)].get_text().set_weight("bold")

        if row in color_map:
            bg_color = color_map[row]
            cells[(row, 0)].set_facecolor(bg_color)
            cells[(row, 1)].set_facecolor(bg_color)

            if bg_color in DARK_COLORS:
                cells[(row, 0)].get_text().set_color("white")
                cells[(row, 1)].get_text().set_color("white")
                cells[(row, 0)].get_text().set_weight("bold")
                cells[(row, 1)].get_text().set_weight("bold")


    # 存檔
    save_path_png = os.path.join(os.path.expanduser("~"), "Desktop", "SA_REPORT.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")

    save_path_csv = os.path.join(os.path.expanduser("~"), "Desktop", "SA_REPORT.csv")
    with open(save_path_csv, mode="w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["Engineer", "Group"] + [f"D{d + 1} ({WEEKDAYS_CH[d % 7]})" for d in range(NUM_DAYS)])
        for i in range(NUM_ENGINEERS):
            writer.writerow([ENGINEER_NAMES[i], DEFAULT_GROUPS[i]] + final_schedule[i])

    print(
        f"\n✅ 執行完畢，耗時{duration}秒。班表已生成請至桌面查看檔案：\n"
        f" - 圖片: {save_path_png}\n"
        f" - 表格: {save_path_csv}"
    )
