#---------- 集合定義----------
set ENGINEERS;                     # 所有工程師
set DAYS = 1..30;                  # 30天排程區間
set SHIFTS = {'D', 'E', 'N', 'O'}; # 班別：早、午、晚、休 
set WORK_SHIFTS = SHIFTS diff {'O'};

# ----------參數定義----------
param DefaultShift{ENGINEERS} symbolic;       # 預設班別群組   
param Demand{DAYS, WORK_SHIFTS} >= 0;         # 每日人力需求  
param IsWeekend{DAYS} binary;                 # 是否為週末    
param PreAssign{ENGINEERS, DAYS} symbolic default ' '; # 預先指定班別  

# ----------決策變數---------- 
var x{ENGINEERS, DAYS, SHIFTS} binary;

# 懲罰項輔助變數 (用於最小化目標函數) 
var p_consec6{ENGINEERS, 1..25} binary;      # 連續上班6天 
var p_night_next{ENGINEERS, 1..29} binary;    # 夜接早/午 
var p_eve_day{ENGINEERS, 1..29} binary;      # 午接早 
var p_day_night{ENGINEERS, 1..29} binary;     # 早/午接夜 
var p_diff_default{ENGINEERS, DAYS} binary;   # 違反預設班別   
var shortfall_off{ENGINEERS} >= 0, integer;  # 月休 < 9天 
var shortfall_weekend{ENGINEERS} >= 0, integer;# 記錄週末休假不足 4 天
var p_single_off{ENGINEERS, 2..29} binary;    # 僅排休1日 

#連續休假問題
# 標記工程師 i 在第 d 天是否開始了一個「連續休假」
var is_consec_start{ENGINEERS, 1..29} binary; 
# 每月連休次數 < 2 的懲罰變數
var p_consec_low{ENGINEERS} binary;



# 目標函數：最小化總懲罰值 
minimize Total_Penalty:
    sum{i in ENGINEERS, d in 1..25} 1 * p_consec6[i,d] +
    sum{i in ENGINEERS, d in 1..29} (1 * p_night_next[i,d] + 1 * p_eve_day[i,d] + 1 * p_day_night[i,d]) +
    sum{i in ENGINEERS, d in DAYS} 0.2 * p_diff_default[i,d] +
    sum{i in ENGINEERS}  (0.1 * shortfall_weekend[i] +0.1 * p_consec_low[i])+
    sum{i in ENGINEERS, d in 2..29} 0.1 * p_single_off[i,d]+
    sum{i in ENGINEERS} 0.1 * shortfall_off[i];
    
# --- 硬限制條件 ---

# 1. 每日每人只能排一個班別 
subject to OneShiftPerDay{i in ENGINEERS, d in DAYS}:
    sum{s in SHIFTS} x[i,d,s] = 1;

# 2. 滿足每日人力需求 
subject to MeetDemand{d in DAYS, s in WORK_SHIFTS}:
    sum{i in ENGINEERS} x[i,d,s] = Demand[d,s];

# 3. 尊重預先指定的班別 
subject to RespectPreAssign{i in ENGINEERS, d in DAYS: PreAssign[i,d] != ' '}:
    x[i,d,PreAssign[i,d]] = 1;

# --- 懲罰邏輯線性化 ---

# 連續上班6天懲罰 
subject to Consec6Limit{i in ENGINEERS, d in 1..25}:
    sum{dd in d..d+5, s in WORK_SHIFTS} x[i,dd,s] <= 5 + p_consec6[i,d];

# 夜接隔早/午 
subject to NightNextRule{i in ENGINEERS, d in 1..29}:
    x[i,d,'N'] + x[i,d+1,'D'] + x[i,d+1,'E'] <= 1 + p_night_next[i,d];
# 午接隔早
subject to EveDayRule{i in ENGINEERS, d in 1..29}:
    x[i,d,'E'] + x[i,d+1,'D'] <= 1 + p_eve_day[i,d];
# 早接隔夜 
subject to DayNightRule{i in ENGINEERS, d in 1..29}:
    x[i,d,'D'] + x[i,d,'E'] + x[i,d+1,'N'] <= 1 + p_day_night[i,d];
    
# 違反預設班別懲罰 
subject to DefaultShiftRule{i in ENGINEERS, d in DAYS}:
    x[i,d,DefaultShift[i]] + x[i,d,'O'] >= 1 - p_diff_default[i,d];

# 月休天數限制是否>9 
subject to TotalOffRule{i in ENGINEERS}:
    sum{d in DAYS} x[i,d,'O'] + shortfall_off[i] >= 9;


#周末休<4天 

subject to WeekendOff_Step1{i in ENGINEERS}:
    sum{d in DAYS: IsWeekend[d] == 1} x[i,d,'O'] <= 3 ==> shortfall_weekend[i] >= 1;

subject to WeekendOff_Step2{i in ENGINEERS}:
    sum{d in DAYS: IsWeekend[d] == 1} x[i,d,'O'] <= 2 ==> shortfall_weekend[i] >= 2;

subject to WeekendOff_Step3{i in ENGINEERS}:
    sum{d in DAYS: IsWeekend[d] == 1} x[i,d,'O'] <= 1 ==> shortfall_weekend[i] >= 3;

subject to WeekendOff_Step4{i in ENGINEERS}:
    sum{d in DAYS: IsWeekend[d] == 1} x[i,d,'O'] <= 0 ==> shortfall_weekend[i] >= 4;

    
#單日休假
# 1. (工作-休-工作) 
subject to SingleOff_Base{i in ENGINEERS, d in 2..29}:
    x[i,d,'O'] <= x[i,d-1,'O'] + x[i,d+1,'O'] + p_single_off[i,d];

# 2. 進階割平面：強迫 p 必須偵測到「休假開始」與「休假結束」的孤立狀態
subject to SingleOff_Left{i in ENGINEERS, d in 2..29}:
    p_single_off[i,d] >= x[i,d,'O'] - x[i,d-1,'O'] - x[i,d+1,'O'];

# 3. 核心加固：防止 0.5 + 0.5 的平攤 (最重要！)
subject to SingleOff_Tight{i in ENGINEERS, d in 2..29}:
    p_single_off[i,d] >= x[i,d,'O'] + (1 - x[i,d-1,'O']) + (1 - x[i,d+1,'O']) - 2;
     
# 4. 原理：總休假天數 = 單日休天數 + 連續休天數因為連續休最少是 2 天，所以 (總休 - 單日休) 必須能被 2 次連休起點覆蓋
subject to Global_Off_Balance{i in ENGINEERS}:
    sum{d in DAYS} x[i,d,'O'] <= sum{d in 2..29} p_single_off[i,d] + 30 * sum{d in 1..29} is_consec_start[i,d];


#連休次數
# 1. 偵測連休起點 
subject to DetectConsec_Logic{i in ENGINEERS, d in 2..29}:
    # 只要其中一個條件不滿足，start 就必須為 0
    3 * is_consec_start[i,d] <= x[i,d,'O'] + x[i,d+1,'O'] + (1 - x[i,d-1,'O']);

subject to DetectConsec_LB{i in ENGINEERS, d in 2..29}:
    # 只有三個條件都滿足，start 才強制為 1
    is_consec_start[i,d] >= x[i,d,'O'] + x[i,d+1,'O'] - x[i,d-1,'O'] - 1;

# 2. 總次數平衡 
subject to ConsecStartUB{i in ENGINEERS}:
    2 * sum{d in 1..29} is_consec_start[i,d] <= sum{d in DAYS} x[i,d,'O'];

# 3. 強化的處罰邏輯 
# 確保只要連休起點總和 < 2，p_consec_low 就必須是 1
subject to ConsecLow_Tight {i in ENGINEERS}:
    2 * p_consec_low[i] >= 2 - sum{d in 1..29} is_consec_start[i,d];

    
# ---------- 強化的前綴順序對稱性限制 (Prefix Ordering) ----------

# Group 1: engineer_6, engineer_7 (D班無預派組)
# 強制讓 6 號在任何時間點累積的休假天數都不少於 7 號
subject to Symmetry_D_Group{d in DAYS}:
    sum{dd in 1..d} x['engineer_6', dd, 'O'] >= sum{dd in 1..d} x['engineer_7', dd, 'O'];

# Group 2: engineer_8, 9, 11, 12 (E班無預派組)
# 形成 8 >= 9 >= 11 >= 12 的前綴鏈
subject to Symmetry_E_Group_8_9{d in DAYS}:
    sum{dd in 1..d} x['engineer_8', dd, 'O'] >= sum{dd in 1..d} x['engineer_9', dd, 'O'];

subject to Symmetry_E_Group_9_11{d in DAYS}:
    sum{dd in 1..d} x['engineer_9', dd, 'O'] >= sum{dd in 1..d} x['engineer_11', dd, 'O'];

subject to Symmetry_E_Group_11_12{d in DAYS}:
    sum{dd in 1..d} x['engineer_11', dd, 'O'] >= sum{dd in 1..d} x['engineer_12', dd, 'O'];

# Group 3: engineer_14, 15 (N班無預派組)
subject to Symmetry_N_Group{d in DAYS}:
    sum{dd in 1..d} x['engineer_14', dd, 'O'] >= sum{dd in 1..d} x['engineer_15', dd, 'O'];
    
