# ==========================================
# 1. KHAI BÁO THƯ VIỆN VÀ TẢI MÔ HÌNH ĐÃ TRAIN
# ==========================================
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import lightgbm as lgb
import json
import os
import argparse

MODEL_PATH = 'lgb_gold_triple_barrier.txt'
model_lgb = lgb.Booster(model_file=MODEL_PATH)
HISTORY_PATH = 'weekly_eval_history.json'

# ==========================================
# 2. THAM SỐ DÒNG LỆNH — CHO PHÉP CHẠY TỰ ĐỘNG KHÔNG CẦN NHẬP TAY
# ==========================================
# FIX: input() sẽ treo vô thời hạn nếu script chạy qua cron/Task
# Scheduler không có người ngồi trực. Thêm cờ --auto để bỏ qua hoàn
# toàn bước hỏi, luôn lấy 7 ngày gần nhất, dùng cho lịch tự động.
# Không truyền cờ gì -> giữ nguyên hành vi hỏi tương tác như cũ, dùng
# khi bạn tự kiểm tra/backtest thủ công.
parser = argparse.ArgumentParser()
parser.add_argument('--auto', action='store_true',
                     help='Chạy tự động, không hỏi input, luôn lấy 7 ngày gần nhất.')
args = parser.parse_args()

if not mt5.initialize():
    print("Kết nối MT5 thất bại, vui lòng kiểm tra lại!")
    mt5.shutdown()
    raise SystemExit("Dừng script vì không kết nối được MT5.")

# ==========================================
# 3. CẤU HÌNH THỜI GIAN
# ==========================================
hien_tai = pd.Timestamp.now()
dau_tuan_truoc = hien_tai - pd.Timedelta(days=7)
start_date_eval = dau_tuan_truoc
end_date_eval = hien_tai

# FIX: đánh dấu rõ record này thuộc chế độ nào. "auto" = giám sát định
# kỳ, dùng để quyết định retrain. "manual" = backtest/kiểm tra tùy ý,
# KHÔNG được tính vào rolling average quyết định retrain.
eval_mode = "auto"

if args.auto:
    print(f"=> [--auto] Chạy tự động: lấy 7 ngày gần nhất "
          f"({start_date_eval.strftime('%Y-%m-%d')} -> {end_date_eval.strftime('%Y-%m-%d')})")
else:
    print("\n=== CẤU HÌNH THỜI GIAN KIỂM TRA ĐÁNH GIÁ ===")
    print("1. Chạy tự động (Lấy dữ liệu 7 ngày gần nhất) - Nhấn ENTER để chọn")
    print("2. Nhập khoảng thời gian tùy chỉnh thủ công")

    luong_chon = input("Nhập lựa chọn của bạn (Ấn Enter hoặc gõ 2): ").strip()

    if luong_chon == "2":
        eval_mode = "manual"  # FIX: mọi lần nhập tay đều là backtest, không phải giám sát
        while True:
            dinh_dang_nhap = input("Nhập khoảng thời gian (Ví dụ: 20240101-20240118): ").strip()
            try:
                start_str, end_str = dinh_dang_nhap.split('-')
                start_date_eval = pd.to_datetime(start_str.strip(), format='%Y%m%d')
                end_date_eval = pd.to_datetime(end_str.strip(), format='%Y%m%d') + pd.Timedelta(hours=23, minutes=59)

                # FIX nhỏ: validate start < end ngay từ đầu thay vì để rơi
                # xuống báo lỗi mơ hồ ở bước dropna phía sau.
                if start_date_eval >= end_date_eval:
                    print("❌ Ngày bắt đầu phải nhỏ hơn ngày kết thúc. Nhập lại.")
                    continue

                print(f"=> Chế độ tùy chỉnh (backtest thủ công, KHÔNG tính vào giám sát retrain): "
                      f"{start_date_eval.strftime('%Y-%m-%d')} đến {end_date_eval.strftime('%Y-%m-%d')}")
                break
            except Exception:
                print("❌ Định dạng nhập vào bị sai! Vui lòng nhập đúng dạng YYYYMMDD-YYYYMMDD "
                      "(Ví dụ: 20240101-20240118).")
    else:
        print(f"=> Chế độ tự động: 7 ngày gần nhất "
              f"({start_date_eval.strftime('%Y-%m-%d')} đến {end_date_eval.strftime('%Y-%m-%d')})")

# ==========================================
# 4. CÀO DỮ LIỆU TỪ MT5 VỚI OFFSET MỒI
# ==========================================
SYMBOL = "XAUUSD"
print(f"\nĐang cào dữ liệu của {SYMBOL} từ MT5...")

def get_clean_mt5_data(symbol, timeframe, start_date, end_date):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise SystemExit(f"❌ Không lấy được dữ liệu cho khung {timeframe}. Hãy kiểm tra lịch sử giá trên MT5.")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df.sort_values('time').reset_index(drop=True)

offset_days = pd.Timedelta(days=15)
df_m5  = get_clean_mt5_data(SYMBOL, mt5.TIMEFRAME_M5,  start_date_eval - offset_days, end_date_eval)
df_m15 = get_clean_mt5_data(SYMBOL, mt5.TIMEFRAME_M15, start_date_eval - offset_days, end_date_eval)
df_h1  = get_clean_mt5_data(SYMBOL, mt5.TIMEFRAME_H1,  start_date_eval - offset_days, end_date_eval)
df_h4  = get_clean_mt5_data(SYMBOL, mt5.TIMEFRAME_H4,  start_date_eval - offset_days, end_date_eval)

mt5.shutdown()

# ==========================================
# 5. TÍNH TOÁN FEATURE TƯƠNG ĐỐI
# ==========================================
print("Đang đồng bộ chỉ báo theo cấu trúc hệ thống...")

df_m5.ta.ema(length=20, append=True)
df_m5.ta.rsi(length=14, append=True)
df_m5.ta.atr(length=14, append=True)
df_m5.ta.bands(length=20, std=2, append=True)

df_m5['ret_close'] = df_m5['close'].pct_change()
df_m5['pct_body'] = (df_m5['close'] - df_m5['open']) / df_m5['open']
df_m5['atr_norm_upper_shadow'] = (df_m5['high'] - df_m5[['open', 'close']].max(axis=1)) / df_m5['ATRr_14']
df_m5['atr_norm_lower_shadow'] = (df_m5[['open', 'close']].min(axis=1) - df_m5['low']) / df_m5['ATRr_14']
df_m5['dist_EMA20_pct'] = (df_m5['close'] - df_m5['EMA_20']) / df_m5['EMA_20']
df_m5['dist_BBU_pct'] = (df_m5['BBU_20_2.0'] - df_m5['close']) / df_m5['close']
df_m5['dist_BBL_pct'] = (df_m5['close'] - df_m5['BBL_20_2.0']) / df_m5['close']

df_m15.ta.ema(length=20, append=True)
df_m15.ta.macd(append=True)
df_m15['dist_EMA20_M15_pct'] = (df_m15['close'] - df_m15['EMA_20']) / df_m15['EMA_20']
df_m15['macd_norm'] = df_m15['MACD_12_26_9'] / df_m15['close']
df_m15_features = df_m15[['time', 'dist_EMA20_M15_pct', 'macd_norm']]

df_h1.ta.ema(length=20, append=True)
df_h1.ta.rsi(length=14, append=True)
df_h1['dist_EMA20_H1_pct'] = (df_h1['close'] - df_h1['EMA_20']) / df_h1['EMA_20']
df_h1_features = df_h1[['time', 'dist_EMA20_H1_pct', 'RSI_14']].rename(columns={'RSI_14': 'RSI_14_H1'})

df_h4.ta.ema(length=20, append=True)
df_h4.ta.rsi(length=14, append=True)
df_h4['dist_EMA20_H4_pct'] = (df_h4['close'] - df_h4['EMA_20']) / df_h4['EMA_20']
df_h4_features = df_h4[['time', 'dist_EMA20_H4_pct', 'RSI_14']].rename(columns={'RSI_14': 'RSI_14_H4'})

merged_df = pd.merge_asof(df_m5, df_m15_features, on='time', direction='backward')
merged_df = pd.merge_asof(merged_df, df_h1_features, on='time', direction='backward')
merged_df = pd.merge_asof(merged_df, df_h4_features, on='time', direction='backward')

merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================
# 6. GÁN NHÃN TRIPLE BARRIER ĐỐI XỨNG HAI CHIỀU
# ==========================================
print("Đang gán nhãn Triple-Barrier riêng cho chiều MUA và chiều BÁN...")

high_prices  = merged_df['high'].values
low_prices   = merged_df['low'].values
close_prices = merged_df['close'].values
atrs         = merged_df['ATRr_14'].values
n = len(merged_df)

PT_SL = [3.0, 2.0]
MAX_HOLD = 12

labels_long  = np.full(n, np.nan)
labels_short = np.full(n, np.nan)

for i in range(n - MAX_HOLD):
    if np.isnan(atrs[i]):
        continue

    entry_price = close_prices[i]
    atr_now = atrs[i]

    tp_long = entry_price + PT_SL[0] * atr_now
    sl_long = entry_price - PT_SL[1] * atr_now
    tp_short = entry_price - PT_SL[0] * atr_now
    sl_short = entry_price + PT_SL[1] * atr_now

    label_long_i  = 0.0
    label_short_i = 0.0
    long_done  = False
    short_done = False

    for j in range(1, MAX_HOLD + 1):
        curr_high = high_prices[i + j]
        curr_low  = low_prices[i + j]

        if not long_done:
            if curr_low <= sl_long and curr_high >= tp_long:
                label_long_i = -1.0
                long_done = True
            elif curr_low <= sl_long:
                label_long_i = -1.0
                long_done = True
            elif curr_high >= tp_long:
                label_long_i = 1.0
                long_done = True

        if not short_done:
            if curr_high >= sl_short and curr_low <= tp_short:
                label_short_i = -1.0
                short_done = True
            elif curr_high >= sl_short:
                label_short_i = -1.0
                short_done = True
            elif curr_low <= tp_short:
                label_short_i = 1.0
                short_done = True

        if long_done and short_done:
            break

    labels_long[i]  = label_long_i
    labels_short[i] = label_short_i

merged_df['target_long']  = labels_long
merged_df['target_short'] = labels_short

weekly_df = merged_df[(merged_df['time'] >= start_date_eval) & (merged_df['time'] <= end_date_eval)].dropna().copy()

# ==========================================
# 7. ÉP AI PHÁN ĐOÁN VÀ ĐÁNH GIÁ CHẤT LƯỢNG KỲ LỌC
# ==========================================
if len(weekly_df) == 0:
    print("❌ Không có đủ dữ liệu trong khoảng thời gian đã chọn sau khi dropna. Dừng tiến trình!")
    raise SystemExit()

feature_cols = [
    'volume', 'spread', 'RSI_14',
    'ret_close', 'pct_body', 'atr_norm_upper_shadow', 'atr_norm_lower_shadow',
    'dist_EMA20_pct', 'dist_BBU_pct', 'dist_BBL_pct',
    'dist_EMA20_M15_pct', 'macd_norm',
    'dist_EMA20_H1_pct', 'RSI_14_H1',
    'dist_EMA20_H4_pct', 'RSI_14_H4'
]

X_weekly = weekly_df[feature_cols]

pred_probs = model_lgb.predict(X_weekly)
prob_giam = pred_probs[:, 0]
prob_tang = pred_probs[:, 2]

PROB_THRESHOLD = 0.65
signal = np.zeros(len(weekly_df))
signal[prob_tang >= PROB_THRESHOLD] = 1.0
signal[prob_giam >= PROB_THRESHOLD] = -1.0

trades_executed = signal != 0
total_trades = int(np.sum(trades_executed))

print("\n" + "="*60)
print(f"BÁO CÁO SỨC KHỎE MÔ HÌNH TRONG GIAI ĐOẠN [{eval_mode.upper()}]: "
      f"({start_date_eval.strftime('%Y-%m-%d')} -> {end_date_eval.strftime('%Y-%m-%d')})")
print("="*60)
print(f"-> Tổng số cơ hội nến M5 xuất hiện: {len(weekly_df)}")
print(f"-> Số lệnh thực tế Bot phát sinh (đạt ngưỡng >= 65%): {total_trades}")

POINT_VALUE = 0.01
avg_atr = weekly_df['ATRr_14'].mean()
avg_spread_cost_atr = (weekly_df['spread'] * POINT_VALUE).mean() / avg_atr if avg_atr > 0 else 0.0

# FIX: thêm field "mode" để phân biệt auto (giám sát định kỳ) vs
# manual (backtest tùy ý) khi tổng hợp lịch sử.
result_record = {
    "mode": eval_mode,
    "period_start": start_date_eval.strftime('%Y-%m-%d'),
    "period_end": end_date_eval.strftime('%Y-%m-%d'),
    "total_candles": len(weekly_df),
    "total_trades": total_trades,
}

if total_trades > 0:
    idx_buy  = trades_executed & (signal == 1.0)
    idx_sell = trades_executed & (signal == -1.0)

    y_buy_real  = weekly_df.loc[idx_buy, 'target_long'].values
    y_sell_real = weekly_df.loc[idx_sell, 'target_short'].values

    n_buy, n_sell = len(y_buy_real), len(y_sell_real)
    win_buy  = np.sum(y_buy_real == 1.0)
    win_sell = np.sum(y_sell_real == 1.0)

    win_rate_buy  = win_buy / n_buy if n_buy > 0 else np.nan
    win_rate_sell = win_sell / n_sell if n_sell > 0 else np.nan
    overall_win_rate = (win_buy + win_sell) / total_trades

    def pnl_from_labels(y_real):
        pnl = np.where(y_real == 1.0, 3.0, np.where(y_real == -1.0, -2.0, -0.5))
        return pnl - avg_spread_cost_atr

    pnl_all = np.concatenate([pnl_from_labels(y_buy_real), pnl_from_labels(y_sell_real)])
    expectancy_atr = np.mean(pnl_all)

    print(f"-> Lệnh BUY:  {n_buy} lệnh | Win-rate: {win_rate_buy:.2%}" if n_buy > 0 else "-> Lệnh BUY: 0 lệnh")
    print(f"-> Lệnh SELL: {n_sell} lệnh | Win-rate: {win_rate_sell:.2%}" if n_sell > 0 else "-> Lệnh SELL: 0 lệnh")
    print(f"-> Win-rate tổng: {overall_win_rate:.2%}")
    print(f"-> Chi phí spread trung bình quy đổi: {avg_spread_cost_atr:.4f} * ATR / lệnh")
    print(f"-> Kỳ vọng lợi nhuận toán học (Expectancy, đã trừ spread): {expectancy_atr:.4f} * ATR / mỗi lệnh")

    result_record.update({
        "n_buy": int(n_buy), "n_sell": int(n_sell),
        "win_rate_buy": float(win_rate_buy) if n_buy > 0 else None,
        "win_rate_sell": float(win_rate_sell) if n_sell > 0 else None,
        "overall_win_rate": float(overall_win_rate),
        "expectancy_atr": float(expectancy_atr),
    })
else:
    print("-> Không có lệnh nào đủ điều kiện vượt ngưỡng 65%. Bot đứng ngoài bảo toàn vốn.")
    result_record.update({"n_buy": 0, "n_sell": 0, "expectancy_atr": None})

# ==========================================
# 8. TÍCH LŨY LỊCH SỬ — CHỈ RÚT KẾT LUẬN TỪ CÁC KỲ "AUTO"
# ==========================================
history = []
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'r', encoding='utf-8') as f:
        history = json.load(f)

history.append(result_record)
with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

ROLLING_WEEKS = 6

# FIX #1: chỉ lấy các record mode == "auto" -> không để backtest thủ
# công làm méo quyết định retrain.
auto_records = [h for h in history if h.get("mode") == "auto" and h.get("expectancy_atr") is not None]
recent = auto_records[-ROLLING_WEEKS:]

print("\n" + "="*60)
print(f"ĐÁNH GIÁ TÍCH LŨY {len(recent)} KỲ GIÁM SÁT TỰ ĐỘNG GẦN NHẤT "
      f"(đã loại các kỳ backtest thủ công)")
print("="*60)

if len(recent) < 3:
    print(f"=> Chưa đủ dữ liệu tích lũy (chỉ có {len(recent)} kỳ auto có lệnh). "
          f"CHƯA đưa ra kết luận retrain, cần tiếp tục theo dõi thêm.")
else:
    # FIX #2: tính trung bình có trọng số theo số lệnh mỗi kỳ, thay vì
    # trung bình đơn giản giữa các kỳ có số lệnh rất khác nhau.
    weights = np.array([h["n_buy"] + h["n_sell"] for h in recent], dtype=float)
    expectancies = np.array([h["expectancy_atr"] for h in recent])
    win_rates = np.array([h["overall_win_rate"] for h in recent])

    total_w = weights.sum()
    avg_expectancy = np.sum(expectancies * weights) / total_w if total_w > 0 else np.mean(expectancies)
    avg_win_rate = np.sum(win_rates * weights) / total_w if total_w > 0 else np.mean(win_rates)

    print(f"-> Tổng số lệnh trong {len(recent)} kỳ: {int(total_w)}")
    print(f"-> Expectancy trung bình (có trọng số theo số lệnh): {avg_expectancy:.4f} * ATR/lệnh")
    print(f"-> Win-rate trung bình (có trọng số theo số lệnh): {avg_win_rate:.2%}")

    if avg_expectancy > 0:
        print("=> KẾT LUẬN: Mô hình ổn định theo xu hướng nhiều kỳ giám sát. Tiếp tục chạy.")
    else:
        print("=> KẾT LUẬN: Expectancy trung bình (có trọng số) đang âm. "
              "Cần rà soát hoặc tiến hành Retrain.")
print("="*60)