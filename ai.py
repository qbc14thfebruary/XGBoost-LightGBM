# ==========================================
# 1. CÀI ĐẶT VÀ KHAI BÁO THƯ VIỆN
# ==========================================
!pip install pandas numpy pandas-ta-classic lightgbm xgboost

import pandas as pd
import numpy as np
import pandas_ta_classic  as ta
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# ==========================================
# 2. ĐỌC DỮ LIỆU ĐA KHUNG THỜI GIAN
# ==========================================
def load_and_clean(filepath):
    df = pd.read_csv(filepath, parse_dates=['time'])
    return df.sort_values('time').reset_index(drop=True)

print("Đang tải dữ liệu gốc cho Vàng...")
df_m5  = load_and_clean('gold_m5.csv')
df_m15 = load_and_clean('gold_m15.csv')
df_h1  = load_and_clean('gold_h1.csv')
df_h4  = load_and_clean('gold_h4.csv')

# ==========================================
# 3. FEATURE ENGINEERING VỚI GIÁ TƯƠNG ĐỐI (STATIONARY FEATURES)
# ==========================================
print("Đang tính toán các đặc trưng tương đối (Loại bỏ giá tuyệt đối)...")

# TÍNH TOÁN TRÊN KHUNG M5 (Khung vào lệnh gốc)
df_m5.ta.ema(length=20, append=True)
df_m5.ta.rsi(length=14, append=True)
df_m5.ta.atr(length=14, append=True)
df_m5.ta.bands(length=20, std=2, append=True)

# Chuyển đổi các đặc trưng M5 sang dạng STATIONARY (Tương đối)
df_m5['ret_close'] = df_m5['close'].pct_change()
df_m5['pct_body'] = (df_m5['close'] - df_m5['open']) / df_m5['open']
df_m5['atr_norm_upper_shadow'] = (df_m5['high'] - df_m5[['open', 'close']].max(axis=1)) / df_m5['ATRr_14']
df_m5['atr_norm_lower_shadow'] = (df_m5[['open', 'close']].min(axis=1) - df_m5['low']) / df_m5['ATRr_14']
df_m5['dist_EMA20_pct'] = (df_m5['close'] - df_m5['EMA_20']) / df_m5['EMA_20']
df_m5['dist_BBU_pct'] = (df_m5['BBU_20_2.0'] - df_m5['close']) / df_m5['close']
df_m5['dist_BBL_pct'] = (df_m5['close'] - df_m5['BBL_20_2.0']) / df_m5['close']

# TÍNH TOÁN TRÊN KHUNG M15
df_m15.ta.ema(length=20, append=True)
df_m15.ta.macd(append=True)
df_m15['dist_EMA20_M15_pct'] = (df_m15['close'] - df_m15['EMA_20']) / df_m15['EMA_20']
df_m15['macd_norm'] = df_m15['MACD_12_26_9'] / df_m15['close'] # Chuẩn hóa theo tỷ lệ giá
df_m15_features = df_m15[['time', 'dist_EMA20_M15_pct', 'macd_norm']]

# TÍNH TOÁN TRÊN KHUNG H1
df_h1.ta.ema(length=20, append=True)
df_h1.ta.rsi(length=14, append=True)
df_h1['dist_EMA20_H1_pct'] = (df_h1['close'] - df_h1['EMA_20']) / df_h1['EMA_20']
df_h1_features = df_h1[['time', 'dist_EMA20_H1_pct', 'RSI_14']]
df_h1_features = df_h1_features.rename(columns={'RSI_14': 'RSI_14_H1'})

# TÍNH TOÁN TRÊN KHUNG H4
df_h4.ta.ema(length=20, append=True)
df_h4.ta.rsi(length=14, append=True)
df_h4['dist_EMA20_H4_pct'] = (df_h4['close'] - df_h4['EMA_20']) / df_h4['EMA_20']
df_h4_features = df_h4[['time', 'dist_EMA20_H4_pct', 'RSI_14']]
df_h4_features = df_h4_features.rename(columns={'RSI_14': 'RSI_14_H4'})

# GỘP ĐA KHUNG THỜI GIAN
merged_df = pd.merge_asof(df_m5, df_m15_features, on='time', direction='backward')
merged_df = pd.merge_asof(merged_df, df_h1_features, on='time', direction='backward')
merged_df = pd.merge_asof(merged_df, df_h4_features, on='time', direction='backward')

# ==========================================
# 4. GÁN NHÃN TRIPLE-BARRIER METHOD (DỰA TRÊN SL/TP THỰC TẾ)
# ==========================================
print("Đang gán nhãn dữ liệu bằng thuật toán Triple-Barrier...")

def apply_triple_barrier(df, pt_sl=[3.0, 2.0], max_holding_candles=12):
    """
    HÀM GÁN NHÃN (Đã vá lỗi từ bản trước):
      - FIX #1 (đã có từ bản kỹ sư gửi): dùng High/Low để check va chạm rào cản,
        không chỉ dùng giá đóng cửa (close) -> phản ánh đúng việc bị quét râu nến.
      - FIX #2 (MỚI): các dòng cuối cùng không đủ dữ liệu tương lai để xác nhận
        nhãn thật thì gán NaN thay vì mặc định 0 (Sideway). Trước đây dùng
        np.zeros() nên các dòng "chưa biết nhãn" bị hiểu nhầm thành "Sideway
        thật", và dropna() phía sau không loại được vì giá trị là 0.0 chứ
        không phải NaN.
      - Đổi tên hàm khớp với dòng gọi (trước đây định nghĩa
        apply_triple_barrier nhưng gọi apply_triple_barrier_fixed -> NameError).

    pt_sl: Hệ số nhân với ATR để tính khoảng cách [TP, SL].
           Mặc định đổi thành [3.0, 2.0] để khớp RISK_REWARD_RATIO = 1.5
           trong config gốc (TP = 3*ATR, SL = 2*ATR -> R:R = 1.5).
           Bản trước dùng [2.0, 2.0] tương đương R:R = 1:1, không khớp thiết kế.
    max_holding_candles: Số nến tối đa nắm giữ vị thế trước khi coi là
           "hết thời gian, không xác định" (không phải Sideway thắng/thua).
    """
    high_prices  = df['high'].values
    low_prices   = df['low'].values
    close_prices = df['close'].values
    atrs         = df['ATRr_14'].values

    n = len(df)
    # FIX #2: khởi tạo bằng NaN thay vì np.zeros, để phân biệt rõ:
    #   NaN = chưa xác định được nhãn (thiếu dữ liệu tương lai hoặc ATR NaN)
    #   0   = đã quét đủ nến nhưng không chạm SL/TP nào -> Sideway THẬT
    labels = np.full(n, np.nan)

    # Chỉ duyệt tới n - max_holding_candles vì cần đủ max_holding_candles
    # nến phía sau để xác nhận nhãn. Các dòng còn lại giữ nguyên NaN
    # và sẽ bị dropna() loại bỏ đúng cách.
    for i in range(n - max_holding_candles):
        if np.isnan(atrs[i]):
            continue  # giữ NaN, không gán 0

        entry_price = close_prices[i]
        atr_now = atrs[i]

        tp_barrier = entry_price + pt_sl[0] * atr_now
        sl_barrier = entry_price - pt_sl[1] * atr_now

        label_i = 0.0  # mặc định nếu quét hết mà không chạm gì -> Sideway thật
        for j in range(1, max_holding_candles + 1):
            curr_high = high_prices[i + j]
            curr_low  = low_prices[i + j]

            # TÌNH HUỐNG 1: nến quét trúng cả SL và TP cùng lúc (giật mạnh)
            # -> chọn kịch bản an toàn (conservative): coi như dính SL trước
            if curr_low <= sl_barrier and curr_high >= tp_barrier:
                label_i = -1.0
                break

            # TÌNH HUỐNG 2: chạm Stop Loss trước
            elif curr_low <= sl_barrier:
                label_i = -1.0
                break

            # TÌNH HUỐNG 3: chạm Take Profit trước
            elif curr_high >= tp_barrier:
                label_i = 1.0
                break

            # nếu chưa chạm gì ở nến j này thì tiếp tục vòng lặp,
            # label_i vẫn giữ giá trị mặc định 0.0 cho tới khi có breakout

        labels[i] = label_i

    df['target'] = labels
    return df


# ==========================================
# GỌI HÀM (đã đồng bộ tên hàm với phần định nghĩa ở trên)
# ==========================================
merged_df = apply_triple_barrier(merged_df, pt_sl=[3.0, 2.0], max_holding_candles=12)
merged_df.dropna(inplace=True)

# Khuyến nghị: kiểm tra phân bố nhãn trước khi train, để phát hiện
# mất cân bằng lớp (thường Sideway=0 sẽ chiếm đa số).
print(merged_df['target'].value_counts(normalize=True))

# ==========================================
# 5. ĐỊNH NGHĨA FEATURES MỚI VÀ CHIA TẬP DATA 3 PHẦN ĐỘC LẬP
# ==========================================
# Danh sách này TUYỆT ĐỐI không có cột chứa giá tuyệt đối (open, high, low, close, bbl, bbu...)
feature_cols = [
    'volume', 'spread', 'RSI_14', 
    'ret_close', 'pct_body', 'atr_norm_upper_shadow', 'atr_norm_lower_shadow', 
    'dist_EMA20_pct', 'dist_BBU_pct', 'dist_BBL_pct',
    'dist_EMA20_M15_pct', 'macd_norm',
    'dist_EMA20_H1_pct', 'RSI_14_H1',
    'dist_EMA20_H4_pct', 'RSI_14_H4'
]

merged_df.set_index('time', inplace=True)

# Tách 3 tập dữ liệu độc lập theo thời gian nghiêm ngặt
train_data = merged_df.loc[:'2023-12-31']
val_data   = merged_df.loc['2024-01-01':'2024-12-31']
test_data  = merged_df.loc['2025-01-01':]

X_train, y_train = train_data[feature_cols], train_data['target']
X_val, y_val     = val_data[feature_cols], val_data['target']
X_test, y_test   = test_data[feature_cols], test_data['target']

print(f"Mẫu tập Train (2020-2023): {len(X_train)}")
print(f"Mẫu tập Validation (2024): {len(X_val)}")
print(f"Mẫu tập Test (2025-Nay):    {len(X_test)}")

# ==========================================
# 6. HUẤN LUYỆN LIGHTGBM (AN TOÀN CHO EVAL_SET)
# ==========================================
print("\n=== HUẤN LUYỆN LIGHTGBM (Sử dụng tập Validation riêng biệt) ===")
model_lgb = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=6,
    class_weight='balanced',
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)
model_lgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)], # CHỈ DÙNG TẬP VAL ĐỂ EARLY STOPPING, TẬP TEST ĐƯỢC GIỮ BÍ MẬT 100%
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)

# ==========================================
# 7. HUẤN LUYỆN XGBOOST (AN TOÀN CHO EVAL_SET)
# ==========================================
print("\n=== HUẤN LUYỆN XGBOOST (Sử dụng tập Validation riêng biệt) ===")
# Chuyển nhãn từ [-1, 0, 1] thành [0, 1, 2] vì XGBoost Classifier yêu cầu nhãn đa lớp từ 0 đến N-1
y_train_xgb = y_train + 1
y_val_xgb   = y_val + 1
y_test_xgb  = y_test + 1

model_xgb = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=6,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
    eval_metric="mlogloss"
)
model_xgb.fit(
    X_train, y_train_xgb,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val_xgb)], # CHỈ DÙNG TẬP VAL ĐỂ EARLY STOPPING
    verbose=False
)

# ==========================================
# 8. ĐÁNH GIÁ THỰC TẾ TRÊN TẬP TEST (ĐÃ ĐƯỢC CÔ LẬP)
# ==========================================
print("\n" + "="*60)
print("BÁO CÁO ĐÁNH GIÁ CHUẨN TRÊN TẬP TEST BÍ MẬT (2025 - NAY)")
print("="*60)

# Đánh giá LightGBM
y_pred_lgb = model_lgb.predict(X_test)
print("\n[LIGHTGBM EVALUATION] - Nhãn: -1 (Thua/Chạm SL), 0 (Sideway), 1 (Thắng/Chạm TP):")
print(classification_report(y_test, y_pred_lgb))

# Đánh giá XGBoost
y_pred_xgb = model_xgb.predict(X_test) - 1 # Đổi ngược nhãn về lại [-1, 0, 1]
print("\n[XGBOOST EVALUATION] - Nhãn: -1 (Thua/Chạm SL), 0 (Sideway), 1 (Thắng/Chạm TP):")
print(classification_report(y_test, y_pred_xgb))

# Lưu cấu trúc mô hình
model_lgb.booster_.save_model('lgb_gold_triple_barrier.txt')
model_xgb.save_model('xgb_gold_triple_barrier.json')