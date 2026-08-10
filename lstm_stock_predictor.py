#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from scipy.stats import binomtest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ============================================================
# เปลี่ยนแค่บรรทัดนี้บรรทัดเดียวเพื่อสลับไปหุ้นตัวอื่น เช่น "COST", "KO"
# ============================================================
STOCK_TICKER = "KO"

BENCHMARK_TICKER = "SPY"    # proxy ของตลาดรวม — เปลี่ยนเป็น None หรือ USE_BENCHMARK=False เพื่อปิด
USE_BENCHMARK = True

n_steps = 20                 # จำนวนวันย้อนหลังที่ใช้ทำนายวันถัดไป
max_daily_return = 0.15      # เพดานการเปลี่ยนแปลงราคาต่อวันตอนพยากรณ์ล่วงหน้า

# --- Walk-forward evaluation settings (สำหรับวัดความน่าเชื่อถือ) ---
N_FOLDS = 4                  # จำนวนช่วงเวลาที่ใช้ทดสอบซ้ำ
EVAL_REGION_FRAC = 0.40      # สัดส่วนข้อมูลล่าสุดที่กันไว้เป็นพื้นที่ walk-forward evaluation ทั้งหมด
VAL_FRAC_WITHIN_FOLD = 0.15  # ในแต่ละ fold กันส่วนท้ายของ train ไว้เป็น val สำหรับ early stopping

# --- Production model settings (โมเดลจริงที่ใช้พยากรณ์อนาคต) ---
PRODUCTION_TRAIN_FRAC = 0.85  # ใช้ข้อมูลเกือบทั้งหมดเทรน ส่วนที่เหลือเป็น val สำหรับ early stopping
N_ENSEMBLE_PRODUCTION = 3     # จำนวนโมเดลที่เฉลี่ยผลสำหรับพยากรณ์จริง

LSTM_UNITS = (64, 32)         # ลดขนาดลงจาก (128, 64) เดิม เพื่อลด overfitting กับข้อมูลขนาดนี้
L2_REG = 1e-4

OUTPUT_CHART_PATH = "lstm_stock_forecast.png"   # ไฟล์กราฟที่จะถูกบันทึกไว้ในเครื่อง


def clean_yf_columns(df):
    """ทำความสะอาดชื่อคอลัมน์ รองรับ MultiIndex ของ yfinance กับ ticker ใดก็ได้"""
    new_columns = []
    for col in df.columns:
        if isinstance(col, tuple):
            new_columns.append(str(col[0]).strip().lower())
        else:
            new_columns.append(str(col).strip().lower())
    df.columns = new_columns

    date_col_found = next((c for c in df.columns if "date" in c), None)
    if date_col_found is None:
        raise KeyError("ไม่พบคอลัมน์วันที่หลังจากปรับชื่อคอลัมน์แล้ว")
    if date_col_found != "date":
        df.rename(columns={date_col_found: "date"}, inplace=True)

    if "adj_close" in df.columns:
        df["close"] = df["adj_close"]
        df.drop(columns=["adj_close"], inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df


def add_technical_features(df):
    df = df.copy()
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_3"] = df["close"].shift(3)
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["return"] = df["close"].pct_change()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_14"] = df["rsi_14"].fillna(50)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    ma_20 = df["close"].rolling(window=20).mean()
    std_20 = df["close"].rolling(window=20).std()
    df["bb_width"] = (4 * std_20) / ma_20

    df["volume_change"] = df["volume"].pct_change()
    df["volatility_10"] = df["return"].rolling(window=10).std()

    df.dropna(inplace=True)
    return df


def create_sequences(data, n_steps, target_idx):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps)])
        y.append(data[i + n_steps, target_idx])
    return np.array(X), np.array(y)


def build_model(input_shape, units=LSTM_UNITS, l2_reg=L2_REG):
    m = Sequential([
        Bidirectional(LSTM(units[0], return_sequences=True, kernel_regularizer=l2(l2_reg)), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(units[1], activation="relu", kernel_regularizer=l2(l2_reg))),
        Dropout(0.3),
        Dense(1),
    ])
    m.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return m


def main():
    start_date = "2022-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    # --- 1. ดึงข้อมูลราคาหุ้น ---

    print(f"กำลังดึงข้อมูลหุ้น {STOCK_TICKER} ตั้งแต่ {start_date} ถึง {end_date}...")

    df_stock = yf.download(STOCK_TICKER, start=start_date, end=end_date, interval="1d")

    if df_stock.empty:
        raise ValueError(
            f"ไม่พบข้อมูลสำหรับหุ้น '{STOCK_TICKER}' — เช็คว่าสะกด ticker ถูกต้องหรือไม่ "
            f"(เช่น Alphabet คือ GOOGL ไม่ใช่ GO)"
        )

    df_stock = df_stock[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    print(f"จำนวนข้อมูลทั้งหมด: {len(df_stock)}")

    df_stock = clean_yf_columns(df_stock)

    core_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df_stock.columns]
    df_stock = df_stock[core_cols].copy()

    missing_cols = set(["open", "high", "low", "close", "volume"]) - set(df_stock.columns)
    if missing_cols:
        raise KeyError(f"ข้อมูลหุ้น {STOCK_TICKER} ขาดคอลัมน์: {missing_cols}")

    # --- 2. เพิ่ม Benchmark Return ---

    use_benchmark = USE_BENCHMARK
    if use_benchmark and BENCHMARK_TICKER:
        try:
            print(f"กำลังดึงข้อมูล benchmark {BENCHMARK_TICKER}...")
            df_bench = yf.download(BENCHMARK_TICKER, start=start_date, end=end_date, interval="1d")
            df_bench = df_bench[["Close"]].reset_index()
            df_bench = clean_yf_columns(df_bench)
            df_bench["benchmark_return"] = df_bench["close"].pct_change()
            df_stock = df_stock.join(df_bench[["benchmark_return"]], how="left")
            df_stock["benchmark_return"] = df_stock["benchmark_return"].ffill()
        except Exception as e:
            print(f"[Warning] ดึงข้อมูล benchmark ไม่สำเร็จ ({e}) — จะรันต่อโดยไม่ใช้ feature นี้")
            use_benchmark = False

    # --- 3. Feature Engineering ---

    df_stock = add_technical_features(df_stock)
    df_stock.reset_index(inplace=True)

    target_column_name = "return"
    feature_cols = [c for c in df_stock.columns if c != "date"]
    target_column_index = feature_cols.index(target_column_name)

    print("Features for LSTM:", feature_cols)

    n_total = len(df_stock)
    min_required = n_steps + 40
    if n_total < min_required * 3:
        raise ValueError(
            f"ข้อมูลหุ้น {STOCK_TICKER} มีแค่ {n_total} แถวหลังทำ feature engineering "
            f"ซึ่งน้อยเกินไป ลองขยาย start_date ให้ย้อนหลังมากขึ้น"
        )

    # ============================================================
    # ส่วนที่ 1: WALK-FORWARD CROSS-VALIDATION
    # ใช้สำหรับ "วัดความน่าเชื่อถือ" ของโมเดลเท่านั้น — ไม่ใช้ผลจากส่วนนี้ไปพยากรณ์อนาคตจริง
    # ============================================================

    print("\n" + "#" * 60)
    print("WALK-FORWARD CROSS-VALIDATION (ประเมินความน่าเชื่อถือ)")
    print("#" * 60)

    eval_start_idx = int(n_total * (1 - EVAL_REGION_FRAC))
    remaining = n_total - eval_start_idx
    fold_size = remaining // N_FOLDS

    fold_results = []
    all_actual_price, all_pred_price = [], []
    all_actual_return, all_pred_return = [], []
    all_naive_return = []
    all_dates = []

    for k in range(N_FOLDS):
        test_start = eval_start_idx + k * fold_size
        test_end = test_start + fold_size if k < N_FOLDS - 1 else n_total

        train_all = df_stock.iloc[:test_start]
        fold_test = df_stock.iloc[test_start:test_end]

        val_size = max(int(len(train_all) * VAL_FRAC_WITHIN_FOLD), n_steps + 10)
        fold_train = train_all.iloc[:-val_size]
        fold_val = train_all.iloc[-val_size:]

        if len(fold_train) <= n_steps or len(fold_val) <= n_steps or len(fold_test) <= n_steps:
            print(f"[Fold {k + 1}] ข้อมูลไม่พอสำหรับ n_steps={n_steps} — ข้าม fold นี้")
            continue

        fold_scaler = MinMaxScaler(feature_range=(0, 1))
        fold_train_scaled = fold_scaler.fit_transform(fold_train[feature_cols])
        fold_val_scaled = fold_scaler.transform(fold_val[feature_cols])
        fold_test_scaled = fold_scaler.transform(fold_test[feature_cols])

        X_tr, y_tr = create_sequences(fold_train_scaled, n_steps, target_column_index)
        X_va, y_va = create_sequences(fold_val_scaled, n_steps, target_column_index)
        X_te, y_te = create_sequences(fold_test_scaled, n_steps, target_column_index)

        if len(X_te) == 0:
            print(f"[Fold {k + 1}] ไม่มี test sequence — ข้าม fold นี้")
            continue

        tf.random.set_seed(k)
        np.random.seed(k)
        fold_model = build_model((n_steps, X_tr.shape[2]))
        es = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=0)
        fold_model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=300, batch_size=32, callbacks=[es], verbose=0)

        pred_scaled = fold_model.predict(X_te, verbose=0).flatten()
        dummy = np.zeros((len(pred_scaled), len(feature_cols)))
        dummy[:, target_column_index] = pred_scaled
        pred_return = fold_scaler.inverse_transform(dummy)[:, target_column_index]

        prev_close_fold = fold_test["close"].values[n_steps - 1:-1]
        pred_price = prev_close_fold * (1 + pred_return)
        actual_price = fold_test["close"].values[n_steps:]
        actual_return = fold_test[target_column_name].values[n_steps:]
        naive_return_fold = fold_test["return"].values[n_steps - 1:-1]

        fold_mae = mean_absolute_error(actual_price, pred_price)
        fold_dir_acc = float(np.mean(np.sign(pred_return) == np.sign(actual_return))) * 100

        fold_results.append({
            "fold": k + 1,
            "period_start": fold_test["date"].iloc[0].date(),
            "period_end": fold_test["date"].iloc[-1].date(),
            "n_test": len(actual_return),
            "mae": fold_mae,
            "directional_accuracy": fold_dir_acc,
        })

        all_actual_price.append(actual_price)
        all_pred_price.append(pred_price)
        all_actual_return.append(actual_return)
        all_pred_return.append(pred_return)
        all_naive_return.append(naive_return_fold)
        all_dates.append(fold_test["date"].values[n_steps:])

        print(f"[Fold {k + 1}] {fold_test['date'].iloc[0].date()} → {fold_test['date'].iloc[-1].date()} "
              f"| n={len(actual_return)} | MAE=${fold_mae:.2f} | DirAcc={fold_dir_acc:.2f}%")

    if not all_actual_price:
        raise ValueError("ไม่มี fold ไหนมีข้อมูลพอให้ประเมินผลได้ ลองลด N_FOLDS หรือเพิ่ม EVAL_REGION_FRAC")

    pooled_actual_price = np.concatenate(all_actual_price)
    pooled_pred_price = np.concatenate(all_pred_price)
    pooled_actual_return = np.concatenate(all_actual_return)
    pooled_pred_return = np.concatenate(all_pred_return)
    pooled_naive_return = np.concatenate(all_naive_return)

    pooled_mae = mean_absolute_error(pooled_actual_price, pooled_pred_price)
    pooled_rmse = np.sqrt(mean_squared_error(pooled_actual_price, pooled_pred_price))
    pooled_r2 = r2_score(pooled_actual_price, pooled_pred_price)

    pooled_dir_correct = (np.sign(pooled_pred_return) == np.sign(pooled_actual_return))
    pooled_dir_acc = float(np.mean(pooled_dir_correct)) * 100
    n_pooled = len(pooled_actual_return)
    n_pooled_correct = int(pooled_dir_correct.sum())

    naive_dir_correct = (np.sign(pooled_naive_return) == np.sign(pooled_actual_return))
    naive_dir_acc = float(np.mean(naive_dir_correct)) * 100

    sig_test = binomtest(n_pooled_correct, n_pooled, p=0.5, alternative="greater")
    p_value = sig_test.pvalue

    print("\n" + "=" * 70)
    print(f"{STOCK_TICKER} — POOLED WALK-FORWARD RESULTS (รวม {len(fold_results)} folds, n={n_pooled})")
    print("=" * 70)
    print(pd.DataFrame(fold_results).to_string(index=False))
    print("-" * 70)
    print(f"Pooled MAE ($):              {pooled_mae:.4f}")
    print(f"Pooled RMSE ($):             {pooled_rmse:.4f}")
    print(f"Pooled R2:                   {pooled_r2:.4f}")
    print(f"Pooled Directional Accuracy: {pooled_dir_acc:.2f}%   (Naive baseline: {naive_dir_acc:.2f}%)")
    print(f"P-value (H0: acc<=50%):      {p_value:.4f}")
    if p_value < 0.05 and pooled_dir_acc > naive_dir_acc:
        print("-> โมเดลมี edge เหนือการเดาสุ่มและ naive baseline อย่างมีนัยสำคัญทางสถิติ")
    else:
        print("-> ยังไม่มีหลักฐานเพียงพอว่าโมเดลมี edge เหนือการเดาสุ่ม/naive baseline "
              "(ผลนี้ยังถือว่ามีคุณค่า — สะท้อนความยากของการพยากรณ์ราคาหุ้นจาก price history อย่างเดียว)")
    print("=" * 70)

    # ============================================================
    # ส่วนที่ 2: PRODUCTION MODEL — เทรนบนข้อมูลเกือบทั้งหมดเพื่อพยากรณ์อนาคตจริง
    # ============================================================

    print("\n" + "#" * 60)
    print("PRODUCTION MODEL (สำหรับพยากรณ์ 3 วันข้างหน้าจริง)")
    print("#" * 60)

    prod_train_end = int(n_total * PRODUCTION_TRAIN_FRAC)
    df_prod_train = df_stock.iloc[:prod_train_end].copy()
    df_prod_val = df_stock.iloc[prod_train_end:].copy()

    prod_scaler = MinMaxScaler(feature_range=(0, 1))
    prod_train_scaled = prod_scaler.fit_transform(df_prod_train[feature_cols])
    prod_val_scaled = prod_scaler.transform(df_prod_val[feature_cols])

    X_ptr, y_ptr = create_sequences(prod_train_scaled, n_steps, target_column_index)
    X_pva, y_pva = create_sequences(prod_val_scaled, n_steps, target_column_index)

    production_models = []
    for seed in range(N_ENSEMBLE_PRODUCTION):
        print(f"Training production ensemble member {seed + 1}/{N_ENSEMBLE_PRODUCTION}...")
        tf.random.set_seed(100 + seed)
        np.random.seed(100 + seed)
        p_model = build_model((n_steps, X_ptr.shape[2]))
        es = EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True, verbose=0)
        p_model.fit(X_ptr, y_ptr, validation_data=(X_pva, y_pva), epochs=500, batch_size=32, callbacks=[es], verbose=0)
        production_models.append(p_model)

    # --- ทำนายล่วงหน้า 3 วัน (ensemble เฉลี่ยผลทุก step) ---

    print("\nPredicting for the next 3 days (production ensemble)...")

    history_df = df_stock.set_index("date").copy()
    future_core_cols = [c for c in ["open", "high", "low", "close", "volume", "benchmark_return"] if c in history_df.columns]

    future_predictions = []
    last_date = history_df.index[-1]
    last_close = history_df["close"].iloc[-1]

    for _ in range(3):
        last_window_raw = history_df[feature_cols].tail(n_steps).values
        current_sequence = prod_scaler.transform(last_window_raw)
        x_input = current_sequence.reshape(1, n_steps, len(feature_cols))

        step_preds_scaled = [m.predict(x_input, verbose=0)[0][0] for m in production_models]
        next_return_scaled = float(np.mean(step_preds_scaled))

        dummy_next = np.zeros((1, len(feature_cols)))
        dummy_next[:, target_column_index] = next_return_scaled
        next_return = prod_scaler.inverse_transform(dummy_next)[:, target_column_index][0]
        next_return_clipped = float(np.clip(next_return, -max_daily_return, max_daily_return))
        next_price = last_close * (1 + next_return_clipped)

        prediction_date = last_date + timedelta(days=1)
        while prediction_date.weekday() > 4:
            prediction_date += timedelta(days=1)

        future_predictions.append({"date": prediction_date, "Predicted": next_price})

        new_row = {
            "open": next_price, "high": next_price, "low": next_price,
            "close": next_price, "volume": history_df["volume"].iloc[-1],
        }
        if "benchmark_return" in future_core_cols:
            new_row["benchmark_return"] = history_df["benchmark_return"].iloc[-1]

        history_df = pd.concat([history_df[future_core_cols], pd.DataFrame([new_row], index=[prediction_date])])
        history_df.sort_index(inplace=True)
        history_df = add_technical_features(history_df)

        last_close = next_price
        last_date = prediction_date

    future_predictions_df = pd.DataFrame(future_predictions)
    print("\nFuture 3-day Predictions:")
    print(future_predictions_df)

    # --- Plot: ใช้ผลจาก walk-forward evaluation + future forecast ---

    plt.figure(figsize=(16, 9))

    eval_dates = np.concatenate(all_dates)

    plt.plot(eval_dates, pooled_actual_price, label=f"Actual {STOCK_TICKER} Price (walk-forward test)",
              color="blue", marker="o", markersize=3, linewidth=1)
    plt.plot(eval_dates, pooled_pred_price, label=f"Predicted {STOCK_TICKER} Price (walk-forward test)",
              color="red", linestyle="--", marker="x", markersize=3, linewidth=1)

    if not future_predictions_df.empty:
        plt.scatter(future_predictions_df["date"], future_predictions_df["Predicted"],
                     color="green", s=100, label="Next 1, 2, 3 Day Forecast (production model)", zorder=5, marker="*")

    plt.title(f"{STOCK_TICKER}: Walk-Forward Evaluation vs Actual (+3-Day Production Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # บันทึกกราฟเป็นไฟล์เสมอ (ใช้งานได้ทั้งเครื่องที่มี/ไม่มีหน้าจอ)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_CHART_PATH)
    plt.savefig(output_path, dpi=150)
    print(f"\nบันทึกกราฟไว้ที่: {output_path}")

    # พยายามเปิดหน้าต่างแสดงกราฟ ถ้าเครื่องรองรับ GUI (ถ้าไม่รองรับจะข้ามไปเฉยๆ ไม่ error)
    try:
        plt.show()
    except Exception:
        print("(ไม่สามารถเปิดหน้าต่างแสดงกราฟได้บนเครื่องนี้ — ดูผลลัพธ์ได้จากไฟล์ภาพที่บันทึกไว้แทน)")


if __name__ == "__main__":
    main()
