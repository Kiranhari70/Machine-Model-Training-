import pandas as pd
import numpy as np
import pickle
import gzip
 
# FEATURE EXTRACTION
def feature_extraction(data, target_len=30):
    features = []
 
    for row in data:
        row = np.array(row, dtype=float)
 
        if len(row) != target_len:
            row = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len(row)),
                row
            )
 
        n = len(row)
 
        # time-domain stats
        mean_val = np.mean(row)
        std_val = np.std(row)
        min_val = np.min(row)
        max_val = np.max(row)
        rng_val = max_val - min_val
        median_val = np.median(row)
 
        # diff / velocity
        diff1 = np.diff(row)
        max_vel = np.max(diff1) if len(diff1) > 0 else 0
        min_vel = np.min(diff1) if len(diff1) > 0 else 0
        mean_vel = np.mean(diff1) if len(diff1) > 0 else 0
        std_vel = np.std(diff1) if len(diff1) > 0 else 0
 
        # second diff
        diff2 = np.diff(diff1)
        max_acc = np.max(diff2) if len(diff2) > 0 else 0
        mean_acc = np.mean(diff2) if len(diff2) > 0 else 0
 
        # fft stuff
        fft_vals = np.abs(np.fft.rfft(row))
        top_power = np.sort(fft_vals)[-5:]
        nf = len(fft_vals)
        low_power  = np.sum(fft_vals[1:max(2, nf // 6)])
        mid_power  = np.sum(fft_vals[max(2, nf // 6):max(3, nf // 3)])
        high_power = np.sum(fft_vals[max(3, nf // 3):])
        spectral_entropy = -np.sum(
            (fft_vals / (np.sum(fft_vals) + 1e-9)) *
            np.log(fft_vals / (np.sum(fft_vals) + 1e-9) + 1e-9)
        )
 
        # area under curve
        auc = np.trapezoid(row) / n
 
        # percentiles
        p25 = np.percentile(row, 25)
        p75 = np.percentile(row, 75)
        iqr = p75 - p25
 
        # other useful stuff
        peak_idx = np.argmax(row) / (n - 1)
        trough_idx = np.argmin(row) / (n - 1)
        peak_to_peak = max_val - min_val
 
        first_half_mean = np.mean(row[:n // 2])
        second_half_mean = np.mean(row[n // 2:])
        mean_shift = second_half_mean - first_half_mean
 
        early_slope = row[min(n - 1, round(n * 0.2))] - row[0]
        late_slope = row[-1] - row[max(0, round(n * 0.8))]
        peak_from_start = max_val - row[0]
        end_minus_start = row[-1] - row[0]
 
        peak_abs_idx = np.argmax(row)
        post_peak_drop = max_val - np.mean(row[peak_abs_idx:]) if peak_abs_idx < n - 1 else 0.0
 
        cv = std_val / (mean_val + 1e-9)
 
        feature_row = [
            mean_val, std_val, min_val, max_val, rng_val, median_val,
            max_vel, min_vel, mean_vel, std_vel,
            max_acc, mean_acc,
            top_power[0], top_power[1], top_power[2], top_power[3], top_power[4],
            low_power, mid_power, high_power, spectral_entropy,
            auc,
            p25, p75, iqr,
            peak_idx, trough_idx, peak_to_peak,
            mean_shift, early_slope, late_slope,
            peak_from_start, end_minus_start,
            post_peak_drop, cv,
        ]
        features.append(feature_row)
 
    return np.array(features)
 
# LOAD MODEL
def load_model():
    with gzip.open("model.pkl", "rb") as f:
        saved = pickle.load(f)
 
    model = saved["model"]
    scaler = saved["scaler"]
    target_len = saved.get("target_len", 30)
    return model, scaler, target_len
 
# LOAD TEST DATA
def load_test_data():
    test_df = pd.read_csv("test.csv", header=None)
    return test_df
 
# CLEAN TEST DATA
def clean_test_data(test_df, target_len):
    test_df = (
        test_df
        .interpolate(method="linear", axis=1, limit_direction="both")
        .fillna(0)
    )
    raw = test_df.values.astype(float)
 
    resampled = np.array([
        np.interp(np.linspace(0, 1, target_len), np.linspace(0, 1, len(row)), row)
        for row in raw
    ])
    return resampled
 
# PREDICT
def predict_test_data(model, scaler, test_values, target_len):
    test_features = feature_extraction(test_values, target_len=target_len)
    test_scaled = scaler.transform(test_features)
    predictions = model.predict(test_scaled)
    return predictions
 
# SAVE RESULTS
def save_results(predictions):
    pd.DataFrame(predictions.astype(int)).to_csv(
        "Result.csv",
        header=False,
        index=False
    )
 
# MAIN
def main():
    model, scaler, target_len = load_model()
    test_df = load_test_data()
    test_values = clean_test_data(test_df, target_len)
    predictions = predict_test_data(model, scaler, test_values, target_len)
    save_results(predictions)
    print("Predictions saved to Result.csv")
 
if __name__ == "__main__":
    main()
