import pandas as pd
import numpy as np
import pickle
import gzip
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
 
TARGET_LEN = 30
 
# LOAD DATA
def load_data():
    cgm1 = pd.read_csv("CGMData.csv", low_memory=False)
    cgm2 = pd.read_csv("CGM_patient2.csv", low_memory=False)
    insulin1 = pd.read_csv("InsulinData.csv", low_memory=False)
    insulin2 = pd.read_csv("Insulin_patient2.csv", low_memory=False)
    return cgm1, cgm2, insulin1, insulin2
 
# TIMESTAMPS
def timestamps(df):
    df = df.copy()
    df["Date"] = df["Date"].astype(str).str.strip()
    df["Time"] = df["Time"].astype(str).str.strip()
 
    df["tm"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="mixed",
        errors="coerce"
    )
    return df
 
# EXTRACT MEAL DATA
def extract_meal_data(cgm, insulin):
    cgm = timestamps(cgm)
    insulin = timestamps(insulin)
 
    cgm["Sensor Glucose (mg/dL)"] = pd.to_numeric(
        cgm["Sensor Glucose (mg/dL)"], errors="coerce"
    )
    insulin["BWZ Carb Input (grams)"] = pd.to_numeric(
        insulin["BWZ Carb Input (grams)"], errors="coerce"
    )
 
    cgm = cgm.dropna(subset=["tm"])
    insulin = insulin.dropna(subset=["tm"])
 
    carb_events = insulin[
        insulin["BWZ Carb Input (grams)"].notna() &
        (insulin["BWZ Carb Input (grams)"] > 0)
    ]["tm"].sort_values().reset_index(drop=True)
 
    meal_windows = []
    used = set()
 
    for i, tm in enumerate(carb_events):
        if i in used:
            continue
 
        cluster = [i]
        for j in range(i + 1, len(carb_events)):
            if carb_events[j] < tm + pd.Timedelta(hours=2):
                cluster.append(j)
                used.add(j)
            else:
                break
 
        anchor = carb_events[cluster[-1]]
        start = anchor - pd.Timedelta(minutes=30)
        end = anchor + pd.Timedelta(hours=2)
 
        window = cgm[
            (cgm["tm"] >= start) & (cgm["tm"] < end)
        ].sort_values("tm")["Sensor Glucose (mg/dL)"].dropna().values
 
        if len(window) >= 20:
            window = np.interp(
                np.linspace(0, 1, TARGET_LEN),
                np.linspace(0, 1, len(window)),
                window
            )
            meal_windows.append(window)
 
    return np.array(meal_windows) if meal_windows else np.empty((0, TARGET_LEN))
 
# EXTRACT NO MEAL DATA
def extract_no_meal_data(cgm, insulin):
    cgm = timestamps(cgm)
    insulin = timestamps(insulin)
 
    cgm["Sensor Glucose (mg/dL)"] = pd.to_numeric(
        cgm["Sensor Glucose (mg/dL)"], errors="coerce"
    )
    insulin["BWZ Carb Input (grams)"] = pd.to_numeric(
        insulin["BWZ Carb Input (grams)"], errors="coerce"
    )
 
    cgm = cgm.dropna(subset=["tm"])
    insulin = insulin.dropna(subset=["tm"])
 
    carb_events = insulin[
        insulin["BWZ Carb Input (grams)"].notna() &
        (insulin["BWZ Carb Input (grams)"] > 0)
    ]["tm"].sort_values().reset_index(drop=True)
 
    no_meal_windows = []
 
    for tm in carb_events:
        start = tm + pd.Timedelta(hours=2)
        end = start + pd.Timedelta(hours=2)
 
        blocking = insulin[
            insulin["BWZ Carb Input (grams)"].notna() &
            (insulin["BWZ Carb Input (grams)"] > 0) &
            (insulin["tm"] > start) &
            (insulin["tm"] <= end)
        ]
 
        if not blocking.empty:
            continue
 
        window = cgm[
            (cgm["tm"] >= start) & (cgm["tm"] < end)
        ].sort_values("tm")["Sensor Glucose (mg/dL)"].dropna().values
 
        if len(window) >= 16:
            window = np.interp(
                np.linspace(0, 1, TARGET_LEN),
                np.linspace(0, 1, len(window)),
                window
            )
            no_meal_windows.append(window)
 
    return np.array(no_meal_windows) if no_meal_windows else np.empty((0, TARGET_LEN))
 
# CLEAN DATA
def clean_data(data, max_missing_frac=0.15):
    if len(data) == 0:
        return data
 
    df = pd.DataFrame(data.astype(float))
    threshold = int(df.shape[1] * (1 - max_missing_frac))
    df = df.dropna(thresh=threshold)
    df = df.interpolate(method="linear", axis=1, limit_direction="both")
    df = df.bfill(axis=1).ffill(axis=1)
    df = df.dropna()
 
    return df.values
 
# FEATURE EXTRACTION
def feature_extraction(data):
    features = []
 
    for row in data:
        row = np.array(row, dtype=float)
 
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
 
# PREP DATA
def prepare_training_data(meal_features, no_meal_features):
    X = np.vstack((meal_features, no_meal_features))
    Y = np.hstack((
        np.ones(len(meal_features)),
        np.zeros(len(no_meal_features))
    ))
    return X, Y
 
# TRAIN MODEL
def train_model(X, Y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X, Y)
    return model
 
# EVAL MODEL
def eval_model(model, X, Y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, Y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, Y, cv=cv, scoring="f1")
 
    print(f"Accuracy: {np.mean(scores):.6f}")
    print(f"F1 Score: {np.mean(f1_scores):.6f}")
 
# SAVE MODEL
def save_model(model, scaler):
    with gzip.open("model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "target_len": TARGET_LEN}, f)
 
# MAIN
def main():
    cgm1, cgm2, insulin1, insulin2 = load_data()
 
    meal1 = extract_meal_data(cgm1, insulin1)
    meal2 = extract_meal_data(cgm2, insulin2)
 
    if len(meal1) and len(meal2):
        meal_raw = np.vstack((meal1, meal2))
    elif len(meal1):
        meal_raw = meal1
    else:
        meal_raw = meal2
 
    no_meal1 = extract_no_meal_data(cgm1, insulin1)
    no_meal2 = extract_no_meal_data(cgm2, insulin2)
 
    if len(no_meal1) and len(no_meal2):
        no_meal_raw = np.vstack((no_meal1, no_meal2))
    elif len(no_meal1):
        no_meal_raw = no_meal1
    else:
        no_meal_raw = no_meal2
 
    meal_clean = clean_data(meal_raw)
    no_meal_clean = clean_data(no_meal_raw)
 
    meal_features = feature_extraction(meal_clean)
    no_meal_features = feature_extraction(no_meal_clean)
 
    X, Y = prepare_training_data(meal_features, no_meal_features)
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    model = train_model(X_scaled, Y)
    eval_model(model, X_scaled, Y)
    save_model(model, scaler)
 
if __name__ == "__main__":
    main()
