import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =========================
# CONFIG (editable)
# =========================
SHEET_NAME = "IMU_Combined"

TIME_COL = "Time (s)"
ACCEL_X_COL = "Accel_X"
ACCEL_Y_COL = "Accel_Y"
ACCEL_Z_COL = "Accel_Z"
GYRO_X_COL  = "Gyro_X"
GYRO_Y_COL  = "Gyro_Y"
GYRO_Z_COL  = "Gyro_Z"

HP_FC_HZ = 20.0                 # High-pass cutoff frequency (Hz)
STAT_WIN_S = 0.20               # Rolling stats window (seconds)
PEAK_MIN_INTERVAL_S = 0.08      # Min spacing between detected peaks (seconds)
MAD_WIN_S = 0.50                # Only used in rolling MAD mode


# =========================
# Small helpers
# =========================
def estimate_fs(time_s):                            #Takes in the time from excel and convert it to sampling freq calculting dt
    t = np.asarray(time_s, dtype=float)
    if len(t) < 3:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return 1.0 / np.median(dt)


def win_tag(seconds):
    # Win0p20s style
    return f"Win{seconds:.2f}".replace(".", "p") + "s"


def rc_highpass(x, fs_hz, fc_hz):
    """
    Discrete RC high-pass:
      y[n] = a * (y[n-1] + x[n] - x[n-1])
      a = RC / (RC + dt), RC = 1/(2*pi*fc), dt = 1/fs
    """
    x = np.asarray(x, dtype=float)  #Takes in input magnitude
    if len(x) == 0:
        return x.copy()

    if not np.isfinite(fs_hz) or fs_hz <= 0 or fc_hz <= 0:  #validation -- DC removal
        return x - np.nanmedian(x)

    dt = 1.0 / fs_hz
    rc = 1.0 / (2.0 * np.pi * fc_hz)
    a = rc / (rc + dt)

    y = np.zeros_like(x, dtype=float)                       #initialize with 0's 
    for i in range(1, len(x)):                              #maintain causality so y[0] = 0... system starts at 0
        y[i] = a * (y[i - 1] + x[i] - x[i - 1])             #difference equation
    return y


def rolling_rms(x, win):
    s = pd.Series(np.asarray(x, dtype=float))
    return np.sqrt((s * s).rolling(win, min_periods=1, center=True).mean()) #rolling 99<win> samples and taking mean over the mid window value


def rolling_sd(x, win):
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(win, min_periods=1, center=True).std(ddof=0)       #If you consider the data within the win 
                                                                        # samples to be your complete set of data for that moment in 
                                                                        # time (common in energy detection), then ddof=0 is appropriate.'''#here impulse detection
                                                                        #If you consider the win samples to be a subset of an infinitely long,
                                                                        #larger signal, then you might prefer ddof=1 for a statistically unbiased estimate.'''


def rolling_peak(x, win):
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(win, min_periods=1, center=True).max()


def mad_sigma_global(x):
    x = np.asarray(x, dtype=float)
    m = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - m)))
    sigma = 1.4826 * mad if mad > 0 else 0.0
    return m, sigma


def mad_sigma_rolling(x, win):
    s = pd.Series(np.asarray(x, dtype=float))
    med = s.rolling(win, min_periods=1, center=True).median()

    def _mad(v):                                                #v = np.array([...])
        mv = float(np.median(v))
        return float(np.median(np.abs(v - mv)))

    mad = s.rolling(win, min_periods=1, center=True).apply(_mad, raw=True) #raw=True -> pandas passes a NumPy array directly into _mad(v)
    sigma = 1.4826 * mad                                                   #Gaussian deviation as reference for deviation
    return med.to_numpy(dtype=float), sigma.to_numpy(dtype=float)


def detect_peaks_mad(sig, fs_hz, peak_k, min_interval_s, mad_mode="global"):
    """
    Qualified peak:
      - local max
      - above threshold
      - at least min_interval_s apart
    """
    sig = np.asarray(sig, dtype=float)
    n = len(sig)
    if n < 3:
        return np.zeros(n, dtype=int), np.nan

    # threshold
    if mad_mode == "rolling" and np.isfinite(fs_hz) and fs_hz > 0:
        win = max(3, int(round(MAD_WIN_S * fs_hz)))         #0.5 * 500->250appprox
        med, sigma = mad_sigma_rolling(sig, win)
        thr = med + peak_k * sigma
        thr_arr = thr
        mid = np.arange(1, n - 1)   #neglecting 0th and nth indices as they dont have left and right neighbour
        cond = np.isfinite(thr_arr[mid]) & (sig[mid] > thr_arr[mid]) & (sig[mid] > sig[mid - 1]) & (sig[mid] >= sig[mid + 1]) #3 values for median, mid value > than others then set the index
        candidates = mid[cond]
        threshold_out = thr_arr
    else:
        m, sigma = mad_sigma_global(sig)
        thr = m + peak_k * sigma
        candidates = np.where((sig[1:-1] > thr) & (sig[1:-1] > sig[:-2]) & (sig[1:-1] >= sig[2:]))[0] + 1 #+1 as we start from index 1
        threshold_out = float(thr)                                                                          #[0] fixed column ... like Array of pointers... pointer = index containing windowed slice

    # spacing
    min_samples = max(1, int(round(min_interval_s * fs_hz))) if np.isfinite(fs_hz) and fs_hz > 0 else 1
    peaks = []
    last = -10**9
    for idx in candidates:              #minimum wait rule
        idx = int(idx)
        if idx - last >= min_samples:
            peaks.append(idx)
            last = idx

    flag = np.zeros(n, dtype=int)
    flag[peaks] = 1                     #array indices set
    return flag, threshold_out


# =========================
# Feature builder
# =========================
def add_features(df, mad_mode="global", peak_k=6.0, add_threshold_cols=False):
    df = df.copy().sort_values(TIME_COL).reset_index(drop=True)

    t = df[TIME_COL].to_numpy(dtype=float)
    fs_hz = estimate_fs(t)

    # Window in samples
    stat_win = max(3, int(round(STAT_WIN_S * fs_hz))) if np.isfinite(fs_hz) and fs_hz > 0 else 3    #convert time to sample... update STAT_WIN_S to increase/decrease samples
    wtag = win_tag(STAT_WIN_S)

    # ---- Magnitude ----
    accel_mag = np.sqrt(df[ACCEL_X_COL]**2 + df[ACCEL_Y_COL]**2 + df[ACCEL_Z_COL]**2).to_numpy(dtype=float)
    gyro_mag  = np.sqrt(df[GYRO_X_COL]**2  + df[GYRO_Y_COL]**2  + df[GYRO_Z_COL]**2 ).to_numpy(dtype=float)

    df["Accel_Mag"] = accel_mag
    df["Gyro_Mag"]  = gyro_mag

    # ---- High-pass ----
    df["Accel_Mag_HP"] = rc_highpass(accel_mag, fs_hz, HP_FC_HZ)
    df["Gyro_Mag_HP"]  = rc_highpass(gyro_mag,  fs_hz, HP_FC_HZ)

    # ---- ABS impulse ----
    df["Accel_Mag_HP_ABS"] = np.abs(df["Accel_Mag_HP"].to_numpy(dtype=float))
    df["Gyro_Mag_HP_ABS"]  = np.abs(df["Gyro_Mag_HP"].to_numpy(dtype=float))

    # ---- Rolling stats on impulse ----
    a_imp = df["Accel_Mag_HP_ABS"].to_numpy(dtype=float)
    g_imp = df["Gyro_Mag_HP_ABS"].to_numpy(dtype=float)

    df[f"Accel_Mag_HP_ABS_RMS_{wtag}"]  = rolling_rms(a_imp, stat_win)
    df[f"Accel_Mag_HP_ABS_SD_{wtag}"]   = rolling_sd(a_imp,  stat_win)
    df[f"Accel_Mag_HP_ABS_Peak_{wtag}"] = rolling_peak(a_imp, stat_win)
    df[f"Accel_Mag_HP_ABS_Crest_{wtag}"] = df[f"Accel_Mag_HP_ABS_Peak_{wtag}"] / (df[f"Accel_Mag_HP_ABS_RMS_{wtag}"] + 1e-12) #Note:float64:1e-12 to prevent /by0 returns NAN or infinity

    df[f"Gyro_Mag_HP_ABS_RMS_{wtag}"]   = rolling_rms(g_imp, stat_win)
    df[f"Gyro_Mag_HP_ABS_SD_{wtag}"]    = rolling_sd(g_imp,  stat_win)
    df[f"Gyro_Mag_HP_ABS_Peak_{wtag}"]  = rolling_peak(g_imp, stat_win)
    df[f"Gyro_Mag_HP_ABS_Crest_{wtag}"] = df[f"Gyro_Mag_HP_ABS_Peak_{wtag}"] / (df[f"Gyro_Mag_HP_ABS_RMS_{wtag}"] + 1e-12)

    # ---- Axial Direction-preserving SD (per-axis HP_ABS) ----
    for col in [ACCEL_X_COL, ACCEL_Y_COL, ACCEL_Z_COL]:
        hp = rc_highpass(df[col].to_numpy(dtype=float), fs_hz, HP_FC_HZ)
        df[f"Accel_{col.split('_')[-1]}_HP_ABS_SD_{wtag}"] = rolling_sd(np.abs(hp), stat_win)

    for col in [GYRO_X_COL, GYRO_Y_COL, GYRO_Z_COL]:
        hp = rc_highpass(df[col].to_numpy(dtype=float), fs_hz, HP_FC_HZ)
        df[f"Gyro_{col.split('_')[-1]}_HP_ABS_SD_{wtag}"] = rolling_sd(np.abs(hp), stat_win)

    # ---- Peak detection ----
    accel_flag, accel_thr = detect_peaks_mad(a_imp, fs_hz, peak_k, PEAK_MIN_INTERVAL_S, mad_mode=mad_mode)
    
    gyro_flag,  gyro_thr  = detect_peaks_mad(g_imp, fs_hz, peak_k, PEAK_MIN_INTERVAL_S, mad_mode=mad_mode) ##

    df["Accel_ClickFlag"] = accel_flag
    df["Gyro_ClickFlag"]  = gyro_flag
    df["IMU_ClickFlag_OR"]  = ((df["Accel_ClickFlag"] == 1) | (df["Gyro_ClickFlag"] == 1)).astype(int)
    df["IMU_ClickFlag_AND"] = ((df["Accel_ClickFlag"] == 1) & (df["Gyro_ClickFlag"] == 1)).astype(int)

    if add_threshold_cols:
        df["Accel_Threshold"] = accel_thr if np.isscalar(accel_thr) else np.asarray(accel_thr, dtype=float)
        df["Gyro_Threshold"]  = gyro_thr  if np.isscalar(gyro_thr)  else np.asarray(gyro_thr,  dtype=float)

    meta = {
        "fs_hz": float(fs_hz) if np.isfinite(fs_hz) else np.nan,
        "HP_FC_HZ": HP_FC_HZ,
        "STAT_WIN_S": STAT_WIN_S,
        "STAT_WIN_SAMPLES": int(stat_win),
        "MAD_MODE": mad_mode,
        "PEAK_K": float(peak_k),
        "PEAK_MIN_INTERVAL_S": PEAK_MIN_INTERVAL_S,
        "ACCEL_PEAKS": int(df["Accel_ClickFlag"].sum()),
        "GYRO_PEAKS": int(df["Gyro_ClickFlag"].sum()),
        "IMU_OR_PEAKS": int(df["IMU_ClickFlag_OR"].sum()),
        "IMU_AND_PEAKS": int(df["IMU_ClickFlag_AND"].sum()),
    }

    if np.isscalar(accel_thr):
        meta["ACCEL_THR"] = float(accel_thr)
    if np.isscalar(gyro_thr):
        meta["GYRO_THR"] = float(gyro_thr)

    return df, meta


# =========================
# File handling
# =========================
def process_one_file(in_path, mad_mode, peak_k, add_threshold_cols):
    out_path = in_path.with_name(in_path.stem + "_with_IMU_features.xlsx")

    xls = pd.ExcelFile(in_path, engine="openpyxl")
    sheets = {name: pd.read_excel(xls, sheet_name=name) for name in xls.sheet_names}

    if SHEET_NAME not in sheets:
        raise SystemExit(f"[{in_path.name}] Missing sheet '{SHEET_NAME}'. Found: {xls.sheet_names}")

    df0 = sheets[SHEET_NAME]
    df_feat, meta = add_features(df0, mad_mode=mad_mode, peak_k=peak_k, add_threshold_cols=add_threshold_cols)

    sheets[SHEET_NAME] = df_feat
    sheets["IMU_Feature_Meta"] = pd.DataFrame([meta])

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, sdf in sheets.items():
            sdf.to_excel(writer, sheet_name=name, index=False)

    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Excel file OR folder containing Excel files")
    ap.add_argument("--mad_mode", choices=["global", "rolling"], default="global")
    ap.add_argument("--peak_k", type=float, default=6.0)
    ap.add_argument("--add_threshold_cols", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        process_one_file(input_path, args.mad_mode, args.peak_k, args.add_threshold_cols)
        return

    if not input_path.is_dir():
        raise SystemExit("Input must be a file or folder")

    files = [p for p in input_path.glob("*.xlsx") if p.is_file() and not p.name.startswith("~$")]
    if not files:
        raise SystemExit("No .xlsx files found in folder")

    for f in files:
        # skip already processed
        if f.name.endswith("_with_IMU_features.xlsx"):
            continue
        process_one_file(f, args.mad_mode, args.peak_k, args.add_threshold_cols)


if __name__ == "__main__":
    main()