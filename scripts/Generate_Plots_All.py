import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------- helpers ----------
def estimate_fs(t):
    t = np.asarray(t, float)
    if len(t) < 4:
        return np.nan
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return np.nan
    return 1.0 / float(np.median(dt))


def mad_sigma(x):
    x = np.asarray(x, float)
    m = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - m)))
    return m, 1.4826 * mad


def pick_time_col(df):
    for c in ["Time_s", "Time (s)"]:
        if c in df.columns:
            return c
    raise KeyError("Missing time column: Time_s or Time (s)")


def pick_prefix(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return cols[0] if cols else None


def logic_peak_times(df, t, flag_col="IMU_ClickFlag_OR"):
    if flag_col in df.columns:
        idx = np.where(df[flag_col].to_numpy(int) == 1)[0]
        return t[idx]
    return np.array([], float)


def scipy_peaks(sig, fs, k=6.0, min_interval_s=0.08, prom_mult=1.0, width_s=0.0):
    if not SCIPY_OK or not np.isfinite(fs) or fs <= 0:
        return np.array([], int), np.nan

    m, sigma = mad_sigma(sig)
    thr = m + k * sigma

    distance = max(1, int(round(min_interval_s * fs)))
    prominence = max(0.0, prom_mult * sigma)
    width = None if width_s <= 0 else max(1, int(round(width_s * fs)))

    peaks, _ = find_peaks(
        sig,
        height=thr,
        distance=distance,
        prominence=prominence if prominence > 0 else None,
        width=width,
    )
    return peaks.astype(int), thr


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def slice_window(df, t1, t2):
    z = df[(df["Time_s"] >= t1) & (df["Time_s"] <= t2)].copy()
    return z if len(z) >= 10 else None


# ---------- plots ----------
def plot_full_raw_mag(df, t, out_png):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_title("Full: Raw Magnitudes")
    ax.plot(t, df["Accel_Mag"], label="Accel_Mag (m/s²)", color="tab:blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Accel_Mag (m/s²)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(t, df["Gyro_Mag"], label="Gyro_Mag (rad/s)", color="tab:orange")
    ax2.set_ylabel("Gyro_Mag (rad/s)")

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    save(fig, out_png)


def plot_full_hpabs_peaks_compare(df, t, fs, out_png, k, min_interval, prom_mult, width_s):
    a = df["Accel_Mag_HP_ABS"].to_numpy(float)
    g = df["Gyro_Mag_HP_ABS"].to_numpy(float)
    y_top = 1.05 * max(np.nanmax(a), np.nanmax(g))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_title("Full: HP_ABS with Peaks (Logic vs SciPy)")
    ax.plot(t, a, label="Accel_Mag_HP_ABS (m/s²)", color="tab:blue")
    ax.plot(t, g, label="Gyro_Mag_HP_ABS (rad/s)", color="tab:orange")

    # Logic peaks (your pipeline)
    tl = logic_peak_times(df, t, "IMU_ClickFlag_OR")
    if len(tl):
        ax.scatter(tl, np.full_like(tl, y_top), s=35, marker="o",
                   color="black", label="Logic Peaks (IMU_ClickFlag_OR)")

    # SciPy peaks (independent check)
    ap, _ = scipy_peaks(a, fs, k, min_interval, prom_mult, width_s)
    gp, _ = scipy_peaks(g, fs, k, min_interval, prom_mult, width_s)

    if len(ap):
        ax.scatter(t[ap], a[ap], s=28, marker="o", color="darkred", label="SciPy Peaks (Accel)")
    if len(gp):
        ax.scatter(t[gp], g[gp], s=28, marker="x", color="maroon", label="SciPy Peaks (Gyro)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("HP_ABS (Accel m/s², Gyro rad/s)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    save(fig, out_png)


def plot_full_rms_sd(df, t, out_png):
    a_rms = pick_prefix(df, "Accel_Mag_HP_ABS_RMS_")
    g_rms = pick_prefix(df, "Gyro_Mag_HP_ABS_RMS_")
    a_sd  = pick_prefix(df, "Accel_Mag_HP_ABS_SD_")
    g_sd  = pick_prefix(df, "Gyro_Mag_HP_ABS_SD_")
    if not all([a_rms, g_rms, a_sd, g_sd]):
        return

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 7))
    fig.suptitle("Full: RMS + SD of Mag_HP_ABS")

    ax[0].plot(t, df[a_rms], label=f"{a_rms} (m/s²)", color="tab:blue")
    ax[0].plot(t, df[g_rms], label=f"{g_rms} (rad/s)", color="tab:orange")
    ax[0].set_ylabel("RMS")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(loc="upper right")

    ax[1].plot(t, df[a_sd], label=f"{a_sd} (m/s²)", color="tab:blue")
    ax[1].plot(t, df[g_sd], label=f"{g_sd} (rad/s)", color="tab:orange")
    ax[1].set_ylabel("SD")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper right")

    save(fig, out_png)


def plot_raw_axes(df, t, out_png, sensor_name, cols, ylabel):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_title(f"Full: Raw Axes {sensor_name} (X/Y/Z)")
    ax.plot(t, df[cols[0]], label=f"{cols[0]}", color="tab:blue")
    ax.plot(t, df[cols[1]], label=f"{cols[1]}", color="tab:orange")
    ax.plot(t, df[cols[2]], label=f"{cols[2]}", color="tab:green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    save(fig, out_png)


def plot_axis_sd(df, t, out_png, sensor_name, axis_prefixes, ylabel):
    # Example prefixes: ["Accel_X_HP_ABS_SD_", "Accel_Y_HP_ABS_SD_", "Accel_Z_HP_ABS_SD_"]
    cols = [pick_prefix(df, p) for p in axis_prefixes]
    if any(c is None for c in cols):
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_title(f"Full: Axis SD of HP_ABS {sensor_name} (X/Y/Z)")
    ax.plot(t, df[cols[0]], label=cols[0], color="tab:blue")
    ax.plot(t, df[cols[1]], label=cols[1], color="tab:orange")
    ax.plot(t, df[cols[2]], label=cols[2], color="tab:green")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    save(fig, out_png)


def plot_zoom_dashboard(dfz, out_png):
    tz = dfz["Time_s"].to_numpy(float)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(16, 8))
    fig.suptitle("Zoom: HP / HP_ABS / RMS+SD (with logic peaks)")

    ax[0].plot(tz, dfz["Accel_Mag_HP"], label="Accel_Mag_HP (m/s²)", color="tab:blue")
    ax[0].plot(tz, dfz["Gyro_Mag_HP"],  label="Gyro_Mag_HP (rad/s)", color="tab:orange")
    ax[0].set_ylabel("HP")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(loc="upper right")

    ax[1].plot(tz, dfz["Accel_Mag_HP_ABS"], label="Accel_Mag_HP_ABS (m/s²)", color="tab:blue")
    ax[1].plot(tz, dfz["Gyro_Mag_HP_ABS"],  label="Gyro_Mag_HP_ABS (rad/s)", color="tab:orange")

    tl = logic_peak_times(dfz, tz, "IMU_ClickFlag_OR")
    if len(tl):
        y_top = 1.05 * max(np.nanmax(dfz["Accel_Mag_HP_ABS"]), np.nanmax(dfz["Gyro_Mag_HP_ABS"]))
        ax[1].scatter(tl, np.full_like(tl, y_top), s=35, marker="o", color="black", label="Logic Peaks")

    ax[1].set_ylabel("HP_ABS")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper right")

    a_rms = pick_prefix(dfz, "Accel_Mag_HP_ABS_RMS_")
    g_rms = pick_prefix(dfz, "Gyro_Mag_HP_ABS_RMS_")
    a_sd  = pick_prefix(dfz, "Accel_Mag_HP_ABS_SD_")
    g_sd  = pick_prefix(dfz, "Gyro_Mag_HP_ABS_SD_")
    if all([a_rms, g_rms, a_sd, g_sd]):
        ax[2].plot(tz, dfz[a_rms], label="Accel RMS", color="tab:blue")
        ax[2].plot(tz, dfz[g_rms], label="Gyro RMS",  color="tab:orange")
        ax[2].plot(tz, dfz[a_sd],  label="Accel SD",  color="tab:blue", linestyle="--")
        ax[2].plot(tz, dfz[g_sd],  label="Gyro SD",   color="tab:orange", linestyle="--")
        ax[2].set_ylabel("RMS + SD")
        ax[2].grid(True, alpha=0.25)
        ax[2].legend(loc="upper right")

    ax[2].set_xlabel("Time (s)")
    save(fig, out_png)


def plot_zoom_raw_axes(dfz, out_png):
    tz = dfz["Time_s"].to_numpy(float)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 7))
    fig.suptitle("Zoom: Raw Axes (Accel XYZ + Gyro XYZ)")

    ax[0].plot(tz, dfz["Accel_X"], label="Accel_X", color="tab:blue")
    ax[0].plot(tz, dfz["Accel_Y"], label="Accel_Y", color="tab:orange")
    ax[0].plot(tz, dfz["Accel_Z"], label="Accel_Z", color="tab:green")
    ax[0].set_ylabel("Accel (m/s²)")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(loc="upper right")

    ax[1].plot(tz, dfz["Gyro_X"], label="Gyro_X", color="tab:blue")
    ax[1].plot(tz, dfz["Gyro_Y"], label="Gyro_Y", color="tab:orange")
    ax[1].plot(tz, dfz["Gyro_Z"], label="Gyro_Z", color="tab:green")
    ax[1].set_ylabel("Gyro (rad/s)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper right")

    save(fig, out_png)


def plot_zoom_axis_sd(dfz, out_png):
    tz = dfz["Time_s"].to_numpy(float)

    ax_cols = [pick_prefix(dfz, "Accel_X_HP_ABS_SD_"),
               pick_prefix(dfz, "Accel_Y_HP_ABS_SD_"),
               pick_prefix(dfz, "Accel_Z_HP_ABS_SD_")]
    gy_cols = [pick_prefix(dfz, "Gyro_X_HP_ABS_SD_"),
               pick_prefix(dfz, "Gyro_Y_HP_ABS_SD_"),
               pick_prefix(dfz, "Gyro_Z_HP_ABS_SD_")]

    if any(c is None for c in ax_cols + gy_cols):
        return

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(16, 7))
    fig.suptitle("Zoom: Axis SD of HP_ABS (Directional Activity)")

    ax[0].plot(tz, dfz[ax_cols[0]], label=ax_cols[0], color="tab:blue")
    ax[0].plot(tz, dfz[ax_cols[1]], label=ax_cols[1], color="tab:orange")
    ax[0].plot(tz, dfz[ax_cols[2]], label=ax_cols[2], color="tab:green")
    ax[0].set_ylabel("Accel SD (m/s²)")
    ax[0].grid(True, alpha=0.25)
    ax[0].legend(loc="upper right")

    ax[1].plot(tz, dfz[gy_cols[0]], label=gy_cols[0], color="tab:blue")
    ax[1].plot(tz, dfz[gy_cols[1]], label=gy_cols[1], color="tab:orange")
    ax[1].plot(tz, dfz[gy_cols[2]], label=gy_cols[2], color="tab:green")
    ax[1].set_ylabel("Gyro SD (rad/s)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True, alpha=0.25)
    ax[1].legend(loc="upper right")

    save(fig, out_png)


# ---------- file processing ----------
def process_file(path, sheet, out_dir, k, min_interval, prom_mult, width_s, zoom_len, zoom_start, zoom_end):
    df = pd.read_excel(path, sheet_name=sheet)
    tcol = pick_time_col(df)
    df = df.sort_values(tcol).reset_index(drop=True)

    # ensure standardized columns exist (created by feature script)
    need = ["Accel_X","Accel_Y","Accel_Z","Gyro_X","Gyro_Y","Gyro_Z",
            "Accel_Mag","Gyro_Mag","Accel_Mag_HP","Gyro_Mag_HP","Accel_Mag_HP_ABS","Gyro_Mag_HP_ABS"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"{path.name}: missing '{c}' (run feature script first)")

    df["Time_s"] = df[tcol].astype(float)
    t = df["Time_s"].to_numpy(float)
    fs = estimate_fs(t)

    plot_full_raw_mag(df, t, out_dir / "01_Full_RawMag.png")
    plot_full_hpabs_peaks_compare(df, t, fs, out_dir / "02_Full_HPABS_PeaksCompare.png",
                                  k, min_interval, prom_mult, width_s)
    plot_full_rms_sd(df, t, out_dir / "03_Full_RMS_SD.png")

    plot_raw_axes(df, t, out_dir / "04_Full_RawAxes_Accel_XYZ.png",
                  "Accel", ["Accel_X","Accel_Y","Accel_Z"], "Accel (m/s²)")
    plot_raw_axes(df, t, out_dir / "05_Full_RawAxes_Gyro_XYZ.png",
                  "Gyro", ["Gyro_X","Gyro_Y","Gyro_Z"], "Gyro (rad/s)")

    plot_axis_sd(df, t, out_dir / "06_Full_AxisSD_Accel_XYZ.png",
                 "Accel", ["Accel_X_HP_ABS_SD_","Accel_Y_HP_ABS_SD_","Accel_Z_HP_ABS_SD_"], "SD (m/s²)")
    plot_axis_sd(df, t, out_dir / "07_Full_AxisSD_Gyro_XYZ.png",
                 "Gyro", ["Gyro_X_HP_ABS_SD_","Gyro_Y_HP_ABS_SD_","Gyro_Z_HP_ABS_SD_"], "SD (rad/s)")

    # zoom window selection
    if zoom_start is not None and zoom_end is not None:
        z1, z2 = zoom_start, zoom_end
    else:
        tl = logic_peak_times(df, t, "IMU_ClickFlag_OR")
        center = float(tl[0]) if len(tl) else float(t[len(t)//2])
        z1, z2 = center - zoom_len/2, center + zoom_len/2               #just like median array

    dfz = slice_window(df, z1, z2)
    if dfz is not None:
        plot_zoom_dashboard(dfz, out_dir / "08_Zoom_Dashboard.png")
        plot_zoom_raw_axes(dfz, out_dir / "09_Zoom_RawAxes.png")
        plot_zoom_axis_sd(dfz, out_dir / "10_Zoom_AxisSD.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default="./ARTIFACTS")
    ap.add_argument("--sheet", default="IMU_Combined")
    ap.add_argument("--glob", default="*_with_IMU_features.xlsx")
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--k", type=float, default=6.0)
    ap.add_argument("--min_interval", type=float, default=0.08)
    ap.add_argument("--prom_mult", type=float, default=1.0)
    ap.add_argument("--width_s", type=float, default=0.0)

    ap.add_argument("--zoom_len", type=float, default=2.0)
    ap.add_argument("--zoom_start", type=float, default=None)
    ap.add_argument("--zoom_end", type=float, default=None)
    args = ap.parse_args()

    root = Path(args.artifacts_dir)
    files = sorted(root.rglob(args.glob) if args.recursive else root.glob(args.glob))
    files = [f for f in files if f.suffix.lower() == ".xlsx"]
    if not files:
        raise SystemExit(f"No Excel files found in {root} matching {args.glob}")

    if not SCIPY_OK:
        print("WARNING: SciPy not installed -> SciPy peak comparison skipped.")
        print("Install: pip install scipy")

    for f in files:
        out_dir = root / (f.stem[:40] + "_plots")
        process_file(f, args.sheet, out_dir,
                     args.k, args.min_interval, args.prom_mult, args.width_s,
                     args.zoom_len, args.zoom_start, args.zoom_end)
        print("Saved:", out_dir)


if __name__ == "__main__":
    main()