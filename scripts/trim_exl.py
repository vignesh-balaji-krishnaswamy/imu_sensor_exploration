import os
import re
import argparse
import numpy as np
import pandas as pd

ACC_SHEET = "Accelerometer"
GYR_SHEET = "Gyroscope"
TIME_COL  = "Time (s)"

ACC_COLS = ["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]
GYR_COLS = ["Gyroscope x (rad/s)", "Gyroscope y (rad/s)", "Gyroscope z (rad/s)"]


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[<>:\"/\\|?*\n\r\t]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def discover_excel_files(root: str) -> list[str]:
    """If root is a file => return [root]. If folder => recursive find .xls/.xlsx."""
    root = os.path.abspath(root)
    if os.path.isfile(root) and root.lower().endswith((".xls", ".xlsx")):
        return [root]

    files = []
    for r, dirs, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".xls", ".xlsx")):
                files.append(os.path.join(r, f))
    return sorted(files)


def reader_engine(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "openpyxl" if ext == ".xlsx" else "xlrd"  # xlrd needed for .xls check phyphox option while exporting datasets


def sync_trim_and_combine(acc_df, gyr_df, start_s, end_s):
    # common overlap
    t0 = max(float(acc_df[TIME_COL].min()), float(gyr_df[TIME_COL].min()))
    t1 = min(float(acc_df[TIME_COL].max()), float(gyr_df[TIME_COL].max()))

    lo = t0 + float(start_s)
    hi = t1 - float(end_s)

    # trim BOTH by the same window
    acc_t = acc_df[(acc_df[TIME_COL] >= lo) & (acc_df[TIME_COL] <= hi)].copy()
    gyr_t = gyr_df[(gyr_df[TIME_COL] >= lo) & (gyr_df[TIME_COL] <= hi)].copy()

    # combine by exact time match (since you said timestamps match)
    combined = acc_t[[TIME_COL] + ACC_COLS].merge(
        gyr_t[[TIME_COL] + GYR_COLS],
        on=TIME_COL,
        how="inner"
    )


    # optional: clean combined column names (easier later)
    combined = combined.rename(columns={
        ACC_COLS[0]: "Accel_X", ACC_COLS[1]: "Accel_Y", ACC_COLS[2]: "Accel_Z",
        GYR_COLS[0]: "Gyro_X",  GYR_COLS[1]: "Gyro_Y",  GYR_COLS[2]: "Gyro_Z",
    })

    return combined#acc_out, gyr_out, combined


def trim_one_file(input_path: str, out_dir: str, start_s: float, end_s: float) -> str:
    eng = reader_engine(input_path)

    xls = pd.ExcelFile(input_path, engine=eng)

    sheets = {}
    for name in xls.sheet_names:
     df = pd.read_excel(input_path, sheet_name=name, engine=eng)
     sheets[name] = df
    
    #combined synchronized trimming
    if ACC_SHEET in sheets and GYR_SHEET in sheets:
        combined = sync_trim_and_combine(
        sheets[ACC_SHEET],
        sheets[GYR_SHEET],
        start_s,
        end_s
    )

    sheets["IMU_Combined"] = combined
    
    parent = sanitize_name(os.path.basename(os.path.dirname(input_path)))[:60]
    base = sanitize_name(os.path.splitext(os.path.basename(input_path))[0])[:80]
    out_path = os.path.join(out_dir, f"{parent}_{base}_TRIMMED.xlsx")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input Excel file OR root folder (recursive).")
    ap.add_argument("--start", type=float, default=1.0, help="Trim first N seconds (default 1.0).")
    ap.add_argument("--end", type=float, default=1.0, help="Trim last N seconds (default 1.0).")
    args = ap.parse_args()

    out_dir = os.path.join(os.getcwd(), "..", "ARTIFACTS")
    ensure_dir(out_dir)

    files = discover_excel_files(args.input)
    if not files:
        raise SystemExit("No .xls/.xlsx files found.")

    for f in files:
        out = trim_one_file(f, out_dir, args.start, args.end)
        print("Saved:", out)

if __name__ == "__main__":
    main()