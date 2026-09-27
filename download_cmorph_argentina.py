#!/usr/bin/env python3
"""
CMORPH CDR daily precipitation -> Argentina subset
===================================================

WHAT THIS SCRIPT DOES
---------------------
For each day between START and END it:
  1. Downloads the global daily NetCDF file (0.25 deg) from NOAA NCEI
     into a temporary folder.
  2. Crops it to the Argentina domain (lat -55..-20, lon -74..-53).
  3. Saves the small cropped file in cmorph_argentina/daily/YYYY/.
  4. Deletes the global file immediately, so disk usage stays tiny.

Because step 4 happens right after each download, you never need space for
the whole global dataset - only for a few global files at a time
(N_WORKERS files, ~a few MB each).

The script is RESUMABLE: if a cropped file already exists, that day is
skipped. So if your connection drops, just run it again.

USAGE
-----
    python download_cmorph_argentina.py --test     # only 3 days, to check everything works
    python download_cmorph_argentina.py --download # full period START-END
    python download_cmorph_argentina.py            # no flag: shows a menu and asks what to run
    python download_cmorph_argentina.py --merge    # join daily files into one file per year
    python download_cmorph_argentina.py --merge-all  # join ALL daily files into a single file

REQUIREMENTS
------------
    pip install requests xarray netCDF4 numpy dask
"""
import argparse
import time
from datetime import date, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
import xarray as xr

# =============================================================================
# CONFIGURATION - edit this section if you need to change something
# =============================================================================

# Base URL of the daily 0.25 deg files. The full URL of one file looks like:
# .../access/daily/0.25deg/2024/01/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_20240101.nc
BASE_URL = ("https://www.ncei.noaa.gov/data/"
            "cmorph-high-resolution-global-precipitation-estimates/access/daily/0.25deg")

# File name pattern; {ymd} will be replaced by the date as YYYYMMDD.
FILE_PATTERN = "CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_{ymd}.nc"

# Default period to download (inclusive). Change only these values to switch
# the year(s) used by the script for normal runs and for the quick test run.
YEAR_START = 2000
YEAR_END = 2025

START = date(YEAR_START, 1, 1)
END = date(YEAR_END, 12, 31)
TEST_START = date(YEAR_START, 1, 1)
TEST_END = date(YEAR_START, 1, 3)

# Argentina domain.
# Latitude: -55 (south) to -20 (north). The file goes from -60 to 60.
LAT_MIN, LAT_MAX = -55.0, -20.0

# Longitude: I give it in the usual -180..180 form (74W to 53W).
# The file uses 0..360, so the script converts it below:
#   -74 -> 286 ,  -53 -> 307
LON_MIN, LON_MAX = -74.0, -53.0

# Output folders (created automatically).
OUT_DIR = Path("data/cmorph_argentina")
TMP_DIR = OUT_DIR / "tmp"        # global files live here only for a few seconds
DAILY_DIR = OUT_DIR / "daily"    # cropped daily files
YEARLY_DIR = OUT_DIR / "yearly"  # optional merged yearly files

# Name of the single file with ALL days merged (created by --merge-all).
FINAL_FILE = OUT_DIR / f"cmorph_argentina_{YEAR_START}_{YEAR_END}.nc"

# Number of downloads running at the same time. 4 is polite to the server;
# don't set it much higher or NOAA may throttle/block you.
N_WORKERS = 4

# Chunks used for the long-term merged products. We keep time blocks to about
# one year so that monthly/seasonal aggregation is fast and does not require
# re-chunking in the notebook later.
MERGE_CHUNKS = {"time": 365, "lat": -1, "lon": -1}

# How many times to retry a failed download before giving up on that day.
MAX_RETRIES = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def url_for(d: date) -> str:
    """Build the download URL for a given day.
    Example: date(2000, 2, 3) -> .../2000/02/CMORPH_..._20000203.nc"""
    ymd = d.strftime("%Y%m%d")
    return f"{BASE_URL}/{d:%Y}/{d:%m}/{FILE_PATTERN.format(ymd=ymd)}"


def daily_out_path(d: date) -> Path:
    """Where the cropped file for a given day is saved (one subfolder per year)."""
    return DAILY_DIR / f"{d:%Y}" / f"cmorph_arg_{d:%Y%m%d}.nc"


def daterange(start: date, end: date):
    """Yield every date from start to end, both included."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download(url: str, dest: Path) -> None:
    """Download `url` to the file `dest`, retrying on errors.

    stream=True + iter_content writes the file in 1 MB chunks, so we never
    hold the whole file in memory.
    Between retries we wait 2, 4, 8... seconds (exponential backoff), which
    helps when the server is temporarily busy.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()   # raises an error on 404, 500, etc.
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            return  # success
        except Exception:
            if attempt == MAX_RETRIES:
                raise              # out of retries: report the failure
            time.sleep(2 ** attempt)


def crop(src: Path, dst: Path) -> None:
    """Open the global file `src`, cut the Argentina domain, save it to `dst`."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(src) as ds:
        # Coordinate names can differ between products; support both.
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"

        # --- Longitude -------------------------------------------------------
        # The file is 0..360, so convert -74 -> 286 and -53 -> 307.
        # (The modulo 360 does exactly that; if the file were already
        # -180..180 we leave the values untouched.)
        lon_min, lon_max = LON_MIN, LON_MAX
        if float(ds[lon_name].max()) > 180:
            lon_min, lon_max = lon_min % 360, lon_max % 360

        # --- Latitude --------------------------------------------------------
        # .sel() with a slice needs the bounds in the same order as the axis.
        # If latitude is ascending (-60 -> 60) use (min, max); otherwise flip.
        lat_ascending = bool(ds[lat_name][0] < ds[lat_name][-1])
        lat_slice = slice(LAT_MIN, LAT_MAX) if lat_ascending else slice(LAT_MAX, LAT_MIN)

        # Select the box. Only the cropped part is read into memory (.load()).
        sub = ds.sel({lat_name: lat_slice, lon_name: slice(lon_min, lon_max)}).load()

        # Save with zlib compression to make the small files even smaller.
        encoding = {v: {"zlib": True, "complevel": 4} for v in sub.data_vars}
        sub.to_netcdf(dst, encoding=encoding)


def process_day(d: date) -> tuple[date, str]:
    """Full pipeline for one day: download -> crop -> delete global file."""
    out = daily_out_path(d)

    # Resume support: if this day is already done, do nothing.
    if out.exists():
        return d, "skipped (already exists)"

    tmp = TMP_DIR / FILE_PATTERN.format(ymd=d.strftime("%Y%m%d"))
    try:
        download(url_for(d), tmp)
        crop(tmp, out)
        return d, "ok"
    finally:
        # 'finally' runs even if download/crop fails, so the big global file
        # is ALWAYS removed and never piles up on your disk.
        tmp.unlink(missing_ok=True)


# =============================================================================
# MAIN ACTIONS
# =============================================================================

def run_download(start: date, end: date) -> None:
    """Process every day in [start, end], N_WORKERS days in parallel."""
    days = list(daterange(start, end))
    print(f"Processing {len(days)} days: {start} -> {end}")
    failed = []

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        # Launch one task per day.
        futures = {ex.submit(process_day, d): d for d in days}

        # as_completed gives results in the order they finish.
        for i, fut in enumerate(as_completed(futures), 1):
            d = futures[fut]
            try:
                _, status = fut.result()
                # Print every day for short runs (test), every 50 for long ones.
                if len(days) <= 10 or i % 50 == 0:
                    print(f"[{i}/{len(days)}] {d} {status}")
            except Exception as e:
                print(f"[FAILED] {d}: {e}")
                failed.append(d)

    # Save the list of failed days; re-running the script retries them,
    # because the successful days are skipped automatically.
    if failed:
        (OUT_DIR / "failed_days.txt").write_text(
            "\n".join(d.isoformat() for d in sorted(failed)))
        print(f"{len(failed)} days failed (see failed_days.txt). Re-run the script to retry.")
    else:
        print("All done.")


def merge_encoding(ds: xr.Dataset) -> dict:
    """Return NetCDF encoding for a merged dataset.

    We explicitly set chunk sizes for the long-term merged products so that
    monthly and seasonal aggregation works efficiently without requiring users
    to re-chunk the dataset later.
    """
    encoding = {}
    for name, var in ds.variables.items():
        if name in {"time", "time_bounds"}:
            encoding[name] = {
                "units": var.encoding.get("units", "seconds since 1970-01-01"),
                "calendar": var.encoding.get("calendar", "standard"),
                "dtype": var.encoding.get("dtype", np.dtype("int64")),
            }
        elif not np.issubdtype(var.dtype, np.datetime64) and not np.issubdtype(var.dtype, np.timedelta64):
            chunks = [ds.sizes[dim] for dim in var.dims]
            # keep roughly yearly blocks along time; keep lat/lon intact.
            if "time" in var.dims:
                chunks[var.dims.index("time")] = min(ds.sizes["time"], 365)
            encoding[name] = {
                "zlib": True,
                "complevel": 4,
                "chunksizes": tuple(chunks),
            }
    return encoding


def run_merge() -> None:
    """Join the daily cropped files into one NetCDF per year (optional step)."""
    YEARLY_DIR.mkdir(parents=True, exist_ok=True)
    for year_dir in sorted(DAILY_DIR.glob("*")):
        files = sorted(year_dir.glob("*.nc"))
        if not files:
            continue
        out = YEARLY_DIR / f"cmorph_arg_{year_dir.name}.nc"
        if out.exists():
            continue  # already merged
        with xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks=MERGE_CHUNKS,
            parallel=True,
        ) as ds:
            ds = ds.sortby("time")
            ds = ds.chunk(MERGE_CHUNKS)
            ds.to_netcdf(out, encoding=merge_encoding(ds))
        print(f"Merged {len(files)} files -> {out}")


def run_merge_all() -> None:
    """Join ALL the daily cropped files into ONE single NetCDF (FINAL_FILE).

    The final file is intentionally written with small, resample-friendly chunks
    along the time axis, so monthly/seasonal aggregation is easy and fast in
    downstream notebooks without manual chunk adjustments.
    """
    # Collect every cropped daily file (daily/YYYY/cmorph_arg_YYYYMMDD.nc).
    files = sorted(DAILY_DIR.glob("*/*.nc"))
    if not files:
        print(f"No daily files found in {DAILY_DIR}. Run the download first.")
        return
    if FINAL_FILE.exists():
        print(f"{FINAL_FILE} already exists. Rebuilding with smaller chunk layout for long-term aggregation.")
        FINAL_FILE.unlink(missing_ok=True)

    print(f"Merging {len(files)} daily files -> {FINAL_FILE}")

    with xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks=MERGE_CHUNKS,
        parallel=True,
    ) as ds:
        ds = ds.sortby("time")
        ds = ds.chunk(MERGE_CHUNKS)

        n_expected = (END - START).days + 1
        print(f"Time steps: {ds.sizes['time']} (expected {n_expected} for {START} -> {END})")
        if ds.sizes["time"] != n_expected:
            print("WARNING: number of days differs from expected. Check failed_days.txt or re-run the download.")

        ds.to_netcdf(FINAL_FILE, encoding=merge_encoding(ds))

    print(f"Done: {FINAL_FILE}")
    print("Once you have checked it, you can delete the 'daily' and 'yearly' "
          "folders to free space.")


def ask_user() -> str:
    """Show a small text menu and return the chosen action.

    The user can type the option name (e.g. "merge all") or its number.
    Keeps asking until the answer is valid. Returns one of:
    "test", "download", "merge", "merge_all", "quit".
    """
    # Map every accepted answer -> internal action name.
    options = {
        "1": "test",      "test": "test",
        "2": "download",  "download": "download",
        "3": "merge",     "merge": "merge",
        "4": "merge_all", "merge all": "merge_all", "merge-all": "merge_all",
        "5": "quit",      "quit": "quit", "exit": "quit",
    }
    print("\nWhat do you want to run?")
    print(f"  1) test       - download 3 days only ({TEST_START} -> {TEST_END})")
    print(f"  2) download   - full download {START} -> {END}")
    print("  3) merge      - merge daily files into yearly files")
    print("  4) merge all  - merge ALL daily files into a single file")
    print("  5) quit")
    while True:
        answer = input("> ").strip().lower()
        if answer in options:
            return options[answer]
        print("Not valid. Type a name or a number from the list above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and crop CMORPH CDR daily precipitation.")
    ap.add_argument("--test", action="store_true", help="Only process 3 days (1-3 Jan 2012)")
    ap.add_argument("--download", action="store_true", help="Full download for START -> END")
    ap.add_argument("--merge", action="store_true", help="Merge daily files into yearly files")
    ap.add_argument("--merge-all", action="store_true",
                    help="Merge ALL daily files into a single NetCDF file")
    args = ap.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Decide the action:
    #  - if a flag was given, use it (handy for automation / non-interactive runs)
    #  - if NO flag was given, ask the user in the terminal
    if args.merge_all:
        action = "merge_all"
    elif args.merge:
        action = "merge"
    elif args.test:
        action = "test"
    elif args.download:
        action = "download"
    else:
        action = ask_user()

    # Run the chosen action.
    if action == "merge_all":
        run_merge_all()
    elif action == "merge":
        run_merge()
    elif action == "test":
        run_download(TEST_START, TEST_END)
    elif action == "download":
        run_download(START, END)
    else:
        print("Nothing to do. Bye.")
