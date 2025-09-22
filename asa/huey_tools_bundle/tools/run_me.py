#!/usr/bin/env python3
"""
Unified entry point for Huey Tools Bundle with timestamped logging.

Edit the paths in the USER EDIT SECTION, then run:
    python tools/run_me.py

This will:
  1) Align two spaces with pseudo‑R (real/imag separate; no scaling).
  2) Run the covariance Hebbian demo.
It writes a timestamped log file under tools/logs/.
"""
import subprocess
from pathlib import Path
from datetime import datetime

# --- USER EDIT SECTION ---
SRC = "tools/templates/en_coords.csv"            # replace with your source CSV
TGT = "tools/templates/zh_coords.csv"            # replace with your target CSV
MAP = "tools/templates/mapping.csv"              # optional, or set to None
FREE = "tools/templates/free_concepts.txt"       # optional, or set to None
# -------------------------

LOGS = Path("tools/logs")
LOGS.mkdir(parents=True, exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOGFILE = LOGS / f"run_{STAMP}.txt"

def log(header, text):
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"\n=== {header} ===\n")
        f.write(text if isinstance(text, str) else str(text))
        f.write("\n")
    # Echo to console as well
    print(f"\n=== {header} ===\n{text}\n")

def run_and_log(cmd, header):
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = (res.stdout or '').strip()
    err = (res.stderr or '').strip()
    log(header + " CMD", " ".join(cmd))
    if out:
        log(header + " STDOUT", out)
    if err:
        log(header + " STDERR", err)
    if res.returncode != 0:
        log(header + " RETURN CODE", str(res.returncode))
        raise SystemExit(f"Command failed: {' '.join(cmd)} (see {LOGFILE})")
    return res

def run_alignment():
    cmd = ["python", "tools/galileo_pseudoR_align.py", "--src", SRC, "--tgt", TGT]
    if MAP:  cmd.extend(["--map", MAP])
    if FREE: cmd.extend(["--free", FREE])
    log("PARAMS", f"SRC={SRC}\nTGT={TGT}\nMAP={MAP}\nFREE={FREE}\nLOGFILE={LOGFILE}")
    print(">>> Running alignment tool...")
    run_and_log(cmd, "ALIGNMENT")
    print("Alignment complete.")

def run_hebb_demo():
    print(">>> Running covariance Hebbian demo...")
    run_and_log(["python", "-m", "tools.huey.examples.cov_hebb_demo"], "HEBB_DEMO")
    print("Hebbian demo complete.")

if __name__ == "__main__":
    log("START", f"Huey Tools Bundle run at {STAMP}")
    run_alignment()
    run_hebb_demo()
    log("DONE", "All steps completed successfully.")
    print(f"\nAll done. Log written to: {LOGFILE}")
