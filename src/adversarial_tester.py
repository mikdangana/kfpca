"""adversarial_tester.py

Generate *adversarial / unseen* versions of a CPU‑usage time‑series CSV,
run `test_pca()` from **ekf.py** on every generated file, collect the RMSE
(or whatever error `test_pca()` returns), and plot how the error grows with
severity.

Scenario generators implemented
-------------------------------
1. **Gaussian noise** – add N(0, σ) to every sample; σ in {1 %, 2 %, … 20 % of
   the original series std}.
2. **Spike noise** – insert `k` random spikes of height `spike_mag × σ`;
   k∈{1 %, 5 %, 10 % of len(series)}.
3. **Linear drift** – add a ramp that reaches `drift_mag × σ` at the end.

You can add more by appending to the `SCENARIOS` list.

Outputs
-------
* `data_adversarial/SCENARIO_#.csv` – mutated data sets
* `results.csv`            – table: scenario, severity, error
* `results.png`            – matplotlib plot of error vs. severity.

Run:
-----
```
python adversarial_tester.py  \
       -f data/cpu_trace.csv \
       -x P              # metric column (default 'P')
```
"""

import argparse, os, sys, subprocess, shutil, random, tempfile, csv, json
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------------

def gaussian(series: np.ndarray, severity: float) -> np.ndarray:
    """Add iid Gaussian noise. severity = std‑multiplier (e.g. 0.05→ 5 %)."""
    sigma = severity * np.std(series)
    return series + np.random.normal(0, sigma, size=series.shape)

def spikes(series: np.ndarray, severity: float) -> np.ndarray:
    """Inject random spikes. severity = fraction of samples to spike."""
    out = series.copy()
    k = int(len(series) * severity)
    idx = np.random.choice(len(series), k, replace=False)
    #spike_sigma = 5 * np.std(series)
    spike_sigma = 4 * np.std(series)
    out[idx] += np.random.normal(0, spike_sigma, size=k)
    return out

def drift(series: np.ndarray, severity: float) -> np.ndarray:
    """Add linear drift reaching ±severity·σ at the end."""
    sigma = np.std(series)
    ramp = np.linspace(0, severity * sigma, len(series))
    return series + ramp

SCENARIOS = [
    ("gaussian",  gaussian, [0.01, 0.03, 0.05, 0.1, 0.2]),
    ("spikes",    spikes,   [0.01, 0.05, 0.1, 0.2]),
    ("drift",     drift,    [0.01, 0.05, 0.1, 0.2])
]

# -----------------------------------------------------------------------------
# Harness to run ekf.test_pca on a generated file
# -----------------------------------------------------------------------------

def run_test_pca(csv_path: Path, column: str, std) -> float:
    """Invoke ekf.test_pca() programmatically and return its error value."""
    import importlib, importlib.util, types
    import ekf
    from importlib import reload

    # Build fake argv for ekf.test_pca()
    save_argv = sys.argv[:]
    sys.argv = ["ekf.py", "--testpcacsv", "-f", str(csv_path), "-x", column, "-y", column]
    try:
        err = ekf.test_pca()   # returns error metric
    finally:
        sys.argv = save_argv
    return float(err[1 if std in sys.argv else 0])

# -----------------------------------------------------------------------------
# Main experiment driver
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", default="data/cpu_analysis.csv", help="Source CSV with baseline CPU metric")
    ap.add_argument("-x", default="CPU", help="Metric column to mutate (default CPU)")
    ap.add_argument("--outdir", default="data_adversarial", help="Directory to write mutated CSVs")
    ap.add_argument("--std", default=False, help="Analyze error mean or std")
    args = ap.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(args.f)
    if args.x not in df.columns:
        raise ValueError(f"Column {args.x} not found in {args.f}")

    baseline = df[args.x].values.astype(float)

    results: List[Tuple[str, float, float]] = []

    for name, func, severities in SCENARIOS:
        for sev in severities:
            mutated = func(baseline, sev)
            mdf = df.copy()
            mdf[args.x] = mutated
            fname = out_dir / f"{name}_{sev:.3f}.csv"
            mdf.to_csv(fname, index=False)
            err = run_test_pca(fname, args.x, args.std)
            results.append((name, sev, err))
            print(f"{name:8s} sev={sev:<5} -> err={err:.4f}")

    # Save results CSV
    res_df = pd.DataFrame(results, columns=["scenario", "severity", "error"])
    res_df.to_csv(out_dir / "results.csv", index=False)

    # Plot
    plt.figure(figsize=(8,5))
    for name in res_df["scenario"].unique():
        sub = res_df[res_df.scenario==name]
        plt.plot(sub["severity"], sub["error"], marker="o", label=name)
    plt.xlabel("Severity parameter (fraction or σ‑multiplier)")
    plt.ylabel("Error returned by test_pca()")
    plt.title("EKF PCA error under adversarial perturbations (p=32)")
    plt.legend()
    plt.grid(True, ls="--", alpha=.4)
    plt.tight_layout()
    plt.savefig(out_dir / "results.png", dpi=180)
    plt.show()

if __name__ == "__main__":
    main()

