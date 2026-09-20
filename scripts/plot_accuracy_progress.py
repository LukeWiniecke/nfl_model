"""Accuracy/calibration companion to plot_progress.py, for the
autoresearch/sep19 run.

plot_progress.py charts development-era ATS ROI, which was the aug18 run's
only objective. The sep19 run added a second objective at the human's
request: make the ensemble accurate and calibrated against the market, not
just profitable on the backtest. ROI alone cannot show whether that worked,
and on its own it makes every anchoring experiment look like a regression.

Reads accuracy_log.tsv (one row per completed experiment, extracted from
each run's summary.json) and writes accuracy_progress.png with two panels:

  left   development-era MAE gap vs the closing line, per experiment. Zero
         means the model matches the market; below zero it beats it. This is
         the metric the run was actually trying to move.
  right  the trade-off frontier: development ATS ROI against that same MAE
         gap, so an experiment that bought accuracy with ROI is visible as
         movement down and to the left rather than down and to the right.
         Marker size is the development bet count, because an ROI figure on
         a collapsed bet universe is not comparable to one on a full book.

Run:
    .venv\\Scripts\\python.exe scripts\\plot_accuracy_progress.py
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, "accuracy_log.tsv")
OUT_PATH = os.path.join(ROOT, "accuracy_progress.png")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
GRAY = "#898781"
INK = "#0b0b0b"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows():
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        for k in ("ats_dev_roi", "ats_hold_roi", "ml_dev_roi", "dev_mae_gap", "hold_mae_gap",
                  "dev_rmse_gap", "dev_brier_ml_gap", "dev_slope_ml", "dev_slope_ats", "dev_ece_ml"):
            r[k] = _f(r.get(k))
        for k in ("ats_dev_n", "ats_hold_n", "ml_dev_n"):
            try:
                r[k] = int(float(r.get(k)))
            except (TypeError, ValueError):
                r[k] = 0
    return rows


def main():
    rows = load_rows()
    if not rows:
        print("accuracy_log.tsv has no rows -- nothing to plot.")
        return

    base = rows[0]
    base_mae = base["dev_mae_gap"]
    base_roi = base["ats_dev_roi"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- left: MAE gap per experiment -------------------------------------
    xs = list(range(1, len(rows) + 1))
    maes = [r["dev_mae_gap"] for r in rows]
    colors = [GREEN if (m is not None and m <= 0) else (BLUE if (m is not None and m < base_mae) else ORANGE)
              for m in maes]
    ax1.bar(xs, [0 if m is None else m for m in maes], color=colors, zorder=3)
    ax1.axhline(0, color=INK, linewidth=1.2, zorder=4)
    ax1.axhline(base_mae, color=GRAY, linewidth=1, linestyle="--", zorder=2,
                label=f"baseline gap {base_mae:+.3f}")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r["id"] for r in rows], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Development MAE minus market MAE (points)")
    ax1.set_title("Accuracy vs the closing line\n(0 = matches the market, below 0 = beats it)",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3, zorder=0)
    ax1.legend(loc="upper right", fontsize=9)

    # --- right: trade-off frontier ----------------------------------------
    # Several experiments land on exactly the same point -- a change that only
    # touches calibration leaves ROI and MAE untouched, and the anchored
    # variants share one bet universe. Plotting them separately just stacks
    # labels on top of each other, so coincident points are merged.
    merged = {}
    for r in rows:
        if r["dev_mae_gap"] is None or r["ats_dev_roi"] is None:
            continue
        key = (round(r["ats_dev_roi"], 4), round(r["dev_mae_gap"], 3))
        merged.setdefault(key, []).append(r)

    for (roi, mae), group in merged.items():
        rep = max(group, key=lambda r: r["ats_dev_n"])
        size = max(30.0, rep["ats_dev_n"] / 2.0)
        is_base = any(r["id"] == "BASE" for r in group)
        ax2.scatter(roi * 100, mae, s=size,
                    color=(GRAY if is_base else (GREEN if mae <= 0 else BLUE)),
                    edgecolor=INK, linewidth=0.8, zorder=3, alpha=0.85)
        ax2.annotate("/".join(r["id"] for r in group), (roi * 100, mae),
                     textcoords="offset points", xytext=(7, 6), fontsize=8, color=INK)

    ax2.axhline(0, color=INK, linewidth=1.2, zorder=2)
    ax2.axvline(base_roi * 100, color=GRAY, linewidth=1, linestyle="--", zorder=1)
    ax2.axhline(base_mae, color=GRAY, linewidth=1, linestyle="--", zorder=1)
    ax2.set_xlabel("Development ATS ROI (%)")
    ax2.set_ylabel("Development MAE minus market MAE (points)")
    ax2.set_title("Trade-off frontier\n(marker size = development bet count)",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, zorder=0)

    # Best = smallest MAE gap, breaking ties on ROI first and bet count
    # second, so the headline names the variant actually worth keeping.
    # Ties are frequent and must be treated as ties: a calibration-only change
    # moves neither ROI nor MAE, and several rows share one bet universe.
    # Rounded to 3 decimals deliberately -- a thousandth of a point of MAE is
    # far below anything meaningful here, and comparing at full float
    # precision let a row parsed from a 4-decimal summary.txt lose to an
    # identical one carried at full precision from a workflow result.
    best = min(
        (r for r in rows if r["dev_mae_gap"] is not None),
        key=lambda r: (round(r["dev_mae_gap"], 3), -(r["ats_dev_roi"] or 0), -r["ats_dev_n"]),
    )
    fig.suptitle(
        f"Autoresearch sep19: accuracy and calibration against the market\n"
        f"baseline MAE gap {base_mae:+.3f} at {base_roi * 100:+.2f}% ROI"
        f"   ->   best {best['id']} gap {best['dev_mae_gap']:+.3f} at {best['ats_dev_roi'] * 100:+.2f}%"
        f" on {best['ats_dev_n']} bets",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH} ({len(rows)} experiments)")


if __name__ == "__main__":
    main()
