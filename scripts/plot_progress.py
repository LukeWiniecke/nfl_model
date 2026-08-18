"""Fixed autoresearch progress chart -- DO NOT MODIFY during an autoresearch run.

Reads results.tsv (every experiment logged so far) and writes progress.png:
development-era ensemble ATS ROI per experiment, kept vs. discarded/crashed,
with a running-best step line -- the same style of chart used to track the
gas_ibs autoresearch loop's Calmar Ratio.

Run any time during or after a loop:
    .venv\\Scripts\\python.exe scripts\\plot_progress.py
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_PATH = "results.tsv"
OUT_PATH = "progress.png"


def load_rows():
    with open(RESULTS_PATH, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    rows = load_rows()
    if not rows:
        print("results.tsv has no experiments logged yet -- nothing to plot.")
        return

    xs, rois, kept, labels = [], [], [], []
    running_best = None
    best_xs, best_ys = [], []

    for i, row in enumerate(rows, start=1):
        roi = float(row["dev_ats_roi"])
        status = row["status"]
        xs.append(i)
        rois.append(roi * 100)  # percentage points
        kept.append(status == "keep")
        labels.append(row["description"])

        if status == "keep":
            if running_best is None or roi > running_best:
                running_best = roi
                best_xs.append(i)
                best_ys.append(roi * 100)
        elif running_best is None:
            # first row is always the baseline run and should always be "keep";
            # this only fires if that convention was broken.
            running_best = roi
            best_xs.append(i)
            best_ys.append(roi * 100)

    # extend the running-best line flat to the last experiment, matching the
    # reference chart's step-plot look
    if best_xs and best_xs[-1] != xs[-1]:
        best_xs.append(xs[-1])
        best_ys.append(best_ys[-1])

    fig, ax = plt.subplots(figsize=(14, 7))

    kept_xs = [x for x, k in zip(xs, kept) if k]
    kept_ys = [y for y, k in zip(rois, kept) if k]
    disc_xs = [x for x, k in zip(xs, kept) if not k]
    disc_ys = [y for y, k in zip(rois, kept) if not k]

    ax.plot(best_xs, best_ys, color="tab:blue", linewidth=2, label="Running best dev ATS ROI", zorder=2)
    ax.scatter(disc_xs, disc_ys, color="tab:red", marker="x", s=70, label="Discarded/crash", zorder=3)
    ax.scatter(kept_xs, kept_ys, color="tab:green", marker="o", s=70, label="Kept (new best)", zorder=3)

    # annotate each kept point with its description
    for x, y, k, label in zip(xs, rois, kept, labels):
        if k:
            ax.annotate(
                f"{y:+.2f}%\n{label[:40]}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 8),
                fontsize=7,
                color="tab:green",
            )

    baseline = rois[0]
    best = best_ys[-1] if best_ys else rois[0]
    ax.set_title(
        f"Autoresearch Progress: Ensemble Dev ATS ROI Across All Experiments\n"
        f"Baseline -> Best: {baseline:+.2f}% -> {best:+.2f}%  ({best - baseline:+.2f}pp)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Development-era ATS ROI (%)")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Wrote {OUT_PATH} ({len(rows)} experiments, baseline {baseline:+.2f}% -> best {best:+.2f}%)")


if __name__ == "__main__":
    main()
