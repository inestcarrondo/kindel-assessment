#!/usr/bin/env python3
"""
Relative noise report of full dataset for replicate counts with a
dropout analysis and comparison figures.

This script does the following:
Downloads data (1M version) if missing
Uses the full file
Outputs to  benchmark_outputs/noise_relative/<target>/
"""

from pathlib import Path
import subprocess
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

DATA_ROOT       = "s3://kin-del-2024/data"
LOCAL_DATA_DIR  = "./data"
RESULTS_DIR     = "benchmark_outputs"
TARGETS         = ["mapk14", "ddr1"]
EPS = 1e-9
DROPOUT_THRESHOLD = 1.0   # if the count ≥ threshold = "present" in a replicate

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor":   "#0e1117",
    "savefig.facecolor":"#0e1117",
    "axes.edgecolor":   "#c9d1d9",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#c9d1d9",
    "ytick.color":      "#c9d1d9",
    "text.color":       "#e6edf3",
    "grid.color":       "#30363d",
    "axes.grid":        False,
})

def ensure_full_parquet(target: str) -> Path:
    # Makes sure that  LOCAL_DATA_DIR/target_1M.parquet exists and  download if not
    data_root = DATA_ROOT.rstrip("/")
    s3 = f"{data_root}/{target}_1M.parquet"
    local_in = Path(LOCAL_DATA_DIR) / f"{target}_1M.parquet"
    if not local_in.exists():
        local_in.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {s3} -> {local_in}")
        subprocess.run(["aws", "s3", "cp", s3, str(local_in), "--no-sign-request"], check=True)
    return local_in

def find_replicate_cols(df: pd.DataFrame, prefix: str) -> list:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort()
    return cols

def compute_cv_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    vals = df[cols].astype(float).values
    mean = vals.mean(axis=1)
    std  = vals.std(axis=1, ddof=1)
    cv   = std / (mean + EPS)
    return pd.DataFrame({"mean": mean, "std": std, "cv": cv}, index=df.index)

def pairwise_log2fc(df: pd.DataFrame, cols: list) -> dict:
    out = {}
    for a, b in itertools.combinations(cols, 2):
        ratio = (df[a].astype(float) + EPS) / (df[b].astype(float) + EPS)
        out[(a, b)] = np.log2(ratio)
    return out

def ma_data(df: pd.DataFrame, a: str, b: str):
    la = np.log2(df[a].astype(float) + EPS)
    lb = np.log2(df[b].astype(float) + EPS)
    A = 0.5 * (la + lb)
    M = la - lb
    return A, M

def savefig_if_missing(fig: plt.Figure, path: Path, pdf: PdfPages | None):
    path = Path(path)
    if path.exists():
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    if pdf: pdf.savefig(fig)
    plt.close(fig)
    return True

# PLOTS
def plot_cv_hist(cv_df, title, out_png, pdf=None):
    if Path(out_png).exists(): return
    fig, ax = plt.subplots()
    ax.hist(cv_df["cv"].replace([np.inf, -np.inf], np.nan).dropna(), bins=60)
    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title} — CV distribution")
    savefig_if_missing(fig, out_png, pdf)

def plot_cv_vs_mean(cv_df, title, out_png, pdf=None):
    if Path(out_png).exists(): return
    fig, ax = plt.subplots()
    ax.scatter(cv_df["mean"], cv_df["cv"], s=6, alpha=0.7)
    ax.set_xlabel("Mean count (linear)")
    ax.set_ylabel("CV (std / mean)")
    ax.set_title(f"{title} — CV vs Mean")
    savefig_if_missing(fig, out_png, pdf)

def plot_pairwise_log2fc(log2fc_dict, title_prefix, outdir, pdf=None):
    for (a, b), s in log2fc_dict.items():
        out_png = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_log2fc_{a}_vs_{b}.png"
        if out_png.exists(): continue
        fig, ax = plt.subplots()
        ax.hist(s.replace([np.inf, -np.inf], np.nan).dropna(), bins=80)
        ax.set_xlabel(f"log2 fold-change [{a} / {b}]")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title_prefix} — log2FC: {a} vs {b}")
        savefig_if_missing(fig, out_png, pdf)

def plot_ma_pairs(df, cols, title_prefix, outdir, pdf=None):
    for a, b in itertools.combinations(cols, 2):
        out_png = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_MA_{a}_vs_{b}.png"
        if out_png.exists(): continue
        A, M = ma_data(df, a, b)
        fig, ax = plt.subplots()
        ax.scatter(A, M, s=6, alpha=0.7)
        ax.set_xlabel("A = mean log2 count")
        ax.set_ylabel(f"M = log2 ratio ({a} - {b})")
        ax.set_title(f"{title_prefix} — MA: {a} vs {b}")
        savefig_if_missing(fig, out_png, pdf)

def plot_spearman_heatmap(df, cols, title, out_png, pdf=None):
    if Path(out_png).exists(): return
    if len(cols) < 2: return
    corr_df = df[cols].astype(float).corr(method="spearman")
    corr = corr_df.values
    fig, ax = plt.subplots()
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    ax.set_title(title)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig_if_missing(fig, out_png, pdf)

def avg_pairwise_spearman(df, cols, min_unique=2, min_n=3):
    if len(cols) < 2:
        return np.nan

    rhos = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            x = df[cols[i]].to_numpy(dtype=float)
            y = df[cols[j]].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]; y = y[mask]
            if x.size < min_n:
                continue
            if np.unique(x).size < min_unique or np.unique(y).size < min_unique:
                continue
            r = spearmanr(x, y).correlation
            if np.isfinite(r):
                rhos.append(r)

    return float(np.mean(rhos)) if rhos else np.nan


def plot_spearman_vs_mean_count(
    df, cols, target_name, label_prefix, out_png, pdf=None,
    n_bins=12, min_bin_n=200
):
    out_png = Path(out_png)
    if out_png.exists() or len(cols) < 2:
        return

    m = df[cols].astype(float).mean(axis=1)
    edges = np.unique(np.quantile(m[~m.isna()], np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.quantile(m[~m.isna()], np.linspace(0, 1, 4)))

    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (m >= lo) & (m < hi)
        sub = df.loc[mask, cols]
        if len(sub) < min_bin_n:
            continue
        rho = avg_pairwise_spearman(sub, cols)
        if np.isfinite(rho):
            xs.append(float(np.median(m[mask])))
            ys.append(float(rho))
            ns.append(int(len(sub)))

    if not xs:
        return

    xs, ys, ns = map(np.array, (xs, ys, ns))

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    # annotate points with n
    for x, y, n in zip(xs, ys, ns):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8)

    # dynamic y-limits + margins
    y_max = float(np.nanmax(ys)) if ys.size else 0.0
    y_min = float(np.nanmin(ys)) if ys.size else 0.0
    top = min(1.05, y_max + 0.12)
    bottom = max(-1.0, min(y_min - 0.06, -0.06))
    ax.set_ylim(bottom, top)

    ax.set_xscale("log")
    ax.margins(x=0.02)
    ax.set_xlabel("Mean count across replicates (log)")
    ax.set_ylabel("Avg pairwise Spearman")
    ax.set_title(f"{target_name} — {label_prefix} — Spearman vs mean count", pad=18)

    fig.tight_layout(pad=1.6)

    savefig_if_missing(fig, out_png, pdf)
    
def plot_spearman_by_presence_threshold(
    df, cols, target_name, label_prefix, out_png, pdf=None,
    thresholds=(1, 2, 5, 10, 20), require_reps=2, min_n=200
):
    
    #Plot of avg pairwise Spearman after requiring presence (count >= t) in >= require_reps replicates
    
    out_png = Path(out_png)
    if out_png.exists() or len(cols) < 2:
        return

    vals = df[cols].astype(float).values
    rhos, labels, ns = [], [], []

    for t in thresholds:
        present = (vals >= float(t)).astype(int)
        keep = present.sum(axis=1) >= int(require_reps)
        idx = np.where(keep)[0]
        ns.append(int(idx.size))
        labels.append(str(t))
        if idx.size < min_n:
            rhos.append(np.nan)
            continue
        sub = df.iloc[idx][cols]
        rhos.append(avg_pairwise_spearman(sub, cols))

    if not np.isfinite(rhos).any():
        return

    fig, ax = plt.subplots()
    heights = [r if np.isfinite(r) else 0.0 for r in rhos]
    bars = ax.bar(labels, heights, alpha=0.85)

    finite_rhos = np.array([r for r in rhos if np.isfinite(r)])
    ymax_val = float(finite_rhos.max()) if finite_rhos.size else 0.0
    ymin_val = float(finite_rhos.min()) if finite_rhos.size else 0.0
    top = min(1.05, ymax_val + 0.18)           # headroom above tallest bar
    bottom = max(-1.0, min(ymin_val - 0.08, -0.08))  # a bit of room below
    ax.set_ylim(bottom, top)

    y_offset = 0.03 * (top - bottom)

    for bar, r, n in zip(bars, rhos, ns):
        y = 0.0 if not np.isfinite(r) else r
        txt = ("NA" if not np.isfinite(r) else f"{r:.2f}") + f"\n(n={n})"
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                y + y_offset, txt, ha="center", va="bottom", fontsize=9)

    ax.set_xlabel(f"Presence threshold (count ≥ t) with ≥{require_reps} reps present")
    ax.set_ylabel("Avg pairwise Spearman")
    ax.set_title(f"{target_name} — {label_prefix} — Spearman after presence filtering", pad=18)

    fig.tight_layout(pad=1.6)

    savefig_if_missing(fig, out_png, pdf)



# Dropout 
def plot_dropout_effect(df, cols, title_prefix, outdir, pdf=None, threshold=DROPOUT_THRESHOLD):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    values = df[cols].astype(float).values
    present = (values >= float(threshold)).astype(int)
    n_reps = present.shape[1]
    num_present = present.sum(axis=1)
    num_zeros = n_reps - num_present
    mean_counts = values.mean(axis=1)
    min_counts = values.min(axis=1)

    #Presence bar chart
    frac = [(num_present == k).mean() for k in range(n_reps + 1)]
    out1 = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_dropout_presence.png"
    if not out1.exists():
        fig, ax = plt.subplots()
        ax.bar(range(n_reps + 1), frac)
        ax.set_xticks(range(n_reps + 1))
        ax.set_xlabel("Number of replicates with count ≥ threshold")
        ax.set_ylabel("Fraction of molecules")
        ax.set_title(f"{title_prefix} — replicate presence (threshold={threshold})")
        savefig_if_missing(fig, out1, pdf)

    #Mean vs # of zeros
    out2 = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_mean_vs_zero_reps.png"
    if not out2.exists():
        fig, ax = plt.subplots()
        ax.scatter(mean_counts, num_zeros, s=3, alpha=0.5)
        ax.set_xlabel("Mean count across replicates (linear)")
        ax.set_ylabel("Number of zero replicates")
        ax.set_title(f"{title_prefix} — mean vs #zero replicates")
        savefig_if_missing(fig, out2, pdf)

    #Min-count histogram
    out3 = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_min_count_hist.png"
    if not out3.exists():
        fig, ax = plt.subplots()
        ax.hist(min_counts, bins=80)
        ax.set_xlabel("Minimum count across replicates")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title_prefix} — min replicate count")
        savefig_if_missing(fig, out3, pdf)

    #Max |log2FC| when ≥1 zero replicate
    out4 = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_dropout_log2fc_tails.png"
    if not out4.exists():
        l2 = pairwise_log2fc(df, cols)
        max_abs_l2 = np.max(np.vstack([np.abs(s.values) for s in l2.values()]), axis=0)
        mask_dropout = (num_zeros >= 1)
        fig, ax = plt.subplots()
        ax.hist(max_abs_l2[mask_dropout], bins=80)
        ax.set_xlabel("Max |log2FC| across replicate pairs")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{title_prefix} — log2FC tails (dropout cases)")
        savefig_if_missing(fig, out4, pdf)

    #summary csv
    summary_csv = Path(outdir) / f"{title_prefix.replace(' ','_').lower()}_dropout_summary.csv"
    if not summary_csv.exists():
        summary = {
            "threshold": [threshold],
            **{f"frac_present_{k}": [frac[k]] for k in range(len(frac))},
            "mean_num_zeros": [float(num_zeros.mean())],
            "median_num_zeros": [float(np.median(num_zeros))],
        }
        pd.DataFrame(summary).to_csv(summary_csv, index=False)

def make_dropout_side_by_side_summary(df, t_cols, m_cols, out_png, threshold=DROPOUT_THRESHOLD, pdf=None):
    out_png = Path(out_png)
    if out_png.exists(): return
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Dropout comparison — Targets vs Matrices")

    def presence_frac(vals):
        pres = (vals >= float(threshold)).astype(int)
        n = pres.shape[1]
        num_present = pres.sum(axis=1)
        return [(num_present == k).mean() for k in range(n+1)]

    #targets presence
    ax11 = fig.add_subplot(2, 2, 1)
    tv = df[t_cols].astype(float).values
    tfrac = presence_frac(tv)
    ax11.bar(range(len(tfrac)), tfrac)
    ax11.set_xticks(range(len(tfrac)))
    ax11.set_title(f"Targets — presence (≥{threshold})")
    ax11.set_xlabel("# replicates present"); ax11.set_ylabel("Fraction")

    #matrices presence
    ax12 = fig.add_subplot(2, 2, 2)
    mv = df[m_cols].astype(float).values
    mfrac = presence_frac(mv)
    ax12.bar(range(len(mfrac)), mfrac)
    ax12.set_xticks(range(len(mfrac)))
    ax12.set_title(f"Matrices — presence (≥{threshold})")
    ax12.set_xlabel("# replicates present"); ax12.set_ylabel("Fraction")

    #targets dropout log2FC tails
    ax21 = fig.add_subplot(2, 2, 3)
    tl2 = pairwise_log2fc(df, t_cols)
    tmax_abs = np.max(np.vstack([np.abs(s.values) for s in tl2.values()]), axis=0)
    t_num_present = ((tv >= float(threshold)).astype(int)).sum(axis=1)
    t_mask = (t_num_present < tv.shape[1])
    ax21.hist(tmax_abs[t_mask], bins=80)
    ax21.set_title("Targets — max |log2FC| (dropout cases)")
    ax21.set_xlabel("Max |log2FC|"); ax21.set_ylabel("Frequency")

    #matrices dropout log2FC tails
    ax22 = fig.add_subplot(2, 2, 4)
    ml2 = pairwise_log2fc(df, m_cols)
    mmax_abs = np.max(np.vstack([np.abs(s.values) for s in ml2.values()]), axis=0)
    m_num_present = ((mv >= float(threshold)).astype(int)).sum(axis=1)
    m_mask = (m_num_present < mv.shape[1])
    ax22.hist(mmax_abs[m_mask], bins=80)
    ax22.set_title("Matrices — max |log2FC| (dropout cases)")
    ax22.set_xlabel("Max |log2FC|"); ax22.set_ylabel("Frequency")

    savefig_if_missing(fig, out_png, pdf)

# existing side-by-side summary for corr + CV vs mean
def make_side_by_side_summary(df, t_cols, m_cols, out_png, title="Targets vs Matrices — Summary", pdf=None):
    out_png = Path(out_png)
    if out_png.exists(): return

    t_corr = df[t_cols].astype(float).corr(method="spearman") if len(t_cols) >= 2 else None
    m_corr = df[m_cols].astype(float).corr(method="spearman") if len(m_cols) >= 2 else None
    cv_t = compute_cv_table(df, t_cols) if len(t_cols) >= 2 else None
    cv_m = compute_cv_table(df, m_cols) if len(m_cols) >= 2 else None

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(title)

    ax11 = fig.add_subplot(2, 2, 1)
    if t_corr is not None:
        im = ax11.imshow(t_corr.values, vmin=-1, vmax=1)
        ax11.set_xticks(range(len(t_corr))); ax11.set_xticklabels(t_corr.columns, rotation=45, ha="right")
        ax11.set_yticks(range(len(t_corr))); ax11.set_yticklabels(t_corr.columns)
        ax11.set_title("Spearman — Targets")
        for i in range(t_corr.shape[0]):
            for j in range(t_corr.shape[1]):
                ax11.text(j, i, f"{t_corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax11, fraction=0.046, pad=0.04)
    else:
        ax11.axis("off"); ax11.set_title("Spearman — Targets (N/A)")

    ax12 = fig.add_subplot(2, 2, 2)
    if m_corr is not None:
        im = ax12.imshow(m_corr.values, vmin=-1, vmax=1)
        ax12.set_xticks(range(len(m_corr))); ax12.set_xticklabels(m_corr.columns, rotation=45, ha="right")
        ax12.set_yticks(range(len(m_corr))); ax12.set_yticklabels(m_corr.columns)
        ax12.set_title("Spearman — Matrices")
        for i in range(m_corr.shape[0]):
            for j in range(m_corr.shape[1]):
                ax12.text(j, i, f"{m_corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax12, fraction=0.046, pad=0.04)
    else:
        ax12.axis("off"); ax12.set_title("Spearman — Matrices (N/A)")

    ax21 = fig.add_subplot(2, 2, 3)
    if cv_t is not None:
        ax21.scatter(cv_t["mean"], cv_t["cv"], s=6, alpha=0.7)
        ax21.set_xlabel("Mean count (Targets)"); ax21.set_ylabel("CV (std/mean)")
        ax21.set_title("CV vs Mean — Targets")
    else:
        ax21.axis("off"); ax21.set_title("CV vs Mean — Targets (N/A)")

    ax22 = fig.add_subplot(2, 2, 4)
    if cv_m is not None:
        ax22.scatter(cv_m["mean"], cv_m["cv"], s=6, alpha=0.7)
        ax22.set_xlabel("Mean count (Matrices)"); ax22.set_ylabel("CV (std/mean)")
        ax22.set_title("CV vs Mean — Matrices")
    else:
        ax22.axis("off"); ax22.set_title("CV vs Mean — Matrices (N/A)")

    savefig_if_missing(fig, out_png, pdf)

def run_for_target(target: str):
    src = ensure_full_parquet(target)
    df = pd.read_parquet(src)

    t_cols = find_replicate_cols(df, "seq_target_")
    m_cols = find_replicate_cols(df, "seq_matrix_")

    out_dir = Path(RESULTS_DIR) / "noise_relative" / target
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{target}_relative_noise_report.pdf"
    pdf = PdfPages(pdf_path) if not Path(pdf_path).exists() else None

    # ---- TARGETS ----
    if len(t_cols) >= 2:
        cv_targets_csv = out_dir / "cv_targets.csv"
        if not cv_targets_csv.exists():
            compute_cv_table(df, t_cols).to_csv(cv_targets_csv, index=False)

        plot_cv_hist(pd.read_csv(cv_targets_csv), "Targets",
                     out_dir / "targets_cv_hist.png", pdf)
        plot_cv_vs_mean(pd.read_csv(cv_targets_csv), "Targets",
                        out_dir / "targets_cv_vs_mean.png", pdf)
        plot_pairwise_log2fc(pairwise_log2fc(df, t_cols), "Targets", out_dir, pdf)
        plot_ma_pairs(df, t_cols, "Targets", out_dir, pdf)
        plot_spearman_heatmap(df, t_cols,
                              f"{target.upper()} — Spearman correlation — Targets",
                              out_dir / "targets_spearman_heatmap.png", pdf)
        plot_dropout_effect(df, t_cols, "Targets", out_dir, pdf)

        plot_spearman_vs_mean_count(
            df, t_cols, target.upper(), "Targets",
            out_png=out_dir / "targets_spearman_vs_mean.png",
            pdf=pdf
        )
        plot_spearman_by_presence_threshold(
            df, t_cols, target.upper(), "Targets",
            out_png=out_dir / "targets_spearman_by_presence.png",
            pdf=pdf,
            thresholds=(1, 2, 5, 10, 20),
            require_reps=2
        )

    #matrices
    if len(m_cols) >= 2:
        cv_matrices_csv = out_dir / "cv_matrices.csv"
        if not cv_matrices_csv.exists():
            compute_cv_table(df, m_cols).to_csv(cv_matrices_csv, index=False)

        plot_cv_hist(pd.read_csv(cv_matrices_csv), "Matrices",
                     out_dir / "matrices_cv_hist.png", pdf)
        plot_cv_vs_mean(pd.read_csv(cv_matrices_csv), "Matrices",
                        out_dir / "matrices_cv_vs_mean.png", pdf)
        plot_pairwise_log2fc(pairwise_log2fc(df, m_cols), "Matrices", out_dir, pdf)
        plot_ma_pairs(df, m_cols, "Matrices", out_dir, pdf)
        plot_spearman_heatmap(df, m_cols,
                              f"{target.upper()} — Spearman correlation — Matrices",
                              out_dir / "matrices_spearman_heatmap.png", pdf)
        plot_dropout_effect(df, m_cols, "Matrices", out_dir, pdf)

        plot_spearman_vs_mean_count(
            df, m_cols, target.upper(), "Matrices",
            out_png=out_dir / "matrices_spearman_vs_mean.png",
            pdf=pdf
        )
        plot_spearman_by_presence_threshold(
            df, m_cols, target.upper(), "Matrices",
            out_png=out_dir / "matrices_spearman_by_presence.png",
            pdf=pdf,
            thresholds=(1, 2, 5, 10, 20),
            require_reps=2
        )

    make_side_by_side_summary(
        df, t_cols, m_cols,
        out_png=out_dir / "side_by_side_summary.png",
        title=f"{target.upper()} — Targets vs Matrices",
        pdf=pdf
    )
    make_dropout_side_by_side_summary(
        df, t_cols, m_cols,
        out_png=out_dir / "dropout_side_by_side_summary.png",
        threshold=DROPOUT_THRESHOLD,
        pdf=pdf
    )

    if pdf:
        pdf.close()
        print(f"[report] wrote -> {pdf_path}")
    else:
        print(f"[report] skipped PDF because it already exists -> {pdf_path}")

def main():
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    for t in TARGETS:
        run_for_target(t)

if __name__ == "__main__":
    main()
