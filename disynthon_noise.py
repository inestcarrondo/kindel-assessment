#!/usr/bin/env python3
"""
disynthon_noise.py

# diagnoses why different disynthon splits result in  different model performances
Steps:
Replicate noise (CV, replicate correlation) in test and train sets
Dropout asymmetry (presence bars, min count, log2FC tails) in test
Train vs Test distribution shifts  per split
Chemical diversity shift

Outputs:
  disynthon_noise/
    <target>/
      split<k>/
        figures/*.png
        split_metrics.csv
      summary_<target>.csv)
"""

import os
import argparse
from pathlib import Path
import itertools
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr, kendalltau, wasserstein_distance
from tqdm import tqdm

from rdkit import Chem
from kindel.utils.fingerprint_feat import CircularFingerprint

DATA_ROOT_DEFAULT = "s3://kin-del-2024/data"
OUT_ROOT = "disynthon_noise"
EPS = 1e-9

def load_main_table(target: str) -> pd.DataFrame:
    
    #Load main table for a target.
    
    local_root = os.environ.get("DATA_ROOT", DATA_ROOT_DEFAULT)
    nrows = os.environ.get("NROWS")
    df_path = None

    if nrows is not None:
        subsample_name = f"{target}_{int(nrows) // 1000}k_stratified.parquet"
        candidate = os.path.join(local_root, subsample_name)
        if os.path.exists(candidate):
            print(f"[load] Using subsample: {candidate}")
            df_path = candidate

    if df_path is None:
        df_path = os.path.join(DATA_ROOT_DEFAULT, f"{target}_1M.parquet")
        print(f"[load] Using full file: {df_path}")

    df = pd.read_parquet(df_path).rename({"target_enrichment": "y"}, axis="columns")
    return df


def load_splits(target: str, split_type: str) -> pd.DataFrame:
    split_path = os.path.join(DATA_ROOT_DEFAULT, "splits", f"{target}_{split_type}.parquet")
    print(f"[load] Splits: {split_path}")
    return pd.read_parquet(split_path)


def get_train_valid_test(df: pd.DataFrame, df_split: pd.DataFrame, split_index: int):
    split_col = f"split{split_index}"
    if split_col not in df_split.columns:
        raise ValueError(f"Split column {split_col} not found in split file.")
    tr = df[df_split[split_col] == "train"]
    va = df[df_split[split_col] == "valid"]
    te = df[df_split[split_col] == "test"]
    return tr, va, te


# Functions for metrics
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

def replicate_corr(df: pd.DataFrame, cols: list, method="spearman"):
    if len(cols) < 2:
        return None
    return df[cols].astype(float).corr(method=method)

def presence_fractions(values: np.ndarray, threshold: float = 1.0):
    present = (values >= float(threshold)).astype(int)
    n_reps = present.shape[1]
    num_present = present.sum(axis=1)
    return [(num_present == k).mean() for k in range(n_reps + 1)]

def min_counts(values: np.ndarray):
    return values.min(axis=1)

def dropout_mask(values: np.ndarray, threshold: float = 1.0):
    present = (values >= float(threshold)).astype(int)
    n_reps = present.shape[1]
    return (present.sum(axis=1) < n_reps)

def kl_divergence_hist(p, q, bins=100, range_=None):
    hist_p, edges = np.histogram(p, bins=bins, range=range_, density=True)
    hist_q, _     = np.histogram(q, bins=bins, range=range_, density=True)
    hist_p = hist_p + EPS
    hist_q = hist_q + EPS
    hist_p /= hist_p.sum()
    hist_q /= hist_q.sum()
    kl_pq = np.sum(hist_p * np.log(hist_p / hist_q))
    kl_qp = np.sum(hist_q * np.log(hist_q / hist_p))
    return 0.5 * (kl_pq + kl_qp)

# Chemistry helpers
def featurize_smiles(smiles_list):
    fingerprinter = CircularFingerprint()
    fps = []
    for smi in tqdm(smiles_list, desc="Featurizing (CircularFP)"):
        mol = Chem.MolFromSmiles(smi) if pd.notna(smi) else None
        if mol is None:
            fps.append(None)
        else:
            fp = fingerprinter._featurize(mol)
            fps.append(fp)
    return fps  # list of np arrays (binary or counts)

def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    # robust to counts: intersection = sum(min(ai, bi)), union = sum(ai) + sum(bi) - intersection
    inter = np.minimum(a, b).sum()
    denom = a.sum() + b.sum() - inter
    return float(inter / denom) if denom > 0 else 0.0

def approx_train_test_chem_overlap(train_smiles, test_smiles, max_chems=3000, seed=42):
    """
    Approximates chemical similarity shift:
    subsample up to max_chems from train/test
    for each test fp, computes the max tanimoto to any train fp
    return mean/max/median of these maxima as 'coverage' proxies
    """
    rng = np.random.default_rng(seed)
    tsm = np.array(train_smiles, dtype=object)
    vsm = np.array(test_smiles, dtype=object)

    if len(tsm) == 0 or len(vsm) == 0:
        return {"chem_max_tani_mean": np.nan, "chem_max_tani_median": np.nan, "chem_max_tani_p90": np.nan}

    tr_idx = rng.choice(len(tsm), size=min(max_chems, len(tsm)), replace=False)
    te_idx = rng.choice(len(vsm), size=min(max_chems, len(vsm)), replace=False)
    tsm = tsm[tr_idx]; vsm = vsm[te_idx]

    fps_tr = featurize_smiles(tsm)
    fps_te = featurize_smiles(vsm)

    # drop None
    fps_tr = [x for x in fps_tr if x is not None]
    fps_te = [x for x in fps_te if x is not None]
    if len(fps_tr) == 0 or len(fps_te) == 0:
        return {"chem_max_tani_mean": np.nan, "chem_max_tani_median": np.nan, "chem_max_tani_p90": np.nan}

    # brute-force max over train for each test (subsampled)
    max_tanis = []
    for fp_te in tqdm(fps_te, desc="Max Tanimoto(test->train)"):
        best = 0.0
        for fp_tr in fps_tr:
            best = max(best, tanimoto(np.asarray(fp_te), np.asarray(fp_tr)))
        max_tanis.append(best)

    max_tanis = np.array(max_tanis, dtype=float)
    return {
        "chem_max_tani_mean": float(np.mean(max_tanis)),
        "chem_max_tani_median": float(np.median(max_tanis)),
        "chem_max_tani_p90": float(np.percentile(max_tanis, 90.0)),
    }

#plots
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def plot_corr_heatmap(corr_df: pd.DataFrame, title: str, out_png: Path):
    if corr_df is None:
        return
    fig, ax = plt.subplots()
    mat = corr_df.values
    im = ax.imshow(mat, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns))); ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)));  ax.set_yticklabels(corr_df.index)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, out_png)

def plot_cv_vs_mean(cv_df: pd.DataFrame, title: str, out_png: Path):
    fig, ax = plt.subplots()
    ax.scatter(cv_df["mean"], cv_df["cv"], s=5, alpha=0.6)
    ax.set_xlabel("Mean count")
    ax.set_ylabel("CV (std/mean)")
    ax.set_title(title)
    savefig(fig, out_png)

def plot_presence_bars(values: np.ndarray, title: str, out_png: Path, threshold: float = 1.0):
    frac = presence_fractions(values, threshold=threshold)
    fig, ax = plt.subplots()
    ax.bar(range(len(frac)), frac)
    ax.set_xlabel("# replicates with count ≥ threshold")
    ax.set_ylabel("Fraction of molecules")
    ax.set_title(title)
    savefig(fig, out_png)

def plot_min_count_hist(values: np.ndarray, title: str, out_png: Path):
    mins = min_counts(values)
    fig, ax = plt.subplots()
    ax.hist(mins, bins=80)
    ax.set_xlabel("Minimum count across replicates")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    savefig(fig, out_png)

def plot_dropout_log2fc_tails(df: pd.DataFrame, cols: list, values: np.ndarray, title: str, out_png: Path, threshold: float = 1.0):
    l2 = pairwise_log2fc(df, cols)
    if len(l2) == 0:
        return
    max_abs_l2 = np.max(np.vstack([np.abs(s.values) for s in l2.values()]), axis=0)
    mask = dropout_mask(values, threshold=threshold)
    fig, ax = plt.subplots()
    ax.hist(max_abs_l2[mask], bins=80)
    ax.set_xlabel("Max |log2FC| across replicate pairs")
    ax.set_ylabel("Frequency")
    ax.set_title(title + " (dropout cases)")
    savefig(fig, out_png)

def plot_hist_overlay(a, b, title: str, label_a: str, label_b: str, out_png: Path, bins=100):
    fig, ax = plt.subplots()
    ax.hist(a, bins=bins, alpha=0.5, label=label_a, density=True)
    ax.hist(b, bins=bins, alpha=0.5, label=label_b, density=True)
    ax.set_title(title)
    ax.legend()
    savefig(fig, out_png)

# per-split analysis
def analyze_split(target: str, split_type: str, split_index: int, out_root: Path,
                  threshold: float = 1.0, max_chems: int = 3000, chem_smiles_col: str = "smiles_b"):
    df = load_main_table(target)
    splits = load_splits(target, split_type)
    train, valid, test = get_train_valid_test(df, splits, split_index)

    t_cols = find_replicate_cols(df, "seq_target_")
    m_cols = find_replicate_cols(df, "seq_matrix_")

    # out dirs
    split_dir = out_root / target / f"split{split_index}"
    fig_dir   = split_dir / "figures"
    ensure_dir(fig_dir)

    # replicate noise: CV & correlations (test set)
    results = {"target": target, "split_type": split_type, "split_index": split_index}

    if len(t_cols) >= 2:
        cv_t_test = compute_cv_table(test, t_cols)
        plot_cv_vs_mean(cv_t_test, "Targets (TEST) — CV vs Mean", fig_dir / "targets_test_cv_vs_mean.png")
        corr_t_test = replicate_corr(test, t_cols, method="spearman")
        plot_corr_heatmap(corr_t_test, "Spearman — Targets (TEST)", fig_dir / "targets_test_spearman_heatmap.png")
        results["test_targets_cv_mean"] = float(np.nanmean(cv_t_test["cv"]))
        results["test_targets_cv_median"] = float(np.nanmedian(cv_t_test["cv"]))
        results["test_targets_corr_mean"] = float(np.nanmean(corr_t_test.values)) if corr_t_test is not None else np.nan

    if len(m_cols) >= 2:
        cv_m_test = compute_cv_table(test, m_cols)
        plot_cv_vs_mean(cv_m_test, "Matrices (TEST) — CV vs Mean", fig_dir / "matrices_test_cv_vs_mean.png")
        corr_m_test = replicate_corr(test, m_cols, method="spearman")
        plot_corr_heatmap(corr_m_test, "Spearman — Matrices (TEST)", fig_dir / "matrices_test_spearman_heatmap.png")
        results["test_matrices_cv_mean"] = float(np.nanmean(cv_m_test["cv"]))
        results["test_matrices_cv_median"] = float(np.nanmedian(cv_m_test["cv"]))
        results["test_matrices_corr_mean"] = float(np.nanmean(corr_m_test.values)) if corr_m_test is not None else np.nan

    # dropout diagnostics (test set)
    if len(t_cols) >= 2:
        tv = test[t_cols].astype(float).values
        plot_presence_bars(tv, f"Targets (TEST) — replicate presence (≥{threshold})", fig_dir / "targets_test_dropout_presence.png", threshold=threshold)
        plot_min_count_hist(tv, "Targets (TEST) — min count", fig_dir / "targets_test_min_count_hist.png")
        plot_dropout_log2fc_tails(test, t_cols, tv, "Targets (TEST) — log2FC tails", fig_dir / "targets_test_dropout_log2fc_tails.png", threshold=threshold)
        # simple dropout rates
        pres = (tv >= threshold).astype(int)
        nreps = pres.shape[1]
        num_present = pres.sum(axis=1)
        for k in range(nreps+1):
            results[f"targets_frac_present_{k}"] = float((num_present == k).mean())

    if len(m_cols) >= 2:
        mv = test[m_cols].astype(float).values
        plot_presence_bars(mv, f"Matrices (TEST) — replicate presence (≥{threshold})", fig_dir / "matrices_test_dropout_presence.png", threshold=threshold)
        plot_min_count_hist(mv, "Matrices (TEST) — min count", fig_dir / "matrices_test_min_count_hist.png")
        plot_dropout_log2fc_tails(test, m_cols, mv, "Matrices (TEST) — log2FC tails", fig_dir / "matrices_test_dropout_log2fc_tails.png", threshold=threshold)
        presm = (mv >= threshold).astype(int)
        nrepsm = presm.shape[1]
        num_present_m = presm.sum(axis=1)
        for k in range(nrepsm+1):
            results[f"matrices_frac_present_{k}"] = float((num_present_m == k).mean())

    # train vs test distribution shifts
    # enrichment distributions
    plot_hist_overlay(train["y"].values, test["y"].values,
                      "Enrichment (y) — Train vs Test", "Train", "Test",
                      fig_dir / "train_test_enrichment_hist.png")
    results["y_wasserstein"] = float(wasserstein_distance(train["y"].values, test["y"].values))
    results["y_symKL"] = float(kl_divergence_hist(train["y"].values, test["y"].values))

    # mean counts (targets and matrices)
    if len(t_cols) > 0:
        tr_mean_t = train[t_cols].astype(float).mean(axis=1).values
        te_mean_t = test[t_cols].astype(float).mean(axis=1).values
        plot_hist_overlay(tr_mean_t, te_mean_t, "Mean Target Count — Train vs Test",
                          "Train", "Test", fig_dir / "train_test_mean_target_counts_hist.png")
        results["mean_t_wasserstein"] = float(wasserstein_distance(tr_mean_t, te_mean_t))
        results["mean_t_symKL"] = float(kl_divergence_hist(tr_mean_t, te_mean_t))

    if len(m_cols) > 0:
        tr_mean_m = train[m_cols].astype(float).mean(axis=1).values
        te_mean_m = test[m_cols].astype(float).mean(axis=1).values
        plot_hist_overlay(tr_mean_m, te_mean_m, "Mean Matrix Count — Train vs Test",
                          "Train", "Test", fig_dir / "train_test_mean_matrix_counts_hist.png")
        results["mean_m_wasserstein"] = float(wasserstein_distance(tr_mean_m, te_mean_m))
        results["mean_m_symKL"] = float(kl_divergence_hist(tr_mean_m, te_mean_m))

    # chemical diversity shift
    # choose one SMILES column as chemical proxy
    smiles_col = chem_smiles_col if chem_smiles_col in df.columns else "smiles_b"
    train_smiles = train[smiles_col].astype(str).values
    test_smiles  = test[smiles_col].astype(str).values
    chem_stats = approx_train_test_chem_overlap(train_smiles, test_smiles, max_chems=max_chems)
    results.update(chem_stats)

    split_metrics_csv = split_dir / "split_metrics.csv"
    pd.DataFrame([results]).to_csv(split_metrics_csv, index=False)
    print(f"[write] {split_metrics_csv}")

    pdf_path = split_dir / f"split{split_index}_diagnostics.pdf"
    with PdfPages(pdf_path) as pdf:
        for img in sorted(fig_dir.glob("*.png")):
            fig = plt.figure()
            img_arr = plt.imread(img)
            plt.imshow(img_arr); plt.axis("off"); plt.title(img.name)
            pdf.savefig(fig); plt.close(fig)
    print(f"[write] {pdf_path}")

    return results


def rollup_summary(all_results, out_path: Path):
    df = pd.DataFrame(all_results)
    df.to_csv(out_path, index=False)
    print(f"[write] {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=["mapk14", "ddr1"])
    p.add_argument("--split-type", default="disynthon", choices=["disynthon", "random"],
                   help="Split type to analyze (default disynthon)")
    p.add_argument("--split-ids", nargs="+", type=int, default=[1,2,3,4,5],
                   help="Split indices to analyze (default 1..5)")
    p.add_argument("--outdir", default=OUT_ROOT)
    p.add_argument("--threshold", type=float, default=1.0,
                   help="Presence threshold for dropout diagnostics (default 1)")
    p.add_argument("--max-chems", type=int, default=3000,
                   help="Max molecules to subsample for chemistry overlap (per set)")
    p.add_argument("--chem-smiles-col", type=str, default="smiles_b",
                   help="SMILES column to characterize chemistry shift (e.g., smiles_b or smiles_c)")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.outdir)
    ensure_dir(out_root)

    all_results = []
    for tgt in args.targets:
        per_target_results = []
        for sid in args.split_ids:
            res = analyze_split(
                target=tgt,
                split_type=args.split_type,
                split_index=sid,
                out_root=out_root,
                threshold=args.threshold,
                max_chems=args.max_chems,
                chem_smiles_col=args.chem_smiles_col,
            )
            per_target_results.append(res)
            all_results.append(res)
        # write per-target rollup
        rollup_summary(per_target_results, out_root / tgt / f"summary_{tgt}.csv")

    # write global rollup
    rollup_summary(all_results, out_root / "summary_all_targets.csv")


if __name__ == "__main__":
    main()
