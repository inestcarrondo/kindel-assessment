#!/usr/bin/env python3

import argparse, os, sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# reuse functions from benchmarking.py
sys.path.append(os.getcwd())
from benchmarking import (
    create_stratified_subsample,
    run_redun,
    extract_yaml,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=2000)
    p.add_argument("--models", default="compose_s,xgboost,rf")
    p.add_argument("--targets", default="mapk14,ddr1")
    p.add_argument("--splits", default="random,disynthon")
    p.add_argument("--split-idx", default="1,2,3,4,5")
    p.add_argument("--data-root", default="s3://kin-del-2024/data")
    p.add_argument("--local-data-dir", default="./data")
    p.add_argument("--results-dir", default="hypotheses_results")
    p.add_argument("--csv-out", default="hypotheses_results/hypothesis_results.csv")
    p.add_argument("--full-data", action="store_true",
                   help="Use full dataset instead of stratified subsample")
    return p.parse_args()


def main():
    args = parse_args()
    models = args.models.split(",")
    targets = args.targets.split(",")
    splits  = args.splits.split(",")
    split_idxs = [int(x) for x in args.split_idx.split(",")]

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.csv_out)).mkdir(parents=True, exist_ok=True)

    # stratified subsampling or full dataset
    if args.full_data:
        print("Using full datasets")
        subs = {t: f"{args.local_data_dir}/{t}.parquet" for t in targets}
    else:
        subs = {t: create_stratified_subsample(t, args.nrows,
                                               args.data_root, args.local_data_dir)
                for t in targets}

    rows = []

    # H1/H3/H4 in-target
    for m in models:
        for t in targets:
            for sp in splits:
                print(f"[training {m} on {t} split={sp}]")
                run_redun(m, t, sp, args)
                for idx in split_idxs:
                    metrics = extract_yaml(args.results_dir, m, t, sp, idx)
                    if metrics:
                        rows.append({
                            "model": m, "target": t, "split": sp,
                            "split_idx": idx, "transfer_pair": "in_target",
                            **metrics
                        })

    # H2 cross-target
    from kindel.utils.data import get_training_data, get_testing_data, spearman, kendall, rmse
    from kindel.run import get_model
    def train_and_test(model_name, train_target, test_target):
        print(f"[CROSS] train {model_name} on {train_target} and test on {test_target}")
        df_train, df_valid, df_test = get_training_data(train_target, split_index=1, split_type="random")
        model = get_model(model_name, {}, wandb_logger=None)
        data = model.prepare_dataset(df_train, df_valid, df_test)
        model.train()
        testing = get_testing_data(test_target)
        for condition in ("on","off"):
            feats = model.featurize(testing[condition])
            X, y = feats[:2]
            preds = model.predict(X)
            rows.append({
                "model":model_name,"target":train_target,"split":"cross",
                "transfer_pair":f"{train_target}->{test_target}",
                "test_rho": spearman(preds,y),
                "test_tau": kendall(preds,y),
                "test_rmse": rmse(preds,y),
                "all_off_rho": spearman(preds,y),
            })

    for m in models:
        train_and_test(m,"mapk14","ddr1")
        train_and_test(m,"ddr1","mapk14")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv_out, index=False)
    print("⇒", args.csv_out)
    print(df.head())

    # plots
    sns.set_style("whitegrid")
    out_dir = Path(os.path.dirname(args.csv_out))
    pdf = PdfPages(out_dir/"hypothesis_plots.pdf")

    sub = df[df.transfer_pair=="in_target"]
    if not sub.empty:
        plt.figure()
        sns.barplot(data=sub, x="split", y="test_rmse", hue="model", errorbar="sd")
        plt.title("H1+H3: Split type vs RMSE")
        plt.savefig(out_dir/"H1_split.png",dpi=200); pdf.savefig(); plt.close()

    cross = df[df.split=="cross"]
    if not cross.empty:
        plt.figure()
        sns.barplot(data=cross, x="transfer_pair", y="test_rho", hue="model")
        plt.title("H2: Cross-target Spearman")
        plt.xticks(rotation=45)
        plt.savefig(out_dir/"H2_cross_rho.png",dpi=200); pdf.savefig(); plt.close()

    if not sub.empty:
        spread = sub.groupby(["split","target","model"])["test_rmse"].std().reset_index()
        plt.figure()
        sns.barplot(data=spread,x="split",y="test_rmse",hue="target")
        plt.title("H4: Std(RMSE) in splits")
        plt.savefig(out_dir/"H4_variance.png",dpi=200); pdf.savefig(); plt.close()

        cc = sub[sub.model.isin(["xgboost","compose_s"])]
        if not cc.empty:
            plt.figure()
            sns.scatterplot(data=cc,x="test_rmse",y="all_off_rho",
                            hue="model",style="target",s=100)
            plt.title("H5: Fit vs Off-DNA generalization")
            plt.savefig(out_dir/"H5_generalization.png",dpi=200); pdf.savefig(); plt.close()

    pdf.close()
    print("✓ plots in", out_dir)

if __name__ == "__main__":
    main()
