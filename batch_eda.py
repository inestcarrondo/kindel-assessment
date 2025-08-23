#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from kindel.utils.data import get_training_data, get_testing_data

# settings
targets        = ["ddr1", "mapk14"]
split_configs  = {
    "random":     [1],          # <-- only split1
    "disynthon":  [1, 2, 3, 4, 5]   # <-- all
}
LOCAL_DATA     = os.path.join(os.path.dirname(__file__), "kindel", "data")
os.environ["DATA_ROOT"] = LOCAL_DATA

OUT_DIR = os.path.join(os.path.dirname(__file__), "batch_eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set(style="whitegrid", font_scale=1.2)

def calc_physchem(smiles):
    m = Chem.MolFromSmiles(smiles)
    return {
        "MW": Descriptors.MolWt(m),
        "LogP": Descriptors.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
    }

# get data
records         = []
enrich_kd_list  = []
physchem_cache  = {}   # speed up

for tgt in targets:
    for split_type, split_idxs in split_configs.items():
        for sidx in split_idxs:
            print(f"Loading {tgt} {split_type} split{sidx}")
            try:
                df_train, df_valid, df_test = get_training_data(
                    tgt, split_index=sidx, split_type=split_type
                )
            except Exception as e:
                print("  >> Skipped due to:", e)
                continue

            df_split = pd.concat([
                df_train.assign(split="train"),
                df_valid.assign(split="valid"),
                df_test.assign(split="test")
            ])

            for _, row in df_split.iterrows():
                smi = row["smiles"]
                if smi not in physchem_cache:
                    physchem_cache[smi] = calc_physchem(smi)
                pc = physchem_cache[smi]

                # only track physchem and replicate noise globally
                rec = {
                    **pc,
                    "target": tgt,
                    "split_type": split_type,
                    "split_index": sidx,
                    "split": row["split"],
                    "count_mean": row[["seq_target_1","seq_target_2","seq_target_3"]].mean(),
                    "count_cv":   row[["seq_target_1","seq_target_2","seq_target_3"]].std() /
                                  (row[["seq_target_1","seq_target_2","seq_target_3"]].mean() + 1e-9),
                }
                # only track control count distributions for disynthon
                if split_type == "disynthon":
                    rec["ctrl_mean"] = row[["seq_matrix_1","seq_matrix_2","seq_matrix_3"]].mean()
                records.append(rec)

            # on DNA only enrichment vs kd
            heldout_on = get_testing_data(tgt)["on"]
            merged = heldout_on.merge(df_split[["molecule_hash","y"]], on="molecule_hash",how="left")
            merged = merged.rename(columns={"y_x":"kd","y_y":"enrichment"}).dropna()
            merged["log_kd"] = np.log(merged["kd"] + 1e-9)
            for _, r in merged.iterrows():
                enrich_kd_list.append({
                    "target":tgt,
                    "split_type":split_type,
                    "split_index":sidx,
                    "enrichment":r["enrichment"],
                    "log_kd":r["log_kd"]
                })

records_df   = pd.DataFrame(records)
enrich_kd_df = pd.DataFrame(enrich_kd_list)

# plots
pdfpath = os.path.join(OUT_DIR, "combined_EDA.pdf")
pdf = PdfPages(pdfpath)

# physchem violins
for desc in ["MW","LogP","TPSA"]:
    plt.figure(figsize=(10,6))
    sns.violinplot(
        data=records_df,
        x="target", y=desc,
        hue="split_type",
        split=True
    )
    plt.title(f"{desc} distribution")
    png = os.path.join(OUT_DIR, f"combined_{desc}.png")
    plt.savefig(png,dpi=300); pdf.savefig(); plt.close()

# replicate noise
g = sns.FacetGrid(
    records_df[records_df["split"]=="train"],
    row="target", col="split_type",
    hue="split_index", height=3, legend_out=True
)
g.map(plt.scatter, "count_mean","count_cv",s=5,alpha=0.4).set(xscale="log")
g.add_legend(title="split")
g.fig.suptitle("Replicate noise",y=1.02)
png = os.path.join(OUT_DIR, "combined_replicate_noise.png")
g.savefig(png); pdf.savefig(g.fig); plt.close()

# enrichment-vs-logKd
g = sns.FacetGrid(
    enrich_kd_df,
    row="target", col="split_type",
    hue="split_index", height=3, legend_out=True
)
g.map(plt.scatter,"enrichment","log_kd",s=4,alpha=0.4)
g.add_legend(title="split")
g.fig.suptitle("enrichment → log(Kd)",y=1.02)
png = os.path.join(OUT_DIR,"combined_enrichment_kd.png")
g.savefig(png); pdf.savefig(g.fig); plt.close()

# residuals
enrich_kd_df["residual"] = np.nan
for name, group in enrich_kd_df.groupby(["target","split_type","split_index"]):
    lr = LinearRegression().fit(group[["enrichment"]],group["log_kd"])
    enrich_kd_df.loc[group.index,"residual"] = group["log_kd"] - lr.predict(group[["enrichment"]])

g = sns.FacetGrid(
    enrich_kd_df,
    row="target", col="split_type",
    hue="split_index", height=3, legend_out=True
)
g.map(plt.scatter,"enrichment","residual",s=4,alpha=0.4)
g.add_legend(title="split")
g.fig.suptitle("residuals",y=1.02)
png = os.path.join(OUT_DIR,"combined_residuals.png")
g.savefig(png); pdf.savefig(g.fig); plt.close()

# control plots – disynthon only
ctrls = records_df[ records_df["split_type"]=="disynthon" ].copy()
if not ctrls.empty:
    g = sns.FacetGrid(
        ctrls[ctrls["split"]=="train"],
        row="target", col="split_type",
        hue="split_index", height=3, legend_out=True
    )
    g.map(plt.hist,"ctrl_mean",bins=50,alpha=0.6).set(yscale="log")
    g.add_legend()
    g.fig.suptitle("control count dist",y=1.02)
    png = os.path.join(OUT_DIR,"combined_control_dist.png")
    g.savefig(png); pdf.savefig(g.fig); plt.close()

    g = sns.FacetGrid(
        ctrls[ctrls["split"]=="train"],
        row="target", col="split_type",
        hue="split_index", height=3, legend_out=True
    )
    g.map(plt.scatter,"count_mean","ctrl_mean",s=4,alpha=0.4).set(xscale="log",yscale="log")
    g.add_legend()
    g.fig.suptitle("target vs control",y=1.02)
    png = os.path.join(OUT_DIR,"combined_control_vs_target.png")
    g.savefig(png); pdf.savefig(g.fig); plt.close()

pdf.close()

print(f"\n✓ Plots and PDF saved in {OUT_DIR}")
