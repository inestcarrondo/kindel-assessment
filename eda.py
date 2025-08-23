"""
Example on how to run: python eda.py ddr1 random 1
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

#ARGS
if len(sys.argv) != 4:
    print("How to run: python eda.py <target> <split_type> <split_index>")
    sys.exit(1)

TARGET      = sys.argv[1].lower()
SPLIT_TYPE  = sys.argv[2].lower()
SPLIT_INDEX = int(sys.argv[3]) #[1-5]

THIS_DIR = os.path.dirname(__file__)
LOCAL_DATA = os.path.join(THIS_DIR, "data")
RESULTS_DIR = os.path.join(THIS_DIR, "eda_results", f"{TARGET}_{SPLIT_TYPE}_{SPLIT_INDEX}")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ["DATA_ROOT"] = LOCAL_DATA

# Loading Data
from kindel.utils.data import get_training_data, get_testing_data

print(f"Loading {TARGET}, split={SPLIT_INDEX}, split_type={SPLIT_TYPE}")
train_df, valid_df, test_df = get_training_data(TARGET, SPLIT_INDEX, SPLIT_TYPE)
train_df["split"] = "train"
valid_df["split"] = "valid"
test_df["split"]  = "test"
df_all = pd.concat([train_df, valid_df, test_df], ignore_index=True)

heldout = get_testing_data(TARGET)
heldout_on  = heldout["on"]

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (6,4)

#noise replication
rep_cols = ["seq_target_1","seq_target_2","seq_target_3"]
train_df["count_mean"] = train_df[rep_cols].mean(axis=1)
train_df["count_cv"]   = train_df[rep_cols].std(axis=1) / (train_df["count_mean"]+1e-6)

plt.figure()
plt.scatter(train_df["count_mean"], train_df["count_cv"], s=5)
plt.xscale('log')
plt.xlabel("Mean Target Counts")
plt.ylabel("CV across replicates")
plt.title("Replicate noise")
plt.savefig(os.path.join(RESULTS_DIR,"replicate_noise.png"), dpi=300)
plt.close()

# enrichment-Kd
merged = heldout_on.merge(df_all[["molecule_hash","y"]], on="molecule_hash",how="left")
merged = merged.rename(columns={"y_x":"kd", "y_y":"enrichment"}).dropna()
merged["log_kd"] = np.log(merged["kd"]+1e-9)

plt.figure()
sns.scatterplot(x="enrichment",y="log_kd",data=merged,s=10,alpha=0.4)
sns.regplot(x="enrichment",y="log_kd",data=merged,lowess=True,scatter=False,
            line_kws={'linewidth':2})
plt.title("log(Kd) vs enrichment")
plt.savefig(os.path.join(RESULTS_DIR,"enrichment_vs_kd.png"), dpi=300)
plt.close()

lr = LinearRegression().fit(merged[["enrichment"]], merged["log_kd"])
merged["residual"] = merged["log_kd"] - lr.predict(merged[["enrichment"]])

plt.figure()
sns.scatterplot(x="enrichment",y="residual",data=merged,s=10,alpha=0.4)
plt.axhline(0,ls="--",c="black"); plt.title("Residual trend")
plt.savefig(os.path.join(RESULTS_DIR,"residual.png"),dpi=300)
plt.close()

#covariate shift
def calc_props(smiles):
    m = Chem.MolFromSmiles(smiles)
    return pd.Series({
        "MW": Descriptors.MolWt(m),
        "LogP": Descriptors.MolLogP(m),
        "TPSA": rdMolDescriptors.CalcTPSA(m),
        "Rings": rdMolDescriptors.CalcNumRings(m),
        "HBA": Descriptors.NumHAcceptors(m),
        "HBD": Descriptors.NumHDonors(m),
    })

desc_list = []
for _, row in df_all.iterrows():
    desc = calc_props(row["smiles"])
    desc["split"] = row["split"]
    desc_list.append(desc)
desc_df = pd.DataFrame(desc_list)

for d in ["MW","LogP","TPSA"]:
    plt.figure()
    sns.violinplot(x="split",y=d,data=desc_df)
    plt.title(d)
    plt.savefig(os.path.join(RESULTS_DIR,f"{d}_shift.png"),dpi=300)
    plt.close()

X = desc_df[["MW","LogP","TPSA","Rings","HBA","HBD"]]
y = (desc_df["split"]=="test").astype(int)
Xt,Xv,yt,yv = train_test_split(X,y,stratify=y,test_size=0.3)
clf = RandomForestClassifier(n_estimators=200).fit(Xt,yt)
auc = roc_auc_score(yv,clf.predict_proba(Xv)[:,1])
print("Distribution shift AUC:",auc)

#scaffold diversity
def scaffold(smiles):
    try:
        return MurckoScaffoldSmiles(smiles=smiles)
    except:
        return None

df_all["scaffold"] = df_all["smiles"].apply(scaffold)
scount = df_all.groupby(["scaffold","split"]).size().unstack(fill_value=0)
only_train = ((scount["train"]>0) & (scount.get("test",0)==0)).sum()
only_test  = ((scount.get("test",0)>0) & (scount["train"]==0)).sum()
print(f"Unique scaffolds in train: {only_train} / in test: {only_test}")

plt.figure()
plt.hist(scount.sum(axis=1),bins=50)
plt.title("Compounds per scaffold")
plt.savefig(os.path.join(RESULTS_DIR,"scaffold_hist.png"),dpi=300)
plt.close()

# synthon-level
for part in ["smiles_a","smiles_b","smiles_c"]:
    if part in train_df:
        agg = train_df.groupby(part)["y"].mean().sort_values(ascending=False)
        print(f"\nTop synthons for {part}:\n", agg.head(5))

#control matrix
ctrl_cols=["seq_matrix_1","seq_matrix_2","seq_matrix_3"]
train_df["ctrl_mean"] = train_df[ctrl_cols].mean(axis=1)

plt.figure()
plt.hist(train_df["ctrl_mean"],bins=80)
plt.yscale("log")
plt.title("Control count distribution")
plt.savefig(os.path.join(RESULTS_DIR,"control_dist.png"),dpi=300)
plt.close()

plt.figure()
sns.scatterplot(x="count_mean",y="ctrl_mean",data=train_df,s=8,alpha=0.4)
plt.xscale("log"); plt.yscale("log")
plt.title("Target vs Control counts")
plt.savefig(os.path.join(RESULTS_DIR,"control_vs_target.png"),dpi=300)
plt.close()

print(f"Plots saved to: {RESULTS_DIR}")
