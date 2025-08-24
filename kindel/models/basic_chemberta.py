import os
os.environ["ACCELERATE_DISABLE_MAPPING"] = "true"  # to prevent meta-tensor loading

from abc import abstractmethod
from collections import namedtuple
import pandas as pd
import torch
import numpy as np

try:
    from datasets import Dataset as HFDataset, DatasetDict
except Exception as e:
    raise ImportError(
        "Missing 'datasets' package. Install it in your pixi env: "
        "pixi add --pypi datasets"
    ) from e

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import hashlib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pathlib import Path
from kindel.utils.data import featurize
import json
Dataset = namedtuple("Dataset", ["train", "valid", "test"])
Example = namedtuple("Example", ["x", "y"])


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true") 
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, os.cpu_count() - 1)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, os.cpu_count() - 1)))
try:
    torch.set_num_threads(max(1, os.cpu_count() - 1))
except Exception:
    pass

class Model:
    def __init__(self, **hyperparameters):
        self.model = self._create_model(**hyperparameters)

    @abstractmethod
    def prepare_dataset(self, df_train, df_valid, df_test): ...

    @abstractmethod
    def train(self): ...

    @abstractmethod
    def predict(self, x): ...

    @abstractmethod
    def featurize(self, df): ...

    @abstractmethod
    def _create_model(self, hyperparameters): ...


# ChemBERTa encoder

class ChemBERTaEncoder:
    """
    Loads a pretrained ChemBERTa model + tokenizer and returns frozen embeddings
    from the last-hidden-state CLS token.
    """

    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1", device=None):
        from transformers import AutoModel, AutoTokenizer

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = torch.device("cpu")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=None,
            low_cpu_mem_usage=False,
            use_safetensors=True,
            torch_dtype=torch.float32,
#            trust_remote_code=False
        ).to(self.device)

        self.model.eval()

        self.model_name = model_name

        try:
            next(self.model.parameters()).device.type
            assert all(p.device.type != "meta" for p in self.model.parameters()), "Model stayed on meta!"
        except StopIteration:
            pass

    def encode(self, smiles_list, batch_size=256, max_length=128):
         return self._encode_impl(smiles_list, batch_size, max_length)

    def _cache_dir(self):
        # where to store .npz files
        return Path(os.getenv("EMB_CACHE_DIR", "./emb_cache"))

    def _cache_key(self, smiles_list, max_length):
        # deterministic key for a set of SMILES and settings
        h = hashlib.sha1()
        for s in sorted(set(smiles_list)):
            h.update(s.encode("utf-8"))
        h.update((self.model_name + f"|L={max_length}").encode("utf-8"))
        return h.hexdigest()

    def encode_cached(self, smiles_list, batch_size=256, max_length=128):
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(smiles_list, max_length)
        npz_path = cache_dir / f"{key}.npz"
        meta_path = cache_dir / f"{key}.json"

        if npz_path.exists() and meta_path.exists():
            data = np.load(npz_path, allow_pickle=True)
            smiles_arr = data["smiles"].tolist()
            embs = data["embs"]
            index = {s: i for i, s in enumerate(smiles_arr)}
            X = np.vstack([embs[index[s]] for s in smiles_list])
            return X

        # compute for uniques, save, then map back to order
        uniq = list(dict.fromkeys(smiles_list)) 
        embs = self._encode_impl(uniq, batch_size, max_length)
        np.savez_compressed(npz_path, smiles=np.array(uniq, dtype=object), embs=embs)
        meta_path.write_text(json.dumps({"model": self.model_name, "max_length": max_length}))
        # map back
        index = {s: i for i, s in enumerate(uniq)}
        X = np.vstack([embs[index[s]] for s in smiles_list])
        return X

    def _encode_impl(self, smiles_list, batch_size=256, max_length=128):
        all_embs = []
        with torch.inference_mode():
            for i in range(0, len(smiles_list), batch_size):
                batch = smiles_list[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                # move only tensors we need
                enc = {k: v.to(self.device, non_blocking=True) for k, v in enc.items()}
                out = self.model(**enc)
                cls_emb = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
                all_embs.append(cls_emb)
        return np.vstack(all_embs)

# Downstream models

class ChemBERTaModel(Model):
    """
    Handles featurization using frozen ChemBERTa.
    """

    def __init__(self, chemberta_name="seyonec/ChemBERTa-zinc-base-v1", **hyperparameters):
        self.encoder = ChemBERTaEncoder(model_name=chemberta_name)
        super().__init__(**hyperparameters)

    def prepare_dataset(self, df_train, df_valid, df_test):

        # Compute embeddings ONCE for the union of all rows used by this split
        all_smiles = pd.concat([df_train["smiles"], df_valid["smiles"], df_test["smiles"]]).tolist()
        # builds/uses a dataset-level cache file
        all_X = self.encoder.encode_cached(all_smiles, batch_size=256, max_length=128)

        n_tr = len(df_train); n_va = len(df_valid)
        X_train = all_X[:n_tr]
        X_valid = all_X[n_tr : n_tr + n_va]
        X_test  = all_X[n_tr + n_va :]

        y_train = df_train["y"].values
        y_valid = df_valid["y"].values
        y_test  = df_test["y"].values

        self.data = Dataset(
            train=Example(x=X_train, y=y_train),
            valid=Example(x=X_valid, y=y_valid),
            test=Example(x=X_test, y=y_test),
        )
        return self.data

    def train(self):
        self.model.fit(self.data.train.x, self.data.train.y)

    def predict(self, x):
        return self.model.predict(x)

    def featurize(self, df):
        smiles = list(df["smiles"])
        X = self.encoder.encode(smiles)
        y = df["y"].values
        return X, y


class RandomForestChemBERTa(ChemBERTaModel):
    def _create_model(self, **hyperparameters):
        return RandomForestRegressor(**hyperparameters, n_jobs=-1)


class XGBoostChemBERTa(ChemBERTaModel):
    def _create_model(self, **hyperparameters):
        return XGBRegressor(**hyperparameters)

class ChemBERTaRegressor(Model):
    """
    Fine-tuning of ChemBERTa for regression using HuggingFace Trainer.
    """

    def __init__(
        self,
        chemberta_ckpt="DeepChem/ChemBERTa-77M-MLM",
        max_length=128,
        lr=2e-5,
        train_batch_size=16,
        eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        **hyperparameters,
    ):
        self.chemberta_ckpt = chemberta_ckpt
        self.max_length = int(max_length)
        self.lr = float(lr)
        self.train_bs = int(train_batch_size)
        self.eval_bs = int(eval_batch_size)
        self.num_train_epochs = int(num_train_epochs)
        self.weight_decay = float(weight_decay)
        self.logging_steps = int(logging_steps)

        self.tokenizer = AutoTokenizer.from_pretrained(self.chemberta_ckpt, use_fast=True)

        super().__init__(  # sets self.model via _create_model()
            chemberta_ckpt=self.chemberta_ckpt,
            **hyperparameters
        )

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=None)
        self.train_ds = None
        self.valid_ds = None
        self.test_ds = None
        self._trainer = None

    def _create_model(self, chemberta_ckpt: str):
        os.environ.setdefault("ACCELERATE_DISABLE_MAPPING", "true")
    
        kwargs = dict(
            num_labels=1,
            problem_type="regression",
            use_safetensors=True,
            low_cpu_mem_usage=False,
            device_map=None,           
            torch_dtype=torch.float32,
            trust_remote_code=False,
        )
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                chemberta_ckpt, **kwargs
            )
        except ValueError as e:
            kwargs.pop("use_safetensors", None)
            model = AutoModelForSequenceClassification.from_pretrained(
                chemberta_ckpt, **kwargs
            )

        try:
            any_meta = any(p.device.type == "meta" for p in model.parameters())
        except StopIteration:
            any_meta = False

        if any_meta:
            raise RuntimeError(
             "Model weights are still on 'meta'. This usually means a lazy/dispatch load path ran. "
             "Ensure low_cpu_mem_usage=False and device_map=None, and that"
                "ACCELERATE_DISABLE_MAPPING=true. If loading a .bin checkpoint, torch must be >=2.6."
            )

        return model

    def _build_hf_ds(self, df: pd.DataFrame, label_col: str):
    # guard
        df = self._clean_df(df, label_col)

        def tok_fn(batch):
            enc = self.tokenizer(
                batch["smiles"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            enc["labels"] = batch[label_col]
            return enc

        ds = HFDataset.from_pandas(df)                       # length == len(df)
        ds = ds.map(tok_fn, batched=True, remove_columns=list(df.columns))
        assert len(ds) == len(df), f"HF dataset length {len(ds)} != df length {len(df)}"
        return ds, df  # return the cleaned df too

    def prepare_dataset(self, df_train, df_valid, df_test):
        label_col = "y" if "y" in df_train.columns else "target_enrichment"

        self.train_ds, tr_df = self._build_hf_ds(df_train, label_col)
        self.valid_ds, va_df = self._build_hf_ds(df_valid, label_col)
        self.test_ds,  te_df = self._build_hf_ds(df_test,  label_col)

        self.data = Dataset(
            train=Example(x=self.train_ds, y=tr_df[label_col].to_numpy()),
            valid=Example(x=self.valid_ds, y=va_df[label_col].to_numpy()),
            test= Example(x=self.test_ds,  y=te_df[label_col].to_numpy()),
        )
        return self.data 

    def train(self):

        base = dict(
            output_dir="chemberta_ft_out",
         learning_rate=self.lr,
         per_device_train_batch_size=self.train_bs,
         per_device_eval_batch_size=self.eval_bs,
         num_train_epochs=self.num_train_epochs,
         weight_decay=self.weight_decay,
         save_strategy="epoch",
         load_best_model_at_end=True,
         logging_steps=self.logging_steps,
         report_to="none",
         fp16=torch.cuda.is_available(),
        )
        try:
            args = TrainingArguments(eval_strategy="epoch", **base)
        except TypeError:
            args = TrainingArguments(evaluation_strategy="epoch", **base)

        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.valid_ds,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        self._trainer.train()

    def predict(self, x):
        """
        Accept either an HF Dataset (preferred) or a pandas DataFrame.
        """
        if isinstance(x, pd.DataFrame):
            x, _ = self.featurize(x)  # returns (dataset, y)

        ds = x
        input_cols = ["input_ids", "attention_mask"]
        if "token_type_ids" in ds.column_names:
            input_cols.append("token_type_ids")

        ds = ds.remove_columns([c for c in ds.column_names if c not in (input_cols + ["labels"])])
        ds.set_format(type="torch", columns=input_cols) 

        from torch.utils.data import DataLoader

        loader = DataLoader(
            ds,
            batch_size=self.eval_bs,
            shuffle=False,
            drop_last=False,
            collate_fn=self.data_collator,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        preds = []
        with torch.inference_mode():
            for batch in loader:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in input_cols}
                logits = self.model(**batch).logits  # shape (B, 1)
                preds.append(logits.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0).reshape(-1)
        n_ds = len(x)
        if preds.shape[0] != n_ds:
            raise RuntimeError(f"predict length mismatch: preds={preds.shape[0]} vs dataset={n_ds}")
        return preds

    def featurize(self, df):
        label_col = "y" if "y" in df.columns else "target_enrichment"
        ds, df_clean = self._build_hf_ds(df, label_col)
        y = df_clean[label_col].to_numpy()
        assert len(ds) == len(y), f"featurize mismatch: ds {len(ds)} vs y {len(y)}"
        return ds, y
    
    def _clean_df(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        # only keep the columns we need, drop rows with nan/none
        keep = ["smiles", label_col]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Have: {list(df.columns)}")
        df = df[keep].dropna(subset=keep).reset_index(drop=True)
        return df
