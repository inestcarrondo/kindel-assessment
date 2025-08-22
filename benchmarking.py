#!/usr/bin/env python3

"""
Benchmark KinDEL models with stratified subsampled training data.
Uses precomputed target_enrichment for stratification.
Runs the redun training pipeline for each model, parses YAML result files,
and aggregates into a CSV.
Skips re-downloading existing data, overwrites output files, and handles YAML file access robustly.
"""

import argparse
import os
import subprocess
import yaml
import csv
from pathlib import Path
import pandas as pd
import logging
import time
import sys
import datetime

   # Add kindel directory to Python path
sys.path.append(os.path.join(os.getcwd(),  "kindel"))
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODELS = ["xgboost", "rf"]



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=2000, help="Number of rows to subsample")
    parser.add_argument("--models", type=lambda s: s.split(","), default=DEFAULT_MODELS, help="Comma-separated list of models")
    parser.add_argument("--data-root", type=str, default="s3://kin-del-2024/data", help="Root S3 path for data (no trailing slash)")
    parser.add_argument("--local-data-dir", type=str, default="./data", help="Local directory to store subsampled data")
    parser.add_argument("--results-dir", type=str, default="benchmark_outputs", help="Directory for model outputs")
    parser.add_argument("--csv-out", type=str, default="benchmark_results.csv", help="Output CSV file for results")
    return parser.parse_args()

def create_stratified_subsample(target, nrows, data_root, local_data_dir, random_state=42):
    """Create a stratified subsample of nrows molecules, saving to local_data_dir."""
    # Normalize data_root to avoid double slashes
    data_root = data_root.rstrip("/")
    # Define input and output paths
    input_path = f"{data_root}/{target}_1M.parquet"
    local_input_path = Path(local_data_dir) / f"{target}_1M.parquet"
    output_path = Path(local_data_dir) / f"{target}_{nrows//1000}k_stratified.parquet"
    
    # Log the S3 path for debugging
    logger.info(f"S3 input path: {input_path}")
    
    # Check if output already exists
    if output_path.exists():
        logger.info(f"Overwriting existing subsample: {output_path}")
    
    # Check if input data exists locally, download if not
    if not local_input_path.exists():
        logger.info(f"Downloading {input_path} to {local_input_path}...")
        try:
            local_input_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = ["aws", "s3", "cp", input_path, str(local_input_path), "--no-sign-request"]
            logger.info(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download {input_path}: {e}")
            raise
    else:
        logger.info(f"Using existing input data: {local_input_path}")
    
    # Load data
    logger.info(f"Loading {local_input_path}...")
    try:
        df = pd.read_parquet(local_input_path)
    except Exception as e:
        logger.error(f"Failed to load parquet {local_input_path}: {e}")
        raise
    
    # Verify required columns
    for col in ['smiles', 'target_enrichment']:
        if col not in df.columns:
            logger.error(f"No '{col}' column found in {local_input_path}. Columns: {df.columns}")
            raise ValueError(f"{col} column missing")
    
    # Stratify by target_enrichment
    logger.info(f"Stratifying by target_enrichment for {target}...")
    df['enrich_bin'] = pd.qcut(df['target_enrichment'], q=10, duplicates='drop')
    frac = nrows / len(df)
    logger.info(f"Subsampling {nrows} rows (frac={frac:.4f})...")
    subsample = df.groupby('enrich_bin', group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))
    
    # Save subsample, overwriting if exists
    logger.info(f"Saving stratified subsample to {output_path}...")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop the interval bin column (not needed after subsampling)
        subsample = subsample.drop(columns=["enrich_bin"])
        subsample.to_parquet(output_path, index=False)
    except Exception as e:
        logger.error(f"Failed to save parquet {output_path}: {e}")
        raise
    logger.info(f"Saved stratified subsample: {output_path}")
    return str(output_path)

from pathlib import Path  # make sure this import is at the top

def run_redun(model, target, split_type, args):
    """Run the Redun pipeline for a model, target, and split type."""
    env = os.environ.copy()
    env["NROWS"] = str(args.nrows)
    # Absolute path so Redun can find <DATA_ROOT>/<target>_<n>k_stratified.parquet
    env["DATA_ROOT"] = str(Path(args.local_data_dir).resolve())

    # Add folder that contains the 'kindel/' package
    repo_root = Path(__file__).resolve().parent  # .../kindel_project/kindel_repo
    env["PYTHONPATH"] = f"{repo_root}:" + env.get("PYTHONPATH", "")
    
    env["WANDB_DISABLED"] = "true"
    env["ACCELERATE_DISABLE_MAPPING"] = "true"
    env["CUDA_VISIBLE_DEVICES"] = ""
    results_dir = Path(args.results_dir).resolve()

    cmd = [
        "redun", "-c", f"{repo_root}/kindel/.redun", "run", "kindel.run", "train",
        "--model", model,
        "--targets", target,
        "--splits", split_type,
        "--output-dir", str(results_dir),
    ]

    print("PYTHONPATH for redun:", env["PYTHONPATH"])
    print("DATA_ROOT for redun:", env["DATA_ROOT"])
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)  # <-- no cwd so results go where you expect
def extract_yaml(results_dir, model, target, split_type, split_idx):
    """Extract metrics from YAML output file with retry for file availability."""
    yml_path = Path(results_dir).resolve() / model / f"results_{split_type}" / f"results_metrics_s{split_idx}_{target}.yml"
    logger.info(f"Checking for YAML: {yml_path}")
    
    # Retry up to 3 times with delay to handle race conditions
    for attempt in range(3):
        if yml_path.exists():
            try:
                with open(yml_path, "r") as f:
                    content = yaml.load(f, Loader=yaml.Loader)
                logger.info(f"Read YAML: {yml_path}")
                return {
                    "test_rmse": content["test"]["rmse"],
                    "test_rho": content["test"]["rho"],
                    "test_tau": content["test"]["tau"],
                    "all_on_rho": content["all"]["on"]["rho"],
                    "all_off_rho": content["all"]["off"]["rho"],
                    "lib_on_rho": content["lib"]["on"]["rho"],
                    "lib_off_rho": content["lib"]["off"]["rho"],
                }
            except Exception as e:
                logger.error(f"Failed to read YAML {yml_path}: {e}")
                return None
        else:
            logger.warning(f"YAML file not found on attempt {attempt + 1}: {yml_path}")
            time.sleep(2)  # Wait 2 seconds before retrying
    logger.error(f"YAML file not found after retries: {yml_path}")
    return None

def main():
    t0 = time.perf_counter()
    args = parse_args()
    logger.info(f"Starting benchmark with args: {args}")
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    
    targets = ["mapk14", "ddr1"]
    split_types = ["random", "disynthon"]
    split_idx = [1, 2]
    
    # Create stratified subsamples for each target
    for target in targets:
        create_stratified_subsample(target, args.nrows, args.data_root, args.local_data_dir)
    
    # Run benchmarks
    rows = []
    for model in args.models:
        for target in targets:
            for split in split_types:
                run_start = time.perf_counter()

                run_redun(model, target, split, args)

                run_dur = time.perf_counter() - run_start
                
                print(f"[TIMING] {model}/{target}/{split} took {run_dur:.1f} sec")
                
                for idx in split_idx:
                    metrics = extract_yaml(args.results_dir, model, target, split, idx)
                    if not metrics:
                        continue
                    row = {
                        "model": model,
                        "target": target,
                        "split_type": split,
                        "split_index": idx,
                        "nrows": args.nrows,
                        **metrics,
                    }
                    rows.append(row)
    
    # Write results to CSV (overwrite if exists)
    logger.info(f"Writing results to {args.csv_out}")
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "model", "target", "split_type", "split_index", "nrows",
            "test_rmse", "test_rho", "test_tau", "all_on_rho", "all_off_rho", "lib_on_rho", "lib_off_rho"
        ])
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"CSV summary written to {args.csv_out}")

    total_dur = time.perf_counter() - t0
    print(f"\n[TIMING] Full benchmarking run took {total_dur/60:.2f} minutes")

if __name__ == "__main__":
    main()
