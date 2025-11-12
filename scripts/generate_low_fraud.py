#!/usr/bin/env python3
"""Generate a low-fraud (10%) variant of the bundled dataset.
This script reads the existing `data/upi_transactions_2025.csv` and writes
`data/upi_transactions_2025_10pct.csv` with the same columns but adjusted
`fraud_flag` proportion (10%).
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC = DATA_DIR / "upi_transactions_2025.csv"
OUT = DATA_DIR / "upi_transactions_2025_10pct.csv"

if not SRC.exists():
    raise SystemExit(f"Source data not found: {SRC}")

print(f"Reading source: {SRC}")
df = pd.read_csv(SRC)

n = len(df)
num_fraud = int(n * 0.10)
print(f"Original rows: {n}; target fraud rows: {num_fraud} ({num_fraud/n:.2%})")

# If dataset already has fraud_flag column, sample from existing fraud/non-fraud rows to assemble the new set
if "fraud_flag" in df.columns:
    fraud_rows = df[df["fraud_flag"] == 1]
    nonfraud_rows = df[df["fraud_flag"] == 0]

    # sample fraud rows (with replacement if necessary)
    if len(fraud_rows) >= num_fraud:
        fraud_sample = fraud_rows.sample(num_fraud, replace=False, random_state=42)
    else:
        fraud_sample = fraud_rows.sample(num_fraud, replace=True, random_state=42)

    nonfraud_needed = n - num_fraud
    if len(nonfraud_rows) >= nonfraud_needed:
        nonfraud_sample = nonfraud_rows.sample(nonfraud_needed, replace=False, random_state=42)
    else:
        nonfraud_sample = nonfraud_rows.sample(nonfraud_needed, replace=True, random_state=42)

    new_df = pd.concat([fraud_sample, nonfraud_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
else:
    # No label column: create a random label with desired fraud_rate
    new_df = df.copy()
    choices = np.zeros(n, dtype=int)
    choices[:num_fraud] = 1
    np.random.RandomState(42).shuffle(choices)
    new_df["fraud_flag"] = choices

OUT.parent.mkdir(parents=True, exist_ok=True)
new_df.to_csv(OUT, index=False)
print(f"Wrote: {OUT} (fraud fraction: {new_df['fraud_flag'].mean():.3f})")
