#!/usr/bin/env sh
# Regenerates every file in samples/ from a fixed seed. Run from the project root.
set -e
python -m labellag simulate --out samples/_full --seed 11 --days 280 --txns-per-day 120
python - <<'PY'
import pandas as pd
tx = pd.read_csv("samples/_full/transactions.csv")
cb = pd.read_csv("samples/_full/chargebacks.csv")
hist = tx[tx["day"] <= 250]
hist.to_csv("samples/transactions.csv", index=False)
cb[cb["report_day"] <= 250].to_csv("samples/chargebacks.csv", index=False)   # as known on day 250
tx[(tx["day"] > 250) & (tx["day"] <= 260)].to_csv("samples/new_transactions.csv", index=False)
PY
rm -rf samples/_full
python -m labellag maturity --transactions samples/transactions.csv --chargebacks samples/chargebacks.csv --cutoff 250 > samples/expected_maturity.json
python -m labellag train --transactions samples/transactions.csv --chargebacks samples/chargebacks.csv --cutoff 250 --strategy ipw --out samples/_model.pkl > samples/expected_train.json
python -m labellag score --model samples/_model.pkl --input samples/new_transactions.csv --budget 0.01 --out samples/expected_scores.csv
rm -f samples/_model.pkl
