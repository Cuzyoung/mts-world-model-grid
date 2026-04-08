#!/bin/bash
# Download ETT and ECL datasets

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "=== Downloading ETT datasets ==="
mkdir -p "$DATA_DIR/ETT-small"
for f in ETTh1.csv ETTh2.csv ETTm1.csv; do
    if [ ! -f "$DATA_DIR/ETT-small/$f" ]; then
        echo "Downloading $f..."
        wget -q "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/$f" \
             -O "$DATA_DIR/ETT-small/$f"
        echo "  -> $DATA_DIR/ETT-small/$f"
    else
        echo "  $f already exists, skipping."
    fi
done

echo "=== Downloading ECL dataset ==="
mkdir -p "$DATA_DIR/ECL"
if [ ! -f "$DATA_DIR/ECL/electricity.csv" ]; then
    echo "Downloading electricity.csv..."
    wget -q "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt" \
         -O "$DATA_DIR/ECL/electricity.txt"
    # Convert to CSV format: add header
    python3 -c "
import pandas as pd
import numpy as np
data = np.loadtxt('$DATA_DIR/ECL/electricity.txt', delimiter=',')
cols = [f'MT_{i:03d}' for i in range(data.shape[1])]
df = pd.DataFrame(data, columns=cols)
df.insert(0, 'date', pd.date_range('2012-01-01', periods=len(df), freq='h'))
df.to_csv('$DATA_DIR/ECL/electricity.csv', index=False)
print(f'ECL data saved: {df.shape}')
"
    rm -f "$DATA_DIR/ECL/electricity.txt"
else
    echo "  electricity.csv already exists, skipping."
fi

echo ""
echo "=== Done! ==="
echo "ETT files: $(ls $DATA_DIR/ETT-small/)"
echo "ECL files: $(ls $DATA_DIR/ECL/)"
