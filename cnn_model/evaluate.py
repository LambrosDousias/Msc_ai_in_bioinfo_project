"""Usage: python evaluate.py reference.tsv[.gz] predictions.tsv [threshold=0.5]
reference: gene, noncodingRNA, label | predictions: gene, noncodingRNA, score (same row order)"""
import sys

import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

ref = pd.read_csv(sys.argv[1], sep="\t")
pred = pd.read_csv(sys.argv[2], sep="\t")
thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

# predictions may come from anyone's code: check they line up with the reference before scoring
assert len(ref) == len(pred), f"{len(pred)} predictions for {len(ref)} reference rows"
cols = ["gene", "noncodingRNA"]
assert (ref[cols].values == pred[cols].values).all(), "prediction rows don't match reference order"

y, s = ref["label"], pred["score"]
yhat = (s >= thr).astype(int)
print(f"AUROC     {roc_auc_score(y, s):.3f}")
print(f"AUPRC     {average_precision_score(y, s):.3f}")
print(f"Accuracy  {accuracy_score(y, yhat):.3f}   (threshold {thr})")
print(f"F1        {f1_score(y, yhat):.3f}")
print(f"MCC       {matthews_corrcoef(y, yhat):.3f}")
print("Confusion [[TN FP] [FN TP]]:\n", confusion_matrix(y, yhat))
