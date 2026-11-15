# miRNA–target prediction: CNN on a pairing matrix

Predicts whether a miRNA binds a 50-nt target site, trained on miRBench's
`AGO2_CLASH_Hejret2023` data. The only inputs are the `noncodingRNA` (miRNA) and `gene`
(target site) sequences. No other columns are used, because some of them (for example
`Nunique`) reveal the label.

## Files

| File | What it does |
|---|---|
| `mirna_cnn.ipynb` | The whole pipeline: data, encoding, model, training, validation, final test, plots |
| `evaluate.py` | Separate, model-agnostic evaluator: scores any predictions file against reference labels |

## Setup

```bash
pip install torch pandas numpy scikit-learn matplotlib jupyter
```

Download these from [Zenodo record 20540907](https://zenodo.org/records/20540907) into
the same folder as the notebook:

- `AGO2_CLASH_Hejret2023_train.tsv.gz`
- `AGO2_CLASH_Hejret2023_test.tsv.gz` (needed only for the final test)

## Reproduce

Open `mirna_cnn.ipynb` and choose **Run All**. It writes:

| Output | Content |
|---|---|
| `val_split.tsv` | The 20% validation split (stratified, seed 42) |
| `cnn.pt` | Trained CNN weights (from the epoch with the best validation loss) |
| `pred_cnn_val.tsv`, `pred_baseline_val.tsv` | Validation predictions from the CNN and the baseline |

The final test only runs if you set `RUN_TEST = True` in the last cell. Run it **once**,
after all model choices are fixed.

The split and model initialisation are seeded, but PyTorch training is not bit-for-bit
deterministic (especially on a GPU), so reruns can differ by about ±0.01 in the metrics.

## Approach

1. **Split.** The training file is split 80/20 into train and validation sets
   (stratified, seed 42). The official test set is kept unseen.
2. **Encoding.** Each pair becomes a 50 × 26 grid with target positions as rows and miRNA
   positions as columns:
   - 1 = Watson–Crick pair (A–T, C–G)
   - 0.5 = G–T wobble pair
   - 0 = no pair, or padding

   miRNAs (17–26 nt) are padded on the 3' end, so the seed region stays in the same
   columns for every miRNA. A run of consecutive base pairs shows up as a diagonal in the
   grid.

      <img src="pairing_matrix.png" width="550" alt="Pairing matrix example">

   *Toy example (8-nt miRNA, 10-nt target; the real grid is 50 × 26). The bound miRNA's
   8 consecutive base pairs (top) become one diagonal run in the grid (bottom).*

3. **Model.** Three 3×3 convolution layers with a 2×2 max pool, then global max pooling,
   dropout 0.3 and a linear output layer (about 56k parameters). The convolutions detect
   diagonal pairing patterns up to about seed length, and global max pooling keeps each
   filter's best match anywhere in the grid.
4. **Training.** Adam optimiser (learning rate 1e-3), batch size 64, binary cross-entropy
   loss. Training uses early stopping on validation loss (patience 5) and keeps the best
   epoch's weights.
5. **Baseline.** A rule-based model: score = the longest unbroken run of paired bases
   along any binding diagonal. It checks whether the CNN learns more than "long
   complementary stretch = binding."

## Evaluating any model

`evaluate.py` only reads files, so it can score any model's predictions:

```bash
python evaluate.py reference.tsv[.gz] predictions.tsv [threshold=0.5]
```

- **Reference file:** TSV with columns `gene`, `noncodingRNA`, `label`. Extra columns are
  ignored.
- **Predictions file:** TSV with columns `gene`, `noncodingRNA`, `score`, in the same row
  order as the reference file. A higher score means "more likely to bind."
- **Safety check:** files with a different row count or a different row order are
  rejected, not silently scored wrong.

| Metric | Meaning | Random guessing gives |
|---|---|---|
| **AUROC** (main metric) | Probability that a random binding pair scores higher than a random negative pair. Doesn't depend on a threshold. | 0.5 |
| AUPRC | Precision averaged over all recall levels; focuses on the positive class | ≈ share of positives (0.5) |
| Accuracy | Share of correct yes/no calls at the threshold. Fair here because the classes are balanced. | 0.5 |
| F1 | Harmonic mean of precision and recall | – |
| MCC | Correlation between predictions and labels, using all four confusion-matrix cells (−1 to 1) | 0 |
| Confusion matrix | Counts of true negatives, false positives, false negatives and true positives | – |

AUROC and AUPRC are threshold-free, so they compare models whose scores are on different
scales. The baseline's score is a run length, so its threshold is 7, the length of a seed
match.

## Results (validation set, 1628 pairs)

| Model | AUROC | AUPRC | Accuracy | F1 | MCC |
|---|---|---|---|---|---|
| CNN (threshold 0.5) | 0.843 | 0.864 | 0.773 | 0.761 | 0.549 |
| Longest-run baseline (threshold 7) | 0.761 | 0.737 | 0.694 | 0.726 | 0.398 |

The CNN beats the baseline on every metric. The baseline finds most binders but has many
false positives (340), because negative pairs often contain long complementary runs by
chance. The CNN is more precise (143 false positives) but misses more binders (227 false
negatives).

Test-set results: *to be added after the single final run.*

## Limitations

- **Possible optimism in validation scores.** The 80/20 split is random, so the same miRNA
  (and possibly the same target site) can appear in both train and validation.
- **Hand-chosen encoding and settings.** The 0.5 wobble score, the architecture and the
  hyperparameters were chosen by hand and only lightly tuned.
- **Small run-to-run variation.** Training isn't fully deterministic (see Reproduce).
- **Truncated long miRNAs.** miRNAs longer than 26 nt would lose their 3' end. None occur
  in the training data.
