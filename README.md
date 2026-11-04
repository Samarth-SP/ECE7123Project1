# ECE-GY 7123 Project 1: Efficient ResNet Architecture Challnge on CIFAR-10

## Contents

```
proj.ipynb       full pipeline: data, model, sweep, long runs, final run, submission
results.csv      log of every run (config, accuracy, per-epoch curves); the notebook skips runs already logged
final_1.pt       weights of the submitted model
submission.csv   test-set predictions (id,label)
report/          AAAI-format report (report.tex, report.pdf, figure, bibliography)
```

## Method

A greedy sweep with 15-epoch runs over three axes, in order: architecture, optimizer, regularization.
The best configurations were then retrained for 150 epochs.
The final model is a three-stage ResNet with widths 96-160-256 and 4-4-2 BasicBlocks.
It is trained with SGD (Nesterov, peak lr 0.2, wd 5e-4) under a one-cycle schedule, with label smoothing 0.1,
TrivialAugment, random erasing, and horizontal-flip test-time augmentation.
For submission it was retrained on all 50k images for 250 epochs. See `report/report.pdf` for details.

## Results

|                                       | Accuracy |
| ------------------------------------- | -------- |
| Held-out validation (5,000 images)    | 97.06%   |
| Public leaderboard (all-data retrain) | 88.4%    |

## Reproducing

1. Download the competition data into `data/`:
   ```
   data/train_images/  data/train_labels.csv  data/test_images/  data/test.csv
   ```
2. Install PyTorch, torchvision, numpy, pandas, Pillow, and matplotlib.
3. Run `proj.ipynb` top to bottom.
   - With `results.csv` present, every logged run is skipped, and the last cell rebuilds `submission.csv` from `final_1.pt`.
   - Delete `results.csv` to rerun everything from scratch.
     That took roughly 25 hours on an Apple-silicon GPU M3 Max (MPS); the code also uses CUDA when it is available.
