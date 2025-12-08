# Notip vs pARI : Comparison of Post Hoc inference methods for fMRI

This repository contains the code to reproduce all experiments of the paper "Cluster Size Matters: A Comparative Study of Notip and pARI for Post Hoc Inference in fMRI"
by Nils Peyrouset, Pierre Neuvial, and Bertrand Thirion.
Preprint available at: https://arxiv.org/abs/2511.02422

Note that the first time these scripts are run, the fMRI data fetching from Neurovault will take place, which takes a significant amount of time. This only needs to be done once.

## Installing dependencies

```
python -m pip install -r requirements.txt
```

## Reproducing figures

To reproduce the figures, run :

```
python3 scripts/compute_thresholds.py
```

This computes the thresholds needed to reproduce the figures.


### Tables

To reproduce all tables from the paper, run :

```
python3 scripts/tables.py
```

These results can be visualized as follows (Figure 3, using R):

```r
source("scripts/figure-3.R")
```

### Brain Visualization (Figure 1)

To generate the brain visualization, run :

```
python3 scripts/brain_plot.py
```

### Plot TDP curves (Figure 2)

To generate the TDP curves, run :

```
python3 scripts/confidence_curve_plot.py
```
