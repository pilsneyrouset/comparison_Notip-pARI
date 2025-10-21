# Notip vs pARI : Comparison of Post Hoc inference methods for fMRI

This repository contains the code to reproduce all experiments of the paper (METTRE LIEN VERS ARTICLE).
Note that the first time you run one of those scripts, the fMRI data fetching from Neurovault will take place, which takes a significant amount of time. This only needs to be done once.

### Installing dependencies

```
python -m pip install -r requirements.txt
```

### Reproducing figures

#### Tables
To reproduce all tables from the paper, run :

```
python3 scripts/code_tables.py
```
#### Brain Visualization
To generate the brain visualization, run :

```
python3 scripts/brain_plot.py
```

#### Plot TDP curves
First, compute the confidence curves :

```
python3 scripts/confidence_curve.py
```

Then, plot them using :

```
python3 scripts/confidence_curve_plot.py
```
