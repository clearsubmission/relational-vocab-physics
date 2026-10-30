# How Far Can a Relational Vocabulary Carry Physical Prediction?

Code for *entity-inductive temporal knowledge graph (TKG) forecasting* on two simulator benchmarks:

- **CLEVRER**: forecast which object a given object collides with next (ranking; MRR / Hits@1).
- **Physion**: predict whether the designated agent and patient objects come into contact (AUROC), using Physion's *all-but-one* (held-out scenario) and *all* protocols.

Simulator state is converted into timestamped facts `(s, r, o, t)` under a four-rung **vocabulary ladder**
(events → + kinematics → + relative motion → strongest rung). The vocabulary is crossed with models of increasing expressiveness:
hand-written rules, a linear model on the facts (`symbolic_lr`), learned history encoders (anchor attention `AAE`, `GRU`, `mean`),
and reference models on engineered physical-state features (`physics_lr`, `physics_mlp`, `physics_gbt`).
Every test scene contributes new object entities, so no test entity is seen in training.

---

## Repository layout

```
aae/       Anchor-attention encoder (AAE) model code (imported by the encoder scripts)
clevrer/   CLEVRER: download, fact conversion, baselines, encoders, ablations, bootstrap
physion/   Physion: HDF5 state extraction, fact conversion, baselines, encoders, statistics
```

| Script | Purpose |
|---|---|
| `clevrer/download_clevrer.sh` | Download CLEVRER annotations (no videos needed) |
| `clevrer/clevrer_to_tkg.py` | Annotations → TKG facts and queries for one vocabulary rung |
| `clevrer/baselines.py` | Rule baselines (random, no_repeat, activity, active_not_partner, sym_rule, …) |
| `clevrer/clevrer_learned_baselines.py` | `symbolic_lr`, `physics_lr`, and (`--mlp`) `physics_mlp` conditional-logit rankers |
| `clevrer/clevrer_encoders.py` | History encoders: `--encoder anchor` (AAE), `gru`, `mean`; ablation flags below |
| `clevrer/relation_ablation.py` | Leave-one-relation-out for `symbolic_lr` |
| `clevrer/clevrer_bootstrap.py` | Paired bootstrap over collisions (both query directions resampled together) |
| `clevrer/make_scene_panel.py` | Top-down scene panel used in the overview figure |
| `physion/extract_states.py` | Stream a Physion HDF5 archive, keep object states, collisions, labels |
| `physion/physion_to_tkg.py` | States → TKG facts and queries (`--vocab`, `--holdout`) |
| `physion/physion_baselines.py` | Rules, `symbolic_lr`, `physics_lr` |
| `physion/physion_encoders.py` | AAE / GRU / mean encoders on Physion |
| `physion/physion_nonlinear.py` | `physics_mlp` and `physics_gbt` (scikit-learn) |
| `physion/physion_stats2.py` | Stratified bootstrap CIs, paired differences, future-only subset |
| `physion/physion_featsets_par.py` | Physical-state feature sets: motion / + contact / + chain_hops |
| `physion/summarize_physion*.py` | Per-scenario tables and vocabulary-ladder summaries |

---

## Requirements

Python ≥ 3.10, PyTorch (GPU recommended for the encoders), NumPy, SciPy, scikit-learn, h5py, matplotlib.

```bash
pip install torch numpy scipy scikit-learn h5py matplotlib
```

---

## CLEVRER

### 1. Data
```bash
bash clevrer/download_clevrer.sh          # -> clevrer_ann/ (~15,000 annotation JSON files)
```

### 2. Build the four vocabulary rungs
Splits are by video (80/10/10, seed 0); the first half of each valid/test video (8 of 16 steps, 8 frames per step) is the history.
All thresholds (speed buckets, `near`, the `heading_toward` miss distance) are fit on training videos only and stored in `meta.json`.
```bash
python clevrer/clevrer_to_tkg.py --ann-dir clevrer_ann --out-dir data/clevrer_events --no-kinematics              # (i) events
python clevrer/clevrer_to_tkg.py --ann-dir clevrer_ann --out-dir data/clevrer_kin                                 # (ii) + kinematics
python clevrer/clevrer_to_tkg.py --ann-dir clevrer_ann --out-dir data/clevrer_rel_h32  --relative-motion --horizon 32   # (iii)
python clevrer/clevrer_to_tkg.py --ann-dir clevrer_ann --out-dir data/clevrer_rel_h128 --relative-motion --horizon 128  # (iv)
```

### 3. Baselines
```bash
for D in data/clevrer_events data/clevrer_kin data/clevrer_rel_h32 data/clevrer_rel_h128; do
  python clevrer/baselines.py --data-dir $D
done
python clevrer/clevrer_learned_baselines.py --ann-dir clevrer_ann \
  --data-dirs data/clevrer_events data/clevrer_kin data/clevrer_rel_h32 data/clevrer_rel_h128 --mlp
```
Results are written to `<data-dir>/baselines_test.json` (including per-query reciprocal ranks for the bootstrap).

### 4. History encoders (3 seeds)
```bash
D=data/clevrer_rel_h128
for s in 42 1 2; do
  python clevrer/clevrer_encoders.py --data-dir $D --encoder anchor --seed $s --tag aae_s$s
  python clevrer/clevrer_encoders.py --data-dir $D --encoder gru    --seed $s --tag gru_s$s
  python clevrer/clevrer_encoders.py --data-dir $D --encoder mean   --seed $s --tag mean_s$s
done
```
Each run writes `<data-dir>/aae_results_<tag>.json` with test MRR, Hits@k, MRR by horizon, per-query reciprocal ranks,
a test-time shuffled-history score, and scores under three random identifier permutations.

Ablation flags:

| Flag | Effect |
|---|---|
| `--no-ids` | Identifier-free: zero the frozen entity identifiers |
| `--shuffle-hist` | Shuffle event order in every history (train/valid/test); events keep their timestamps |
| `--train-nb` | Train the identifier embeddings instead of freezing them |
| `--max-history N` | History length (default 20) |

### 5. Analyses
```bash
python clevrer/relation_ablation.py --data-dir data/clevrer_rel_h128
python clevrer/clevrer_bootstrap.py --data-dir data/clevrer_rel_h128 --enc AAE=aae_s GRU=gru_s Mean=mean_s
```

---

## Physion

### 1. Data
Download the per-scenario HDF5 archives from the official Physion release
(readout training sets `<Scenario>_readout_training_HDF5s` and test sets `<Scenario>_testing_HDF5s`
for Collide, Contain, Dominoes, Drape, Drop, Link, Roll, Support), then extract compact states:
```bash
for A in archives/*.tar.gz; do python physion/extract_states.py $A --out states; done
```

### 2. Convert: vocabulary ladder × held-out scenario
`--holdout none` is the in-distribution (*all*) protocol; a scenario name is the *all-but-one* protocol.
Facts use a 45-frame (1.5 s) observation window; thresholds are fit on the training split of the training scenarios only.
```bash
for V in events kin rel full; do
  for H in none Collide Contain Dominoes Drape Drop Link Roll Support; do
    python physion/physion_to_tkg.py --states states --vocab $V --holdout $H --out tkg_$V/holdout_$H
    python physion/physion_baselines.py --data-dir tkg_$V/holdout_$H
  done
done
```

### 3. Encoders (strongest rung, 3 seeds)
```bash
for H in none Collide Contain Dominoes Drape Drop Link Roll Support; do for s in 42 1 2; do
  python physion/physion_encoders.py --data-dir tkg_full/holdout_$H --encoder anchor --seed $s --tag s$s
  python physion/physion_encoders.py --data-dir tkg_full/holdout_$H --encoder gru    --seed $s --tag gru_s$s
  python physion/physion_encoders.py --data-dir tkg_full/holdout_$H --encoder mean   --seed $s --tag mean_s$s
done; done
```

### 4. Statistics and physical-state analyses
```bash
python physion/summarize_physion.py tkg_full auroc
python physion/physion_nonlinear.py --root tkg_full
python physion/physion_stats2.py --root tkg_full \
  --enc "AAE=aae_results_s*.json" "GRU=aae_results_gru_s*.json" "Mean=aae_results_mean_s*.json"
python physion/physion_featsets_par.py --root tkg_full --boot 5000
```

---

## Reproducing the paper

| Paper result | Produced by |
|---|---|
| CLEVRER vocabulary × model table, ladder figure (a) | CLEVRER steps 2–4 on all four rungs |
| CLEVRER MRR by horizon | `clevrer_encoders.py` / `clevrer_learned_baselines.py` (`_by_horizon` in the JSON outputs) |
| Relation ablation | `relation_ablation.py` |
| Order shuffle, identifier-free, identifier permutation | `clevrer_encoders.py` with `--shuffle-hist` / `--no-ids`; permutation scores in every run's JSON |
| CLEVRER bootstrap intervals | `clevrer_bootstrap.py` |
| Physion per-scenario AUROC table, ladder figure (b) | Physion steps 2–3, `summarize_physion*.py` |
| Physion nonlinear physical-state baselines | `physion_nonlinear.py` |
| Physion bootstrap intervals, future-only subset | `physion_stats2.py` |
| Physion feature-set table (motion / + contact / + chain_hops) | `physion_featsets_par.py` |

Default hyperparameters in each script match the paper (dimension 200, 16 anchors, σ = 0.5, Adam 1e-3,
early stopping on validation; CLEVRER history 20 and forecasting-style cutoffs {4, 6, 8, 10, 12}; Physion history 50).

## License
Released for research use. Dataset licenses: see the CLEVRER and Physion project pages.
