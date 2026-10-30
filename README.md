# How far can a relational vocabulary carry physical prediction?
Entity-inductive temporal KG forecasting on CLEVRER (collision forecasting) and Physion (object-contact prediction):
vocabulary-ladder fact conversion, rule/linear baselines, physical-state baselines (linear/MLP/trees),
history encoders (anchor-attention `aae/`, GRU, mean), and bootstrap statistics.
- `aae/` anchor-attention encoder model
- `clevrer/` download, conversion (`clevrer_to_tkg.py`), baselines, encoders (`clevrer_encoders.py --encoder {anchor,gru,mean}`), relation ablation, bootstrap
- `physion/` state extraction, conversion (`physion_to_tkg.py --vocab ... --holdout <scenario|none>`), baselines, encoders, statistics
Run any script with `--help`. Requirements: Python 3.10+, PyTorch, NumPy, scikit-learn, h5py, matplotlib.
