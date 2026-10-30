#!/usr/bin/env python3
import glob, json, os, sys
import numpy as np
METRIC = sys.argv[1] if len(sys.argv) > 1 else "auroc"
ROOTS = [("events", "tkg_vocab/events"), ("kin", "tkg_vocab/kin"), ("rel", "tkg_vocab/rel"), ("full", "tkg")]
H = ["Collide", "Contain", "Dominoes", "Drape", "Drop", "Link", "Roll", "Support"]
def get(root, h, m):
    d = os.path.join(root, f"holdout_{h}")
    b = json.load(open(os.path.join(d, "baselines.json"))) if os.path.exists(os.path.join(d, "baselines.json")) else {}
    k = [json.load(open(f))[METRIC] for f in glob.glob(os.path.join(d, "aae_results_*.json"))]
    if m == "AAE":
        return (np.mean(k) if k else np.nan), (np.std(k) if len(k) > 1 else 0.0)
    return b.get(m, {}).get(METRIC, np.nan), 0.0
print(f"metric: {METRIC}")
print(f"{'vocab':8s} {'symbolic_lr LOSO':>17s} {'physics_lr LOSO':>16s} {'AAE LOSO':>11s} {'symbolic_lr all8':>17s} {'AAE all8':>14s}")
for name, root in ROOTS:
    if not os.path.isdir(root):
        continue
    lo = {m: np.nanmean([get(root, h, m)[0] for h in H]) for m in ("symbolic_lr", "physics_lr", "AAE")}
    a8s = get(root, "none", "symbolic_lr")[0]; a8k, a8sd = get(root, "none", "AAE")
    print(f"{name:8s} {lo['symbolic_lr']:17.3f} {lo['physics_lr']:16.3f} {lo['AAE']:11.3f} {a8s:17.3f} {a8k:9.3f}±{a8sd:.3f}")
