#!/usr/bin/env python3
"""Leave-one-scenario-out table: rows = held-out scenario, cols = baselines + AAE (mean±std over seeds)."""
import glob, json, os, sys
import numpy as np
ROOT = sys.argv[1] if len(sys.argv) > 1 else "tkg"
METRIC = sys.argv[2] if len(sys.argv) > 2 else "acc"
B = ["majority", "contact_obs", "gap_end", "extrapolate", "chain_reach", "physics_lr", "symbolic_lr"]
rows = ["Collide", "Contain", "Dominoes", "Drape", "Drop", "Link", "Roll", "Support", "none"]
print(f"metric: {METRIC}   (row 'none' = trained on all 8 scenarios, tested on all)")
print(f"{'held-out':10s} " + " ".join(f"{b:>12s}" for b in B) + f" {'AAE':>14s}")
cols = {b: [] for b in B + ["AAE"]}
for h in rows:
    d = os.path.join(ROOT, f"holdout_{h}")
    if not os.path.isdir(d):
        continue
    bl = json.load(open(os.path.join(d, "baselines.json"))) if os.path.exists(os.path.join(d, "baselines.json")) else {}
    ks = [json.load(open(f))[METRIC] for f in glob.glob(os.path.join(d, "aae_results_*.json"))]
    vals = [bl.get(b, {}).get(METRIC, np.nan) for b in B]
    kstr = f"{np.mean(ks):.3f}±{np.std(ks):.3f}" if len(ks) > 1 else (f"{ks[0]:.3f}" if ks else "-")
    print(f"{h:10s} " + " ".join(f"{v:12.3f}" for v in vals) + f" {kstr:>14s}")
    if h != "none":
        for b, v in zip(B, vals):
            cols[b].append(v)
        if ks:
            cols["AAE"].append(np.mean(ks))
print(f"{'LOSO mean':10s} " + " ".join(f"{np.nanmean(cols[b]) if cols[b] else np.nan:12.3f}" for b in B)
      + f" {np.mean(cols['AAE']) if cols['AAE'] else np.nan:14.3f}")
