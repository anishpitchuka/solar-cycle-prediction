import json

with open("/sessions/compassionate-brave-bardeen/mnt/solar_cycle_pred2/phase5/phase5_trajectory_forecasting.ipynb") as f:
    nb = json.load(f)

def code(src):
    return {"cell_type":"code","metadata":{},"source":src,"outputs":[],"execution_count":None}
def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src}

# ── Cell A: Improved LOCO CV ──────────────────────────────────────────────────
cell_a = code("""## ── IMPROVEMENT 1+2+3: Augmentation + Cycle-Norm + Ensemble ────────────────
# 1. Data augmentation  — 3 training examples per cycle (windows 12, 24, 36m)
# 2. Cycle-normalised input — divide input AND target by the window mean
#    (model learns trajectory SHAPE relative to early activity level)
# 3. Ensemble — average predictions from GRU, LSTM, TCN

AUG_WINDOWS = [12, 24, 36]

def loco_cv_improved(aug_windows=AUG_WINDOWS):
    records = []
    for test_cid in cycle_ids:
        train_cids = [c for c in cycle_ids if c != test_cid]

        # ── Build augmented, cycle-normalised training set ──────────────────
        X_aug, Y_aug, M_aug = [], [], []
        for c in train_cids:
            for w in aug_windows:
                # Zero-pad raw input to WINDOW length
                obs      = np.zeros(WINDOW)
                tlen     = min(len(trajectories[c]), w)
                obs[:tlen] = trajectories[c][:tlen]

                # Cycle normalisation — scale by window mean (clip to avoid /0)
                win_mean = max(float(trajectories[c][:tlen].mean()), 5.0)

                X_aug.append(obs              / win_mean)   # normalised input
                Y_aug.append(padded_trajs[c]  / win_mean)   # normalised target
                M_aug.append(traj_masks[c])

        X_tr = np.array(X_aug)   # (N*3, WINDOW)
        Y_tr = np.array(Y_aug)   # (N*3, MAX_LEN)
        M_tr = np.array(M_aug)   # (N*3, MAX_LEN)

        # ── Test observation (full WINDOW, cycle-normalised) ─────────────────
        obs_test = np.zeros(WINDOW)
        tlen     = min(len(trajectories[test_cid]), WINDOW)
        obs_test[:tlen] = trajectories[test_cid][:tlen]
        test_scale = max(float(trajectories[test_cid][:tlen].mean()), 5.0)
        obs_test_norm = obs_test / test_scale

        # ── Train all three models and ensemble ──────────────────────────────
        preds = []
        for mcls in model_classes.values():
            model = train_model(mcls, X_tr, Y_tr, M_tr, output_size=MAX_LEN)
            with torch.no_grad():
                out = model(torch.FloatTensor(obs_test_norm).unsqueeze(0).unsqueeze(-1)).numpy()[0]
            preds.append(out * test_scale)   # scale back to absolute SN

        pred_traj   = np.mean(preds, axis=0)    # ensemble average
        actual_traj = trajectories[test_cid]
        actual_len  = len(actual_traj)

        pred_peak_mag   = float(pred_traj[:actual_len].max())
        pred_peak_month = int(pred_traj[:actual_len].argmax()) + 1
        actual_peak_mag   = float(actual_traj.max())
        actual_peak_month = int(actual_traj.argmax()) + 1

        records.append({
            "Cycle_ID":          test_cid,
            "actual_peak_mag":   actual_peak_mag,
            "pred_peak_mag":     pred_peak_mag,
            "mag_error":         pred_peak_mag - actual_peak_mag,
            "mag_abs_error":     abs(pred_peak_mag - actual_peak_mag),
            "actual_peak_month": actual_peak_month,
            "pred_peak_month":   pred_peak_month,
            "time_error":        pred_peak_month - actual_peak_month,
            "time_abs_error":    abs(pred_peak_month - actual_peak_month),
            "pred_traj":         pred_traj[:actual_len].tolist(),
            "actual_traj":       actual_traj.tolist(),
        })
        print(f"  SC{test_cid:2d} | mag_err={records[-1]['mag_error']:+6.1f} SN | "
              f"time_err={records[-1]['time_error']:+4.0f}m")

    return pd.DataFrame(records)

print("Improved LOCO CV function ready.")
print(f"Training samples per fold: {(len(cycle_ids)-1) * len(AUG_WINDOWS)}  "
      f"(was {len(cycle_ids)-1} before augmentation)")""")

# ── Cell B: Run improved LOCO CV ──────────────────────────────────────────────
cell_b = code("""print("Running improved LOCO CV (augmentation + cycle-norm + ensemble)...")
t0 = time.time()
df_improved = loco_cv_improved()
elapsed = time.time() - t0

mag_mae   = df_improved["mag_abs_error"].mean()
mag_rmse  = np.sqrt((df_improved["mag_error"]**2).mean())
time_mae  = df_improved["time_abs_error"].mean()
time_rmse = np.sqrt((df_improved["time_error"]**2).mean())
mag_r2    = r2_score(df_improved["actual_peak_mag"],   df_improved["pred_peak_mag"])
time_r2   = r2_score(df_improved["actual_peak_month"], df_improved["pred_peak_month"])

print(f"\\nImproved model — elapsed: {elapsed:.1f}s")
print(f"  Peak Magnitude: MAE={mag_mae:.2f}  RMSE={mag_rmse:.2f}  R2={mag_r2:.4f}")
print(f"  Peak Timing   : MAE={time_mae:.2f}m  RMSE={time_rmse:.2f}m  R2={time_r2:.4f}")

# Trajectory-level metrics
all_errors = []
for _, row in df_improved.iterrows():
    actual = np.array(row["actual_traj"])
    pred   = np.array(row["pred_traj"])
    all_errors.extend(np.abs(actual - pred).tolist())
traj_mae  = np.mean(all_errors)
traj_rmse = np.sqrt(np.mean(np.array(all_errors)**2))
print(f"  Trajectory    : MAE={traj_mae:.2f}  RMSE={traj_rmse:.2f}")""")

# ── Cell C: Comparison table ──────────────────────────────────────────────────
cell_c = code("""# Compare original vs improved
print("=" * 70)
print("  PHASE 5 — ORIGINAL vs IMPROVED")
print("=" * 70)
print(f"\\n{'Model':<28s}  {'Mag_MAE':>7s}  {'Mag_RMSE':>8s}  {'Mag_R2':>7s}  "
      f"{'Time_MAE':>8s}  {'Time_R2':>7s}  {'Traj_MAE':>8s}")

# Original models
for mname, df in all_loco.items():
    traj_errs = []
    for _, row in df.iterrows():
        traj_errs.extend(np.abs(np.array(row["actual_traj"]) - np.array(row["pred_traj"])).tolist())
    print(f"  {mname:<26s}  "
          f"{df['mag_abs_error'].mean():>7.2f}  "
          f"{np.sqrt((df['mag_error']**2).mean()):>8.2f}  "
          f"{r2_score(df['actual_peak_mag'], df['pred_peak_mag']):>7.4f}  "
          f"{df['time_abs_error'].mean():>8.2f}  "
          f"{r2_score(df['actual_peak_month'], df['pred_peak_month']):>7.4f}  "
          f"{np.mean(traj_errs):>8.2f}")

# Improved ensemble
imp_traj_errs = []
for _, row in df_improved.iterrows():
    imp_traj_errs.extend(np.abs(np.array(row["actual_traj"]) - np.array(row["pred_traj"])).tolist())
print(f"  {'Ensemble+Aug+CycleNorm':<26s}  "
      f"{df_improved['mag_abs_error'].mean():>7.2f}  "
      f"{np.sqrt((df_improved['mag_error']**2).mean()):>8.2f}  "
      f"{r2_score(df_improved['actual_peak_mag'], df_improved['pred_peak_mag']):>7.4f}  "
      f"{df_improved['time_abs_error'].mean():>8.2f}  "
      f"{r2_score(df_improved['actual_peak_month'], df_improved['pred_peak_month']):>7.4f}  "
      f"{np.mean(imp_traj_errs):>8.2f}")

# Save improved results
df_improved[["Cycle_ID","actual_peak_mag","pred_peak_mag","mag_error","mag_abs_error",
             "actual_peak_month","pred_peak_month","time_error","time_abs_error"]].to_csv(
    "results/phase5_improved.csv", index=False)
print("\\nSaved: results/phase5_improved.csv")""")

nb["cells"].extend([
    md("## Improvements: Augmentation + Cycle Normalisation + Ensemble"),
    cell_a,
    cell_b,
    cell_c
])

with open("/sessions/compassionate-brave-bardeen/mnt/solar_cycle_pred2/phase5/phase5_trajectory_forecasting.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook now has {len(nb['cells'])} cells")
