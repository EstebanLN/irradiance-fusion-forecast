# %% [markdown]
# # Hour × Horizon Heatmaps for All Architectures (Ground Tabular)

# %% [markdown]
# ## Libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# %% [markdown]
# ## Configuration

# %%
# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Define directory paths for data and output
DATA_DIR = Path("../data_processed")
OUT_DIR  = Path("../reports/skill_heatmaps_ground_all_archs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Forecast horizons to analyze (in 10-minute steps)
H_LIST = [6, 18, 36]

# Fixed sequence length for sequential models
SEQ_LEN = 12   # Can be changed to 18, etc.

print("Output directory:", OUT_DIR.resolve())

# %% [markdown]
# ## Helper Functions

# %%
def _rmse(a, b):
    """Calculate Root Mean Square Error between two arrays."""
    return float(np.sqrt(mean_squared_error(a, b)))

def skill(y_true, y_pred, y_base):
    """Calculate skill score: 1 - (RMSE_model / RMSE_baseline)."""
    return 1.0 - (_rmse(y_true, y_pred) / (_rmse(y_true, y_base) + 1e-6))

def _build_seq(X_df, y_ser, L):
    """Build sequences without preserving index information."""
    Xv, yv = X_df.values, y_ser.values
    xs, ys = [], []
    for i in range(L-1, len(X_df)):
        block = Xv[i-L+1:i+1]
        if np.isnan(block).any():
            continue
        xs.append(block); ys.append(yv[i])
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="float32")

def build_seq_with_idx(X_df, y_ser, L):
    """Build sequences while preserving index for alignment with baseline and hour extraction."""
    Xv, yv = X_df.values, y_ser.values
    xs, ys, idx = [], [], []
    for i in range(L-1, len(X_df)):
        block = Xv[i-L+1:i+1]
        if np.isnan(block).any():
            continue
        xs.append(block); ys.append(yv[i]); idx.append(X_df.index[i])
    return (
        np.asarray(xs, dtype="float32"),
        np.asarray(ys, dtype="float32"),
        pd.DatetimeIndex(idx)
    )

def hourly_skill(y_true, y_pred, y_base, idx):
    """
    Calculate skill vs persistence by hour of day:
    skill(h) = 1 - RMSE_model(h) / RMSE_base(h).
    Also stores RMSE and MSE by hour for additional heatmaps.
    """
    # Create DataFrame with predictions and baseline
    df = pd.DataFrame({
        "time": idx,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_base": y_base,
    }).set_index("time")
    df["hour"] = df.index.hour

    # Calculate metrics for each hour
    rows = []
    for h in range(24):
        sub = df[df["hour"] == h]
        if len(sub) < 10:  # Skip hours with insufficient data
            continue
        rmse_m = _rmse(sub["y_true"], sub["y_pred"])
        rmse_b = _rmse(sub["y_true"], sub["y_base"])
        mse_m  = rmse_m**2
        mse_b  = rmse_b**2
        skl = 1.0 - rmse_m / (rmse_b + 1e-6)
        rows.append((h, rmse_m, rmse_b, mse_m, mse_b, skl))

    # Return empty DataFrame if no valid hours found
    if not rows:
        return pd.DataFrame(columns=["rmse_model", "rmse_base", "mse_model", "mse_base", "skill"])

    # Create results DataFrame
    res = pd.DataFrame(
        rows,
        columns=["hour", "rmse_model", "rmse_base", "mse_model", "mse_base", "skill"]
    ).set_index("hour")
    return res

# %% [markdown]
# ## Model Builders (simple, without Optuna)

# %%
def build_mlp(input_dim: int):
    """Build a simple Multi-Layer Perceptron model."""
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(1, dtype="float32"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

def build_lstm(L, n_feat):
    """Build LSTM model for sequential data."""
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.LSTM(64, dropout=0.1)(inp)
    out = layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

def build_bilstm(L, n_feat):
    """Build Bidirectional LSTM model."""
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.Bidirectional(layers.LSTM(64, dropout=0.1))(inp)
    out = layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

def build_cnnlstm(L, n_feat):
    """Build CNN-LSTM hybrid model."""
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inp)
    x   = layers.LSTM(64, dropout=0.1)(x)
    out = layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

def build_transformer(L, n_feat):
    """Build Transformer-based model for sequential data."""
    d_model = 64
    heads   = 4
    ff_dim  = 128
    att_do  = 0.1
    do      = 0.1

    inp = layers.Input(shape=(L, n_feat))
    x   = layers.Dense(d_model)(inp)
    x2  = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=d_model // heads,
        dropout=att_do
    )(x, x)
    x   = layers.Add()([x, x2])
    x   = layers.LayerNormalization()(x)
    ff  = layers.Dense(ff_dim, activation="relu")(x)
    ff  = layers.Dense(d_model)(ff)
    x   = layers.Add()([x, ff])
    x   = layers.LayerNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(do)(x)
    out = layers.Dense(1, dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model

# %% [markdown]
# ## Architectures Configuration

# %%
# Define architecture configurations with their types and builders
ARCH_CONFIGS = {
    "MLP": {
        "type": "tabular",  # Uses flat input features
        "builder": lambda input_dim, L, n_feat: build_mlp(input_dim),
    },
    "LSTM": {
        "type": "seq",  # Uses sequential input
        "builder": lambda input_dim, L, n_feat: build_lstm(L, n_feat),
    },
    "BiLSTM": {
        "type": "seq",
        "builder": lambda input_dim, L, n_feat: build_bilstm(L, n_feat),
    },
    "CNN-LSTM": {
        "type": "seq",
        "builder": lambda input_dim, L, n_feat: build_cnnlstm(L, n_feat),
    },
    "Transformer": {
        "type": "seq",
        "builder": lambda input_dim, L, n_feat: build_transformer(L, n_feat),
    },
}

# %% [markdown]
# ## Main Loop: Horizon × Architecture

# %%
# Initialize dictionaries to store results
metrics_by_arch_and_h = {arch: {} for arch in ARCH_CONFIGS}
hourly_skill_by_arch_and_h = {arch: {} for arch in ARCH_CONFIGS}

# Loop through each forecast horizon
for H in H_LIST:
    print("\n" + "="*60)
    print(f"=== Horizon H={H} steps (≈ {H*10} minutes) ===")
    print("="*60)

    # ---------- Data Loading ----------
    train_pq = DATA_DIR / f"ground_train_h{H}.parquet"
    val_pq   = DATA_DIR / f"ground_val_h{H}.parquet"
    test_pq  = DATA_DIR / f"ground_test_h{H}.parquet"

    # Check if required files exist
    if not (train_pq.exists() and val_pq.exists() and test_pq.exists()):
        print(f"Missing parquet files for h={H}, skipping completely.")
        continue

    # Load datasets
    train = pd.read_parquet(train_pq).sort_index()
    val   = pd.read_parquet(val_pq).sort_index()
    test  = pd.read_parquet(test_pq).sort_index()

    # ---------- Target Column Identification ----------
    # Define candidate target columns in order of preference
    candidatos = [f"y_ghi_h{H}", f"y_k_h{H}", f"y_ghi_sg_h{H}"]
    y_cols = [c for c in train.columns if c.startswith("y_")]
    target = None
    
    # Try to find the preferred target column
    for c in candidatos:
        if c in train.columns:
            target = c
            break
    
    # Fallback: any column ending with the horizon
    if target is None:
        posibles = [c for c in y_cols if c.endswith(f"_h{H}")]
        if posibles:
            target = posibles[0]
    
    # Skip if no target found
    if target is None:
        print(f"  → No target column y_* found for h={H}, skipping.")
        continue

    print("  Target used:", target)

    # ---------- Feature Selection ----------
    # Find common numeric columns across all datasets
    common_cols = set(train.columns) & set(val.columns) & set(test.columns)
    feat_cols = sorted([
        c for c in common_cols
        if (c != target) and (not c.startswith("y_")) and
           pd.api.types.is_numeric_dtype(train[c])
    ])

    # Split features and target
    Xtr_df, ytr = train[feat_cols], train[target]
    Xva_df, yva = val[feat_cols],   val[target]
    Xte_df, yte = test[feat_cols],  test[target]

    # Standardize features
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_df)
    Xva = scaler.transform(Xva_df)
    Xte = scaler.transform(Xte_df)

    # ---------- Baseline Model Setup ----------
    # Select appropriate baseline based on target type
    if target.startswith("y_ghi"):
        base_candidates = ["ghi_qc", "ghi_sg_definitive", "ghi_qc_lag1"]
    else:
        base_candidates = ["k_ghi", "k_raw", "k_ghi_lag1", "k_raw_lag1"]

    base_src = None
    for c in base_candidates:
        if c in test.columns:
            base_src = test[c]
            break
    
    # Fallback to median if no baseline column found
    if base_src is None:
        base_src = pd.Series(np.nanmedian(ytr), index=test.index)
        print("  WARNING: Using degenerate baseline (median).")

    y_base = base_src.to_numpy()
    print(f"  Baseline RMSE: {_rmse(yte, y_base):.4f} | MAE: {mean_absolute_error(yte, y_base):.4f}")

    # Create scaled DataFrames for sequential models
    Xtr_s = pd.DataFrame(Xtr, index=Xtr_df.index, columns=feat_cols)
    Xva_s = pd.DataFrame(Xva, index=Xva_df.index, columns=feat_cols)
    Xte_s = pd.DataFrame(Xte, index=Xte_df.index, columns=feat_cols)

    # ---------- Architecture Loop ----------
    for arch_name, cfg in ARCH_CONFIGS.items():
        print(f"\n  --- Architecture: {arch_name} ---")

        arch_type = cfg["type"]

        # Handle sequential architectures
        if arch_type == "seq":
            L = SEQ_LEN
            # Build sequences for training, validation, and test
            Xtr_seq, ytr_seq = _build_seq(Xtr_s, ytr, L)
            Xva_seq, yva_seq = _build_seq(Xva_s, yva, L)
            
            # Skip if insufficient sequences
            if min(map(len, [Xtr_seq, Xva_seq])) == 0:
                print("    No valid sequences found, skipping this architecture.")
                continue
            
            n_feat = Xtr_seq.shape[2]
            model = cfg["builder"](input_dim=Xtr.shape[1], L=L, n_feat=n_feat)

            # Early stopping callback
            es = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=0
            )

            # Train the model
            model.fit(
                Xtr_seq, ytr_seq,
                validation_data=(Xva_seq, yva_seq),
                epochs=100,
                batch_size=256,
                verbose=0,
                callbacks=[es]
            )

            # Prepare test sequences with index for alignment
            Xte_seq, yte_seq, idx = build_seq_with_idx(Xte_s, yte, L)
            if len(Xte_seq) == 0:
                print("    No test sequences available, skipping.")
                continue

            # Get predictions and align baseline
            y_true = yte_seq
            y_pred = model.predict(Xte_seq, verbose=0).squeeze()
            yb     = pd.Series(y_base, index=Xte_df.index).reindex(idx).to_numpy()

        else:  # Tabular architecture (MLP)
            model = cfg["builder"](input_dim=Xtr.shape[1], L=None, n_feat=None)

            # Early stopping callback
            es = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=12,
                restore_best_weights=True,
                verbose=0
            )

            # Train the model
            model.fit(
                Xtr, ytr,
                validation_data=(Xva, yva),
                epochs=100,
                batch_size=256,
                verbose=0,
                callbacks=[es]
            )

            # Get predictions
            idx    = Xte_df.index
            y_true = yte
            y_pred = model.predict(Xte, verbose=0).squeeze()
            yb     = y_base

        # Calculate global metrics
        rmse = _rmse(y_true, y_pred)
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        skl  = skill(y_true, y_pred, yb)

        # Store metrics
        metrics_by_arch_and_h[arch_name][H] = {
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Skill": skl,
            "RMSE_base": _rmse(y_true, yb),
            "MAE_base": mean_absolute_error(y_true, yb),
        }

        print(f"    MODEL Test → RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Skill={skl:.3f}")

        # Calculate hourly skill metrics
        sk_hour = hourly_skill(
            y_true=y_true,
            y_pred=y_pred,
            y_base=yb,
            idx=idx
        )
        hourly_skill_by_arch_and_h[arch_name][H] = sk_hour

# %% [markdown]
# ## Heatmaps by Architecture and Metric

# %%
# Define heatmap parameters
all_hours = range(24)
horiz_minutes = [H*10 for H in sorted(H_LIST)]

# Metrics to visualize in heatmaps
metric_columns = {
    "skill": "skill",
    "rmse_model": "rmse_model",
    "mse_model": "mse_model",
}

# Generate heatmaps for each architecture and metric
for arch_name in ARCH_CONFIGS.keys():
    print("\n" + "#"*70)
    print(f"# Heatmaps for architecture: {arch_name}")
    print("#"*70)

    arch_hourly = hourly_skill_by_arch_and_h[arch_name]
    if not arch_hourly:
        print("  No data available for this architecture (possibly not trained for any H).")
        continue

    # Create heatmap for each metric type
    for label, col_name in metric_columns.items():
        # Initialize matrix with NaN values
        mat = pd.DataFrame(
            data=np.nan,
            index=all_hours,
            columns=horiz_minutes,
            dtype="float32",
        )

        # Fill matrix with metric values
        for H, df_sk in arch_hourly.items():
            col = H * 10
            if col not in mat.columns:
                continue
            if col_name not in df_sk.columns:
                continue
            for h in df_sk.index:
                if h in mat.index:
                    mat.loc[h, col] = df_sk.loc[h, col_name]

        # Save CSV
        csv_path = OUT_DIR / f"ground_{arch_name}_hour_vs_horizon_{label}_matrix.csv"
        mat.to_csv(csv_path)
        print(f"  Saved {arch_name} {label} matrix →", csv_path)

        # Create and save heatmap plot
        plt.figure(figsize=(7, 4.5))
        
        # Set color scale limits based on metric type
        if label == "skill":
            vmin, vmax = -0.5, 1.0
            cbar_label = "Skill vs persistence"
            title = f"{arch_name} — Skill vs persistence (hour × horizon, test)"
        else:
            vmin, vmax = 0.0, np.nanmax(mat.values)
            cbar_label = label
            title = f"{arch_name} — {label} (hour × horizon, test)"

        # Create heatmap
        im = plt.imshow(
            mat.values,
            origin="lower",
            aspect="auto",
            vmin=vmin, vmax=vmax,
        )
        plt.colorbar(im, label=cbar_label)

        # Configure axes
        plt.yticks(
            ticks=np.arange(0, 24, 2),
            labels=np.arange(0, 24, 2)
        )
        plt.xticks(
            ticks=np.arange(len(horiz_minutes)),
            labels=horiz_minutes
        )

        plt.xlabel("Forecast horizon (minutes)")
        plt.ylabel("Hour of day (UTC)")
        plt.title(title)
        plt.tight_layout()

        # Save figure
        fig_path = OUT_DIR / f"ground_{arch_name}_hour_vs_horizon_{label}_heatmap.png"
        plt.savefig(fig_path, dpi=160)
        plt.show()
        print(f"  Saved {arch_name} {label} heatmap →", fig_path)