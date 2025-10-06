#!/usr/bin/env python
# coding: utf-8

# # Ground HPO with Optuna (MLP, LSTM, BiLSTM, CNN-LSTM, Transformer)

# ## Libraries

# In[ ]:


# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # 0=all,1=info,2=warning,3=error
# # os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"    # opcional: desactiva oneDNN por reproducibilidad exacta (puede bajar performance)

# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# for g in gpus:
#     tf.config.experimental.set_memory_growth(g, True)
# print("GPUs visibles:", gpus)


# In[ ]:


# import tensorflow as tf, time
# with tf.device('/GPU:0'):
#     a = tf.random.normal([4000, 4000])
#     b = tf.random.normal([4000, 4000])
#     tf.linalg.matmul(a, b)  # warmup
# t0 = time.time()
# for _ in range(5):
#     with tf.device('/GPU:0'):
#         c = tf.linalg.matmul(a, b)
# _ = c.numpy()
# print("Tiempo 5 matmul GPU:", time.time() - t0, "s")


# In[2]:


import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers, backend as K

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import TFKerasPruningCallback
from optuna.storages import JournalStorage
from optuna.storages import JournalFileStorage, JournalFileOpenLock


# ## Config

# In[3]:


SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

DATA_DIR = Path("../data_processed")
OUT_DIR  = Path("../models"); OUT_DIR.mkdir(parents=True, exist_ok=True)
STUDY_DIR= Path("../optuna_studies"); STUDY_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR  = (OUT_DIR / "optuna_artifacts").resolve(); ART_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PQ = DATA_DIR / "ground_train_h6.parquet"
VAL_PQ   = DATA_DIR / "ground_val_h6.parquet"
TEST_PQ  = DATA_DIR / "ground_test_h6.parquet"
TARGET   = "y_ghi_h6" 

print("Studies dir:", STUDY_DIR.resolve())
print("Artifacts dir:", ART_DIR)


# ### Data loading and preprocessing

# In[4]:


train = pd.read_parquet(TRAIN_PQ).sort_index()
val   = pd.read_parquet(VAL_PQ).sort_index()
test  = pd.read_parquet(TEST_PQ).sort_index()
assert TARGET in train and TARGET in val and TARGET in test, f"{TARGET} missing!"

feat_cols = sorted(list(set(train.columns) & set(val.columns) & set(test.columns) - {TARGET}))
feat_cols = [c for c in feat_cols if pd.api.types.is_numeric_dtype(train[c])]

Xtr_df, ytr = train[feat_cols], train[TARGET]
Xva_df, yva = val[feat_cols],   val[TARGET]
Xte_df, yte = test[feat_cols],  test[TARGET]

scaler = StandardScaler()
Xtr = scaler.fit_transform(Xtr_df)
Xva = scaler.transform(Xva_df)
Xte = scaler.transform(Xte_df)


# ## Helpers

# In[5]:


def _rmse(a,b):
    return float(np.sqrt(mean_squared_error(a,b)))

def skill(y_true, y_pred, y_base):
    return 1.0 - (_rmse(y_true, y_pred) / _rmse(y_true, y_base))

def _build_seq(X_df, y_ser, L):
    """Secuencias sin índice (rápido para objetivos Optuna)."""
    Xv, yv = X_df.values, y_ser.values
    xs, ys = [], []
    for i in range(L-1, len(X_df)):
        block = Xv[i-L+1:i+1]
        if np.isnan(block).any():
            continue
        xs.append(block); ys.append(yv[i])
    return np.asarray(xs, dtype="float32"), np.asarray(ys, dtype="float32")

def build_seq_with_idx(X_df, y_ser, L):
    """Secuencias con índice (para evaluación y plots)."""
    Xv, yv = X_df.values, y_ser.values
    xs, ys, idx = [], [], []
    for i in range(L-1, len(X_df)):
        block = Xv[i-L+1:i+1]
        if np.isnan(block).any():
            continue
        xs.append(block); ys.append(yv[i]); idx.append(X_df.index[i])
    return (np.asarray(xs, dtype="float32"),
            np.asarray(ys, dtype="float32"),
            pd.DatetimeIndex(idx))

def prepare_journal_storage(study_name: str) -> JournalStorage:
    log_path   = STUDY_DIR / f"{study_name}.log"
    lock_path  = STUDY_DIR / f"{study_name}.lock"
    try: lock_path.unlink()
    except FileNotFoundError: pass
    file_storage = JournalFileStorage(str(log_path), lock_obj=JournalFileOpenLock(str(lock_path)))
    return JournalStorage(file_storage)

# def _safe_load_best(study, rebuild_fn=None):
#     """Carga robusta del mejor modelo guardado por el estudio."""
#     p = Path(study.best_trial.user_attrs["model_path"])
#     if not p.exists():
#         # fallback: buscar por nombre
#         hits = list(ART_DIR.rglob(p.name))
#         if hits:
#             p = hits[0]
#         elif rebuild_fn is not None:
#             model = rebuild_fn(study.best_trial.params)
#             p = ART_DIR / "recover.keras"
#             model.save(p)
#         else:
#             raise FileNotFoundError(f"Checkpoint not found: {p}")
#     return tf.keras.models.load_model(p), p


# ### Models

# In[6]:


def build_mlp(input_dim, n1=128, n2=64, do1=0.2, do2=0.1, act="relu", l2w=0.0):
    return tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(n1, activation=act, kernel_regularizer=regularizers.l2(l2w)),
        layers.Dropout(do1),
        layers.Dense(n2, activation=act, kernel_regularizer=regularizers.l2(l2w)),
        layers.Dropout(do2),
        layers.Dense(1, dtype="float32"),
    ])

def build_lstm(L, n_feat, units=64, do=0.0):
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.LSTM(units, dropout=do)(inp)
    out = layers.Dense(1, dtype="float32")(x)
    return tf.keras.Model(inp, out)

def build_bilstm(L, n_feat, units=64, do=0.0):
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.Bidirectional(layers.LSTM(units, dropout=do))(inp)
    out = layers.Dense(1, dtype="float32")(x)
    return tf.keras.Model(inp, out)

def build_cnnlstm(L, n_feat, filt=32, ksz=3, pool=1, lstm_units=64, do=0.0):
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.Conv1D(filt, kernel_size=ksz, padding="causal", activation="relu")(inp)
    x   = (layers.MaxPooling1D(pool_size=pool)(x) if pool>1 else layers.Identity()(x))
    x   = layers.LSTM(lstm_units, dropout=do)(x)
    out = layers.Dense(1, dtype="float32")(x)
    return tf.keras.Model(inp, out)

def build_transformer(L, n_feat, d_model=64, heads=4, ff_dim=128, att_do=0.1, do=0.0):
    inp = layers.Input(shape=(L, n_feat))
    x   = layers.Dense(d_model)(inp)
    x2  = layers.MultiHeadAttention(num_heads=heads, key_dim=d_model//heads, dropout=att_do)(x, x)
    x   = layers.Add()([x, x2]); x = layers.LayerNormalization()(x)
    ff  = layers.Dense(ff_dim, activation="relu")(x); ff = layers.Dense(d_model)(ff)
    x   = layers.Add()([x, ff]); x = layers.LayerNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dropout(do)(x)
    out = layers.Dense(1, dtype="float32")(x)
    return tf.keras.Model(inp, out)


# ### Saving

# In[7]:


def _safe_load_best(study):
    """
    Carga robusta: si el best_trial no tiene los nuevos user_attrs (arch, seq_len_used, n_feat),
    los infiere desde study_name / ruta del checkpoint / params del trial.
    Reconstruye la arquitectura y carga PESOS (.weights.h5 o .keras/.h5 legacy).
    """
    ua = dict(study.best_trial.user_attrs) if study.best_trial.user_attrs else {}

    # 1) Localiza el checkpoint
    wpath = None
    if "model_path" in ua:
        p = Path(ua["model_path"])
        if p.exists():
            wpath = p
        else:
            hits = list(ART_DIR.rglob(p.name))
            if hits:
                wpath = hits[0]
    if wpath is None:
        # Fallback: deduce por nombre de estudio
        # Busca archivos 'best.weights.h5' o 'best.keras' en ART_DIR que coincidan con el estudio
        patt = []
        name = (study.study_name or "").lower()
        if "mlp" in name: patt.append("A_mlp_*")
        if "lstm" in name and "bilstm" not in name: patt.append("B_lstm_*")
        if "bilstm" in name: patt.append("B_bilstm_*")
        if "cnn" in name: patt.append("B_cnnlstm_*")
        if "transformer" in name: patt.append("B_transformer_*")
        candidates = []
        for pat in (patt or ["*"]):
            candidates += list(ART_DIR.glob(f"{pat}/best.weights.h5"))
            candidates += list(ART_DIR.glob(f"{pat}/best.keras"))
            candidates += list(ART_DIR.glob(f"{pat}/best.h5"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found for best trial and no user_attrs['model_path'].")
        # Toma el más reciente
        wpath = max(candidates, key=lambda p: p.stat().st_mtime)

    # 2) Deducir 'arch'
    arch = ua.get("arch")
    base = wpath.parent.name.lower()
    sname = (study.study_name or "").lower()
    if arch is None:
        if "mlp" in sname or base.startswith("a_mlp"):
            arch = "mlp"
        elif "bilstm" in sname or "b_bilstm" in base:
            arch = "bilstm"
        elif ("lstm" in sname and "bilstm" not in sname) or "b_lstm" in base:
            arch = "lstm"
        elif "cnn" in sname or "cnn" in base:
            arch = "cnn-lstm"
        elif "transformer" in sname or "transformer" in base:
            arch = "transformer"
        else:
            raise KeyError("Cannot infer 'arch' from study; please re-run trials or set user_attrs.")

    # 3) Deducir L y n_feat para secuenciales
    params = study.best_trial.params
    L = ua.get("seq_len_used") or params.get("seq_len")
    n_feat = ua.get("n_feat")
    if arch != "mlp":
        if L is None:
            # default razonable si faltara
            L = 12
        if n_feat is None:
            # usar el contexto global ya cargado
            n_feat = int(Xtr_s.shape[1])

    # 4) Reconstruye modelo y carga pesos / modelo
    # (acepta tanto weights-only como modelo completo legacy)
    if wpath.suffix in {".keras", ".h5"} and "weights" not in wpath.name:
        # Legacy: modelo completo; cargar con safe_mode desactivado SOLO si confías en el archivo
        import keras
        try:
            keras.config.enable_unsafe_deserialization()
        except Exception:
            pass
        model = tf.keras.models.load_model(wpath, compile=False, safe_mode=False)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return model, wpath

    # Weights-only (recomendado)
    if arch == "mlp":
        model = build_mlp(
            input_dim=Xtr.shape[1],
            n1=params.get("n1",128),
            n2=params.get("n2",64),
            do1=params.get("do1",0.0),
            do2=params.get("do2",0.0),
            act=params.get("act","relu"),
            l2w=params.get("l2",0.0),
        )
    elif arch == "lstm":
        model = build_lstm(int(L), int(n_feat),
                           units=params.get("units",64),
                           do=params.get("dropout",0.0))
    elif arch == "bilstm":
        model = build_bilstm(int(L), int(n_feat),
                             units=params.get("units",64),
                             do=params.get("dropout",0.0))
    elif arch == "cnn-lstm":
        model = build_cnnlstm(int(L), int(n_feat),
                              filt=params.get("filters",32),
                              ksz=params.get("kernel_size",3),
                              pool=params.get("pool",1),
                              lstm_units=params.get("lstm_units",64),
                              do=params.get("dropout",0.0))
    elif arch == "transformer":
        model = build_transformer(int(L), int(n_feat),
                                  d_model=params.get("d_model",64),
                                  heads=params.get("heads",4),
                                  ff_dim=params.get("ff_dim",128),
                                  att_do=params.get("att_dropout",0.1),
                                  do=params.get("dropout",0.0))
    else:
        raise ValueError(f"Unknown arch: {arch}")

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.load_weights(str(wpath))
    return model, wpath


# ## Baseline

# In[8]:


base_src = None

# for c in ["k_ghi","k_raw","k_ghi_lag1","k_raw_lag1"]:
#     if c in test.columns: base_src = test[c]; break
# if base_src is None:
#     base_src = pd.Series(np.nanmedian(ytr), index=test.index)

for c in ["ghi_qc","ghi_sg_definitive","ghi_qc_lag1"]:
    if c in test.columns: base_src = test[c]; break
if base_src is None:
    base_src = pd.Series(np.nanmedian(ytr), index=test.index)

y_base = base_src.to_numpy()
print(f"Baseline → RMSE: {_rmse(yte, y_base):.4f} | MAE: {mean_absolute_error(yte, y_base):.4f}")


# ## Track A - MLP

# In[9]:


def objective_mlp(trial: optuna.Trial) -> float:
    K.clear_session()
    n1  = trial.suggest_int("n1", 64, 512, step=64)
    n2  = trial.suggest_int("n2", 32, max(64, n1//2), step=32)
    do1 = trial.suggest_float("do1", 0.0, 0.5)
    do2 = trial.suggest_float("do2", 0.0, 0.5)
    lr  = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    l2w = trial.suggest_float("l2", 1e-8, 1e-3, log=True)
    act = trial.suggest_categorical("act", ["relu","selu","gelu"])
    bs  = trial.suggest_categorical("batch", [64, 128, 256, 512])
    eps = trial.suggest_int("epochs", 40, 150)

    model = build_mlp(Xtr.shape[1], n1=n1, n2=n2, do1=do1, do2=do2, act=act, l2w=l2w)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae"])

    tmp_dir  = ART_DIR / f"A_mlp_t{trial.number:04d}"; tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss",
                                  save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]

    model.fit(Xtr, ytr, validation_data=(Xva, yva),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva, verbose=0).squeeze()
    val_rmse = _rmse(yva, yhat)

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("arch", "mlp")
    trial.set_user_attr("input_dim", Xtr.shape[1])
    return val_rmse


# In[10]:


storageA = prepare_journal_storage("ground_trackA_mlp")
studyA = optuna.create_study(direction="minimize",
                             sampler=TPESampler(seed=SEED),
                             pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5),
                             study_name="ground_trackA_mlp",
                             storage=storageA, load_if_exists=True)
print("Running Study A (MLP)…")
studyA.optimize(objective_mlp, n_trials=40, show_progress_bar=True)

best_mlp, bestA_path = _safe_load_best(studyA)
yhatA = best_mlp.predict(Xte, verbose=0).squeeze()
print("Best MLP params:", studyA.best_trial.params)
print(f"MLP test → RMSE: {_rmse(yte, yhatA):.4f} | MAE: {mean_absolute_error(yte, yhatA):.4f} | R2: {r2_score(yte, yhatA):.4f} | Skill: {skill(yte, yhatA, y_base):.3f}")


# ## Track B - Sequentials

# ### Mods

# In[ ]:


Xtr_s = pd.DataFrame(Xtr, index=Xtr_df.index, columns=feat_cols)
Xva_s = pd.DataFrame(Xva, index=Xva_df.index, columns=feat_cols)
Xte_s = pd.DataFrame(Xte, index=Xte_df.index, columns=feat_cols)


# ### LSTM

# In[ ]:


def objective_lstm(trial: optuna.Trial) -> float:
    K.clear_session()
    L   = trial.suggest_categorical("seq_len", [6, 12, 18, 24])
    u   = trial.suggest_int("units", 32, 128, step=32)
    do  = trial.suggest_float("dropout", 0.0, 0.4)
    lr  = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs  = trial.suggest_categorical("batch", [64, 128, 256])
    eps = trial.suggest_int("epochs", 40, 120)

    Xtr_seq, ytr_seq = _build_seq(Xtr_s, ytr, L)
    Xva_seq, yva_seq = _build_seq(Xva_s, yva, L)
    if min(map(len,[Xtr_seq, Xva_seq])) == 0:
        raise optuna.TrialPruned()

    model = build_lstm(L, Xtr_seq.shape[2], units=u, do=do)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])

    tmp_dir  = ART_DIR / f"B_lstm_t{trial.number:04d}"; tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss",
                                  save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    val_rmse = _rmse(yva_seq, yhat)

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("arch", "lstm")
    trial.set_user_attr("seq_len_used", L)
    trial.set_user_attr("n_feat", Xtr_s.shape[1])
    return val_rmse


# In[ ]:


storageB1 = prepare_journal_storage("ground_trackB_lstm")
studyB1 = optuna.create_study(direction="minimize",
                              sampler=TPESampler(seed=SEED),
                              pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5),
                              study_name="ground_trackB_lstm",
                              storage=storageB1, load_if_exists=True)
print("Running Study B1 (LSTM)…")
studyB1.optimize(objective_lstm, n_trials=40, show_progress_bar=True)

best_lstm, _ = _safe_load_best(studyB1)
bestL1 = studyB1.best_trial.user_attrs["seq_len_used"]
Xte_seq, yte_seq, idx_LSTM = build_seq_with_idx(Xte_s, yte, bestL1)
yhatB1 = best_lstm.predict(Xte_seq, verbose=0).squeeze()
y_base_LSTM = pd.Series(y_base, index=Xte_df.index).reindex(idx_LSTM).to_numpy()
print("Best LSTM params:", studyB1.best_trial.params | {"seq_len": bestL1})
print(f"LSTM test → RMSE: {_rmse(yte_seq, yhatB1):.4f} | MAE: {mean_absolute_error(yte_seq, yhatB1):.4f} | R2: {r2_score(yte_seq, yhatB1):.4f} | Skill: {skill(yte_seq, yhatB1, y_base_LSTM):.3f}")


# ### BiLSTM

# In[ ]:


def objective_bilstm(trial: optuna.Trial) -> float:
    K.clear_session()
    L   = trial.suggest_categorical("seq_len", [6, 12, 18, 24])
    u   = trial.suggest_int("units", 32, 128, step=32)
    do  = trial.suggest_float("dropout", 0.0, 0.4)
    lr  = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs  = trial.suggest_categorical("batch", [64, 128, 256])
    eps = trial.suggest_int("epochs", 40, 120)

    Xtr_seq, ytr_seq = _build_seq(Xtr_s, ytr, L)
    Xva_seq, yva_seq = _build_seq(Xva_s, yva, L)
    if min(map(len,[Xtr_seq, Xva_seq])) == 0:
        raise optuna.TrialPruned()

    model = build_bilstm(L, Xtr_seq.shape[2], units=u, do=do)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])

    tmp_dir  = ART_DIR / f"B_bilstm_t{trial.number:04d}"; tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss",
                                  save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    val_rmse = _rmse(yva_seq, yhat)

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("arch", "bilstm")
    trial.set_user_attr("seq_len_used", L)
    trial.set_user_attr("n_feat", Xtr_s.shape[1])
    return val_rmse


# In[ ]:


storageB2 = prepare_journal_storage("ground_trackB_bilstm")
studyB2 = optuna.create_study(direction="minimize",
                              sampler=TPESampler(seed=SEED),
                              pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5),
                              study_name="ground_trackB_bilstm",
                              storage=storageB2, load_if_exists=True)
print("Running Study B2 (BiLSTM)…")
studyB2.optimize(objective_bilstm, n_trials=35, show_progress_bar=True)

best_bi, _ = _safe_load_best(studyB2)
bestL2 = studyB2.best_trial.user_attrs["seq_len_used"]
Xte_seq, yte_seq, idx_BI = build_seq_with_idx(Xte_s, yte, bestL2)
yhatB2 = best_bi.predict(Xte_seq, verbose=0).squeeze()
y_base_BI = pd.Series(y_base, index=Xte_df.index).reindex(idx_BI).to_numpy()
print("Best BiLSTM params:", studyB2.best_trial.params | {"seq_len": bestL2})
print(f"BiLSTM test → RMSE: {_rmse(yte_seq, yhatB2):.4f} | MAE: {mean_absolute_error(yte_seq, yhatB2):.4f} | R2: {r2_score(yte_seq, yhatB2):.4f} | Skill: {skill(yte_seq, yhatB2, y_base_BI):.3f}")


# ### CNN-LSTM

# In[ ]:


def objective_cnnlstm(trial: optuna.Trial) -> float:
    K.clear_session()
    L     = trial.suggest_categorical("seq_len", [6, 12, 18, 24])
    filt  = trial.suggest_int("filters", 16, 64, step=16)
    ksz   = trial.suggest_categorical("kernel_size", [2,3,5])
    pool  = trial.suggest_categorical("pool", [1,2])
    u     = trial.suggest_int("lstm_units", 32, 128, step=32)
    do    = trial.suggest_float("dropout", 0.0, 0.4)
    lr    = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs    = trial.suggest_categorical("batch", [64, 128, 256])
    eps   = trial.suggest_int("epochs", 40, 120)

    Xtr_seq, ytr_seq = _build_seq(Xtr_s, ytr, L)
    Xva_seq, yva_seq = _build_seq(Xva_s, yva, L)
    if min(map(len,[Xtr_seq, Xva_seq])) == 0:
        raise optuna.TrialPruned()

    model = build_cnnlstm(L, Xtr_seq.shape[2], filt=filt, ksz=ksz, pool=pool, lstm_units=u, do=do)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])

    tmp_dir  = ART_DIR / f"B_cnnlstm_t{trial.number:04d}"; tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss",
                                  save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    val_rmse = _rmse(yva_seq, yhat)

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("arch", "cnn-lstm")
    trial.set_user_attr("seq_len_used", L)
    trial.set_user_attr("n_feat", Xtr_s.shape[1])
    return val_rmse


# In[ ]:


storageB3 = prepare_journal_storage("ground_trackB_cnnlstm")
studyB3 = optuna.create_study(direction="minimize",
                              sampler=TPESampler(seed=SEED),
                              pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5),
                              study_name="ground_trackB_cnnlstm",
                              storage=storageB3, load_if_exists=True)
print("Running Study B3 (CNN-LSTM)…")
studyB3.optimize(objective_cnnlstm, n_trials=35, show_progress_bar=True)

best_cnn, _ = _safe_load_best(studyB3)
bestL3 = studyB3.best_trial.user_attrs["seq_len_used"]
Xte_seq, yte_seq, idx_CNN = build_seq_with_idx(Xte_s, yte, bestL3)
yhatB3 = best_cnn.predict(Xte_seq, verbose=0).squeeze()
y_base_CNN = pd.Series(y_base, index=Xte_df.index).reindex(idx_CNN).to_numpy()
print("Best CNN-LSTM params:", studyB3.best_trial.params | {"seq_len": bestL3})
print(f"CNN-LSTM test → RMSE: {_rmse(yte_seq, yhatB3):.4f} | MAE: {mean_absolute_error(yte_seq, yhatB3):.4f} | R2: {r2_score(yte_seq, yhatB3):.4f} | Skill: {skill(yte_seq, yhatB3, y_base_CNN):.3f}")


# ### Transformer

# In[ ]:


def objective_transformer(trial: optuna.Trial) -> float:
    K.clear_session()
    L       = trial.suggest_categorical("seq_len", [6, 12, 18, 24])
    d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
    heads   = trial.suggest_categorical("heads", [2, 4, 8])
    if d_model % heads != 0:
        raise optuna.TrialPruned()
    ff_dim  = trial.suggest_categorical("ff_dim", [64, 96, 128, 192])
    att_do  = trial.suggest_float("att_dropout", 0.0, 0.3)
    do      = trial.suggest_float("dropout", 0.0, 0.4)
    lr      = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    bs      = trial.suggest_categorical("batch", [64, 128, 256])
    eps     = trial.suggest_int("epochs", 40, 120)

    Xtr_seq, ytr_seq = _build_seq(Xtr_s, ytr, L)
    Xva_seq, yva_seq = _build_seq(Xva_s, yva, L)
    if min(map(len,[Xtr_seq, Xva_seq])) == 0:
        raise optuna.TrialPruned()

    model = build_transformer(L, Xtr_seq.shape[2], d_model=d_model, heads=heads,
                              ff_dim=ff_dim, att_do=att_do, do=do)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])

    tmp_dir  = ART_DIR / f"B_transformer_t{trial.number:04d}"; tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = (tmp_dir / "best.weights.h5").resolve()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
        callbacks.ModelCheckpoint(filepath=str(tmp_path), monitor="val_loss",
                                  save_best_only=True, save_weights_only=True),
        TFKerasPruningCallback(trial, "val_loss"),
    ]
    model.fit(Xtr_seq, ytr_seq, validation_data=(Xva_seq, yva_seq),
              epochs=eps, batch_size=bs, verbose=0, callbacks=cbs)

    yhat = model.predict(Xva_seq, verbose=0).squeeze()
    val_rmse = _rmse(yva_seq, yhat)

    trial.set_user_attr("model_path", str(tmp_path))
    trial.set_user_attr("arch", "transformer")
    trial.set_user_attr("seq_len_used", L)
    trial.set_user_attr("n_feat", Xtr_s.shape[1])
    return val_rmse


# In[ ]:


storageB4 = prepare_journal_storage("ground_trackB_transformer")
studyB4 = optuna.create_study(direction="minimize",
                              sampler=TPESampler(seed=SEED),
                              pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=5),
                              study_name="ground_trackB_transformer",
                              storage=storageB4, load_if_exists=True)
print("Running Study B4 (Transformer)…")
studyB4.optimize(objective_transformer, n_trials=40, show_progress_bar=True)

best_tr, _ = _safe_load_best(studyB4)
bestL4 = studyB4.best_trial.user_attrs["seq_len_used"]
Xte_seq, yte_seq, idx_TR = build_seq_with_idx(Xte_s, yte, bestL4)
yhatB4 = best_tr.predict(Xte_seq, verbose=0).squeeze()
y_base_TR = pd.Series(y_base, index=Xte_df.index).reindex(idx_TR).to_numpy()
print("Best Transformer params:", studyB4.best_trial.params | {"seq_len": bestL4})
print(f"Transformer test → RMSE: {_rmse(yte_seq, yhatB4):.4f} | MAE: {mean_absolute_error(yte_seq, yhatB4):.4f} | R2: {r2_score(yte_seq, yhatB4):.4f} | Skill: {skill(yte_seq, yhatB4, y_base_TR):.3f}")


# ## Best

# In[ ]:


best_params = {
    "MLP":        studyA.best_trial.params,
    "LSTM":       studyB1.best_trial.params | {"seq_len": studyB1.best_trial.user_attrs["seq_len_used"]},
    "BiLSTM":     studyB2.best_trial.params | {"seq_len": studyB2.best_trial.user_attrs["seq_len_used"]},
    "CNN_LSTM":   studyB3.best_trial.params | {"seq_len": studyB3.best_trial.user_attrs["seq_len_used"]},
    "Transformer":studyB4.best_trial.params | {"seq_len": studyB4.best_trial.user_attrs["seq_len_used"]},
}
(out := OUT_DIR / "best_hpo_params_all.json")
with open(out, "w") as f:
    json.dump(best_params, f, indent=2)
print("Saved params →", out)


# ## Visualization

# In[ ]:


models_info = {
    "MLP": {
        "type": "tabular",
        "model": best_mlp,
        "idx": Xte_df.index,
        "y_base": y_base
    },
    "LSTM": {
        "type": "seq",
        "model": best_lstm,
        "L": bestL1,
        "idx": idx_LSTM,
        "y_base": y_base_LSTM
    },
    "BiLSTM": {
        "type": "seq",
        "model": best_bi,
        "L": bestL2,
        "idx": idx_BI,
        "y_base": y_base_BI
    },
    "CNN-LSTM": {
        "type": "seq",
        "model": best_cnn,
        "L": bestL3,
        "idx": idx_CNN,
        "y_base": y_base_CNN
    },
    "Transformer": {
        "type": "seq",
        "model": best_tr,
        "L": bestL4,
        "idx": idx_TR,
        "y_base": y_base_TR
    }
}


# In[ ]:


rows = []
OUT_FIG = Path("../reports/figures")
for name, cfg in models_info.items():
    print(f"\n=== {name} ===")
    if cfg["type"] == "tabular":
        y_true = yte
        y_pred = cfg["model"].predict(Xte, verbose=0).squeeze()
        idx    = cfg["idx"]
        yb     = cfg["y_base"]
    else:
        L = int(cfg["L"])
        X_seq, y_seq, idx = build_seq_with_idx(Xte_s, yte, L)
        if len(X_seq) == 0:
            print("No hay secuencias válidas (NaNs). Se omite.")
            continue
        y_true = y_seq
        y_pred = cfg["model"].predict(X_seq, verbose=0).squeeze()
        yb     = pd.Series(y_base, index=Xte_df.index).reindex(idx).to_numpy()

    rmse = _rmse(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    skl  = skill(y_true, y_pred, yb)
    print(f"RMSE={rmse:.4f} | MAE={mae:.4f} | R2={r2:.4f} | Skill={skl:.3f}")

    rows.append({"model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "Skill": skl})

    # 1) Time series (clip)
    N = min(400, len(y_true))
    plt.figure(figsize=(12, 3.6))
    plt.plot(idx[:N], y_true[:N], label="truth", lw=1.4)
    plt.plot(idx[:N], y_pred[:N], label=name, lw=1.1)
    plt.plot(idx[:N], yb[:N], label="baseline", lw=1.0, alpha=0.7)
    plt.title(f"Test — Truth vs {name} vs Baseline ({TARGET})")
    plt.ylabel("GHI (W/m²)" if TARGET.startswith("y_ghi") else "k-index")
    plt.xlabel("Time"); plt.grid(True, ls="--", alpha=0.3); plt.legend()
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(OUT_FIG / f"{name}_ts_test.png", dpi=140)
    plt.show()

    # 2) Scatter
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(4.8, 4.8))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=1.0)
    plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.title(f"{name} — Actual vs Predicted\nRMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")
    plt.grid(True, ls="--", alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_FIG / f"{name}_scatter.png", dpi=140)
    plt.show()

    # 3) Residuals histogram
    resid = y_pred - y_true
    plt.figure(figsize=(6, 3.2))
    plt.hist(resid, bins=50, alpha=0.85)
    plt.axvline(0, color='r', ls='--', lw=1)
    plt.title(f"{name} — Residuals (mean={np.mean(resid):.3f})")
    plt.xlabel("Residual"); plt.ylabel("Frequency")
    plt.grid(True, ls="--", alpha=0.3); plt.tight_layout()
    plt.savefig(OUT_FIG / f"{name}_residuals.png", dpi=140)
    plt.show()


# In[ ]:


results_df = pd.DataFrame(rows).sort_values("RMSE")
print("\n=== Test Summary ===")
print(results_df.round(4))
results_df.to_csv(OUT_DIR / "hpo_models_test_summary.csv", index=False)

