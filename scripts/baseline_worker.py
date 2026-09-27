"""baseline_worker.py — one CPU worker for the TranAD-harness baselines (TranAD, LSTM_AD, USAD, OmniAnomaly).

Usage (launched N times by the notebook, each with a different --worker index):
    python baseline_worker.py --worker 3 --n_workers 16 --datasets SMD,MSL --seeds 0-9 --methods TranAD,LSTM_AD,USAD,OmniAnomaly
    python baseline_worker.py --list            # print the job list and exit
Each worker takes jobs[worker::n_workers], skips jobs whose result JSON exists, and writes results_gpu/baselines/<DS>/.
"""
import os, sys, json, types, argparse, time, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)

ROOT = os.path.abspath(os.path.dirname(__file__)); REPOS = f"{ROOT}/repos"; OUT_BASE = f"{ROOT}/results_gpu/baselines"
SMD_MACHINES = [f"machine-1-{i}" for i in range(1, 9)] + [f"machine-2-{i}" for i in range(1, 10)] + [f"machine-3-{i}" for i in range(1, 12)]
MSL_IDS = ["P-15","M-6","M-1","M-2","S-2","P-10","T-4","T-5","F-7","M-3","M-4","M-5","C-1","C-2","T-12","T-13","F-4","F-5","D-14","T-9","P-14","T-8","P-11","D-15","D-16","M-7","F-8"]
ENTITIES = {"SMD": SMD_MACHINES, "MSL": MSL_IDS}

# ----------------------------------------------------------------------------- data + protocol (identical to the notebooks)
def load_entity(ds, ent):
    if ds == "SMD":
        tr = np.loadtxt(f"{ROOT}/data/SMD/train_{ent}.txt", delimiter=",", dtype=np.float32)
        te = np.loadtxt(f"{ROOT}/data/SMD/test_{ent}.txt", delimiter=",", dtype=np.float32)
        lb = np.loadtxt(f"{ROOT}/data/SMD/test_label_{ent}.txt", dtype=np.int64)
    else:
        tr = np.load(f"{ROOT}/data/{ds}/{ent}_train.npy").astype(np.float32); te = np.load(f"{ROOT}/data/{ds}/{ent}_test.npy").astype(np.float32)
        lb = np.load(f"{ROOT}/data/{ds}/{ent}_test_label.npy").astype(np.int64).reshape(-1)
    return tr, te, lb

def point_metrics(score, label, tau):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    if not (0 < label.sum() < len(label)): return {k: float("nan") for k in ("auroc", "auprc", "f1_unsup_tau", "f1_best")}
    pred = (score > tau).astype(int)
    tp = int(((pred == 1) & (label == 1)).sum()); fp = int(((pred == 1) & (label == 0)).sum()); fn = int(((pred == 0) & (label == 1)).sum())
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn); p, r, _ = precision_recall_curve(label, score)
    return {"auroc": float(roc_auc_score(label, score)), "auprc": float(average_precision_score(label, score)),
            "f1_unsup_tau": 2 * prec * rec / max(1e-12, prec + rec), "f1_best": float(np.max(2 * p * r / np.maximum(p + r, 1e-12)))}

def unsup_tau(train_scores, val_frac=0.1, q=0.99):
    n = max(1, int(len(train_scores) * val_frac)); return float(np.quantile(train_scores[-n:], q))

def result_path(ds, method, ent, seed): return f"{OUT_BASE}/{ds}/{method}__{ent}__{ent}__s{seed}.json"

def save_result(ds, method, ent, seed, train_scores, test_scores, label, extra=None):
    os.makedirs(f"{OUT_BASE}/{ds}", exist_ok=True)
    n = min(len(test_scores), len(label)); test_scores, label = np.asarray(test_scores[:n], float), np.asarray(label[:n], int)
    tau = unsup_tau(train_scores)
    res = {"tag": method, "dataset": ds, "train_entity": ent, "test_entity": ent, "seed": seed, "tau0": tau,
           "n_points_scored": int(n), **point_metrics(test_scores, label, tau), **(extra or {})}
    json.dump(res, open(result_path(ds, method, ent, seed), "w"))
    np.save(f"{OUT_BASE}/{ds}/{method}__{ent}__s{seed}_scores.npy", test_scores.astype(np.float32))
    return res

# ----------------------------------------------------------------------------- TranAD harness (loaded once per worker)
_tm = None
def tranad_harness():
    global _tm
    if _tm is not None: return _tm
    if "dgl" not in sys.modules:                       # dgl is only needed by MTAD_GAT/GDN, which we do not use here
        dgl = types.ModuleType("dgl"); dgl.nn = types.ModuleType("dgl.nn"); dgl.nn.GATConv = object
        dgl.graph = dgl.add_self_loop = lambda *a, **k: None; sys.modules["dgl"] = dgl; sys.modules["dgl.nn"] = dgl.nn
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; plt.style.use = lambda *a, **k: None
    sys.path.insert(0, f"{REPOS}/TranAD"); cwd = os.getcwd(); os.chdir(f"{REPOS}/TranAD")
    sys.argv = ["main.py", "--dataset", "SMD", "--model", "TranAD"]
    import importlib; tm = importlib.import_module("main"); os.chdir(cwd)
    import src.dlutils as _dl                          # torch>=2.1 compatibility for the repo's custom Transformer layers (idempotent)
    if not hasattr(_dl.TransformerEncoderLayer, "_orig_forward"): _dl.TransformerEncoderLayer._orig_forward = _dl.TransformerEncoderLayer.forward
    if not hasattr(_dl.TransformerDecoderLayer, "_orig_forward"): _dl.TransformerDecoderLayer._orig_forward = _dl.TransformerDecoderLayer.forward
    def _enc(self, src, src_mask=None, src_key_padding_mask=None, **kw): return _dl.TransformerEncoderLayer._orig_forward(self, src, src_mask, src_key_padding_mask)
    def _dec(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kw):
        return _dl.TransformerDecoderLayer._orig_forward(self, tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
    _dl.TransformerEncoderLayer.forward = _enc; _dl.TransformerDecoderLayer.forward = _dec
    _tm = tm; return tm

def tranad_minmax(tr, te):                             # preprocess.py normalize3
    mn, mx = tr.min(0), tr.max(0); d = np.where(mx - mn == 0, 1.0, mx - mn)
    return ((tr - mn) / d).astype(np.float32), ((te - mn) / d).astype(np.float32)

def run_tranad_model(ds, ent, method, seed, epochs=5):
    tm = tranad_harness()
    torch.manual_seed(seed); np.random.seed(seed)
    tr, te, lb = load_entity(ds, ent); tr, te = tranad_minmax(tr, te); dims = tr.shape[1]
    cwd = os.getcwd(); os.chdir(f"{REPOS}/TranAD")                      # load_model may look for checkpoints relative to the repo
    try:
        model, optimizer, scheduler, _, _ = tm.load_model(method, dims)
    finally:
        os.chdir(cwd)
    dt = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32
    trainD = torch.tensor(tr, dtype=dt); testD = torch.tensor(te, dtype=dt); trainO, testO = trainD, testD
    tm.args.model = method                                                # convert_to_windows keys on the global CLI arg
    if model.name in ["Attention", "DAGMM", "USAD", "MSCRED", "CAE_M", "GDN", "MTAD_GAT", "MAD_GAN"] or "TranAD" in model.name:
        trainD, testD = tm.convert_to_windows(trainD, model), tm.convert_to_windows(testD, model)
    for e in range(epochs):
        tm.backprop(e, model, trainD, trainO, optimizer, scheduler, training=True)
    tr_loss, _ = tm.backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    te_loss, _ = tm.backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    tr_s = np.asarray(tr_loss).reshape(len(tr), -1).mean(1); te_s = np.asarray(te_loss).reshape(len(te), -1).mean(1)
    return save_result(ds, method, ent, seed, tr_s, te_s, lb, extra={"epochs": epochs})

# ----------------------------------------------------------------------------- job list + main
def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part: a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else: out.append(int(part))
    return out

def job_list(datasets, seeds, methods):
    return [(seed, ds, ent, m) for seed in seeds for ds in datasets for ent in ENTITIES[ds] for m in methods]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--datasets", default="SMD,MSL"); ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--methods", default="TranAD,LSTM_AD,USAD,OmniAnomaly"); ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    jobs = job_list(a.datasets.split(","), parse_seeds(a.seeds), a.methods.split(","))
    mine = jobs[a.worker::a.n_workers]
    if a.list:
        print(f"{len(jobs)} jobs total, {len(mine)} for worker {a.worker}/{a.n_workers}"); print(mine[:5]); sys.exit(0)
    print(f"[worker {a.worker}/{a.n_workers}] {len(mine)} jobs", flush=True)
    for seed, ds, ent, m in mine:
        if os.path.exists(result_path(ds, m, ent, seed)): continue
        t0 = time.time()
        try:
            r = run_tranad_model(ds, ent, m, seed, epochs=a.epochs)
            print(f"[w{a.worker}] {time.strftime('%H:%M:%S')} {ds} {ent:12s} {m:12s} s{seed}  AUROC={r['auroc']:.3f} AUPRC={r['auprc']:.3f} "
                  f"F1@tau={r['f1_unsup_tau']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[w{a.worker}] FAILED {ds} {ent} {m} s{seed}: {e!r}", flush=True); traceback.print_exc()
    print(f"[worker {a.worker}] DONE", flush=True)
