# Masked-View Disagreement (MVD) for Unsupervised Anomaly Detection in Drifting Multi-Sensor Streams

Code, configurations, and per-entity results for the manuscript submitted to *Knowledge-Based Systems*.

## What is here
- `notebooks/mvd_experiments.ipynb` — MVD and its ablation variants (offline training, label-free calibration, gated online adaptation); campaigns for SMD (28 machines), MSL (27 entities), SWaT, cross-machine shift, and sensitivity.
- `notebooks/baselines_official.ipynb`, `notebooks/baselines_parallel.ipynb`, `scripts/baseline_worker.py` — adapters that run the official implementations of LSTM-AD, OmniAnomaly, USAD, TranAD, MTAD-GAT, THOC and M2N2 under the same protocol and export point-wise scores.
- `results/` — per-entity, seed-averaged metrics and full tables used in the paper; raw per-run JSON available from the author on request.
- `figures/` — all figures.

## Protocol (identical for every method)
Point-wise metrics without point adjustment; threshold = 99th percentile of the method's own scores on the last 10% of its training data; every entity; 10 seeds. See Sec. IV-B of the paper.

## Data
SMD is downloaded automatically (NetManAIOps/OmniAnomaly). MSL: NASA telemanom archive (`data.zip` + `labeled_anomalies.csv`). SWaT: request from iTrust, SUTD (not redistributable). See `data/README.md`.

## Reproduce
1. `pip install -r requirements.txt`
2. Open `notebooks/mvd_experiments.ipynb`, run the setup and data-preparation cells once, then the campaign cells.
3. Baselines: `notebooks/baselines_official.ipynb` (GPU models) and `notebooks/baselines_parallel.ipynb` (CPU harness, N workers).

Baseline repositories are fetched by the notebooks at first run: imperial-qore/TranAD, ML4ITS/mtad-gat-pytorch, carrtesy/M2N2, thuml/Time-Series-Library, thuml/Anomaly-Transformer, DAMO-DI-ML/KDD2023-DCdetector.

## Citation
Seunghwan Song, "Masked-view disagreement for unsupervised anomaly detection in drifting multi-sensor streams," submitted to Knowledge-Based Systems, 2026.

## License
MIT. Third-party baseline code retains its own licenses.
