[Under Review] Efficient Recommendation Unlearning via Task Vector Arithmetic in Shared Space

---

The code supports a simple, reproducible workflow:
1) Data preprocessing → 2) Train the Original Model → 3) Train the Retraining Model → 4) Free experiments.


## Requirements
- Python 3.9.0
- PyTorch 2.5.1+cu121 (CUDA 12.1)
- numpy 1.26.3, pandas 2.3.1, scipy 1.13.1, scikit-learn 1.6.1


## Data
Place your datasets under `Data/`. Preprocessing will produce processed files under `Data/Process/<DATASET>/<ATTACK>/`.

## Data Preprocessing

1) Run the preprocessing script before training. Adjust arguments as needed for your datasets.
```bash
python _data_process.py
```

## Models Directory Structure
All model scripts are organized under `models/`:
- `models/original`: Original training scripts
- `models/retrain`: Retraining (ground-truth) scripts
- `models/sisa`: SISA unlearning scripts
- `models/receraser`: RecEraser scripts
- `models/scif`: SCIF-related scripts
- `models/ifru`: IFRU-related scripts
- `models/COVA`: COVA scripts

Legacy top-level scripts are kept as lightweight shims (symlinks) for backward compatibility. Please prefer invoking scripts from `models/` going forward.

## Training
Below we show LightGCN and MF examples on Yelp. Replace `yelp` with `gowalla` or `amazon` variants by choosing the corresponding scripts in this folder.

### Original Model
- LightGCN (Yelp):
```bash
python models/original/original_lightgcn_yelp_bpr.py --attack 0.01 --gcn_layers 1
```

- MF (Yelp):
```bash
python models/original/original_mf_yelp_bpr.py --attack 0.01
```

### Retraining Model
- LightGCN (Yelp):
```bash
python models/retrain/retrain_lightgcn_yelp_bpr.py --attack 0.01 --gcn_layers 1
```

- MF (Yelp):
```bash
python models/retrain/retrain_mf_yelp_bpr.py --attack 0.01
```

### COVA
- LightGCN (Yelp):
```bash
python models/COVA/COVA_lightgcn_yelp.py --attack 0.01 --dataset Yelp --gcn_layers 1
```

- MF (Yelp):
```bash
python models/COVA/COVA_mf_yelp.py --attack 0.01 --dataset Yelp
```

### Outputs
- Model checkpoints are saved under `./Weights/LightGCN/` or `./Weights/MF/` with informative names including key configuration values.
