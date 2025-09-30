# [Under Review] Efficient Recommendation Unlearning via Task Vector Arithmetic in Shared Space

---

The code supports a simple, reproducible workflow:
1) Data preprocessing → 2) Train the Original Model → 3) Train the Retraining Model → 4) Free experiments.


## Requirements
- Python 3.9.0
- PyTorch 2.5.1+cu121 (CUDA 12.1)
- numpy 1.26.3, pandas 2.3.1, scipy 1.13.1, scikit-learn 1.6.1

You can install the core dependencies using:
```
# It is recommended to use a virtual environment
pip install torch==2.5.1
pip install numpy==1.26.3 pandas==2.3.1 scipy==1.13.1 scikit-learn==1.6.1

```

## Data
Place your datasets under `Data/`. Preprocessing will produce processed files under `Data/Process/<DATASET>/<ATTACK>/`.

## Data Preprocessing

1) Run the preprocessing script before training. Adjust arguments as needed for your datasets.
```bash
python _data_process.py
```

2) Preprocessing Command Examples : specify the --dataset argument (Amazon_Book, Gowalla, or Yelp).

- Arguments Example (Gowalla, Custom Settings):
Gowalla dataset is preprocessed with attack=0.05, k-core=10, and a custom seed=42.
```bash
python _data_process.py --dataset Gowalla --attack 0.05 --k 10 --seed 42
```



### Directory Setup
````bash

mkdir -p Weights/MF
mkdir -p Weights/MF_JointSVD
mkdir -p Weights/LightGCN
mkdir -p Weights/LightGCN_JointSVD
````


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


## Usage

This project can be executed with a single script: **main.py**.  
You only need to edit `main.py` to set the desired **model type (model_type)**, **algorithm (algorithm)**, **dataset (dataset)**, and **parameters (args)**.  

### Run Command

If you want to perform fine-grained configuration when running the script, you can run 'main.py'.

Specify the model type, algorithm, dataset, and any additional parameters directly:

```bash
python main.py --model-type cova --algorithm LightGCN --dataset Yelp --attack 0.01 --gcn_layers 2
```

Use predefined configurations inside main.py by adding --use-preset.
Uncomment the desired run_model(...) call in the code, then run:

```bash
python main.py
```

### Outputs
- Model checkpoints are saved under `./Weights/LightGCN/` or `./Weights/MF/` with informative names including key configuration values.
