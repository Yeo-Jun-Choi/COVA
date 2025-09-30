# [Under Review] Efficient Recommendation Unlearning via Task Vector Arithmetic in Shared Space

---

The code supports a simple, reproducible workflow:
1) Data preprocessing → 2) Train the Original Model → 3) Train the Retraining Model → 4) Free experiments.


## Requirements
- Python 3.9.0
- PyTorch 2.5.1+cu121 (CUDA 12.1)
- numpy 1.26.3, pandas 2.3.1, scipy 1.13.1, scikit-learn 1.6.1


## Data
Place your datasets under `Data/`. Preprocessing will produce processed files under `Data/Process/<DATASET>/<ATTACK>/`.

Specify the --dataset argument (Amazon_Book, Gowalla, or Yelp):
```bash
python _data_process.py --dataset Yelp --attack 0.01
```


## Directory Setup
````bash
mkdir -p Weights/MF
mkdir -p Weights/MF_JointSVD
mkdir -p Weights/LightGCN
mkdir -p Weights/LightGCN_JointSVD
````

## Usage & Evaluation

This project can be executed with a single script: **main.py**.  
You only need to edit `main.py` to set the desired **model type (model_type)**, **algorithm (algorithm)**, **dataset (dataset)**, and **parameters (args)**.  

```bash
# Example: LightGCN on Yelp
python main.py --model-type cova --algorithm MF --dataset yelp --attack 0.01 --alpha 35 --beta 0.9
```

### Outputs
- Model checkpoints are saved under `./Weights/LightGCN/` or `./Weights/MF/` with informative names including key configuration values.
