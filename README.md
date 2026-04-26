# El Niño Prediction
## Overview

This project forecasts El Niño events by predicting the **ONI 3.4 index** (Oceanic Niño Index) at lead times ranging from 1 to 24 months. Rather than predicting the index directly, the approach reconstructs it from the **two leading principal components (PC1 and PC2)** of an EOF analysis on sea surface temperature (SST) anomalies — a linear combination of the two PCs recovers 96.7% of the ONI 3.4 variance.

Five models are benchmarked: three classical ML methods and two deep learning architectures.

---

## Repository Structure

```
.
├── sst.mnmean.nc                      # Raw SST dataset (NOAA, 1981–2021)
├── preprocess.py                      # EOF analysis, anomaly computation, PC extraction
├── CNN.py                             # CNN model definition (resolution-agnostic with LazyLinear)
├── el_nino_Introduction.ipynb         # Data exploration and EOF analysis
├── el_nino_Machine_Learning_v2.ipynb  # Linear Regression, Ridge, Random Forest
└── el_nino_Deep_Learning_v2.ipynb     # CNN, ViT (full fine-tune), ViT + LoRA
```

---

## Data

- **Source:** [NOAA](https://www.noaa.gov/) — monthly mean SST, 1981–2021
- **Format:** NetCDF (`.nc`), accessed via `xarray` with shape `(time, lat, lon)`
- **Preprocessing** (see `preprocess.py`):
  1. Remove the seasonal cycle to obtain SST anomalies
  2. Apply area-weighting (`√cos(lat)`) before EOF decomposition
  3. Extract PC1 and PC2 via eigenvalue decomposition of the covariance matrix
  4. Fit a linear regression of PC1 + PC2 → ONI 3.4 on the training set

Two spatial domains are used:
- **Tropical Pacific** — `[20°S–12.5°N] × [120°E–70°W]`
- **Global** — full globe, to test whether remote SST information helps

---

## Methodology

### Feature engineering

| Model family | Input format |
|---|---|
| ML (Linear, Ridge, RF) | Vector of 12 lagged PC values: `(12 × 1, y)` |
| CNN / ViT / LoRA | Stack of 12 monthly SST maps: `(12 × H × W, y)` |

All experiments use a **chronological 80/20 train/test split** (no shuffling). The ONI 3.4 forecast is always derived as a linear combination of the predicted PC1 and PC2, where the combiner weights are fit on the training set only.

### Models

#### Classical ML

| Model | Key detail |
|---|---|
| **Linear Regression** | OLS baseline on 12 lagged PC values |
| **Ridge Regression** | L2 regularisation; α selected via leave-one-out CV over 50 log-spaced candidates (10⁻⁵ → 10⁵) |
| **Random Forest** | 300 trees, `min_samples_leaf=2`, all CPU cores |

#### Deep Learning

| Model | Key detail |
|---|---|
| **CNN** | Two conv blocks (32→16 filters, 3×3, same padding, dropout + MaxPool); `nn.LazyLinear` output head — resolution-agnostic |
| **ViT** | Pretrained `vit_base_patch16_224` (ImageNet); 12-channel SST input; full fine-tuning with AdamW |
| **ViT + LoRA** | Same backbone, fully frozen; LoRA adapters (rank 8) injected into QKV projections layers of every transformer block |



---

## Evaluation Metrics

Each model is evaluated on the held-out test set (last 20% of the time series) using:

- **RMSE** — Root Mean Square Error
- **R²** — Coefficient of determination
- **Pearson correlation** — Linear correlation between predicted and actual values

Metrics are computed separately for PC1, PC2, and the reconstructed ONI 3.4 index.


## Installation

```bash
pip install numpy pandas matplotlib xarray scikit-learn torch torchvision timm
```

---

## Usage

Run the notebooks in order:

```bash
jupyter notebook el_nino_Introduction.ipynb        # Data exploration & EOF
jupyter notebook el_nino_Machine_Learning_v2.ipynb  # ML experiments
jupyter notebook el_nino_Deep_Learning_v2.ipynb     # DL experiments
```

The notebooks are self-contained: `preprocess.py` and `CNN.py` are imported at runtime and `sst.mnmean.nc` must be present in the working directory.
