# 🧬 Pan-Cancer Gene Expression Classifier
Predicting cancer type from RNA-Seq gene expression profiles using unsupervised and supervised machine learning, served through a FastAPI endpoint and deployable via Docker.

**Status:** ✅ ML pipeline, API and Docker deployment complete — frontend demo included

**Tools:** Python 3.11 · Pandas · NumPy · Scikit-learn · FastAPI · Docker · Matplotlib · Seaborn · Conda
**Data:** UCI Gene Expression Cancer RNA-Seq (TCGA PANCAN) · 801 samples · 20,531 genes · 5 cancer types

---

## Biological Question
Can gene expression profiles alone distinguish between five different cancer types — without any prior clinical information — using standard machine learning methods?

This is a classic pan-cancer genomics problem: RNA-Seq measures the activity level of thousands of genes simultaneously, and different cancer types are known to have distinct transcriptomic signatures. This project tests whether that signal is strong enough to separate cancer types both visually (unsupervised) and predictively (supervised) — and whether that signal survives a methodologically honest evaluation.

---

## Dataset

| Metric | Value |
|---|---|
| Source | UCI Machine Learning Repository — Gene Expression Cancer RNA-Seq |
| Origin | TCGA Pan-Cancer Analysis Project, Illumina HiSeq platform |
| Samples | 801 |
| Genes (features) | 20,531 |
| Missing values | 0 |
| Cancer types | 5 |

### Class Distribution (imbalanced)

| Class | Full name | Count |
|---|---|---|
| BRCA | Breast invasive carcinoma | 300 |
| KIRC | Kidney renal clear cell carcinoma | 146 |
| LUAD | Lung adenocarcinoma | 141 |
| PRAD | Prostate adenocarcinoma | 136 |
| COAD | Colon adenocarcinoma | 78 |

> Note: this imbalance is handled explicitly in the modeling stage (stratified train/test split, class-weighted classifier) rather than ignored.

---

## How the Pipeline Works

```
Raw Data (UCI zip)              801 samples × 20,531 genes
        ↓
   Download + Cache             Direct download (bypassing a broken
                                 ucimlrepo import for this dataset),
                                 zip/tar extraction, local caching
        ↓
   Log-Transform                log2(x + 1) — compresses the highly
                                 skewed RNA-Seq expression distribution
                                 (per-sample, so safe to apply before
                                 the train/test split — see note below)
        ↓
  Train/Test Split              70/30, stratified by class — done
                                 BEFORE any statistic is estimated
        ↓
   Variance Filtering           fit on TRAIN only: ~20,531 → ~2,540 genes
        ↓
   Standardization              fit on TRAIN only: zero mean, unit variance
        ↓
   ┌─────────────────┬─────────────────────┐
   ↓                                       ↓
PCA + K-Means                    Random Forest Classifier
(unsupervised check)             (supervised prediction, inside
                                  a single sklearn Pipeline)
   ↓                                       ↓
Cluster vs. true-class           Confusion matrix + F1 per class
comparison plot                  + 5-fold stratified cross-validation
```

---

## 🔍 A Methodology Lesson: Catching and Fixing a Data Leak

The first version of this pipeline scored **99% accuracy** on the test set. An impressive number — and exactly for that reason, a suspicious one. On a 5-class genomics problem with only 801 samples, a score that close to perfect deserves scrutiny before celebration.

Re-reading the pipeline surfaced the issue: the variance filter (20,531 → ~2,500 genes) and the standardization step were being **fit on the entire dataset before the train/test split**. In practice, the choice of which genes were "informative" — and the mean/std used to normalize them — had already "seen" the test samples before training even began. A textbook case of data leakage: the model never truly faced unseen data.

The fix: move the split to **before** any fitting, and fit the variance filter and scaler on the training data only, bundled into a single `sklearn.Pipeline` — which mechanically guarantees that no statistic is ever computed using the test set.

**Result after the fix**, validated with 5-fold stratified cross-validation (not a single split, to avoid getting fooled by luck a second time):

| Metric | Value |
|---|---|
| Accuracy | 99.5% ± 0.7% |
| F1 macro | 99.6% ± 0.6% |

The score barely moved — which is itself valuable information. It confirms that the biological signal separating the five cancer types is real and robust, not an artifact of the data leak.

---

## Results

### 1 — Unsupervised Check: PCA + K-Means

| Metric | Value |
|---|---|
| PCA components | 2 |
| Variance explained (PC1 + PC2) | 23.85% |
| Clustering method | K-Means (k=5, no labels used) |

![PCA Clusters Comparison](results/figures/pca_clusters_comparison.png)

**Interpretation:** Two cancer types (KIRC, PRAD) form tight, well-isolated clusters that K-Means recovers almost perfectly without ever seeing the true labels. Three types (BRCA, LUAD, COAD) occupy a partially overlapping region in 2D — biologically plausible, since these tissues can share broader transcriptomic programs. This confirms a real, detectable biological signal exists before any supervised modeling is attempted.

### 2 — Supervised Classification: Random Forest (leakage-free)

| Metric | Value |
|---|---|
| Model | Random Forest (200 trees, class-weighted), in a single sklearn Pipeline |
| Train/test split | 70/30, stratified by class |
| Test accuracy | 99% |
| 5-fold CV accuracy | 99.5% ± 0.7% |
| 5-fold CV F1 macro | 99.6% ± 0.6% |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| BRCA | 0.97 | 1.00 | 0.98 | 90 |
| COAD | 1.00 | 1.00 | 1.00 | 24 |
| KIRC | 1.00 | 0.98 | 0.99 | 44 |
| LUAD | 1.00 | 0.95 | 0.98 | 42 |
| PRAD | 1.00 | 1.00 | 1.00 | 41 |

![Confusion Matrix](results/figures/confusion_matrix.png)

**Interpretation:** Only 3 misclassifications out of 241 test samples (1 KIRC → BRCA, 2 LUAD → BRCA). COAD — the minority class with only 78 total samples — is still classified perfectly, confirming the class-weighted approach compensates for the imbalance rather than the model defaulting to the majority class. The 5-fold cross-validation confirms this isn't a lucky single split.

---

## API

A FastAPI service serves the trained Pipeline for live predictions.

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service status, model loaded, number of genes expected |
| `/sample` | GET | Returns a real patient from the dataset, pre-filtered to the genes the model expects |
| `/predict` | POST | Takes `{"expression": {"gene_name": value, ...}}` (raw, pre-log values), returns predicted class + per-class probabilities |

```bash
uvicorn api:app --reload --app-dir src
python src/test_api.py
```

## Frontend

A minimal static demo (`frontend/index.html`) lets you either pull a random real patient (via `/sample`) or upload a CSV of gene expression values, and visualizes the predicted class with per-class confidence bars. Point it at your running API (default `http://localhost:8000`).

## Docker

```bash
docker build -t pancancer-api .
docker run -p 8000:8000 pancancer-api
# or:
docker compose up --build
```

The image bundles the trained pipeline (`models/pancancer_pipeline.joblib`) and the cached dataset — no training or download happens at container startup, only model loading.

---

## Project Structure

```
pancancer-genomic-classification/
│
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                ← for Docker (API serving)
├── environment.yml                 ← conda environment (pancancer-genomics, full dev env)
│
├── src/
│   ├── config.py                   ← paths, constants, random seed
│   ├── data_loading.py             ← download, cache, sanity-check dataset
│   ├── preprocessing.py            ← log-transform, variance filter, scaling
│   ├── clustering.py               ← PCA + K-Means, comparison plot
│   ├── classification.py           ← train/test split, Pipeline, Random Forest, evaluation
│   ├── cross_validation.py         ← 5-fold stratified CV on the full Pipeline
│   ├── api.py                      ← FastAPI serving endpoint
│   └── test_api.py                 ← smoke tests for the API
│
├── frontend/
│   └── index.html                  ← static demo UI
│
├── notebooks/                      ← exploratory work
├── reports/                        ← write-ups
│
├── data/
│   ├── raw/                        ← downloaded dataset (gitignored)
│   └── processed/                  ← cleaned/cached data
│
└── results/
    ├── figures/
    │   ├── pca_clusters_comparison.png
    │   └── confusion_matrix.png
    └── tables/
```

---

## Reproduce This Analysis

### Requirements
- Linux or WSL2 (Ubuntu 22.04 / 24.04)
- Miniconda (for development) or Docker (for serving only)

### Setup (development)

```bash
git clone https://github.com/moumouh6/pancancer-genomic-classification.git
cd pancancer-genomic-classification

conda env create -f environment.yml
conda activate pancancer-genomics
```

### Run the pipeline

```bash
python src/data_loading.py       # downloads + caches the dataset
python src/clustering.py         # PCA + K-Means, saves comparison plot
python src/classification.py     # trains Pipeline, saves model + confusion matrix
python src/cross_validation.py   # 5-fold stratified CV
uvicorn api:app --reload --app-dir src   # serves predictions
```

---

## Tools & Versions

| Tool | Role |
|---|---|
| Python 3.11 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | PCA, K-Means, Random Forest, Pipeline, metrics |
| FastAPI / Uvicorn | Model serving |
| Docker | Deployment |
| Matplotlib / Seaborn | Visualization |
| Conda | Environment management (dev) |

---

## Limitations

Being upfront about this matters more than the accuracy number:

- **Small, single-source dataset.** 801 samples from TCGA only. The model has never seen data from another cohort, sequencing platform, or population, so this accuracy may not generalize outside TCGA-like data.
- **No external validation set.** All evaluation (test split and cross-validation) comes from the same 801-sample pool. A truly held-out, independently collected cohort would be the real test.
- **Gene list is dataset-specific.** The ~2,540 genes the model expects were selected by variance on this particular training split; a different sample or a different sequencing batch could shift that selection slightly.
- **Not clinically validated.** This is a methodology and engineering exercise, not a diagnostic tool. Any resemblance to a clinical pipeline stops at the architecture.

---

## Scientific Context

This project reproduces a well-studied benchmark problem in cancer genomics: multi-class tumor classification from RNA-Seq gene expression, using the TCGA Pan-Cancer dataset distributed via the UCI Machine Learning Repository. This type of pipeline — expression-based classification — is directly relevant to genomic surveillance and precision medicine work carried out by institutions such as the Institut Pasteur d'Algérie (IPA), bridging computational methods with practical diagnostic and research applications.
