# Seismological Fault Line Discovery & Earthquake Impact Profiling

**Course**: BITSF464 Machine Learning — Fall 2026  
**Repository**: `https://github.com/Tulip0648/seismic-fault-impact`  
**Project Proposal**: [project_proposal.pdf](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/project_proposal.pdf)  

### Team Members
| Name | ID |
|---|---|
| Aaryan Pandey | 2024A7PS0375G |
| Rhea Jain | 2024A7PS0575G |
| Nivya Nandagopal | 2024ADPS1428G |
| Sahaj Jhamb | 2024A7PS0485G |
| Mridul Sardana | 2024A7PS0547G |
| Harshit Rawat | 2024AAPS0756G |
| Parshav Giya | 2024ADPS0852G |

---

## 1. Project Overview & Motivation
Standard tabular earthquake models treat latitude and longitude as flat Euclidean planes and ignore the underlying 3D tectonic fault geometry that dictates ground shaking.

This project implements a **4-module semi-supervised machine learning pipeline** that:
1. **Unsupervised Fault Discovery**: Identifies tectonic fault boundaries and isolates background tremors using spherical distance clustering (DBSCAN) and Expectation-Maximization Gaussian Mixture Models (GMM) in 3D hypocentral coordinates.
2. **Supervised Impact Prediction**: Predicts earthquake impact severity (PAGER multi-class alert level: green, yellow, orange, red; and continuous/categorical Modified Mercalli Intensity `mmi`) by coupling discovered tectonic fault features into linear, kernel, tree-based, and neural ensemble models.
3. **Two-Stage Evaluation Protocol**: Evaluates all supervised models across two dataset configurations under a fixed stratified split:
   - **Base (Baseline)**: Imputed and standardized raw sensor features + geospatial coordinates.
   - **Augmented (Coupled)**: Base features combined with sensor PCA components, DBSCAN fault distances, and GMM EM posterior probabilities.

---

## 2. Global Architecture & Dependency Graph

```
                            Raw USGS ComCat Catalog
                         (ShakeMap & PAGER Attributes)
                                       │
                                       ▼
             ┌───────────────────────────────────────────────────┐
             │                     MODULE 1                      │
             │        01_geospatial_preprocessing.ipynb          │
             │   - Data cleaning & MICE/Median Imputation        │
             │   - 3D ECEF & Spherical Radian Coordinate Calcs   │
             │   - Categorical Encoding & Sensor Robust Scaling  │
             │   - Sensor PCA (Dimensionality Reduction)         │
             │   - Fixed Stratified Train / Val / Test Splitting │
             └─────────────────────────┬─────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ Base Features (Train/Val/Test)              │ Spatial Coords (x,y,z,depth)
                │ & Stratified Splits                         │ & Sensor PCA
                ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│           MODULE 3            │             │           MODULE 2            │
│ 03_supervised_predictive_     │             │ 02_unsupervised_fault_        │
│    modelling.ipynb            │             │    discovery.ipynb            │
│ - Linear (Logistic L1/L2)     │             │ - Spherical & 3D DBSCAN       │
│ - Probabilistic (NB, LDA, QDA)│             │ - EM Gaussian Mixture Models  │
│ - Instance-based (k-NN)       │             │ - BIC / AIC Model Selection   │
│ - Kernel-based (SVM RBF/Poly) │             │ - Silhouette & Davies-Bouldin │
│                               │             │ - Fault Distance & Posteriors │
│ [Evaluates Stage 1: Base]     │             └───────────────┬───────────────┘
└───────────────┬───────────────┘                             │
                │                                             │ Augmented Features
                │                                             │ (Base + PCA + Fault Features)
                │ Baseline Results &                          ├───────────────────────────────┐
                │ Out-of-Fold (OOF) Probs                     │                               │
                │                                             ▼                               ▼
                │                               ┌───────────────────────────┐   ┌───────────────────────────┐
                │                               │         MODULE 3          │   │         MODULE 4          │
                └──────────────────────────────►│ [Evaluates Stage 2:       │   │ 04_advanced_models_and_   │
                                                │  Augmented Features]      │   │    integration.ipynb      │
                                                └───────────────────────────┘   │ - Tree Ensembles (RF)     │
                                                                                │ - Boosters (LGBM, XGB)    │
                                                                                │ - Deep Tabular Neural Net │
                                                                                │ - Stacking Meta-Learner   │
                                                                                │ - Global Leaderboard      │
                                                                                │ - SHAP Attributions       │
                                                                                │ - End-to-End Pipeline     │
                                                                                └───────────────────────────┘
```

---

## 3. Module Specifications & Notebook Directory

The project follows a **Notebook-First Workflow**. Each module is implemented as a self-contained, reproducible Jupyter Notebook (`.ipynb`). Teammates collaborate without breaking each other's code by exchanging standardized Parquet datasets and Joblib model artifacts:

| Module | Focus & Responsibilities | Notebook & Quick Launch | Specification |
|---|---|---|---|
| **Module 1** | Geospatial preprocessing, 3D ECEF conversion, sensor PCA, stratified split | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module1/01_geospatial_preprocessing.ipynb)<br>`module1/01_geospatial_preprocessing.ipynb` | [module1/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module1/TODO.md) |
| **Module 2** | Spherical DBSCAN fault lines, EM GMM posteriors, BIC/AIC, dataset augmentation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module2/02_unsupervised_fault_discovery.ipynb)<br>`module2/02_unsupervised_fault_discovery.ipynb` | [module2/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module2/TODO.md) |
| **Module 3** | Linear, Probabilistic, $k$-NN, SVM models; Two-Stage Base vs. Augmented evaluation; OOF predictions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module3/03_supervised_predictive_modelling.ipynb)<br>`module3/03_supervised_predictive_modelling.ipynb` | [module3/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module3/TODO.md) |
| **Module 4** | Random Forest, LightGBM, XGBoost, Neural Nets, Stacking Ensemble, SHAP, end-to-end inference API | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tulip0648/seismic-fault-impact/blob/main/module4/04_advanced_models_and_integration.ipynb)<br>`module4/04_advanced_models_and_integration.ipynb` | [module4/TODO.md](file:///run/media/cosmos/Data/mridul/codes/ML/seismic-fault-impact/module4/TODO.md) |

---

## 4. How to Run & Collaborate: Google Colab & Local Jupyter

### 4.1 Running in Google Colab

Every notebook contains a universal initialization cell at the very top that automatically detects whether it is running in Colab or locally.

#### Opening Notebooks in Colab:
- **Option A (One-Click Badges)**: Click any **Open in Colab** badge in the table above.
- **Option B (Direct from Colab)**:
  1. Go to [colab.research.google.com](https://colab.research.google.com).
  2. In the modal, select the **GitHub** tab.
  3. Enter the repository URL: `https://github.com/Tulip0648/seismic-fault-impact`.
  4. Select the desired module notebook from the dropdown.

#### Hardware Accelerators (GPU):
For **Module 4** (LightGBM, XGBoost, CatBoost, and PyTorch Neural Networks), enable GPU acceleration:
- In the top menu: **Runtime -> Change runtime type**.
- Under **Hardware accelerator**, select **T4 GPU** and click **Save**.

---

### 4.2 How to Integrate Google Colab with GitHub

There are three primary ways our team integrates Google Colab with GitHub:

#### Method 1: Direct Colab UI Sync (Recommended for Quick Updates)
Colab has built-in two-way GitHub integration:
1. **Authorize GitHub in Colab**: When opening a notebook from the GitHub tab, Colab will ask for authorization to access your repositories. Check the box to allow access to private repositories if prompted.
2. **Commit changes directly from Colab**:
   - In Colab, click **File -> Save a copy in GitHub...**.
   - Select repository: `Tulip0648/seismic-fault-impact`.
   - Select branch: `main` (or your feature branch, e.g. `feature/module1`).
   - Specify file path: `module1/01_geospatial_preprocessing.ipynb`.
   - Enter a descriptive commit message (e.g. `feat: implement 3D ECEF coordinate transform`).
   - Click **OK**. Your notebook is saved and committed directly to GitHub!

#### Method 2: Git Clone and Push via Colab Bash Cells (Full Terminal Control)
If you prefer standard Git CLI commands inside a Colab cell:
1. Generate a **GitHub Personal Access Token (PAT)**:
   - On GitHub: `Settings -> Developer settings -> Personal access tokens -> Tokens (classic)`.
   - Click **Generate new token**, check `repo` scope, and copy the token.
2. Store the token securely in Colab:
   - In the Colab left sidebar, click the **Secrets** icon (🔑 key symbol).
   - Add a new secret with name `GITHUB_TOKEN` and paste your token value. Enable notebook access.
3. Run the following cell inside Colab to clone with write access:
   ```python
   from google.colab import userdata
   import os

   GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
   USERNAME = "your_github_username"
   REPO = "Tulip0648/seismic-fault-impact"

   # Clone using authenticated token:
   !git clone https://{GITHUB_TOKEN}@github.com/{REPO}.git /content/seismic-fault-impact
   %cd /content/seismic-fault-impact
   !git config user.name "Your Name"
   !git config user.email "your.email@example.com"
   ```
4. Stage, commit, and push updates:
   ```bash
   !git status
   !git add module1/01_geospatial_preprocessing.ipynb
   !git commit -m "docs: add exploratory data analysis plots"
   !git push origin main
   ```

#### Method 3: Persistent Data Storage via Google Drive
Colab instances are ephemeral: all local files inside `/content/` are wiped when the session disconnects. To share large datasets (USGS ComCat raw records, Parquet splits, and trained models) across teammates without re-downloading or re-training:
1. Mount Google Drive at the start of your notebook:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Create a shared project folder in Google Drive (e.g. `/content/drive/MyDrive/seismic-fault-impact/data/`) and configure your notebook paths to read from and write to this persistent directory.

---

### 4.3 Running Locally (JupyterLab / VS Code)

If you prefer developing on your local machine:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tulip0648/seismic-fault-impact.git
   cd seismic-fault-impact
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch JupyterLab or VS Code**:
   ```bash
   jupyter lab
   ```
   Or open any `.ipynb` file directly in VS Code (with the Microsoft Python and Jupyter extensions enabled) and select your `venv` kernel.

---

## 5. Cross-Module Data Contracts

Because team members work on different modules in parallel, notebooks communicate through clean, serialized data contracts saved in Parquet format:

### 5.1 Base Dataset Contract (Module 1 $\rightarrow$ Modules 2, 3, 4)
- Files: `train_base.parquet`, `val_base.parquet`, `test_base.parquet`
- Contains:
  - Coordinate representations: `latitude`, `longitude`, `depth`, `lat_rad`, `lon_rad`, `x_ecef`, `y_ecef`, `z_ecef`
  - Sensor & error metrics: `mag`, `nst`, `gap`, `dmin`, `rms`, `horizontalError`, `depthError`, `magError` (imputed and standardized)
  - Categorical encodings: `magType_*`
  - Dimensionality reduction: `pca_sensor_1`, `pca_sensor_2`, `pca_sensor_3`
  - Targets: `alert` (`green`, `yellow`, `orange`, `red`), `mmi`

### 5.2 Augmented Dataset Contract (Module 2 $\rightarrow$ Modules 3, 4)
- Files: `train_augmented.parquet`, `val_augmented.parquet`, `test_augmented.parquet`
- Contains: All Base dataset columns plus:
  - `dbscan_cluster`: Discovered fault line ID (`-1` = background tremor)
  - `is_background_noise`: Binary indicator (`1` = isolated tremor, `0` = fault line event)
  - `dist_to_nearest_fault_km`: Orthogonal / geodesic distance in km to nearest fault cluster
  - `gmm_cluster`: Argmax mixture component
  - `gmm_log_likelihood`: Hypocentral density score
  - `gmm_prob_0` ... `gmm_prob_{K-1}`: Soft EM fault membership probabilities ($\sum_k p_k = 1.0$)

### 5.3 Stacking Contract (Module 3 $\rightarrow$ Module 4)
- Files: `oof_train_m3.parquet`, `oof_test_m3.parquet`
- Contains: Cross-validated out-of-fold probability distributions from Logistic Regression, LDA, $k$-NN, and Kernel SVM.

---

## 6. Execution Roadmap & Recommended Workflow
1. **Module 1**: Run `01_geospatial_preprocessing.ipynb` to clean raw USGS data, generate 3D coordinates, and export the Base splits (`data/processed/*_base.parquet`).
2. **Module 2**: Run `02_unsupervised_fault_discovery.ipynb` to fit DBSCAN and GMM on the training split and export the Augmented splits (`data/augmented/*_augmented.parquet`).
3. **Module 3**: Run `03_supervised_predictive_modelling.ipynb` to train linear, probabilistic, $k$-NN, and SVM models across Base vs Augmented, log baseline metrics, and export OOF predictions.
4. **Module 4**: Run `04_advanced_models_and_integration.ipynb` to train gradient boosters and neural nets, fit the Stacking Meta-Learner, generate SHAP reports, and test the interactive earthquake prediction widget.
