# ElasticNet-Based Supervised Feature Selection for 50-Gene Lung Cancer Classification Panel

---

## Project Overview

This repository presents a complete supervised machine learning workflow for robust biomarker discovery and classification in lung cancer.

Using ElasticNet-based feature selection with stability-driven ranking, we derived a reproducible 50-gene classification panel from integrated transcriptomic datasets. The model was benchmarked internally using multiple classifiers and independently validated on an external dataset.

The pipeline was implemented end-to-end in Google Colab.

---

## Study Design

### Training Data
- 4 integrated lung cancer gene expression datasets
- Preprocessed machine-learning-ready expression matrix
- Binary classification: Tumor (1) vs Normal (0)

### External Validation Dataset
- 91 independent samples
  - 46 Tumor
  - 45 Normal
- No gene re-selection performed
- No hyperparameter tuning on external data

Strict separation between training and validation data was maintained to prevent data leakage.

---

## Methodological Workflow

### 1. Differential Expression Filtering
Initial dimensionality reduction:
- 551 differentially expressed genes (DEGs) retained

### 2. ElasticNet-Based Feature Selection
- Logistic regression with ElasticNet regularization
- Repeated subsampling for stability selection
- Selection frequency computed per gene
- Genes ranked by reproducibility
- Top 50 genes selected as final biomarker panel

This stability-driven approach reduces overfitting and increases biological robustness.

---

## Internal Model Benchmarking

Models evaluated using the 50-gene panel:

- Random Forest
- Support Vector Machine (RBF kernel)
- XGBoost

Internal Performance:

- Random Forest AUC = 0.954
- SVM AUC = 0.954
- XGBoost AUC = 0.948

Models showed strong and consistent classification performance.

---

## External Validation Results

Final models were trained on the full training dataset and evaluated on an independent cohort.

### External AUC

- ElasticNet (Logistic Regression) AUC = 0.994
- Random Forest AUC = 1.000

### Confusion Matrix
[[43  2] 
[ 0 46]]

### Classification Report

Precision, Recall, F1-score:

Class 0 (Normal):
- Precision = 1.00
- Recall = 0.96
- F1 = 0.98

Class 1 (Tumor):
- Precision = 0.96
- Recall = 1.00
- F1 = 0.98

Overall Accuracy = 97.8%

Predicted probability distributions showed complete separation between groups with minimal overlap.

---

## Key Findings

- Stable 50-gene biomarker panel derived using ElasticNet regularization
- Consistent internal classification performance (AUC ≈ 0.95)
- Near-perfect external validation performance (AUC up to 1.0)
- Minimal misclassification (2 false negatives, 0 false positives)
- Strong class separation in predicted probability space

---

## Repository Structure
elasticnet-50-gene-lung-cancer/ │ 
├── Lung_Cancer_ElasticNet_Stability_External_Validation.ipynb 
├── Top50_ElasticNet_Stable_Biomarkers.csv 
└── README.md

---

## Reproducibility

Environment:
- Python 3.x
- Google Colab

Core libraries:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

To reproduce:

1. Clone repository
2. Install dependencies:
   pip install -r requirements.txt
3. Open and execute the notebook

External dataset must be uploaded locally before running validation steps.

---

## Methodological Strengths

- Stability-based feature selection
- Regularized regression framework
- Multi-model benchmarking
- Strict external validation
- No feature re-selection on validation set
- No external hyperparameter tuning

---

## Future Work

- Survival analysis of 50-gene panel
- Pathway enrichment analysis
- Cross-cohort validation across additional studies
- Clinical interpretability assessment

---

## License

NA
