# 🛡️ Real-Time Fraud Detection System
A machine learning project that detects fraudulent transactions using gradient boosting models (LightGBM, XGBoost) and an Isolation Forest baseline, with SHAP-based explainability and a live Streamlit dashboard.
## 📂 Project Structure
- `analysis3.ipynb` – full EDA, feature engineering, model training & evaluation
- `dashboard/app.py` – Streamlit dashboard for exploring predictions
- `charts/` – saved visualizations (ROC/PR curves, SHAP plots, feature importance, etc.)
- `summary.docx` – written project summary

## ⚙️ Setup
1. Clone this repo and install dependencies:
pip install -r requirements.txt
2. Download the dataset (not included in this repo due to file size — see below) and place the CSV files inside a `data/` folder:
   - `train_transaction.csv`
   - `train_identity.csv`
3. Run `analysis3.ipynb` top to bottom to train and evaluate the models.
4. Launch the dashboard:
streamlit run dashboard/app.py

## 📊 Dataset
This project uses the [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection/data) from Kaggle. Due to GitHub's file size limits, the raw CSVs are not included — download them from Kaggle and place them in `data/` as shown above.
 
## 📈 Model Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| LightGBM (tuned) | ~0.96 | ~0.82 |
| XGBoost | ~0.94 | ~0.79 |
| IsolationForest | ~0.78 | ~0.21 |

## 🧠 Explainability
SHAP is used to explain individual predictions and global feature importance — see `charts/shap_summary.png` and the waterfall plots for example cases (fraud, borderline, and legitimate transactions).
