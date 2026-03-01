# Credit Card Fraud Detection Project

A complete Machine Learning project for detecting fraudulent credit card transactions using XGBoost and Streamlit.

## 📋 Overview

This project implements a fraud detection system that predicts whether a credit card transaction is fraudulent based on various transaction features (V1-V28 from PCA transformation, Time, and Amount).

## 🚀 Features

- **Exploratory Data Analysis (EDA)**: Class distribution, correlation heatmap, Amount/Time distributions
- **Data Preprocessing**: StandardScaler for Time and Amount features
- **Class Imbalance Handling**: Custom SMOTE implementation
- **Machine Learning Models**: 
  - Logistic Regression
  - Random Forest
  - XGBoost
- **Model Evaluation**: Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Visualizations**: ROC Curves, Confusion Matrix heatmaps, Feature Importance
- **Model Explainability**: SHAP integration for XGBoost
- **Web Interface**: Streamlit app for real-time predictions

## 📁 Project Structure

```
credit card fraud detection/
├── app.py                    # Streamlit web application
├── main.py                   # Complete ML pipeline
├── README.md                 # This file
├── TODO.md                   # Project task list
├── creditcard_synthetic.csv  # Synthetic dataset
├── 01_class_distribution.png # EDA visualization
├── 02_correlation_heatmap.png # EDA visualization
└── 03_amount_time_distribution.png # EDA visualization
```

## 🛠️ Tech Stack

- **Python**: Core programming language
- **Pandas, NumPy**: Data manipulation
- **Scikit-learn**: Machine learning models
- **XGBoost**: Gradient boosting classifier
- **SHAP**: Model explainability
- **Matplotlib, Seaborn**: Data visualization
- **Streamlit**: Web application framework

## 📊 Dataset

The project uses a synthetic dataset that mimics the Kaggle Credit Card Fraud Detection dataset with:
- 100,000 transactions
- 30 features (Time, Amount, V1-V28)
- ~0.17% fraud rate (imbalanced)

## 🎯 Risk Levels

| Risk Level | Fraud Probability |
|------------|-------------------|
| 🟢 Low     | < 20%             |
| 🟡 Medium  | 20% - 50%         |
| 🔴 High    | > 50%             |

## 🚦 Getting Started

### Prerequisites

```
bash
pip install streamlit xgboost shap joblib matplotlib seaborn pandas numpy scikit-learn
```

### Running the Application

1. **Start Streamlit App**:
   
```
bash
   streamlit run app.py
   
```

2. **Open in Browser**:
   Navigate to `http://localhost:8501`

3. **Use the App**:
   - Enter transaction features
   - Click "Predict Fraud"
   - View probability and risk level

## 📖 Usage

1. **Input Features**: Enter the transaction Time, Amount, and V1-V28 PCA components
2. **Predict**: Click the "Predict Fraud" button
3. **Results**: View:
   - Fraud probability percentage
   - Risk level (Low/Medium/High)
   - Prediction (Legitimate/Fraudulent)
   - Feature importance chart

## 🔧 Development Notes

- The model is trained on synthetic data for demonstration
- In production, load a pre-trained model using `joblib.load('xgb_model.pkl')`
- Adjust the risk thresholds based on business requirements

## 📝 License

This project is for educational purposes.

## 👨‍💻 Author

Data Scientist & ML Engineer
