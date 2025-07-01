# 🫁 Lung Cancer Prediction App

This project focuses on predicting the likelihood of lung cancer in individuals using a diverse dataset and a suite of supervised machine learning algorithms. The models were evaluated on key performance metrics and a **Streamlit web app** was also built for user-friendly access.

---

## 🚀 Overview

Lung cancer is a leading cause of cancer-related deaths worldwide. Early detection is crucial for improving treatment outcomes. This project aims to build predictive models that can assist in identifying patients at risk based on available data.

---

## 🧠 Algorithms Used

The following machine learning models were trained and compared:

1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Naive Bayes**
6. **Random Forest**

---

## 📊 Evaluation Metrics

To assess the performance of each model, the following metrics were used:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics provide a comprehensive view of each model's predictive capabilities, especially in dealing with imbalanced data.

---

## 🌐 Streamlit Web App

An interactive **Streamlit** app was created to make the lung cancer prediction tool accessible to users with no coding background.

🔗 [Live Streamlit App](https://lungcancerdetection-lakshaymalik.streamlit.app/)  
Deployed version of the streamlit.

---

## 📁 Project Structure

lung_cancer_detection/
├── lung_cancer_app.py # Streamlit web app
├── notebooks/ # Jupyter notebooks for EDA and model building
├── data/ # CSV Dataset files
├── README.md # Project documentation
└── requirements.txt # Python dependencies



---

## 🛠️ Requirements

Install required packages with:

```bash
pip install -r requirements.txt
```

## Streamlit

To run the app :
```bash
streamlit run lung_cancer_app.py
```
