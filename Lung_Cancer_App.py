import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset and clean column names
@st.cache_data
def load_data():
    df = pd.read_csv("survey_lung_cancer.csv")

    # Strip column name spaces
    df.columns = df.columns.str.strip()

    # Map categorical to numeric
    df["GENDER"] = df["GENDER"].map({'M': 0, 'F': 1})
    df["LUNG_CANCER"] = df["LUNG_CANCER"].map({'YES': 1, 'NO': 0})
    return df
df = load_data()

# Sidebar inputs
st.sidebar.header("Patient Information")

def user_input_features():
    GENDER = st.sidebar.selectbox("Gender", ['Male', 'Female'])
    AGE = st.sidebar.slider("Age", 20, 100, 50)
    SMOKING = st.sidebar.selectbox("Smoking", [0, 1])
    YELLOW_FINGERS = st.sidebar.selectbox("Yellow Fingers", [0, 1])
    ANXIETY = st.sidebar.selectbox("Anxiety", [0, 1])
    PEER_PRESSURE = st.sidebar.selectbox("Peer Pressure", [0, 1])
    CHRONIC_DISEASE = st.sidebar.selectbox("Chronic Disease", [0, 1])
    FATIGUE = st.sidebar.selectbox("Fatigue", [0, 1])
    ALLERGY = st.sidebar.selectbox("Allergy", [0, 1])
    WHEEZING = st.sidebar.selectbox("Wheezing", [0, 1])
    ALCOHOL_CONSUMING = st.sidebar.selectbox("Alcohol Consuming", [0, 1])
    COUGHING = st.sidebar.selectbox("Coughing", [0, 1])
    SHORTNESS_OF_BREATH = st.sidebar.selectbox("Shortness of Breath", [0, 1])
    SWALLOWING_DIFFICULTY = st.sidebar.selectbox("Swallowing Difficulty", [0, 1])
    CHEST_PAIN = st.sidebar.selectbox("Chest Pain", [0, 1])

    GENDER = 0 if GENDER == 'Male' else 1

    data = {
        'GENDER': GENDER,
        'AGE': AGE,
        'SMOKING': SMOKING,
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ANXIETY': ANXIETY,
        'PEER_PRESSURE': PEER_PRESSURE,
        'CHRONIC DISEASE': CHRONIC_DISEASE,
        'FATIGUE': FATIGUE,
        'ALLERGY': ALLERGY,
        'WHEEZING': WHEEZING,
        'ALCOHOL CONSUMING': ALCOHOL_CONSUMING,
        'COUGHING': COUGHING,
        'SHORTNESS OF BREATH': SHORTNESS_OF_BREATH,
        'SWALLOWING DIFFICULTY': SWALLOWING_DIFFICULTY,
        'CHEST PAIN': CHEST_PAIN
    }


    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Features and labels
X = df.drop(columns=["LUNG_CANCER"])
y = df["LUNG_CANCER"]

# Split and train KNN model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Prediction
prediction = knn.predict(input_df)[0]
prediction_text = "Lung Cancer Risk Detected" if prediction == 1 else "No Lung Cancer Risk Detected"

# Display input & prediction
st.title("Lung Cancer Prediction App")
st.write("### Patient Input Data")
st.write(input_df)

st.write("## Prediction")
st.success(prediction_text)

# Visualizations
st.write("## Visualizations")

# Heatmap
st.write("### Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax1)
st.pyplot(fig1)

# Class Distribution
st.write("### Lung Cancer Class Distribution")
fig2, ax2 = plt.subplots()
sns.countplot(x="LUNG_CANCER", data=df, ax=ax2)
st.pyplot(fig2)

# Confusion Matrix on test data
st.write("### Confusion Matrix on Test Set")
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
fig3, ax3 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")
st.pyplot(fig3)