# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DISABLE_META_OPTIMIZER"] = "1"

# -----------------------------
# PAGE CONFIG & STYLING
# -----------------------------
st.set_page_config(page_title="Car Price Prediction Dashboard", page_icon="🚗", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f5f7fa;}
    .stButton>button {
        background-color: #4CAF50;
        color:white;
        font-size:16px;
        height:3em;
        width:100%;
        border-radius: 10px;
    }
    .stDataFrame {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .highlight {background-color: #ffffb3; font-weight: bold; padding:5px; border-radius:5px;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🚗 Car Price Prediction Dashboard")
st.markdown("<h5 style='color:#555;'>Compare Linear Regression, Random Forest & Neural Network Models</h5>", unsafe_allow_html=True)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("car_price_dataset_medium.csv")
    return df

df = load_data()
st.subheader("Dataset Sample")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# PREPROCESS DATA
# -----------------------------
target = "Price_USD"
X = df.drop(columns=[target])
y = df[target]

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

X_processed = preprocess.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# -----------------------------
# TRAIN MODELS
# -----------------------------
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# Neural Network
input_dim = X_train.shape[1]
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
y_pred_nn = nn_model.predict(X_test).flatten()
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)

# -----------------------------
# SHOW METRICS
# -----------------------------
st.subheader("📊 Model Performance Metrics")
metrics = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Neural Network"],
    "RMSE": [rmse_lr, rmse_rf, rmse_nn],
    "R²": [r2_lr, r2_rf, r2_nn]
})
st.table(metrics.style.applymap(lambda x: 'background-color: #ffffb3' if isinstance(x, (int,float)) else ''))

# -----------------------------
# VISUAL COMPARISON
# -----------------------------
st.subheader("📈 Predicted vs Actual Prices")
fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(y_test, y_pred_lr, color='blue', alpha=0.6, label='Linear Regression')
ax.scatter(y_test, y_pred_rf, color='green', alpha=0.6, label='Random Forest')
ax.scatter(y_test, y_pred_nn, color='red', alpha=0.6, label='Neural Network')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("Predicted vs Actual Prices", fontsize=16, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# -----------------------------
# NEW CAR PREDICTION
# -----------------------------
st.subheader("🚘 Predict Price for a New Car")

with st.form("new_car_form"):
    Brand = st.selectbox("Brand", df['Brand'].unique())
    Model_Year = st.number_input("Model Year", 2000, 2025, 2020)
    Kilometers_Driven = st.number_input("Kilometers Driven", 0, 500000, 50000)
    Fuel_Type = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
    Transmission = st.selectbox("Transmission", df['Transmission'].unique())
    Owner_Type = st.selectbox("Owner Type", df['Owner_Type'].unique())
    Engine_CC = st.number_input("Engine CC", 500, 5000, 1500)
    Max_Power_bhp = st.number_input("Max Power (bhp)", 20.0, 600.0, 120.0)
    Mileage_kmpl = st.number_input("Mileage (kmpl)", 5.0, 50.0, 18.0)
    Seats = st.number_input("Seats", 2, 10, 5)
    
    submitted = st.form_submit_button("Predict Price")
    
    if submitted:
        new_car = pd.DataFrame({
            'Brand':[Brand],
            'Model_Year':[Model_Year],
            'Kilometers_Driven':[Kilometers_Driven],
            'Fuel_Type':[Fuel_Type],
            'Transmission':[Transmission],
            'Owner_Type':[Owner_Type],
            'Engine_CC':[Engine_CC],
            'Max_Power_bhp':[Max_Power_bhp],
            'Mileage_kmpl':[Mileage_kmpl],
            'Seats':[Seats]
        })
        new_car_processed = preprocess.transform(new_car)
        
        price_lr = lr_model.predict(new_car_processed)[0]
        price_rf = rf_model.predict(new_car_processed)[0]
        price_nn = nn_model.predict(new_car_processed).flatten()[0]
        
        st.markdown("<div class='highlight'>**Predicted Price:**</div>", unsafe_allow_html=True)
        st.write(f"Linear Regression: ${price_lr:,.2f}")
        st.write(f"Random Forest: ${price_rf:,.2f}")
        st.write(f"Neural Network: ${price_nn:,.2f}")
