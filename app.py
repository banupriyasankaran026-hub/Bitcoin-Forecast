import streamlit as st
import pandas as pd
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from models import BitcoinLSTM, BitcoinCNN, BitcoinRNN, BitcoinTransformer

st.set_page_config(
        page_title="Cryptocast",
        layout="wide",
        initial_sidebar_state="expanded"
)

page=st.sidebar.radio("NAVIGATE",["Main Page","Dashboard","Model Comparison"])

@st.cache_data
def load_data():
    return pd.read_csv("Bitcoin Historical Data (1).csv")

df = load_data()

df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
df = df.sort_values(by="Date", ascending=True)

df['Price'] = df['Price'].astype(str).str.replace(',', '')
df['Price'] = df['Price'].replace('-', np.nan).astype(float)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

if page=="Main Page":
    st.title("CryptoCast: Multi-Horizon Bitcoin Price Forecasting Using Deep Learning")
    st.divider()
    st.markdown("""
              - This project focuses on building deep learning models 
                that learn from historical price sequences to forecast Bitcoin prices over 
                multiple future horizons.
              - It trains and compare deep learning architectures to predict:
                1-Day Price
                3-Day Price
                7-Day Price
            """)
    
elif page=="Dashboard":
    st.title("Dashboard")
    st.title("Bitcoin Price Forecasting Dashboard")

    col1, col2 = st.columns([2, 2])

    with col1:
        model_choice = st.selectbox("Model",["Select Model","LSTM", "CNN", "RNN", "Transformer"])

    with col2:
        horizon_choice = st.selectbox("Forecast Horizon",["Select","1-day", "3-day", "7-day"])

    def create_sequences(data, seq_length=60, horizons=[1,3,7]):
        X, y = [], []
        for i in range(len(data) - seq_length - max(horizons)):
            seq_x = data[i:i+seq_length]
            seq_y = [data[i+seq_length+h-1][0] for h in horizons]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    X, y = create_sequences(scaler.transform(df[["Price"]].values), seq_length=60)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    if model_choice == "LSTM":
        model = BitcoinLSTM()
        model.load_state_dict(torch.load("bitcoin_lstm.pth"))
    elif model_choice == "CNN":
        model = BitcoinCNN()
        model.load_state_dict(torch.load("bitcoin_cnn.pth"))
    elif model_choice == "RNN":
        model = BitcoinRNN()
        model.load_state_dict(torch.load("bitcoin_rnn.pth"))
    else:
        model = BitcoinTransformer()
        model.load_state_dict(torch.load("bitcoin_trans.pth"))

    model.eval()
    with torch.no_grad():
        preds = model(X_t).numpy()

    pred_1d = scaler.inverse_transform(preds[:,0].reshape(-1,1))
    pred_3d = scaler.inverse_transform(preds[:,1].reshape(-1,1))
    pred_7d = scaler.inverse_transform(preds[:,2].reshape(-1,1))

    actual_1d = scaler.inverse_transform(y_t[:,0].reshape(-1,1))
    actual_3d = scaler.inverse_transform(y_t[:,1].reshape(-1,1))
    actual_7d = scaler.inverse_transform(y_t[:,2].reshape(-1,1))

    if horizon_choice=="1-day":
        mae = mean_absolute_error(actual_1d,pred_1d)
        rmse = np.sqrt(mean_squared_error(actual_1d,pred_1d))
        r2 = r2_score(actual_1d,pred_1d)
        st.metric("MAE (1-day)", f"{mae:.2f}")
        st.metric("RMSE (1-day)", f"{rmse:.2f}")
        st.metric("R2 (1-day)", f"{r2:.2f}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(actual_1d, label="Actual 1-day")
        ax.plot(pred_1d, label="Predicted 1-day")
        ax.legend()
        st.pyplot(fig)

    elif horizon_choice=="3-day":
        mae = mean_absolute_error(actual_3d,pred_3d)
        rmse = np.sqrt(mean_squared_error(actual_3d,pred_3d))
        r2 = r2_score(actual_3d,pred_3d)
        st.metric("MAE (3-day)", f"{mae:.2f}")
        st.metric("RMSE (3-day)", f"{rmse:.2f}")
        st.metric("R2 (3-day)", f"{r2:.2f}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(actual_3d, label="Actual 3-day")
        ax.plot(pred_3d, label="Predicted 3-day")
        ax.legend()
        st.pyplot(fig)

    elif horizon_choice=="7-day":
        mae = mean_absolute_error(actual_7d,pred_7d)
        rmse = np.sqrt(mean_squared_error(actual_7d,pred_7d))
        r2 = r2_score(actual_7d,pred_7d)
        st.metric("MAE (7-day)", f"{mae:.2f}")
        st.metric("RMSE (7-day)", f"{rmse:.2f}")
        st.metric("R2 (7-day)", f"{r2:.2f}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(actual_7d, label="Actual 7-day")
        ax.plot(pred_7d, label="Predicted 7-day")
        ax.legend()
        st.pyplot(fig)

elif page=="Model Comparison":
    st.title("Model Comparison")
    comparison = pd.read_csv("model_comparison.csv")

    tab1, tab2, tab3 = st.tabs(["1-Day", "3-Day", "7-Day"])

    with tab1:
        st.subheader("1-Day Forecast")
        st.bar_chart(
            comparison.set_index("Model")[["1-Day MAE", "1-Day RMSE"]]
        )

    with tab2:
        st.subheader("3-Day Forecast")
        st.bar_chart(
            comparison.set_index("Model")[["3-Day MAE", "3-Day RMSE"]]
        )

    with tab3:
        st.subheader("7-Day Forecast")
        st.bar_chart(
            comparison.set_index("Model")[["7-Day MAE", "7-Day RMSE"]]
        )
