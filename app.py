import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from model_utils import StockLSTM
from data_collection import mean_std_dict
from datetime import timedelta

# Constants
SEQ_LEN = 20
HIDDEN_SIZE = 64
TICKER_EMBED_SIZE = 8
MODEL_PATH = "stock_lstm_model.pth"
DATA_PATH = "normalized_shuffled_combined_data.csv"

# Load Data
df = pd.read_csv(DATA_PATH)
df['Date'] = pd.to_datetime(df['Date'])

# Encode tickers
ticker_encoder = LabelEncoder()
df['ticker_id'] = ticker_encoder.fit_transform(df['Ticker'])

# Load model
model = StockLSTM(5, HIDDEN_SIZE, len(ticker_encoder.classes_), TICKER_EMBED_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("📈 Stock Price Predictor")
ticker = st.selectbox("Select Ticker", sorted(df['Ticker'].unique()))

# Date input (for display only)
ticker_df = df[df['Ticker'] == ticker].sort_values(by='Date')
latest_date = ticker_df['Date'].max()
default_pred_date = latest_date + timedelta(days=1)

pred_date = st.date_input(
    "Select prediction date (optional)",
    value=default_pred_date,
    min_value=default_pred_date,
    help="Used for displaying prediction; model always uses latest available data."
)

# Prediction
if st.button("Predict Next Close Price"):
    ticker_id = ticker_encoder.transform([ticker])[0]

    if len(ticker_df) >= SEQ_LEN:
        latest_seq = ticker_df[['Open', 'High', 'Low', 'Close', 'Volume']].values[-SEQ_LEN:]
        input_seq = torch.tensor(latest_seq, dtype=torch.float32).unsqueeze(0)
        ticker_tensor = torch.tensor([ticker_id], dtype=torch.long)

        with torch.no_grad():
            pred_norm = model(input_seq, ticker_tensor).item()

        # Denormalize
        mean = mean_std_dict[ticker]["mean"]["Close"]
        std = mean_std_dict[ticker]["std"]["Close"]
        pred_price = pred_norm * std + mean

        st.success(f"🔮 Predicted Normalized Close: {pred_norm:.4f}")
        st.success(f"💵 Predicted Actual Close Price for {pred_date}: {pred_price:.2f}")

        # 📊 Plot
        # last_dates = ticker_df['Date'].iloc[-SEQ_LEN:].tolist()
        # last_closes = ticker_df['Close'].iloc[-SEQ_LEN:].tolist()

        # # Add predicted date and price
        # all_dates = last_dates + [pred_date]
        # all_prices = last_closes + [pred_price]

        # fig, ax = plt.subplots()
        # ax.plot(last_dates, last_closes, marker='o', label="Actual Close Price (last 10)")
        # ax.plot(pred_date, pred_price, marker='X', color='red', label="Predicted Price")
        # ax.set_title(f"{ticker} Close Price Prediction")
        # ax.set_xlabel("Date")
        # ax.set_ylabel("Price")
        # ax.legend()
        # ax.grid(True)
        # plt.xticks(rotation=30)

        # st.pyplot(fig)

        # 📊 Plot with denormalized last 10 prices
        #mean = mean_std_dict[ticker]["mean"]["Close"]
        #std = mean_std_dict[ticker]["std"]["Close"]

        last_dates = ticker_df['Date'].iloc[-SEQ_LEN:].tolist()
        last_closes_norm = ticker_df['Close'].iloc[-SEQ_LEN:].tolist()
        last_closes = [(val * std + mean) for val in last_closes_norm]

        # Add predicted date and price
        all_dates = last_dates + [pred_date]
        all_prices = last_closes + [pred_price]

        fig, ax = plt.subplots()
        ax.plot(last_dates, last_closes, marker='o', label="Actual Close Price (last 10)")
        ax.plot(pred_date, pred_price, marker='X', color='red', label="Predicted Price")
        ax.set_title(f"{ticker} Close Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=30)

        st.pyplot(fig)


    else:
        st.warning("Not enough data to make prediction!")
