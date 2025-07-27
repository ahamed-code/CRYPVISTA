import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from streamlit_autorefresh import st_autorefresh
import smtplib, ssl
from datetime import datetime
import os
import hashlib

# ------------------ EMAIL FUNCTION ------------------
def send_email(receiver_email, subject, message):
    port = 465
    smtp_server = "smtp.gmail.com"
    sender_email = "hitmanbasheer@gmail.com"
    password = "koiirnkvnnmfmxei"  # Consider using environment variables
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            email_message = f"Subject: {subject}\n\n{message}"
            server.sendmail(sender_email, receiver_email, email_message)
    except Exception as e:
        st.warning(f"Email failed: {e}")

# ------------------ USER AUTH SYSTEM ------------------
USER_CSV = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USER_CSV):
        return pd.read_csv(USER_CSV)
    return pd.DataFrame(columns=["email", "password"])

def save_user(email, password):
    df = load_users()
    new_user = pd.DataFrame([[email, hash_password(password)]], columns=["email", "password"])
    df = pd.concat([df, new_user], ignore_index=True)
    df.to_csv(USER_CSV, index=False)

def check_login(email, password):
    df = load_users()
    hashed = hash_password(password)
    return not df[(df["email"] == email) & (df["password"] == hashed)].empty

def user_exists(email):
    df = load_users()
    return email in df["email"].values

# ------------------ CRYPTO PRICE FUNCTIONS ------------------
def get_current_price(crypto_name):
    try:
        file_path = f"{crypto_name}.csv"
        data = pd.read_csv(file_path)
        if 'Close' not in data.columns or data['Close'].dropna().empty:
            st.error(f"No valid 'Close' data in {file_path}.")
            return None
        return data['Close'].dropna().iloc[-1]
    except Exception as e:
        st.error(f"Error reading {crypto_name}.csv: {e}")
        return None

def load_crypto_data(crypto_name):
    try:
        data = pd.read_csv(f"{crypto_name}.csv")
        if "Close" not in data.columns or data["Close"].dropna().empty:
            raise ValueError(f"No 'Close' data in {crypto_name}.csv")
        return data["Close"].dropna()
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.Series()

def predict_prices(close_series, crypto_ticker):
    if len(close_series) < 70:
        raise ValueError("Insufficient data (< 70 rows)")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_series.values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=0)

    test_data = scaled_data[-60:]
    predicted_prices = []
    for _ in range(4):
        X_test = test_data[-60:].reshape(1, 60, 1)
        predicted_price = model.predict(X_test, verbose=0)
        predicted_prices.append(predicted_price[0, 0])
        test_data = np.append(test_data, predicted_price)[1:]

    return scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

# ------------------ MAIN APP DISPLAY ------------------
def display_prediction_app(username):
    st.subheader("🔄 Real-Time Prediction App")
    st.markdown("---")

    available_cryptos = ['Bitcoin', 'Ethereum', 'Solana']
    coin_map = {'Bitcoin': 'BTC-USD', 'Ethereum': 'ETH-USD', 'Solana': 'SOL-USD'}
    selected_name = st.selectbox("Choose Cryptocurrency", available_cryptos)
    selected_crypto = coin_map[selected_name]

    current_price = get_current_price(selected_crypto)
    if current_price is not None:
        st.sidebar.markdown(f"*Current price of {selected_crypto}: ${current_price:.2f}*")

    close_data = load_crypto_data(selected_crypto)
    if close_data.empty:
        return

    st.write(f"### {selected_crypto} - Actual Prices")
    st.line_chart(close_data)

    try:
        predicted_prices = predict_prices(close_data, selected_crypto)
        prediction_dates = pd.date_range(start=datetime.now(), periods=4, freq='D')
        predicted_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted Price': predicted_prices
        })

        st.write("### Predicted Prices (Next 4 Days)")
        st.dataframe(predicted_df)

        csv = predicted_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions as CSV", csv, "predictions.csv", "text/csv")

        fig, ax = plt.subplots()
        ax.plot(prediction_dates, predicted_prices, label='Predicted', color='orange')
        ax.set_title(f"{selected_crypto} - Forecast")
        ax.legend()
        st.pyplot(fig)

        price_threshold = st.number_input("🔔 Alert if price > ", value=60000.0)
        if current_price > price_threshold:
            st.warning(f"🚨 {selected_crypto} above ${price_threshold}!")
            if st.button("Send Email Alert"):
                send_email(username, "Price Alert", f"{selected_crypto} above ${price_threshold}")
                st.success("Alert email sent.")

    except ValueError as e:
        st.error(f"Prediction Error: {e}")

# ------------------ MAIN FUNCTION ------------------
def main():
    st.title('"CRYP VISTA"')
    st.image("IMAGE/CRYPTO.JPG", use_column_width=True)


    if 'display' not in st.session_state:
        st.session_state.display = False

    auth_mode = st.sidebar.radio("Choose Auth Mode:", ["Login", "Signup"])
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Signup":
        confirm = st.sidebar.text_input("Confirm Password", type="password")
        if st.sidebar.button("Create Account"):
            if password != confirm:
                st.error("❌ Passwords don't match.")
            elif user_exists(email):
                st.error("❌ Email already exists.")
            else:
                save_user(email, password)
                send_email(email, "Welcome to CRYP VISTA", "You're registered!")
                st.success("✅ Signup successful.")
    else:
        if st.sidebar.button("Login"):
            if check_login(email, password):
                st.session_state.display = True
                send_email(email, "Login Alert", "You logged in to CRYP VISTA.")
            else:
                st.error("❌ Invalid credentials.")

    if st.session_state.display:
        st_autorefresh(interval=60000, key="auto_refresh")
        display_prediction_app(email)

if __name__ == "__main__":
    main()
