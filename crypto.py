import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import smtplib, ssl
from datetime import datetime
import yfinance as yf

def send_email(receiver_email, subject, message):
    """
    Sends an email notification.
    """
    port = 465
    smtp_server = "smtp.gmail.com"
    sender_email = "hitmanbasheer@gmail.com"  # your email
    password = "ggdw pfhm ogne kkpr"  # app-specific password
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        email_message = f"Subject: {subject}\n\n{message}"
        server.sendmail(sender_email, receiver_email, email_message)

def fetch_crypto_data(crypto_ticker):
    """
    Fetch cryptocurrency data using yfinance.
    """
    data = yf.download(crypto_ticker, period="1y", interval="1d")  # Last 1 year, daily intervals
    return data['Close']

def get_current_price(crypto_ticker):
    """
    Get the current price of the selected cryptocurrency using yfinance.
    """
    data = yf.download(crypto_ticker, period="1d", interval="1m")  # Latest price (1 day, 1-minute intervals)
    current_price = data['Close'].iloc[-1]  # Get the most recent close price
    return current_price

def predict_prices(data, crypto_ticker):
    """
    Predict prices for the next 4 days using Linear Regression.
    If the crypto is reduced (XRP, SOL, DOGE), apply a reduction factor.
    """
    # Define the reduction factor for XRP, Solana, and Dogecoin
    reduced_cryptos = ['XRP-USD', 'SOL-USD', 'DOGE-USD']
    reduction_factor = 0.95  # 5% reduction for the specified coins
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    # Prepare training data
    X = pd.DataFrame(range(len(scaled_data)))
    y = pd.DataFrame(scaled_data)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next 4 days
    future_days = pd.DataFrame(range(len(scaled_data), len(scaled_data) + 4))
    predicted_scaled_prices = model.predict(future_days)

    # If the cryptocurrency is in the reduced list, apply the reduction factor
    if crypto_ticker in reduced_cryptos:
        predicted_scaled_prices *= reduction_factor

    # Transform back to original scale
    predicted_prices = scaler.inverse_transform(predicted_scaled_prices)
    return predicted_prices.flatten()

def display_prediction_app(username):
    selected_cryptos = st.multiselect('Select Cryptocurrencies', ['BTC-USD', 'DOGE-USD', 'XRP-USD', 'SOL-USD'])

    if not selected_cryptos:
        st.warning('Please select at least one cryptocurrency.')
        return

    # Sidebar: Display desired price and tagline below it for each selected cryptocurrency
    for index, selected_crypto in enumerate(selected_cryptos):
        # Get the most recent closing price for the selected cryptocurrency
        current_price = get_current_price(selected_crypto)

        # Display the tagline below the text input for the desired price
        st.sidebar.markdown(f"**Today's price of {selected_crypto}: {current_price:.2f}**")
        
        # Ensure that the key is unique by combining the username, selected_crypto, and index
        desired_price_input = st.sidebar.text_input(f"Set desired price for {selected_crypto}",
                                                   value="", key=f"desired_{username}_{selected_crypto}_{index}")
        
        if desired_price_input:
            try:
                desired_price = float(desired_price_input)
                if desired_price != 0.0 and current_price >= desired_price:
                    st.sidebar.success(f"Desired price for {selected_crypto} has been reached!")

                    # Generate direct link to a cryptocurrency website (e.g., CoinMarketCap)
                    link = ""
                    if selected_crypto == 'BTC-USD':
                        link = "https://www.coinmarketcap.com/currencies/bitcoin/"
                    elif selected_crypto == 'DOGE-USD':
                        link = "https://www.coinmarketcap.com/currencies/dogecoin/"
                    elif selected_crypto == 'XRP-USD':
                        link = "https://www.coinmarketcap.com/currencies/ripple/"
                    elif selected_crypto == 'SOL-USD':
                        link = "https://www.coinmarketcap.com/currencies/solana/"

                    st.sidebar.markdown(f"[Click here to view {selected_crypto} on CoinMarketCap]({link})")

                    notification_subject = f"{selected_crypto} - Desired Price Reached!"
                    notification_message = f"The predicted mean price for {selected_crypto} has reached {desired_price}."
                    send_email(username, notification_subject, notification_message)
            except ValueError:
                st.sidebar.error("Please enter a valid number for the desired price.")

    # Main Display: Show predictions and charts for selected cryptocurrencies
    for selected_crypto in selected_cryptos:
        st.write(f"### {selected_crypto} - Actual Prices")
        st.line_chart(fetch_crypto_data(selected_crypto))

        # Predict the next 4 days
        try:
            predicted_prices = predict_prices(fetch_crypto_data(selected_crypto), selected_crypto)

            # Display predicted prices as a table
            prediction_dates = pd.date_range(start=datetime.now(), periods=4, freq='D')
            prediction_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Price': predicted_prices
            })
            st.write(f"### Predicted Prices for {selected_crypto} (Next 4 Days)")
            st.dataframe(prediction_df)

            # Plot predicted prices
            fig, ax = plt.subplots()
            ax.plot(predicted_prices, label='Predicted Prices (Next 4 Days)', color='orange')
            ax.set_title(f'{selected_crypto} - Predicted Prices')
            ax.set_xlabel('Days')
            ax.set_ylabel('Price')
            ax.legend()
            st.pyplot(fig)

            actual_mean = fetch_crypto_data(selected_crypto).mean()
            predicted_mean = sum(predicted_prices) / len(predicted_prices)
            mean_diff = predicted_mean - actual_mean

            st.write(f"### Summary for {selected_crypto}")
            st.write(f"**Actual Mean Price:** {actual_mean}")
            st.write(f"**Predicted Mean Price:** {predicted_mean}")
            st.write(f"**Mean Price Difference:** {mean_diff}")

        except ValueError as e:
            st.error(f"Error predicting prices for {selected_crypto}: {e}")

    # Final paragraph explaining the predictions with promotion for LSTM
    st.write("""
    ### How the Predictions Are Made
    The price predictions for each cryptocurrency are currently based on the most recent closing prices using **Linear Regression**. This method estimates future prices based on the historical price trends. While this approach is simple and effective for short-term forecasts, there is a more advanced and accurate method for modeling time-series data like cryptocurrency prices: the **LSTM (Long Short-Term Memory)** model.

    **LSTM models** are a type of recurrent neural network (RNN) designed to learn from sequences of data over time. Unlike Linear Regression, which assumes a linear relationship, LSTM can capture complex, non-linear relationships and long-term dependencies in the data. This makes LSTM a great choice for predicting volatile market prices like cryptocurrencies.

    With an LSTM model, we can better account for patterns, trends, and seasonality in the data, providing more robust and accurate predictions. While the Linear Regression model gives us a quick estimate based on recent trends, an LSTM model would enable us to predict price movements over a longer horizon by learning from a larger context of historical data.

    Using **LSTM** for price prediction can significantly enhance the accuracy of forecasting in the cryptocurrency market, providing investors with a more reliable tool for making data-driven decisions.
    """)

def main():
    violet_bg = """
        <style>
            body {
                background-color: violet;
            }
        </style>
    """
    st.markdown(violet_bg, unsafe_allow_html=True)

    st.title('\t\t"CRYP VISTA"')
    image_path = 'CRYPTO.JPG'
    st.image(image_path, caption='Crypto', use_column_width=True)

    if 'display' not in st.session_state:
        st.session_state.display = False

    username = st.sidebar.text_input('Email', key="email")
    password = st.sidebar.text_input('Password', type='password', key="password")

    if st.sidebar.button('Login'):
        if password == "1432":
            st.session_state.display = True
            confirmation_subject = "Welcome to CRYPVISTA"
            confirmation_message = "You have successfully logged in."
            send_email(username, confirmation_subject, confirmation_message)
        else:
            st.error('Incorrect password. Please try again.')

    if st.session_state.display:
        display_prediction_app(username)

if __name__ == "__main__":
    main()
