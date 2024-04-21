import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import smtplib, ssl
from datetime import datetime

def send_email(receiver_email, subject, message):
    port = 465  
    smtp_server = "smtp.gmail.com"
    sender_email = "your mail"#your mail   
    password = "set your password smtp"#abcd efgh ijkl mnop like this    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        email_message = f"Subject: {subject}\n\n{message}"
        server.sendmail(sender_email, receiver_email, email_message)

 
bitcoin_data = pd.read_csv('BTC-USD.csv')
dogecoin_data = pd.read_csv('DOGE-USD.csv')
xrp_data = pd.read_csv('XRP-USD.csv')
solana_data = pd.read_csv('SOL-USD.csv')

 
crypto_data = pd.DataFrame()
crypto_data['Bitcoin'] = bitcoin_data['Close']
crypto_data['Dogecoin'] = dogecoin_data['Close']
crypto_data['XRP'] = xrp_data['Close']
crypto_data['Solana'] = solana_data['Close']

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
        if password == "your password not as smtp pass":   
            st.session_state.display = True
            confirmation_subject = "Welcome to CRYPVISTA"
            confirmation_message = "You have successfully logged in."
            send_email(username, confirmation_subject, confirmation_message)
        else:
            st.error('Incorrect password. Please try again.')

    if st.session_state.display:
        display_prediction_app(username)   

def predict_prices(data):
    return data * 1.1   

def display_prediction_app(username):  
    selected_cryptos = st.multiselect('Select Cryptocurrencies', crypto_data.columns)

    if not selected_cryptos:
        st.warning('Please select at least one cryptocurrency.')
        return

    for selected_crypto in selected_cryptos:
        st.write(f"## {selected_crypto} - Actual Prices")
        st.line_chart(crypto_data[selected_crypto])

        predicted_prices = predict_prices(crypto_data[selected_crypto])

        fig, ax = plt.subplots()
        ax.plot(predicted_prices, label='Predicted Prices', color='orange')
        ax.set_title(f'{selected_crypto} - Predicted Prices')
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()

        st.pyplot(fig)

        actual_mean = crypto_data[selected_crypto].mean()
        predicted_mean = predicted_prices.mean()
        mean_diff = predicted_mean - actual_mean
        
        st.write(f"### Summary for {selected_crypto}")
        st.write(f"**Actual Mean Price:** {actual_mean}")
        st.write(f"**Predicted Mean Price:** {predicted_mean}")
        st.write(f"**Mean Price Difference:** {mean_diff}")

        desired_price = st.sidebar.number_input(f"Set desired price for {selected_crypto}", value=0.0, step=0.01)

        if desired_price != 0.0 and predicted_mean >= desired_price:
            st.sidebar.success(f"Desired price for {selected_crypto} has been reached!")
            
            notification_subject = f"{selected_crypto} - Desired Price Reached!"
            notification_message = f"The predicted mean price for {selected_crypto} has reached {desired_price}."
            send_email(username, notification_subject, notification_message)

if __name__ == "__main__":
    main()
