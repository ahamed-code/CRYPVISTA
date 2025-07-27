# 🚀 CRYPVISTA – AI-Based Cryptocurrency Predictor with Streamlit

**CRYPVISTA** is a real-time, AI-powered cryptocurrency prediction and visualization app built using Streamlit. It offers price forecasting for top cryptocurrencies like Bitcoin, Ethereum, and Solana using LSTM (Long Short-Term Memory) neural networks. The app features built-in user authentication, email alerts, and real-time auto-refreshing dashboards.

---

## 🧠 Features

- 🔐 **User Authentication**: Signup & login system with password hashing (`SHA256`) stored in `users.csv`.
- 📬 **Email Alerts**: Send notifications when a predicted price crosses a user-defined threshold (using Gmail SMTP).
- 📊 **Cryptocurrency Forecasting**: Uses LSTM-based model to predict next 4 days' prices.
- 📈 **Visualizations**: Displays historical close prices and predicted values using matplotlib and Streamlit charts.
- 💾 **Download Option**: Export predictions to a CSV.
- 🔁 **Auto-Refresh**: Keeps predictions updated in real time.

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `Streamlit` | UI & web app framework |
| `TensorFlow / Keras` | Deep learning (LSTM model) |
| `Scikit-learn` | Scaling and preprocessing |
| `Matplotlib` | Plotting prediction graphs |
| `pandas`, `numpy` | Data manipulation |
| `smtplib`, `ssl` | Email functionality |
| `hashlib`, `os` | Security & user storage |

---

## 📂 Folder Structure

```

Crypvista/
├── CRYPTO.JPG
├── Bitcoin.csv / Ethereum.csv / Solana.csv
├── users.csv
├── app.py (or main script file)
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/ahamed-code/Crypvista.git
cd Crypvista
````

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 🧪 Sample Credentials (for testing)

> You can sign up from the app, but here's an example login:

* **Email**: `demo@crypvista.com`
* **Password**: `password123`

---

## 📬 SMTP Email Setup

The app currently uses Gmail's SMTP:

* Sender email: `hitmanbasheer@gmail.com`
* App password (used in code): `koiirnkvnnmfmxei`

> ⚠️ **Important**: Store credentials securely (use environment variables or `.env` files in production).

---

## ✅ To-Do / Future Enhancements

* [ ] Add live crypto API integration (e.g., CoinGecko)
* [ ] Use pre-trained model or fine-tune LSTM with more epochs
* [ ] Add logout & session expiry
* [ ] Deploy on Streamlit Cloud / HuggingFace Spaces

---

## 📸 Screenshots

![Dashboard](assets/dashboard.png)
*Real-time price chart with prediction overlay*

---

## 📄 License

MIT License © [Basheer Ahamed](https://github.com/ahamed-code)

---

## 🙌 Credits

* Streamlit for making UI effortless
* TensorFlow for LSTM support
* Matplotlib & Pandas for smooth data handling

```
