# 🧠 Neural Network Trading Bot

A neural network-powered trading bot that uses LSTM (Long Short-Term Memory) to predict cryptocurrency prices and backtest a simple buy/sell strategy based on those predictions.

## 🚀 Features

- Fetches historical price data from Yahoo Finance (or generates synthetic data)
- Preprocesses and scales time series data
- Builds and trains a deep learning model using LSTM layers
- Predicts future prices
- Generates buy/sell signals
- Simulates trades and backtests strategy
- Visualizes predictions, signals, and portfolio performance

## 📈 Example Output

![Example Chart](https://via.placeholder.com/800x300.png?text=Sample+Chart+Output)

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Yahoo Finance (`yfinance`)

## 📦 Installation

```bash
git clone https://github.com/yourusername/neural-network-trading-bot.git
cd neural-network-trading-bot
pip install -r requirements.txt


⚙️ Usage
You can run the bot directly using:

bash
Copy
Edit
python trading_bot.py
To customize:

python
Copy
Edit
bot = NeuralNetworkTradingBot(
    symbol='BTC-USD',
    start_date='2020-01-01',
    use_synthetic_data=True  # Set to False to fetch real data
)
bot.run()
📊 Backtesting Metrics
The bot calculates and displays:

Total return

Annualized return

Sharpe ratio

Buy/Sell actions

Portfolio value over time

📁 Project Structure
bash
Copy
Edit
.
├── trading_bot.py       # Main bot logic
├── README.md            # This file
├── requirements.txt     # Python dependencies
✅ Requirements
numpy

pandas

matplotlib

scikit-learn

yfinance

tensorflow

You can install them with:

bash
Copy
Edit
pip install -r requirements.txt
📈 Strategy Logic
Buy Signal: If predicted price is >1% higher than previous close.

Sell Signal: If predicted price is >1% lower than previous close.

📌 Limitations
The model is for educational purposes and may not perform well in real markets.

No risk management or transaction fee modeling.

Real-world trading requires more advanced strategy design and evaluation.

📄 License
This project is licensed under the MIT License.

👤 Author
Your Name – Sadman Shoumik Rouf

Disclaimer: This software is for educational purposes only and is not financial advice.
