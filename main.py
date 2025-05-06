import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import yfinance as yf
import datetime
import time
import random

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetworkTradingBot:
    def __init__(self, symbol='BTC-USD', start_date='2020-01-01', end_date=None, window_size=60, use_synthetic_data=False):
        """
        Initialize the trading bot with parameters
        
        Parameters:
        -----------
        symbol : str
            The ticker symbol to trade
        start_date : str
            Start date for historical data in 'YYYY-MM-DD' format
        end_date : str
            End date for historical data in 'YYYY-MM-DD' format (defaults to today)
        window_size : int
            Number of previous days to use for prediction
        use_synthetic_data : bool
            Whether to use synthetic data instead of fetching from API
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.datetime.now().strftime('%Y-%m-%d')
        self.window_size = window_size
        self.use_synthetic_data = use_synthetic_data
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predictions = None
        self.portfolio_value = []
        self.signals = []
        
    def fetch_data(self, max_retries=3, retry_delay=5):
        """
        Fetch historical data for the specified symbol with retry logic
        If fetching fails, generate synthetic data
        """
        if self.use_synthetic_data:
            print("Using synthetic data instead of fetching from API")
            return self.generate_synthetic_data()
            
        print(f"Fetching data for {self.symbol} from {self.start_date} to {self.end_date}")
        
        # Try to fetch data with retries
        for attempt in range(max_retries):
            try:
                self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
                
                if not self.data.empty:
                    print(f"Downloaded {len(self.data)} rows of data")
                    return self.data
                    
                print(f"No data returned on attempt {attempt+1}/{max_retries}, retrying...")
                time.sleep(retry_delay)
                
            except Exception as e:
                print(f"Error on attempt {attempt+1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        print("Failed to fetch data after multiple attempts. Generating synthetic data instead.")
        return self.generate_synthetic_data()
    
    def generate_synthetic_data(self, days=1000):
        """Generate synthetic price data for testing when API calls fail"""
        print(f"Generating {days} days of synthetic data for testing")
        
        # Create date range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate random walk for price data with some trend and volatility
        np.random.seed(42)  # For reproducibility
        
        # Start with a base price (e.g., 10000 for BTC)
        base_price = 10000
        
        # Generate daily returns with a slight positive drift
        daily_returns = np.random.normal(0.001, 0.02, size=len(date_range))
        
        # Add some autocorrelation to simulate market momentum
        for i in range(1, len(daily_returns)):
            daily_returns[i] = 0.7 * daily_returns[i] + 0.3 * daily_returns[i-1]
        
        # Calculate price series
        price_series = base_price * (1 + daily_returns).cumprod()
        
        # Add some cyclical patterns
        cycles = 0.1 * base_price * np.sin(np.linspace(0, 15, len(date_range)))
        price_series = price_series + cycles
        
        # Create volume data (correlated with absolute returns)
        volume = np.abs(daily_returns) * 1000000 + 500000 + np.random.normal(0, 100000, size=len(date_range))
        volume = np.maximum(volume, 100000)  # Ensure minimum volume
        
        # Create OHLC data
        high = price_series * (1 + np.abs(daily_returns) * 0.5)
        low = price_series * (1 - np.abs(daily_returns) * 0.5)
        open_price = price_series - (price_series - low) * np.random.random(size=len(date_range))
        
        # Create DataFrame
        synthetic_data = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price_series,
            'Volume': volume,
            'Adj Close': price_series  # Same as close for simplicity
        }, index=date_range)
        
        self.data = synthetic_data
        print(f"Generated {len(synthetic_data)} rows of synthetic data")
        return synthetic_data
    
    def prepare_data(self):
        """Prepare data for the neural network"""
        if self.data is None or self.data.empty:
            raise ValueError("No data available. Please fetch or generate data first.")
            
        # Extract close prices and convert to numpy array
        close_prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create features and target
        X, y = [], []
        
        for i in range(self.window_size, len(self.scaled_data)):
            X.append(self.scaled_data[i-self.window_size:i, 0])
            y.append(self.scaled_data[i, 0])
            
        # Convert to numpy arrays
        X, y = np.array(X), np.array(y)
        
        # Reshape X to fit LSTM input format: [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split data into training and testing sets (80% train, 20% test)
        split_idx = int(0.8 * len(X))
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def build_model(self):
        """Build the LSTM neural network model"""
        self.model = Sequential()
        
        # Add LSTM layer with 50 units and return sequences
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        
        # Add second LSTM layer
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        
        # Add dense layers
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        
        # Compile the model
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        return self.model
    
    def train_model(self, epochs=20, batch_size=32, verbose=1):
        """Train the neural network model"""
        if self.model is None:
            self.build_model()
            
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=verbose
        )
        
        return history
    
    def make_predictions(self):
        """Make predictions using the trained model"""
        self.predictions = self.model.predict(self.X_test)
        
        # Inverse transform to get actual price values
        self.predictions = self.scaler.inverse_transform(self.predictions)
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(((self.predictions - y_test_actual) ** 2)))
        print(f"Root Mean Squared Error: {rmse}")
        
        return self.predictions
    
    def generate_signals(self):
        """Generate buy/sell signals based on predictions"""
        # Get the actual test data for comparison
        test_data = self.data.iloc[-len(self.predictions):]
        test_data = test_data.reset_index()
        
        # Create a new dataframe with date, actual price, and predicted price
        signal_data = pd.DataFrame({
            'Date': test_data['Date'] if 'Date' in test_data.columns else test_data.index,
            'Actual': test_data['Close'].values,
            'Predicted': self.predictions.flatten()
        })
        
        # Generate signals: 1 for buy, -1 for sell, 0 for hold
        signal_data['Signal'] = 0
        
        # Strategy: Buy when predicted price is higher than actual price, sell when lower
        for i in range(1, len(signal_data)):
            if signal_data['Predicted'][i] > signal_data['Actual'][i-1] * 1.01:  # 1% threshold
                signal_data.loc[i, 'Signal'] = 1  # Buy
            elif signal_data['Predicted'][i] < signal_data['Actual'][i-1] * 0.99:  # 1% threshold
                signal_data.loc[i, 'Signal'] = -1  # Sell
        
        self.signals = signal_data
        return signal_data
    
    def backtest_strategy(self, initial_capital=10000):
        """Backtest the trading strategy"""
        if self.signals is None or len(self.signals) == 0:
            self.generate_signals()
            
        # Initialize portfolio and positions
        portfolio = pd.DataFrame({
            'Date': self.signals['Date'],
            'Close': self.signals['Actual'],
            'Signal': self.signals['Signal'],
            'Position': 0,
            'Cash': initial_capital,
            'Holdings': 0,
            'Total': initial_capital
        })
        
        position = 0
        cash = initial_capital
        
        # Simulate trading
        for i in range(len(portfolio)):
            # Update position based on signal
            if portfolio.loc[i, 'Signal'] == 1 and position == 0:  # Buy signal and no position
                # Calculate how many shares to buy (use 95% of cash to leave some for fees)
                shares_to_buy = int((cash * 0.95) / portfolio.loc[i, 'Close'])
                if shares_to_buy > 0:
                    position = shares_to_buy
                    cash -= position * portfolio.loc[i, 'Close']
            elif portfolio.loc[i, 'Signal'] == -1 and position > 0:  # Sell signal and has position
                # Sell all shares
                cash += position * portfolio.loc[i, 'Close']
                position = 0
                
            # Update portfolio values
            portfolio.loc[i, 'Position'] = position
            portfolio.loc[i, 'Cash'] = cash
            portfolio.loc[i, 'Holdings'] = position * portfolio.loc[i, 'Close']
            portfolio.loc[i, 'Total'] = portfolio.loc[i, 'Cash'] + portfolio.loc[i, 'Holdings']
        
        # Calculate returns
        portfolio['Returns'] = portfolio['Total'].pct_change()
        portfolio['Cumulative_Returns'] = (1 + portfolio['Returns']).cumprod()
        
        # Calculate metrics
        total_return = (portfolio['Total'].iloc[-1] / initial_capital - 1) * 100
        
        # Calculate days between first and last date for annualization
        # Handle different date formats safely
        try:
            # Calculate trading period in years for annualization
            if len(portfolio) > 1:
                # Try to determine the time period in days
                days_diff = 0
                
                # Check if Date is already a datetime
                if pd.api.types.is_datetime64_any_dtype(portfolio['Date']):
                    days_diff = (portfolio['Date'].iloc[-1] - portfolio['Date'].iloc[0]).days
                # Check if Date is a string that can be converted to datetime
                elif isinstance(portfolio['Date'].iloc[0], str):
                    try:
                        first_date = pd.to_datetime(portfolio['Date'].iloc[0])
                        last_date = pd.to_datetime(portfolio['Date'].iloc[-1])
                        days_diff = (last_date - first_date).days
                    except:
                        # If conversion fails, estimate based on number of rows (assuming daily data)
                        days_diff = len(portfolio)
                else:
                    # If all else fails, use the number of rows as an estimate
                    days_diff = len(portfolio)
                
                # Calculate annualized return
                years = max(days_diff / 365, 0.01)  # Avoid division by zero
                annual_return = ((1 + total_return/100) ** (1/years) - 1) * 100
            else:
                annual_return = 0
        except Exception as e:
            print(f"Error calculating annual return: {str(e)}")
            annual_return = 0
            days_diff = 0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * portfolio['Returns'].mean() / portfolio['Returns'].std() if portfolio['Returns'].std() > 0 else 0
        
        print(f"Total Return: {total_return:.2f}%")
        print(f"Trading Period: {days_diff} days")
        print(f"Annualized Return: {annual_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        self.portfolio_value = portfolio
        return portfolio
    
    def plot_results(self):
        """Plot the results of the trading strategy"""
        if self.portfolio_value is None or len(self.portfolio_value) == 0:
            self.backtest_strategy()
            
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot 1: Actual vs Predicted prices
        ax1.set_title('Actual vs Predicted Prices')
        ax1.plot(range(len(self.signals)), self.signals['Actual'], label='Actual Price', color='blue')
        ax1.plot(range(len(self.signals)), self.signals['Predicted'], label='Predicted Price', color='red', linestyle='--')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Buy/Sell Signals
        ax2.set_title('Buy/Sell Signals')
        ax2.plot(range(len(self.signals)), self.signals['Actual'], label='Price', color='blue')
        
        # Plot buy signals
        buy_signals = self.signals[self.signals['Signal'] == 1]
        if not buy_signals.empty:
            buy_indices = buy_signals.index
            ax2.scatter(buy_indices, buy_signals['Actual'], color='green', label='Buy Signal', marker='^', s=100)
        
        # Plot sell signals
        sell_signals = self.signals[self.signals['Signal'] == -1]
        if not sell_signals.empty:
            sell_indices = sell_signals.index
            ax2.scatter(sell_indices, sell_signals['Actual'], color='red', label='Sell Signal', marker='v', s=100)
        
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Portfolio Value
        ax3.set_title('Portfolio Value')
        ax3.plot(range(len(self.portfolio_value)), self.portfolio_value['Total'], label='Portfolio Value', color='purple')
        
        # Plot buy/sell points on portfolio value
        buy_points = []
        sell_points = []
        
        for i in range(1, len(self.portfolio_value)):
            if self.portfolio_value.loc[i, 'Position'] > self.portfolio_value.loc[i-1, 'Position']:
                buy_points.append(i)
            elif self.portfolio_value.loc[i, 'Position'] < self.portfolio_value.loc[i-1, 'Position']:
                sell_points.append(i)
        
        if buy_points:
            ax3.scatter(
                buy_points, 
                self.portfolio_value.loc[buy_points, 'Total'], 
                color='green', marker='^', s=100, label='Buy'
            )
            
        if sell_points:
            ax3.scatter(
                sell_points, 
                self.portfolio_value.loc[sell_points, 'Total'], 
                color='red', marker='v', s=100, label='Sell'
            )
        
        ax3.set_ylabel('Portfolio Value ($)')
        ax3.set_xlabel('Trading Days')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def run(self):
        """Run the complete trading bot workflow"""
        try:
            self.fetch_data()
            self.prepare_data()
            self.build_model()
            self.train_model(epochs=20)  # Reduced epochs for faster execution
            self.make_predictions()
            self.generate_signals()
            self.backtest_strategy()
            self.plot_results()
            
            return self.portfolio_value
        except Exception as e:
            print(f"Error running the trading bot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

# Example usage
if __name__ == "__main__":
    # Create a trading bot for Bitcoin with synthetic data option
    bot = NeuralNetworkTradingBot(symbol='BTC-USD', start_date='2020-01-01', use_synthetic_data=True)
    
    # Run the complete workflow
    results = bot.run()
    
    # Print final portfolio value if available
    if results is not None and not results.empty:
        print(f"Final Portfolio Value: ${results['Total'].iloc[-1]:.2f}")
    else:
        print("Trading simulation failed or produced no results.")
