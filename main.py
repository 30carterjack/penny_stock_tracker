import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt


class main(): 
    def __init__(self):
        pass

    def which_stock(self):
        # need to implement this through user input
        self.current_stock = "RVSN"
        self.stock = yf.Ticker(self.current_stock)

        return self.stock, self.current_stock
    

    def fetch_key_statistics(self):
        market_cap = self.stock.info.get('marketCap')
        volume = self.stock.info.get('volume')
        average_volume = self.stock.info.get('averageVolume')
        high_today = self.stock.info.get('dayHigh')
        low_today = self.stock.info.get('dayLow')

        formatted_market_cap = f"{market_cap:,}"
        formatted_volume = f"{volume:,}"
        formatted_average_volume = f"{average_volume:,}"

        print(f"{self.current_stock} key statistics: \n")
        print(f"Market cap: {formatted_market_cap}")
        print(f"Volume: {formatted_volume}")
        print(f"Average volume: {formatted_average_volume}")
        print(f"High today: ${high_today}")
        print(f"Low today: ${low_today}")


    def download_historical_data(self):
        # need to implement this through user input
        data = yf.download(tickers=self.current_stock, period='1y')
        data.head()
        print(data)

        return data


    def prepare_data(self, data):
        data['Target'] = data['Close'].shift(-1)
        data.dropna(inplace=True)

        X = data[['Close', 'Volume']]
        y = data['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test


    def train_model(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        print("training complete")

    def evaluate_model(self, X_test, y_test):
        if self.model:
            predictions = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)

            print(f"Evaluation metrics:\n")
            print(f"Mean Absolute Error: {mae}")
            print(f"Mean Squared Error: {mse}")
            print(f"Root Mean Squared Error: {rmse}")
        else:
            print("Model has not been trained yet.")
    '''data['Close'].plot(title="Closing price of Rail Vision (RVSN)")
    plt.show()'''

    def run(self):
        self.which_stock()
        self.fetch_key_statistics()

        data = self.download_historical_data()
        X_train, X_test, y_train, y_test = self.prepare_data(data)

        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)


if __name__ == "__main__":
    app = main()
    app.run()