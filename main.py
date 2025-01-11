import yfinance as yf

import matplotlib.pyplot as plt

current_stock = "RVSN"

stock = yf.Ticker(current_stock)

market_cap = stock.info.get('marketCap')
volume = stock.info.get('volume')
average_volume = stock.info.get('averageVolume')
high_today = stock.info.get('dayHigh')
low_today = stock.info.get('dayLow')

formatted_market_cap = f"{market_cap:,}"
formatted_volume = f"{volume:,}"
formatted_average_volume = f"{average_volume:,}"


print(f"{current_stock} key statistics\n")
print(f"Market cap: {formatted_market_cap}")
print(f"Volume: {formatted_volume}")
print(f"Average volume: {formatted_average_volume}")
print(f"High today: ${high_today}")
print(f"Low today: ${low_today}")

'''data['Close'].plot(title="Closing price of Rail Vision (RVSN)")
plt.show()'''