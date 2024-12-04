import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data.csv")
data = pd.read_csv(data_path)

data_log = np.sqrt(data['Close'])
data_diff = data_log.diff().dropna()

def perform_adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values: {result[4]}")
    print("The series is stationary" if result[1] < 0.05 else "The series is non-stationary")

print("ADF Test on Square Root Transformed Data:")
perform_adf_test(data_diff)

plot_acf(data["Close"], lags=40)
plot_pacf(data["Close"], lags=40)
plt.show()

train_size = int(len(data) * 0.8)
train = data[:train_size].copy()
test = data[train_size:].copy()

train_series = train['Close']
test_series = test['Close']

p, d, q = 1, 2, 2
model = ARIMA(train_series, order=(p, d, q))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test_series))[:len(test_series)]

print(model_fit.summary())

residuals = model_fit.resid
residuals.plot()
residuals.plot(kind='kde')
plt.show()

print(f"AIC: {model_fit.aic}")
print(f"BIC: {model_fit.bic}")

mse = mean_squared_error(test_series, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_series, forecast)
r2 = r2_score(test_series, forecast)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

with open("result.txt", "w") as file:
    file.write(f"RMSE (arima): {rmse}\n")
    file.write(f"MAE (arima): {mae}\n")
    file.write(f"MSE (arima): {mse}\n")
    file.write(f"R² Score (arima): {r2}\n")
