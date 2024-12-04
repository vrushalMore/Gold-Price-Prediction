import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pmd

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data.csv")
data = pd.read_csv(data_path)

plot_acf(data['Close'], lags=50)
plt.show()

plot_pacf(data['Close'], lags=50)
plt.show()

decomposition = seasonal_decompose(data['Close'], model='additive', period=12)
decomposition.plot()
plt.show()

def perform_adf_test(series):
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    print(f"Critical Values: {result[4]}")
    print("The series is stationary" if result[1] < 0.05 else "The series is non-stationary")

series = data['Close'].values
print("ADF Test on Original Series:")
perform_adf_test(series)

data_log = np.sqrt(data['Close'])
data_diff = data_log.diff().dropna()

print("ADF Test on Square Root Transformed Data:")
perform_adf_test(data_diff)

model = pmd.auto_arima(data['Close'], start_p=1, start_q=1, test='adf', m=12, seasonal=True, trace=True)
sarima = SARIMAX(data['Close'], order=(1, 1, 2), seasonal_order=(0, 0, 0, 12))
result = sarima.fit()

predicted = result.predict(start=1, end=len(data['Close']))

plt.figure(figsize=(20, 6))
plt.plot(data['Close'], label='Actual')
plt.plot(predicted, label='Predicted', linestyle='dashed')
plt.legend()
plt.show()

mse = mean_squared_error(data['Close'][1:], predicted[1:])
rmse = np.sqrt(mse)
mae = mean_absolute_error(data['Close'][1:], predicted[1:])
r2 = r2_score(data['Close'][1:], predicted[1:])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

with open("result.txt", "w") as file:
    file.write(f"MSE (sarimax): {mse}\n")
    file.write(f"RMSE (sarimax): {rmse}\n")
    file.write(f"MAE (sarimax): {mae}\n")
    file.write(f"R² Score (sarimax): {r2}\n")
