import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

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

sarima_model = ARIMA(data['Close'], order=(1, 1, 2), seasonal_order=(0, 0, 0, 12))
sarima_result = sarima_model.fit()

predicted = sarima_result.predict(start=1, end=len(data['Close']))

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

with open("result_sarima.txt", "w") as file:
    file.write(f"MSE (SARIMA): {mse}\n")
    file.write(f"RMSE (SARIMA): {rmse}\n")
    file.write(f"MAE (SARIMA): {mae}\n")
    file.write(f"R² Score (SARIMA): {r2}\n")
