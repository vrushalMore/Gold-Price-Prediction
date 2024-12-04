import pandas as pd
import warnings
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing

warnings.simplefilter('ignore')

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data.csv")
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'])
time_series = data['Close']

model_single = SimpleExpSmoothing(time_series)
model_single_fit = model_single.fit()
forecast_single = model_single_fit.forecast(12)

forecast_dates = pd.date_range(data['Date'].iloc[-1], periods=13, freq='D')[1:]
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], time_series, label='Original Data')
plt.plot(data['Date'], model_single_fit.fittedvalues, label='Fitted Values')
plt.plot(forecast_dates, forecast_single, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Close values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

actual = time_series.values
predicted = model_single_fit.fittedvalues.values
mae_1 = mean_absolute_error(actual, predicted)
mse_1 = mean_squared_error(actual, predicted)
rmse_1 = np.sqrt(mse_1)
r2_1 = r2_score(actual, predicted)

model_double = Holt(time_series)
model_double_fit = model_double.fit()
forecast_double = model_double_fit.forecast(12)

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], time_series, label='Original Data')
plt.plot(data['Date'], model_double_fit.fittedvalues, label='Fitted Values')
plt.plot(forecast_dates, forecast_double, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Close values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

mae_2 = mean_absolute_error(actual, model_double_fit.fittedvalues.values)
mse_2 = mean_squared_error(actual, model_double_fit.fittedvalues.values)
rmse_2 = np.sqrt(mse_2)
r2_2 = r2_score(actual, model_double_fit.fittedvalues.values)

model_triple = ExponentialSmoothing(time_series, seasonal_periods=12, trend='add', seasonal='add')
model_triple_fit = model_triple.fit()
forecast_triple = model_triple_fit.forecast(12)

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], time_series, label='Original Data')
plt.plot(data['Date'], model_triple_fit.fittedvalues, label='Fitted Values')
plt.plot(forecast_dates, forecast_triple, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Close values')
plt.legend()
plt.xticks(rotation=45)
plt.show()

mae_3 = mean_absolute_error(actual, model_triple_fit.fittedvalues.values)
mse_3 = mean_squared_error(actual, model_triple_fit.fittedvalues.values)
rmse_3 = np.sqrt(mse_3)
r2_3 = r2_score(actual, model_triple_fit.fittedvalues.values)

with open("forecast_results.txt", "w") as file:
    file.write(f"Simple Exponential Smoothing (SES):\n")
    file.write(f"MAE: {mae_1:.4f}\n")
    file.write(f"MSE: {mse_1:.4f}\n")
    file.write(f"RMSE: {rmse_1:.4f}\n")
    file.write(f"R2: {r2_1:.4f}\n\n")
    
    file.write(f"Holt’s Linear Trend Model:\n")
    file.write(f"MAE: {mae_2:.4f}\n")
    file.write(f"MSE: {mse_2:.4f}\n")
    file.write(f"RMSE: {rmse_2:.4f}\n")
    file.write(f"R2: {r2_2:.4f}\n\n")
    
    file.write(f"Triple Exponential Smoothing (Holt-Winters):\n")
    file.write(f"MAE: {mae_3:.4f}\n")
    file.write(f"MSE: {mse_3:.4f}\n")
    file.write(f"RMSE: {rmse_3:.4f}\n")
    file.write(f"R2: {r2_3:.4f}\n")
