import pandas as pd
from prophet import Prophet
import warnings
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

warnings.simplefilter('ignore')

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data.csv")
data = pd.read_csv(data_path)

data['Date'] = pd.to_datetime(data['Date'])
data = data[['Date', 'Close']]
data = data.rename(columns={'Date': 'ds', 'Close': 'y'})

train_data = data.iloc[:-int(len(data) * 0.2)]
test_data = data.iloc[-int(len(data) * 0.2):]

model = Prophet(interval_width=0.95, daily_seasonality=True)
model.fit(train_data)

future = model.make_future_dataframe(periods=len(test_data), freq='D')
forecast = model.predict(future)

forecasted = forecast[-len(test_data):]
predicted = forecasted['yhat'].values
actual = test_data['y'].values

mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

results_path = os.path.join(current_dir, "prophet_results.txt")
with open(results_path, "w") as file:
    file.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
    file.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    file.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    file.write(f"R² Score: {r2:.4f}\n")
