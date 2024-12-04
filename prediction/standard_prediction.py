import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
import numpy as np

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data.csv")

data = pd.read_csv(data_path)
x = data.drop(['Close','Date'], axis=1)
y = data['Close']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

pipeline_lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
pipeline_svr = Pipeline([("scaler", StandardScaler()), ("model", SVR())])
pipeline_dt = Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeRegressor())])
pipeline_rf = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor())])

results = []

pipeline_lr.fit(x_train, y_train)
y_pred = pipeline_lr.predict(x_test)
r2_lr = r2_score(y_test, y_pred)
mae_lr = mean_absolute_error(y_test, y_pred)
mse_lr = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse_lr)
results.append({"Model": "Linear Regression", "R2": r2_lr, "MAE": mae_lr, "MSE": mse_lr, "RMSE": rmse})

pipeline_svr.fit(x_train, y_train)
y_pred_svr = pipeline_svr.predict(x_test)
r2_svr = r2_score(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
results.append({"Model": "SVM", "R2": r2_svr, "MAE": mae_svr, "MSE": mse_svr, "RMSE": rmse_svr})

pipeline_dt.fit(x_train, y_train)
y_pred_dt = pipeline_dt.predict(x_test)
r2_dt = r2_score(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
results.append({"Model": "Decision Tree", "R2": r2_dt, "MAE": mae_dt, "MSE": mse_dt, "RMSE": rmse_dt})

pipeline_rf.fit(x_train, y_train)
y_pred_rf = pipeline_rf.predict(x_test)
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
results.append({"Model": "Random Forest", "R2": r2_rf, "MAE": mae_rf, "MSE": mse_rf, "RMSE": rmse_rf})

results_df = pd.DataFrame(results)

with open("result.txt", "w") as file:
    for result in results:
        file.write(f"MSE ({result['Model']}): {result['MSE']:.4f}\n")
        file.write(f"RMSE ({result['Model']}): {result['RMSE']:.4f}\n")
        file.write(f"MAE ({result['Model']}): {result['MAE']:.4f}\n")
        file.write(f"R² Score ({result['Model']}): {result['R2']:.4f}\n")
        file.write("\n")
