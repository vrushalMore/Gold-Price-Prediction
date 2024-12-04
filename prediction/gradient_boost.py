import pandas as pd
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor
import numpy as np

warnings.simplefilter('ignore')

current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, "..", "data.csv")
data = pd.read_csv(data_path)

x = data.drop(['Close', 'Date'], axis=1)
y = data['Close']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dtrain_reg = xgb.DMatrix(x_train, label=y_train)
dtest_reg = xgb.DMatrix(x_test, label=y_test)
params_xgb = {'objective': 'reg:squarederror', 'max_depth': 3, 'learning_rate': 0.1}
model_xgb = xgb.train(params=params_xgb, dtrain=dtrain_reg, num_boost_round=50)
prediction_xgb = model_xgb.predict(dtest_reg)
mse_xgb = mean_squared_error(y_test, prediction_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, prediction_xgb)
r2_xgb = r2_score(y_test, prediction_xgb)

model_gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, random_state=42, max_features=5)
model_gbr.fit(x_train, y_train)
prediction_gbr = model_gbr.predict(x_test)
mse_gbr = mean_squared_error(y_test, prediction_gbr)
rmse_gbr = np.sqrt(mse_gbr)
mae_gbr = mean_absolute_error(y_test, prediction_gbr)
r2_gbr = r2_score(y_test, prediction_gbr)

params_lgbm = {'task': 'train', 'boosting': 'gbdt', 'objective': 'regression', 'num_leaves': 10, 'learning_rate': 0.05}
lgb_train = lgb.Dataset(x_train, label=y_train)
lgb_eval = lgb.Dataset(x_test, label=y_test, reference=lgb_train)
model_lgbm = lgb.train(params=params_lgbm, train_set=lgb_train, valid_sets=[lgb_train, lgb_eval], num_boost_round=100)
prediction_lgbm = model_lgbm.predict(x_test)
mse_lgbm = mean_squared_error(y_test, prediction_lgbm)
rmse_lgbm = np.sqrt(mse_lgbm)
mae_lgbm = mean_absolute_error(y_test, prediction_lgbm)
r2_lgbm = r2_score(y_test, prediction_lgbm)

model_cat = CatBoostRegressor(loss_function='RMSE', verbose=0)
model_cat.fit(x_train, y_train)
prediction_cat = model_cat.predict(x_test)
mse_cat = mean_squared_error(y_test, prediction_cat)
rmse_cat = np.sqrt(mse_cat)
mae_cat = mean_absolute_error(y_test, prediction_cat)
r2_cat = r2_score(y_test, prediction_cat)

results_path = os.path.join(current_dir, "model_results.txt")
with open(results_path, "w") as file:
    file.write("XGBoost:\n")
    file.write(f"  MSE: {mse_xgb:.4f}\n")
    file.write(f"  RMSE: {rmse_xgb:.4f}\n")
    file.write(f"  MAE: {mae_xgb:.4f}\n")
    file.write(f"  R2: {r2_xgb:.4f}\n\n")
    
    file.write("Gradient Boosting:\n")
    file.write(f"  MSE: {mse_gbr:.4f}\n")
    file.write(f"  RMSE: {rmse_gbr:.4f}\n")
    file.write(f"  MAE: {mae_gbr:.4f}\n")
    file.write(f"  R2: {r2_gbr:.4f}\n\n")
    
    file.write("LightGBM:\n")
    file.write(f"  MSE: {mse_lgbm:.4f}\n")
    file.write(f"  RMSE: {rmse_lgbm:.4f}\n")
    file.write(f"  MAE: {mae_lgbm:.4f}\n")
    file.write(f"  R2: {r2_lgbm:.4f}\n\n")
    
    file.write("CatBoost:\n")
    file.write(f"  MSE: {mse_cat:.4f}\n")
    file.write(f"  RMSE: {rmse_cat:.4f}\n")
    file.write(f"  MAE: {mae_cat:.4f}\n")
    file.write(f"  R2: {r2_cat:.4f}\n")
