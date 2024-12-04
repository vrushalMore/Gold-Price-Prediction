import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
series = data.loc[:, 'Close'].values

def generate_ar_process(lags, coefs, length):
    coefs = np.array(coefs)
    series = [np.random.normal() for _ in range(lags)]
    for _ in range(length):
        prev_vals = series[-lags:][::-1]
        new_val = np.sum(np.array(prev_vals) * coefs) + np.random.normal()
        series.append(new_val)
    return np.array(series)

def perform_adf_test(series):
    result = adfuller(series)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:', result[4])
    if result[1] < 0.05:
        print('The series is stationary')
    else:
        print('The series is non-stationary')

print("\nADF Test on Original Series:")
perform_adf_test(series)







