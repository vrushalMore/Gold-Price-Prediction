import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

data = pd.read_csv('data.csv')

data.head()
data.columns
data.info()
data.describe()
data.dtypes
data.shape

missing_features = [feature for feature in data.columns if data[feature].isnull().sum() > 0]

if missing_features:
    print("Features with missing values:", missing_features)
else:
    print("No missing value found")

for column in data.columns:
    print(column, data[column].nunique())

categorical_features = [feature for feature in data.columns if data[feature].dtypes == 'O']
print(categorical_features)

numerical_features = [feature for feature in data.columns if data[feature].dtypes != 'O']
data[numerical_features].head()

discrete_features = [feature for feature in numerical_features if len(data[feature].unique()) < 25]
print(f"Discrete feature count: {len(discrete_features)}")

continuous_features = [feature for feature in numerical_features if feature not in discrete_features]
print(f"Continuous feature count: {len(continuous_features)}")

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.title('Gold Price Trend')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 5))
sns.lineplot(x=data.index.month, y=data['Close'], errorbar=None)
plt.xlabel('Month')
plt.ylabel('Close Price')
plt.title('Seasonal Plot')
plt.xticks(range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.show()

z_scores = np.abs(stats.zscore(data['Close']))
threshold = 3

outliers = data[z_scores > threshold]

print(f"Outliers: {outliers}")
