from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.pipeline import Pipeline

# Loading and reshaping data for the training set
# Closing prices for Ethereum and Bitcoin are stored in separate CSV files
df_bc = pd.read_csv("BTC-USD.csv", parse_dates=['Date'])
df_eth = pd.read_csv("ETH-USD.csv", parse_dates=['Date'])

bcv = df_bc.Close.values.reshape(-1, 1)
etv = df_eth.Close.values.reshape(-1, 1)

# Split the features into training and testing sets
X_train = bcv[:-20]
X_test = bcv[-20:]

# Split the labels into training and testing sets
y_train = etv[:-20]
y_test = etv[-20:]

scaler_for_labels = StandardScaler().fit(y_train)

pipe = Pipeline(steps=[('data_transformer', StandardScaler()), ('model', linear_model.LinearRegression())])

# Training the model
pipe.fit(X_train, scaler_for_labels.transform(y_train))
# Testing the model
print(pipe.score(X_test, scaler_for_labels.transform(y_test)))
