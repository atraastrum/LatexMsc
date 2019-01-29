from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml


# Loading and reshaping data
# Closing prices for Ethereum and Bitcoin are stored in separate CSV files
df_bc = pd.read_csv("BTC-USD.csv", parse_dates=['Date'])
df_eth = pd.read_csv("ETH-USD.csv", parse_dates=['Date'])

bcv = df_bc.Close.values.reshape(-1, 1)
etv = df_eth.Close.values.reshape(-1, 1)

scaler_for_output = StandardScaler().fit(etv)

scaler_for_prediction = StandardScaler()
scaler_for_prediction.mean_ = -1.0 * scaler_for_output.mean_ / scaler_for_output.scale_
scaler_for_prediction.scale_ = 1 / scaler_for_output.scale_

pmml_pipe = PMMLPipeline(steps=[('data_transformer', StandardScaler()), ('model', linear_model.LinearRegression())],
                         predict_transformer=scaler_for_prediction)
# Training the model
pmml_pipe.fit(bcv, scaler_for_output.transform(etv).reshape(-1,))
print('Learned parameters')
print(pmml_pipe.steps[1][1].coef_)
print(pmml_pipe.steps[1][1].intercept_)

# Testing prediction
print(pmml_pipe.predict([[1.0]]))
print(pmml_pipe.predict_transform([[1.0]]))

# Exporting Model to PMML
sklearn2pmml(pmml_pipe, 'etherium_price_redict_model.xml', with_repr=True)
