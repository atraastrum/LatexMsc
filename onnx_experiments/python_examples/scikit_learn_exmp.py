from sklearn import datasets, linear_model
import load_training_set

# Load the training data set
training_set = load_training_set.get()

# Split the features into training/testing sets
X_train = training_set['features'][:-20]
X_test = training_set['features'][-20:]

# Split the labels into training/testing sets
y_train = training_set['labels'][:-20]
y_test = training_set['labels'][-20:]

# Create linear regression object
linreg = linear_model.LinearRegression()

# Train the model using the training sets
linreg.fit(X_train, y_train)

print(linreg.score(X_test, y_test))

