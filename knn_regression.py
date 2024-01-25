import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'Battery_RUL.csv'
data = pd.read_csv(file_path)

#  Assuming  the dataset has features and a target variable (RUL - Remaining Useful Life)
X = data.drop(columns=['RUL'])
y = data['RUL']

#  Split the data into training and testing sets
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Standardize the features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the value of k (number of neighbors)
k_value = 5

# Create the KNN model
knn_model = KNeighborsRegressor(n_neighbors=k_value)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title("Actual vs Predicted RUL")
plt.show()



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error

# # Load the dataset
# file_path = 'Battery_RUL.csv'
# data = pd.read_csv(file_path)

# # Assuming the dataset has features and a target variable (RUL - Remaining Useful Life)
# X = data.drop(columns=['RUL'])
# y = data['RUL']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features (important for KNN)
# scaler  = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test  = scaler.transform(X_test)

# print(X_test)

# # Choose the value of k (number of neighbors)
# k_value = 5

# # Create the KNN model
# knn_model = KNeighborsRegressor(n_neighbors=k_value)

# # Train the model
# knn_model.fit(X_train,y_train)

# # Make predictions on the test set
# y_pred = knn_model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')