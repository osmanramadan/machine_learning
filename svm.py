# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


# Load your dataset
file_path = 'Battery_RUL.csv'
data      = pd.read_csv(file_path)

# Assuming the dataset has features and a target variable (RUL-Remaining Useful Life)
X = data.drop(columns=['RUL'])
y = data['RUL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (recommended for SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
# Reshape y to a 2D array before scaling
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

X_test_scaled = scaler_X.transform(X_test)
# Reshape y to a 2D array before scaling
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Create an SVR model
svr_model = SVR(kernel='poly', C=1.0)

# Train the model
svr_model.fit(X_train_scaled, y_train_scaled.ravel())


# Make predictions on the test set
y_pred_scaled = svr_model.predict(X_test_scaled)

# Transform predictions back to the original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Evaluate the mean squared error
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# Calculate the accuracy of the model
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# You can also use the trained model to make predictions on new data
new_data = X_test.iloc[:1, :]  # Example new data point
new_data_scaled = scaler_X.transform(new_data)
prediction_scaled = svr_model.predict(new_data_scaled)
prediction = scaler_y.inverse_transform(prediction_scaled)
print(f"Predicted value for new data : {prediction}")
# Plotting the predicted vs. actual values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', linewidth=2)  # Diagonal line for reference
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Actual vs. Predicted RUL')
plt.show()
