# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
file_path = 'Battery_RUL.csv'
data = pd.read_csv(file_path)

# Assuming the dataset has features and a target variable (RUL - Remaining Useful Life)
X = data.drop(columns=['RUL'])
y = data['RUL']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regressor with limited depth and features
tree_regressor = DecisionTreeRegressor(max_depth=3, max_features=5)  # Adjust parameters as needed

# Train the regressor
tree_regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree_regressor.predict(X_test)

# Evaluate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree_regressor, filled=True, feature_names=X.columns)
plt.show()
