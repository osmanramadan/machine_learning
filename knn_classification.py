import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the dataset
file_path = 'Battery_RUL.csv'
data = pd.read_csv(file_path)

# Assuming the dataset has features and a target variable (RUL - Remaining Useful Life)
X = data.drop(columns=['RUL'])
y = data['RUL']

# Convert regression to classification using a threshold (you may adjust this)
threshold = 100
y_class = (y <= threshold).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.4, random_state=42)

# Standardize the features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Choose the value of k (number of neighbors)
k_value = 4

# Create the KNN model
knn_model = KNeighborsRegressor(n_neighbors=k_value)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Convert predictions to binary using the same threshold
y_pred_class = (y_pred <= threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test,y_pred_class)
print(f'Accuracy: {accuracy}')