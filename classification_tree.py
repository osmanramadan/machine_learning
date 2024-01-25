# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
file_path = 'Battery_RUL.csv'
data = pd.read_csv(file_path)

# Assuming the dataset has features and a target variable (RUL - Remaining Useful Life)
X = data.drop(columns=['RUL'])
y = data['RUL']

# Convert the regression problem into a classification problem (example threshold: RUL > 50)
y_class = (y > 50).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Create a decision tree classifier
tree_classifier = DecisionTreeClassifier()

# Train the classifier
tree_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = tree_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(tree_classifier, filled=True, feature_names=X.columns, class_names=['RUL <= 50', 'RUL > 50'])
plt.show()
