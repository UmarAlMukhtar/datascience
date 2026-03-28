# importing required libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the dataset
df = pd.read_csv('iris.csv')

# Inspect the dataset
print("First 10 rows of the dataset:")
print(df.head(10))

print("Dataset Information")
print(df.info())

print("Summary Statistics")
print(df.describe())

print("Missing Values in the Dataset:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Encode Categorical Data
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

print("Encoded Species Column:")
print(df['species'].head())

# Split the dataset into features and target variable
X = df.drop('species', axis=1)
Y = df['species']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the Neural Network model
model = Sequential()

# Layer 1: Input layer with 10 neurons and ReLU activation
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))

# Layer 2: Hidden layer with 8 neurons and ReLU activation
model.add(Dense(8, activation='relu'))

# Layer 3: Output layer with 3 neurons (for 3 classes) and softmax activation
model.add(Dense(3, activation='softmax'))  

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Make predictions on the test set
predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes:", predicted_classes)

print("Actual classes:", Y_test.values)
