import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv('iris.csv')

# print("First 10 rows of the dataset:")
# print(df.head(10))

# print("Dataset Information")
# print(df.info())

# print("Summary Statistics")
# print(df.describe())

# print("Missing Values in the Dataset:")
# print(df.isnull().sum())

df = df.dropna()

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# print("Encoded Species Column:")
# print(df['species'].head())

X = df.drop('species', axis=1)
Y = df['species']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dense(8, activation='relu'))

model.add(Dense(3, activation='softmax'))  


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1
)

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes:", predicted_classes)

print("Actual classes:", Y_test.values)
