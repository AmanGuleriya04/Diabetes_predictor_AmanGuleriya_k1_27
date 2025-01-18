# Diabetes_predictor_AmanGuleriya_
import pandas as pd
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import plot_model

from keras_visualizer import visualizer
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('diabetes.csv')
df

# Split data into features and labels
X = df.iloc[:, :-1].values  # Features (all columns except the last one)
y = df.iloc[:, -1].values   # Labels (last column)

# Scale features for better model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
# Add an Input layer explicitly
model.add(Input(shape=(X.shape[1],)))  # Number of input features = X.shape[1]

# Add the rest of the layers
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=10, callbacks=[early_stopping])

# Visualize the model architecture
visualizer(model, file_format='png', view=True)  # Place this before prediction or evaluation
plot_model(model, to_file='Neural_network.png', show_layer_names=True)

# Step 1 - Evaluate on Training set
_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
train_error = 1 - train_accuracy

# Step 2 - Evaluate on Testing Set
_, test_accuracy = model.evaluate(X_val, y_val, verbose=0)
test_error = 1 - test_accuracy

# Print the results
print(f'Training Accuracy : {train_accuracy * 100:.2f}%')
print(f'Training Error    : {train_error * 100:.2f}%')
print(f'Testing Accuracy  : {test_accuracy * 100:.2f}%')
print(f'Testing Error     : {test_error * 100:.2f}%')
