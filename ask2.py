import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import OneHotEncoder

# Step 1: Generate the data
num_samples = 10000
time_steps = 500
num_features = 6
num_classes = 15

# Create a random array for features with shape (10000, 100, 6)
data = np.random.rand(num_samples, time_steps, num_features)

# Generate random integer labels for 15 classes
labels = np.random.randint(num_classes, size=(num_samples, 1))

# Step 2: One-hot encode the labels
encoder = OneHotEncoder(sparse=False, categories='auto')
one_hot_labels = encoder.fit_transform(labels)

# Step 3: Split the data
train_data, test_data, train_labels, test_labels = train_test_split(
    data, one_hot_labels, test_size=0.2, random_state=42
)

# Define the model
model = Sequential()
# First LSTM layer with return_sequences=True to stack more LSTM layers
model.add(LSTM(128, return_sequences=True, input_shape=(100, 6)))
model.add(Dropout(0.1))
# Second LSTM layer
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.1))
# Output layer for binary classification
model.add(Dense(15, activation='relu'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()

# Assuming train_data and train_labels are prepared
model.fit(train_data, train_labels, epochs=10, batch_size=64, validation_split=0.2)
