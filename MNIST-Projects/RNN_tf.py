# Recurrent Neural Network with tensorflow

import tensorflow as tf

# Importing the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 10
N_INPUT = 28
N_STEPS = 28
N_HIDDEN = 256
N_CLASSES = 10

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, N_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, N_CLASSES)

# Creating LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(N_HIDDEN, return_sequences=True,
                         input_shape=(N_STEPS, N_INPUT)),
    tf.keras.layers.Dropout(0.5),  # Dropout layer
    tf.keras.layers.LSTM(N_HIDDEN),  # Second LSTM layer
    tf.keras.layers.Dense(N_CLASSES, activation='softmax')  # Output layer
])

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE), metrics=['accuracy'])

# Training the model
model.fit(x_train.reshape(-1, N_STEPS, N_INPUT), y_train,
          batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(
    x_test.reshape(-1, N_STEPS, N_INPUT), y_test)

print(f"TEST LOSS: {test_loss:.6f}, TEST ACCURACY: {test_accuracy:.5f}")
