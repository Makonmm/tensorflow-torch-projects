# Multilayer perceptron using tensorflow and MNIST dataset

import tensorflow as tf
import matplotlib.pyplot as plt

# Importing the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 100
DISPLAY_STEP = 1
TRAINING_EPOCHS = 35

# Network parameters
n_hidden1 = 256
n_hidden2 = 256
n_input = 784
n_classes = 10

# Layer 1 weights and biases
W1 = tf.Variable(tf.random.normal([n_input, n_hidden1]))
b1 = tf.Variable(tf.random.normal([n_hidden1]))

# Layer 2 weights and biases
W2 = tf.Variable(tf.random.normal([n_hidden1, n_hidden2]))
b2 = tf.Variable(tf.random.normal([n_hidden2]))

# Output layer weights and biases
W_out = tf.Variable(tf.random.normal([n_hidden2, n_classes]))
b_out = tf.Variable(tf.random.normal([n_classes]))

# Function to define the model


def model(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, W1), b1))
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, W2), b2))
    output_layer = tf.add(tf.matmul(layer2, W_out), b_out)
    return output_layer

# Cost function


def compute_loss(y_true, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))


# Optimizer
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# Lists to store costs and epochs
avg_set = []
epoch_set = []

# Training the model
for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0
    total_batch = int(x_train.shape[0] / BATCH_SIZE)

    for i in range(total_batch):
        batch_xs = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_ys = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        # Using GradientTape to calculate gradients
        with tf.GradientTape() as tape:
            logits = model(batch_xs)
            cost = compute_loss(batch_ys, logits)

        # Applying gradients to update weights
        gradients = tape.gradient(cost, [W1, b1, W2, b2, W_out, b_out])
        optimizer.apply_gradients(
            zip(gradients, [W1, b1, W2, b2, W_out, b_out]))

        avg_cost += cost / total_batch

    if epoch % DISPLAY_STEP == 0:
        print("EPOCH: ", '%04d' % (epoch + 1),
              "COST: ", "{:.9f}".format(avg_cost))
    avg_set.append(avg_cost)
    epoch_set.append(epoch + 1)

print("END")

# Plotting the cost history
plt.plot(epoch_set, avg_set, 'o', label='Training Phase')
plt.ylabel('COST')
plt.xlabel('EPOCH')
plt.title('Training Cost')
plt.legend()
plt.show()

# Evaluating the model's accuracy
logits = model(x_test)
true_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y_test, axis=1))
accuracy = tf.reduce_mean(tf.cast(true_pred, "float"))

print("Model accuracy: {:.4f}".format(accuracy.numpy()))
