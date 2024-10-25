# Convolutional Neural Network with TensorFlow


import tensorflow as tf


# Importing the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Reshaping for convolution
x_train = tf.reshape(x_train, shape=[-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape=[-1, 28, 28, 1])

# Hyperparameters
LEARNING_RATE = 0.001
BATCH_SIZE = 256
DISPLAY_STEP = 10
TRAINING_ITERS = 100000

# Network parameters
num_classes = 10  # Number of output classes

# Convolution function


def conv2d(img, w, b):
    # Perform convolution
    conv = tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='VALID')
    # Checking the shape after convolution
    # print(f"Conv shape: {tf.shape(conv)}")
    # Add bias and apply ReLU activation
    return tf.nn.relu(tf.nn.bias_add(conv, b))

# Pooling function


def max_pool(img, k):
    # Perform max pooling
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')

# Loss function


def compute_loss(y_true, logits):
    # Calculate the loss using softmax cross entropy
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))


# Initialize weights and biases

# Weights for the first convolutional layer
wc1 = tf.Variable(tf.random.normal([5, 5, 1, 32]))
# Weights for the second convolutional layer
wc2 = tf.Variable(tf.random.normal([5, 5, 32, 64]))
# Weights for the fully connected layer
wd1 = tf.Variable(tf.random.normal([4 * 4 * 64, 1024]))
# Weights for the output layer
wout = tf.Variable(tf.random.normal([1024, num_classes]))

bc1 = tf.Variable(tf.random.normal([32]))  # Bias for conv1
bc2 = tf.Variable(tf.random.normal([64]))  # Bias for conv2
# Bias for the fully connected layer
bd1 = tf.Variable(tf.random.normal([1024]))
# Bias for the output layer
bout = tf.Variable(tf.random.normal([num_classes]))

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# Training loop
step = 1
while step * BATCH_SIZE < len(x_train):
    batch_xs = x_train[(step-1) * BATCH_SIZE: step * BATCH_SIZE]
    batch_ys = y_train[(step-1) * BATCH_SIZE: step * BATCH_SIZE]

    if len(batch_xs) == 0:
        break

    with tf.GradientTape() as tape:
        # Apply the first convolutional layer
        conv1 = conv2d(batch_xs, wc1, bc1)

        # Apply the pooling layer
        conv1 = max_pool(conv1, k=2)

        # Apply the second convolutional layer
        conv2 = conv2d(conv1, wc2, bc2)

        # Apply the pooling layer
        conv2 = max_pool(conv2, k=2)

        # Fully connected layer
        dense1 = tf.reshape(conv2, [-1, wd1.shape[0]])  # Flatten the tensor
        # Apply ReLU activation
        dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))

        # Prediction layer
        logits = tf.add(tf.matmul(dense1, wout), bout)

        # Calculate the loss
        loss_value = compute_loss(batch_ys, logits)

    grads = tape.gradient(
        loss_value, [wc1, wc2, wd1, wout, bc1, bc2, bd1, bout])
    optimizer.apply_gradients(
        zip(grads, [wc1, wc2, wd1, wout, bc1, bc2, bd1, bout]))

    if step % DISPLAY_STEP == 0:
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(batch_ys, 1)), tf.float32))
        print(f"ITER: {
              step * BATCH_SIZE}, LOSS: {loss_value:.6f}, ACCURACY (TRAIN DATA): {acc:.5f}")

    step += 1

print("Optimization completed")


# Test accuracy
test_data = []
for i in range(0, len(x_test), BATCH_SIZE):
    batch_xs = x_test[i:i + BATCH_SIZE]
    if len(batch_xs) == 0:
        break
    conv1 = conv2d(batch_xs, wc1, bc1)
    conv1 = max_pool(conv1, k=2)
    conv2 = conv2d(conv1, wc2, bc2)
    conv2 = max_pool(conv2, k=2)
    dense1 = tf.reshape(conv2, [-1, wd1.shape[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, wd1), bd1))
    logits = tf.add(tf.matmul(dense1, wout), bout)
    test_data.append(logits)


test_logits = tf.concat(test_data, axis=0)

true_pred = tf.equal(tf.argmax(test_logits, 1), tf.argmax(y_test, 1))
test_accuracy = tf.reduce_mean(tf.cast(true_pred, tf.float32))

print(f"ACCURACY (TEST DATA): {test_accuracy:.5f}")
