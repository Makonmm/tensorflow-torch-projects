import tensorflow as tf
import matplotlib.pyplot as plt

# Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizing the data
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Hyperparameters
LEARNING_RATE = 0.01
BATCH_SIZE = 100
DISPLAY_STEP = 1
TRAINING_EPOCHS = 35

# Defining the model variables
W = tf.Variable(tf.zeros([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='bias')

# Defining the model and the activation function


def model(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)

# Cross entropy


def compute_loss(y, activation):
    cross_entropy = y * tf.math.log(activation + 1e-10)
    return tf.reduce_mean(-tf.reduce_sum(cross_entropy, axis=1))


# Optimizer
optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

# Training loop
avg_set = []
epoch_set = []

for epoch in range(TRAINING_EPOCHS):
    avg_cost = 0
    total_batch = int(x_train.shape[0] / BATCH_SIZE)

    for i in range(total_batch):
        batch_xs = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_ys = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        with tf.GradientTape() as tape:
            activation = model(batch_xs)
            cost = compute_loss(batch_ys, activation)

        gradients = tape.gradient(cost, [W, b])
        optimizer.apply_gradients(zip(gradients, [W, b]))

        avg_cost += cost / total_batch

    if epoch % DISPLAY_STEP == 0:
        print("EPOCH: ", '%04d' % (epoch + 1),
              "COST: ", "{:.9f}".format(avg_cost))
    avg_set.append(avg_cost)
    epoch_set.append(epoch + 1)

print("END")

# Plotting the cost history
plt.plot(epoch_set, avg_set, 'o',
         label='Logistic Linear Regression (TRAINING PHASE)')
plt.ylabel('COST')
plt.xlabel('EPOCH')
plt.title('Training Cost')
plt.legend()
plt.show()

# Evaluating the model's accuracy
activation = model(x_test)
true_pred = tf.equal(tf.argmax(activation, axis=1), tf.argmax(y_test, axis=1))
accuracy = tf.reduce_mean(tf.cast(true_pred, "float"))

print("Model accuracy: {:.4f}".format(accuracy.numpy()))
