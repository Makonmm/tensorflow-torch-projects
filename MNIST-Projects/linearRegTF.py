import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generating data
num_of_points = 500

x_point = []
y_point = []

a = 0.24
b = 0.76

# Creating random points
for _ in range(num_of_points):
    x = np.random.normal(0.0, 0.5)
    y = a * x + b + np.random.normal(0.0, 0.1)
    x_point.append([x])
    y_point.append([y])

# Convert to numpy arrays for TensorFlow
x_point = np.array(x_point, dtype=np.float32)
y_point = np.array(y_point, dtype=np.float32)

# Visualizing points
plt.plot(x_point, y_point, 'o', label="Input Data")
plt.legend()
plt.show()

# Creating variables
A = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
B = tf.Variable(tf.zeros([1]))

# Training
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

# Number of steps for training
num_steps = 35


logdir = "/mygraph"
summary_writer = tf.summary.create_file_writer(logdir)

for step in range(num_steps):
    with tf.GradientTape() as tape:
        # Calculate predicted y
        y_pred = A * x_point + B

        # Cost func
        cost_func = tf.reduce_mean(tf.square(y_pred - y_point))

    # Compute gradients and update variables
    gradients = tape.gradient(cost_func, [A, B])
    print(f"Step {step}:")
    print(f"Gradient A --> {gradients[0].numpy()}")
    print(f"Gradient B --> {gradients[1].numpy()}")

    # Apply gradients
    optimizer.apply_gradients(zip(gradients, [A, B]))

    with summary_writer.as_default():
        tf.summary.scalar('cost', cost_func, step=num_steps)

    # Updated values of A and B
    print(f"Updated A --> {A.numpy()}")
    print(f"Updated B --> {B.numpy()}")

    if (step % 5) == 0:
        plt.plot(x_point, y_point, 'o', label='step = {}'.format(step))
        plt.plot(x_point, A.numpy() * x_point + B.numpy(), label='Prediction')
        plt.legend()
        plt.show()


print("\nEnd")
