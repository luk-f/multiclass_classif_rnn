import keras
from keras.datasets import mnist
from keras import layers

from matplotlib import pyplot as plt
import numpy as np

def print_digit(the_digit):
    plt.imshow(the_digit, cmap=plt.get_cmap('gray'))


# the MNIST data is split between train and tesextitt sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
y_train_onehot = np.zeros((y_train.size, y_train.max()+1))
y_train_onehot[np.arange(y_train.size), y_train] = 1
print('y_train_onehot: ' + str(y_train_onehot.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))
y_test_onehot = np.zeros((y_test.size, y_test.max()+1))
y_test_onehot[np.arange(y_test.size), y_test] = 1
print('y_test_onehot: ' + str(y_test_onehot.shape))

# print_digit(X_train[0])
# plt.show()

flat_dim = int(28*28)

X_train = X_train.reshape((X_train.shape[0],-1))
X_test = X_test.reshape((X_test.shape[0],-1))

print('X_train: ' + str(X_train.shape))
print('X_test:  '  + str(X_test.shape))

inputs = keras.Input(shape=(flat_dim), name="digit")

x = layers.Dense(100, 
                activation="relu", 
                name="dense_1",
                kernel_initializer='random_normal',
                bias_initializer='zeros')(inputs)

outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary()
)

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.BinaryCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.BinaryAccuracy()],
)

print(X_train[0].shape)

history = model.fit(
    X_train,
    y_train_onehot,
    batch_size=1000,
    epochs=15,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_test, y_test_onehot),
#     validation_steps=10,
    
    # verbose = 0
)

print(history.history)

plt.plot(history.history['binary_accuracy'], label='binary_accuracy')
plt.plot(history.history['val_binary_accuracy'], label='val_binary_accuracy')

plt.legend()
plt.show()