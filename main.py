from keras.datasets import mnist
from matplotlib import pyplot as plt

def print_digit(the_digit):
    plt.imshow(the_digit, cmap=plt.get_cmap('gray'))


# the MNIST data is split between train and tesextitt sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))

print_digit(X_train[0])
plt.show()