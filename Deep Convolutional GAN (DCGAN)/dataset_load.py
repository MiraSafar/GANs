import numpy as np
from tensorflow import keras

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# For dataset to work properly on the DCGAN architecture we need to reshape and rescale the X_train features:
X_train_dcgan = X_train.reshape(-1, 28, 28, 1) * 2. - 1.
