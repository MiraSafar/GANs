import tensorflow as tf
from dataset_load import *
from helpers.helpers import *

# +------------------------------------------------------+
# | MAIN GUIDELINES (as proposed by Alec Radford et al.) |
# +------------------------------------------------------+
#
# They proposed the following guidelines to make the convolutional GAN stable:
#
# 1)
# Replace any pooling layers with stride convolutions (in the discriminator) and transpose the convolutions (in the
# generator).
#
# 2)
# Use Batch Normalization in both the generator and the discriminator, except in the generator's output layer and the
# discriminator's input layer.
#
# 3)
# Remove fully connected hidden layers for deeper architectures.
#
# 4)
# Use ReLU activation in the generator for all layers except the output layer, which should use tanh.
#
# 5) Use leaky ReLU activation in the discriminator for all layers.

codings_size = 100

generator = keras.models.Sequential([
    keras.layers.Dense(7 * 7 * 128, input_shape=[codings_size]),
    keras.layers.Reshape([7, 7, 128]),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same", activation="selu"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh")
])

discriminator = keras.models.Sequential([
    keras.layers.Conv2D(64, kernel_size=5, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2),
                        input_shape=[28, 28, 1]),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(128, kernel_size=5, strides=2, padding="same", activation=keras.layers.LeakyReLU(0.2)),
    keras.layers.Dropout(0.4),
    keras.layers.Flatten(),
    keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator, discriminator])

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train_dcgan).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)


def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):

    generator, discriminator = gan.layers

    generated_images = None

    for epoch in range(n_epochs):

        print("Epoch {}/{}".format(epoch + 1, n_epochs))

        for X_batch in dataset:

            # Phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            x_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.0]] * batch_size)    # labels ?? 0 for fake, 1 for real ??
            discriminator.trainable = True
            discriminator.train_on_batch(x_fake_and_real, y1)

            # Phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)

        plot_multiple_images(generated_images, 8)
        plt.show()


train_gan(gan, dataset, batch_size, codings_size, n_epochs=3)
