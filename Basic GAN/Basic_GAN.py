import tensorflow as tf
from dataset_load import *
from helpers.helpers import *

codings_size = 30

generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28*28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])

discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])

gan = keras.models.Sequential([generator, discriminator])

# +-------------------+
# | COMPILATION SETUP |
# +-------------------+
# NOTES:
#
# As the discriminator is a binary classifier, we can naturally use the binary cross-entropy loss.
# The generator will only be trained through the gan model, so we do not need to compile it at all.
# The gan model is also a binary classifier, so it can use the binary cross-entropy loss.
# Importantly, the discriminator should not be trained during the second phase, so we make it non-trainable before
# compiling the gan model.

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# NOTES:
#
# The 'trainable' attribute is taken into account byt Keras only when compiling a model, so after running this code, the
# discriminator is trainable if we call its fit() method or its train_on_batch() method (which we will be using), while
# it is not trainable when we call these methods on the gan model.
#
# COMPILATION SETUP - END
# ----------------------------------------------------------------------------------------------------------------------


# +--------------+
# | DATASET PREP |
# +--------------+
# NOTES:
#
# Since the training loop is unusual, we cannot use the regular fit() method. Instead, we will write a custom training
# loop. For this, we first need to create a Dataset to iterate thought the images.

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# DATASET PREP - END
# ----------------------------------------------------------------------------------------------------------------------


# +----------------------+
# | CUSTOM TRAINING LOOP |
# +----------------------+

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

# NOTES:
#
# In phase one we feed Gaussian noise to the generator to produce fake images, and we complete this batch by
# concatenating an equal number of real images. The targets y1 are set to 0 for fake images and 1 for real images. Then
# we train the discriminator on this batch. Note that we set the discriminator's 'trainable' attribute to 'True': this
# is only to get rid of a warning that Keras displays when it notices that trainable is now False but was True when the
# model was compiled (or vice versa).
#
# In phase two, we feed the GAN some Gaussian noise. Its generator will start by producing fake images, then the
# discriminator will try to guess whether these images are fake or real. We want the discriminator to believe that the
# fake images are real, so the targets y2 are set to 1. Note that we set the 'trainable' attribute to 'False', once
# again to avoid a warning.

# CUSTOM TRAINING LOOP - END
# ----------------------------------------------------------------------------------------------------------------------


# +----------+
# | TRAINING |
# +----------+

# Change n_epochs to about 50 for best results
train_gan(gan, dataset, batch_size, codings_size, n_epochs=1)
