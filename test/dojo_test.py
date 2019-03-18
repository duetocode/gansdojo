import unittest
import tensorflow as tf
tf.enable_eager_execution()

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, models, backend as K

from gansdojo import Dojo, Config

class DojoTest(unittest.TestCase):

    def test(self):
        config = build_config()
        dojo = Dojo(config)
        dojo.train(epochs=2)


def build_config() -> Config:
    # pylint: disable=unexpected-keyword-arg
    return Config(
        training_ratio=1,
        input_dim=9,
        dataset=create_dataset,
        generator=build_generator,
        discriminator=build_discriminator,
        optimizer_discriminator=tf.train.AdamOptimizer(),
        optimizer_generator=tf.train.AdamOptimizer(),
        discriminator_loss_fn=compute_discriminator_loss,
        generator_loss_fn=compute_generator_loss
    )

def create_dataset() -> tf.data.Dataset:
    source = K.random_normal((20, 3, 3, 3))
    return tf.data.Dataset.from_tensor_slices(source).batch(4)

def build_generator() -> Model:
    model = Sequential()
    model.add(layers.Dense(27, input_shape=(9, )))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((3, 3, 3)))
    return model

def build_discriminator() -> Model:
    model = Sequential()
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))
    return model

def compute_generator_loss(logits_fake, *args, **kwargs):
    return -K.mean(logits_fake)

def compute_discriminator_loss(logits_fake, *args, **kwargs):
    return K.mean(logits_fake)


    
    