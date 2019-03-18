import tensorflow as tf
tf.enable_eager_execution()

from tensorflow.keras import backend as K


import time

class Dojo():
    """
    Dodo for the GANs! This is the main coordinator for the training process.
    """

    def __init__(self, config):
        self.config = config

        assert hasattr(config, 'training_ratio')
        assert hasattr(config, 'input_dim')
        assert hasattr(config, 'dataset')
        assert hasattr(config, 'generator')
        assert hasattr(config, 'discriminator')
        assert hasattr(config, 'optimizer_discriminator')
        assert hasattr(config, 'optimizer_generator')
        assert hasattr(config, 'generator_loss_fn') \
            and callable(config.generator_loss_fn)
        assert hasattr(config, 'discriminator_loss_fn') \
            and callable(config.discriminator_loss_fn)

        self.generator = get(config.generator)
        self.discriminator = get(config.discriminator)

        self.optimizer_discriminator = get(config.optimizer_discriminator)
        self.optimizer_generator = get(config.optimizer_generator)

        if hasattr(config, 'global_step'):
            self.global_step = config.global_step
        else: 
            self.global_step = tf.train.get_or_create_global_step()

    def train_on_batch(self, epoch, iteration, batch):
        self.global_step.assign_add(1)

        loss_d, loss_g = None, None
        with tf.GradientTape() as tape_generator, tf.GradientTape() as tape_discriminator:
            
            z = K.random_uniform((K.int_shape(batch)[0], self.config.input_dim))
            generated = self.generator(z, training=True)
            logits_generated = self.discriminator(generated, training=True)
            logits_real = self.discriminator(batch, training=True)

            loss_d = self.config.discriminator_loss_fn(logits_generated, logits_real, generated, batch)
            if self.global_step.numpy() % self.config.training_ratio == 0:
                loss_g = self.config.generator_loss_fn(logits_generated, logits_real, generated, batch)
        
        _update(loss_d, self.discriminator, self.optimizer_discriminator, tape_discriminator)
        _update(loss_g, self.generator, self.optimizer_generator, tape_generator)

        return loss_g, loss_d

    def train_epoch(self, epoch, dataset):
        for i, batch in enumerate(dataset):
            self.train_on_batch(epoch, i + 1, batch)


    def train(self, epochs=None):
        actual_epochs = epochs if epochs is not None else self.config.epochs

        assert actual_epochs is not None

        dataset = get(self.config.dataset)
        with tf.contrib.summary.always_record_summaries():
            for epoch in range(1, actual_epochs + 1):
                self.train_epoch(epoch, dataset)

    def run(self, batch_size=4):
        z = K.random_uniform((batch_size, self.config.input_dim))
        return self.generator(z, training=False)

def _update(loss, model, optimizer, tape):
    if loss is None:
        return

    gradients = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(zip(gradients, model.variables))

def get(obj):
    if callable(obj):
        return obj()
    else:
        return obj

