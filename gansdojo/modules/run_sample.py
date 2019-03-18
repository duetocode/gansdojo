import tensorflow as tf
from tensorflow.keras import backend as K

class SampleRunner:
    """Run sample and save them to summary regularly."""

    def __init__(self, interval_in_steps=50):
        self.interval = interval_in_steps

    def setup(self, dojo):
        dojo.register('after_train_step', self.run)
        self._dojo = dojo

    def run(self, loss_d, loss_g, epoch, iteration, *args, **kwargs):
        generated = self._dojo.run()
        tf.contrib.summary.image('generated', K.concatenate(generated, axis=2))
        
