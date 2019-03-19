import tensorflow as tf
from tensorflow.keras import backend as K

from gansdojo import ObservableDojo

class SampleRunner:
    """Run sample and save them to summary regularly."""

    def __init__(self, interval_in_steps=50):
        self.interval = interval_in_steps

    def setup(self, observable:ObservableDojo):
        observable.register('after_train_step', self.run)
        self._observable = observable

    def run(self, loss_d, loss_g, epoch, iteration, *args, **kwargs):
        generated = self._observable.run()
        tf.contrib.summary.image('generated', K.expand_dims(K.concatenate(generated, axis=1), axis=0))
        
