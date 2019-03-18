import tensorflow as tf
from time import time

from gansdojo import ObservableDojo

class ConsoleUpdator:

    def before_batch_step(self, *args, **kwargs):
        self.begin = time()

    def after_batch_step(self, loss_g, loss_d, epoch, iteration, *args, **kwargs):
        speed = 1.0 / (time() - self.begin)
        global_step = tf.train.get_global_step().numpy()
        print(f'\r{global_step} E:{epoch} I:{iteration} {speed:.2f} b/s G:{loss_g.numpy():.4e} D:{loss_d.numpy():.4e}')

    def setup(self, observable:ObservableDojo):
        observable.register('before_train_step', self.before_batch_step)
        observable.register('after_train_step', self.after_batch_step)