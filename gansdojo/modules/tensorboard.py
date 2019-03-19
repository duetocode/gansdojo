import tensorflow as tf
import os
import shutil

class TensorBoardLogger:

    def __init__(self, log_dir, writer=None, remove_old_data=False):
        if remove_old_data and os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        
        if writer is None:
            self._writer = tf.contrib.summary.create_file_writer(log_dir, flush_millis=1000)
            self._writer.set_as_default()
        else:
            self._writer = writer

    def after_train_batch(self, loss_g, loss_d, *args, **kwargs):
        tf.contrib.summary.scalar('loss_g', loss_g, family='loss')
        tf.contrib.summary.scalar('loss_d', loss_d, family='loss')

    def setup(self, dojo):
        dojo.register('after_train_step', self.after_train_batch)