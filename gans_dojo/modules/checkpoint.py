import tensorflow as tf
import os
import shutil

class CheckpointSaver:

    def __init__(self, save_dir, remove_old_data=False, auto_restore_latest=False):
        if remove_old_data and os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        self.save_dir = save_dir
        self.auto_restore_lastest = auto_restore_latest

    def setup(self, dojo):
        self.checkpoint_generator = tf.train.Checkpoint(
            optimizer=dojo.optimizer_generator,
            model=dojo.generator)
        self.checkpoint_discriminator = tf.train.Checkpoint(
            optimizer=dojo.optimizer_discriminator,
            model=dojo.discriminator)

        dojo.register('after_epoch', self.save)
        self._dojo = dojo

    def save(self, *args, **kwargs):
        # Create checkpoints
        self.checkpoint_generator.save(os.path.join(self.save_dir, 'generator', 'gen'))
        self.checkpoint_discriminator.save(os.path.join(self.save_dir, 'discriminator', 'dis'))
        
        # Persistent the global steps
        with open(os.path.join(self.save_dir, 'global_step.txt'), 'w') as fd:
            fd.write(str(tf.train.get_global_step().numpy()))