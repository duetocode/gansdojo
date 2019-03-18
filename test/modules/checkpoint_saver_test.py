import unittest
import tensorflow as tf
from tensorflow.keras import models, layers
from tempfile import TemporaryDirectory
import os

from test.dojo_test import build_config
from .mock import MockDojo
from gans_dojo.modules import CheckpointSaver

class CheckpointSaverTest(unittest.TestCase):
    
    def setUp(self):
        # pylint: disable=no-member
        config = build_config()
        dojo = MockDojo(self)

        dojo.generator = config.generator()
        dojo.discriminator = config.discriminator()
        dojo.optimizer_discriminator = config.optimizer_discriminator
        dojo.optimizer_generator = config.optimizer_generator
        
        self.dojo = dojo

    def test(self):
        with TemporaryDirectory() as save_dir:
            saver = CheckpointSaver(save_dir)

            saver.setup(self.dojo)

            tf.train.get_or_create_global_step().assign(150)
            self.dojo.fire('after_epoch', 12, None)
            
            # Check the parent directory
            self.assertTrue(os.path.exists(save_dir))

            # Check the step file
            step_file = os.path.join(save_dir, 'global_step.txt')
            self.assertTrue(os.path.exists(step_file))
            with open(step_file, 'r') as f:
                self.assertEqual('150', f.read())

            # Check the checkpoints of generator
            check(self, os.path.join(save_dir, 'generator'))
            check(self, os.path.join(save_dir, 'discriminator'))

def check(self, save_dir):
    self.assertTrue(os.path.exists(save_dir))
    files = os.listdir(save_dir)
    self.assertEqual(3, len(files))
    self.assertTrue(os.path.exists(os.path.join(save_dir, 'checkpoint')))






        

        


