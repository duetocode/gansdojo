import unittest
import tensorflow as tf
from tensorflow.keras import models, layers
from tempfile import TemporaryDirectory
import os

from test.dojo_test import build_config
from .mock import MockObserableDojo
from gansdojo.modules import CheckpointSaver

class CheckpointSaverTest(unittest.TestCase):
    
    def setUp(self):
        # pylint: disable=no-member
        config = build_config()
        self.obserable = MockObserableDojo(self, config)

    def test(self):
        with TemporaryDirectory() as save_dir:
            saver = CheckpointSaver(save_dir)

            saver.setup(self.obserable)

            tf.train.get_or_create_global_step().assign(150)
            self.obserable.fire('after_epoch', 12, None)
            
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

