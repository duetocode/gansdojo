import unittest
import tensorflow as tf
from  tempfile import TemporaryDirectory
import os

from .mock import MockDojo
from gans_dojo.modules import TensorBoardLogger


class TensorboardTest(unittest.TestCase):

    def test(self):
        with TemporaryDirectory() as playground:
            log_dir = os.path.join(playground, 'logs')
            
            logger = TensorBoardLogger(log_dir=log_dir, remove_old_data=True)

            self.assertTrue(os.path.exists(log_dir))

            mock_dojo = MockDojo(self)
            logger.setup(mock_dojo)

            mock_dojo.fire('after_train_step', tf.constant(10.20), tf.constant(30.40), 1, 20, None)

            self.assertTrue(os.path.exists(log_dir))