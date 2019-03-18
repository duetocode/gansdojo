import unittest
import tensorflow as tf

from gans_dojo.modules import SampleRunner
from .mock import MockDojo

class SampleRunnerTest(unittest.TestCase):
    
    def test(self):
        dojo = MockDojo(self)

        runner = SampleRunner(50)
        runner.setup(dojo)

        dojo.fire('after_train_step', tf.constant(10.20), tf.constant(30.40), 2, 17, None)