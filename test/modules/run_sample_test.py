import unittest
import tensorflow as tf

from gansdojo.modules import SampleRunner
from .mock import MockObserableDojo

class SampleRunnerTest(unittest.TestCase):
    
    def test(self):
        dojo = MockObserableDojo(self)

        runner = SampleRunner(50)
        runner.setup(dojo)

        dojo.fire('after_train_step', tf.constant(10.20), tf.constant(30.40), 2, 17, None)