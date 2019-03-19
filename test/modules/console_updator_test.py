import unittest
import time
import tensorflow as tf

from gansdojo.modules import ConsoleUpdator
from .mock import MockObserableDojo
from test.dojo_test import build_config

class ConsoleUpdatorTest(unittest.TestCase):

    def test(self):
        updator = ConsoleUpdator()
        observable = MockObserableDojo(self, build_config())
        updator.setup(observable)

        tf.train.get_or_create_global_step()

        observable.fire('before_train_step')
        time.sleep(0.5)
        observable.fire('after_train_step', tf.constant(10.2), tf.constant(20.3), 9, 99)

