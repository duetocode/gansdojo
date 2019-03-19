import unittest
from tensorflow.keras import backend as K

class MockObserableDojo:

    def __init__(self, test_case:unittest.TestCase, config):
        self.test_case = test_case
        self.generator = config.generator()
        self.discriminator = config.discriminator()
        self.optimizer_generator = config.optimizer_generator
        self.optimizer_discriminator = config.optimizer_discriminator
        self.handlers = {}

    def register(self, event_name, handler):
        self.handlers[event_name] = handler

    def fire(self, name, *args, **kwargs):
        handler = self.handlers[name]
        handler(*args, **kwargs)

    def run(self, batch_size=4):
        return K.random_normal([batch_size, 3, 3, 3])
