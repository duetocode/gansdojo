import unittest
from tensorflow.keras import backend as K

class MockObserableDojo:

    def __init__(self, test_case:unittest.TestCase, dojo=None):
        self.test_case = test_case
        self.dojo = dojo
        self.handlers = {}

    def register(self, event_name, handler):
        self.handlers[event_name] = handler

    def fire(self, name, *args, **kwargs):
        handler = self.handlers[name]
        handler(*args, **kwargs)

    def run(self, batch_size=4):
        return K.random_normal([batch_size, 3, 3, 3])
