import unittest
from tensorflow.keras import backend as K

class MockObserableDojo:

    def __init__(self, test_case:unittest.TestCase, dojo=None):
        self.test_case = test_case
        self.dojo = dojo

    def register(self, event_name, handler):
        self.handler = handler
        self.event_name = event_name

    def fire(self, name, *args, **kwargs):
        self.test_case.assertEqual(name, self.event_name)
        self.handler(*args, **kwargs)

    def run(self, batch_size=4):
        return K.random_normal([batch_size, 3, 3, 3])
