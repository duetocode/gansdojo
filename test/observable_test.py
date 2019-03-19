import unittest
from gansdojo.observable import ObservableDojo
from .dojo_test import build_config

from tensorflow.keras import backend as K

class ObserableDojoTest(unittest.TestCase):

    def setUp(self):
        self.config = build_config()
        self.dojo = ObservableDojo(self.config)

    def test_constructor(self):
        self.assertIsNotNone(self.dojo)

    def test_before_initialization(self):
        t = self
        class Subscriber:
            def update(self, epochs):
                t.assertEqual(2, epochs)

        sub = Subscriber()
        self.dojo.register('before_initialization', sub.update)
        self.dojo.train(2)
        
    def test_before_after_train_step(self):
        t = self
        invoked_before, invoked_after = False, False
        class SubscriberBefore:
            def update(self, epoch, iteration, batch):
                nonlocal invoked_before
                t.assertEqual(2, epoch)
                t.assertEqual(1, iteration)
                invoked_before = True

        class SubscriberAfter(SubscriberBefore):
            def update(self, loss_g, loss_d, epoch, iteration, batch):
                nonlocal invoked_after
                super(SubscriberAfter, self).update(epoch, iteration, batch)
                t.assertIsNotNone(loss_g)
                t.assertIsNotNone(loss_d)
                invoked_after = True
                
        self.dojo.register('before_train_step', SubscriberBefore().update)
        self.dojo.register('after_train_step', SubscriberAfter().update)

        batch = K.random_normal([4, 3, 3, 3])
        self.dojo.train_on_batch(2, 1, batch)

        self.assertEqual(True, invoked_before)
        self.assertEqual(True, invoked_after)

    def test_train_epoch(self):
        t = self

        update_invoked = False
        class Subscriber:
            def update(self, epoch, dataset):
                nonlocal update_invoked
                t.assertEqual(2, epoch)
                t.assertIsNotNone(dataset)
                update_invoked = True

        self.dojo.register('before_epoch', Subscriber().update)
        self.dojo.register('after_epoch', Subscriber().update)

        self.dojo.train_epoch(2, self.config.dataset())

        self.assertEqual(True, update_invoked)


class MockDojo:

    def train(self, *args, **kwargs):
        pass

    def train_on_batch(self, *args, **kwargs):
        return 7788, 8877

    def train_epoch(self, epoch, dataset):
        pass



if __name__ == "__main__":
    unittest.main()