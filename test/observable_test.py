import unittest
from gansdojo.observable import ObservableDojo

class ObserableDojoTest(unittest.TestCase):

    def setUp(self):
        self.mockDojo = MockDojo()
        self.dojo = ObservableDojo(self.mockDojo)

    def test_constructor(self):
        self.assertIsNotNone(self.dojo)
        self.assertTrue(hasattr(self.dojo, 'dojo'))

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
        class SubscriberBefore:
            def update(self, epoch, iteration, batch):
                t.assertEqual(2, epoch)
                t.assertEqual(1, iteration)
                t.assertEqual('XXXX', batch)

        class SubscriberAfter(SubscriberBefore):
            def update(self, loss_g, loss_d, epoch, iteration, batch):
                super(SubscriberAfter, self).update(epoch, iteration, batch)
                t.assertEqual(7788, loss_g)
                t.assertEqual(8877, loss_d)

        self.dojo.register('before_train_step', SubscriberBefore().update)
        self.dojo.register('after_train_step', SubscriberAfter().update)

        self.dojo.train_on_batch(2, 1, 'XXXX')

    def test_train_epoch(self):
        t = self
        dataset = "dataset"
        class Subscriber:
            def update(self, epoch, dataset):
                t.assertEqual(2, epoch)
                t.assertEqual('dataset', dataset)

        self.dojo.register('before_epoch', Subscriber().update)
        self.dojo.register('after_epoch', Subscriber().update)

        self.dojo.train_epoch(2, dataset)


class MockDojo:

    def train(self, *args, **kwargs):
        pass

    def train_on_batch(self, *args, **kwargs):
        return 7788, 8877

    def train_epoch(self, epoch, dataset):
        pass



if __name__ == "__main__":
    unittest.main()