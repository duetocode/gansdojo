# MIT License
# 
# Copyright (c) 2019 Bowie
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

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