import tensorflow as tf

from .dojo import Dojo

class ObservableDojo(Dojo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__subjects = {}
        def add_subject(obj, name):
            value = Subject() 
            obj.__subjects[name] = value

        add_subject(self, 'before_initialization')
        add_subject(self, 'before_train_step')
        add_subject(self, 'after_train_step')
        add_subject(self, 'before_epoch')
        add_subject(self, 'after_epoch')

    def train_on_batch(self, *args, **kwargs):
        
        self.__subjects['before_train_step'].notify(*args, **kwargs) 

        loss_g, loss_d = super().train_on_batch(*args, **kwargs)

        self.__subjects['after_train_step'].notify(loss_g, loss_d, *args, **kwargs)

        return loss_g, loss_d

    def train_epoch(self, *args, **kwargs):
        self.__subjects['before_epoch'].notify(*args, **kwargs)
        super().train_epoch(*args, **kwargs)
        self.__subjects['after_epoch'].notify(*args, **kwargs)

    def train(self, *args, **kwargs):
        self.__subjects['before_initialization'].notify(*args, **kwargs)
        super().train(*args, **kwargs)

    def register(self, subject, observer):
        self.__subjects[subject].register(observer)


class Subject:

    def __init__(self):
        self.__observers = []

    def register(self, observer):
        self.__observers.append(observer)
    
    def notify(self, *args, **kwargs):
        for observer in self.__observers:
            observer(*args, **kwargs)