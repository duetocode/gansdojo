from collections import namedtuple


Config = namedtuple('Config', [
        'training_ratio',
        'input_dim',
        'dataset',
        'batches_per_epoch',
        'generator',
        'discriminator',
        'optimizer_discriminator',
        'optimizer_generator',
        'generator_loss_fn',
        'discriminator_loss_fn'])