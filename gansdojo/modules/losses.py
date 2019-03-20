import tensorflow as tf
from tensorflow.keras import backend as K

def least_square():
    def compute_generator_loss(logits_generated, logits_real, generated, batch, generator):
        return K.mean((logits_generated - 1.0)**2) / 2.0
    
    def compute_discriminator_loss(logits_generated, logits_real, generated, batch, generator):
        D_loss_real = K.mean((logits_real - 1)**2)
        D_loss_fake = K.mean(logits_generated**2)
        D_loss = (D_loss_real + D_loss_fake) / 2.0
        return D_loss

    return compute_generator_loss, compute_discriminator_loss

def wasserstein_gp(use_gradient_penalty=False):

    def compute_generator_loss(logits_generated, logits_real, generated, batch, generator):
        return -1.0 * K.mean(logits_generated)
    
    def compute_discriminator_loss(logits_generated, logits_real, generated, batch, discriminator):
        loss = K.mean(logits_generated) - K.mean(logits_real)
        if use_gradient_penalty:
            loss += gradient_penalty(generated, batch, discriminator)

        return loss


    def gradient_penalty(generated, real, discriminator):
        # Gradient Penalty
        alpha = K.random_uniform(
                                shape = (K.int_shape(generated)[0], 1, 1, 1),
                                minval=0.0,
                                maxval=1.0)
        differences = generated - real
        interpolates = real + (alpha * differences)
        with tf.GradientTape() as tape:
            tape.watch(interpolates)
            logits_interpolates = discriminator(interpolates, training=False)

        gradients = tape.gradient(logits_interpolates, interpolates)[0]
        slopes = K.sqrt(K.sum(K.square(gradients), axis=[1]))
        gradient_penalty = K.mean(K.square(slopes-1.))

        return 10.0 * gradient_penalty

    return compute_generator_loss, compute_discriminator_loss

