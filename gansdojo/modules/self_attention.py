import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


class SelfAttention(keras.layers.Layer):

    def build(self, input_shape):
        num_channels = input_shape[-1].value

        self.kernel_f, self.bias_f = self._initialize_conv_weight(
            [1, 1, num_channels, num_channels // 8], 'f')
        self.kernel_g, self.bias_g = self._initialize_conv_weight(
            [1, 1, num_channels, num_channels // 8], 'g')
        self.kernel_h, self.bias_h = self._initialize_conv_weight(
            [1, 1, num_channels, num_channels], 'h')
        
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros')

        super().build(input_shape)
    
    def call(self, x):
        f = K.conv2d(x, kernel=self.kernel_f, strides=(1, 1), padding='same')
        f = K.bias_add(f, self.bias_f)
        g = K.conv2d(x, kernel=self.kernel_g, strides=(1, 1), padding='same')
        g = K.bias_add(g, self.bias_g)
        h = K.conv2d(x, kernel=self.kernel_h, strides=(1, 1), padding='same')
        h = K.bias_add(h, self.bias_h)

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
        beta = K.softmax(s, axis=-1)  # attention map
        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]
        o = K.reshape(o, shape=K.int_shape(x))  # [bs, h, w, C]
        return self.gamma * o + x

    def _initialize_conv_weight(self, kernel_shape, name):
        
        kernel_weight = self.add_weight(shape=kernel_shape,
                            initializer='glorot_uniform',
                            name=f'kernel_{name}')
        bias = self.add_weight(shape=[kernel_shape[-1]],
                                      initializer='zeros',
                                      name=f'bias_{name}')

        return kernel_weight, bias

    def compute_output_shape(self, input_shape):
        return input_shape

def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the tensor width and height
    shape = K.int_shape(x)
    return K.reshape(x, [shape[0], -1, shape[-1]]) # return [BATCH, W*H, CHANNELS]