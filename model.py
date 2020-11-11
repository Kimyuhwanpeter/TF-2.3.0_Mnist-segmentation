import tensorflow as tf

l2 = tf.keras.regularizers.l2
class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class Pad(tf.keras.layers.Layer):

    def __init__(self, paddings, mode='CONSTANT', constant_values=0, **kwargs):
        super(Pad, self).__init__(**kwargs)
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, mode=self.mode, constant_values=self.constant_values)

def residual_block_with_conv(x, k=3, weight_decay=0.00001):
    regul = tf.keras.regularizers.l2
    dim = x.shape[-1]

    h = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=dim,
                                kernel_size=k,
                                strides=1,
                                padding='valid',
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.pad(x, [[0,0],[1,1],[1,1],[0,0]], "REFLECT")
    h = tf.keras.layers.Conv2D(filters=dim,
                                kernel_size=k,
                                strides=1,
                                padding='valid',
                                use_bias=False,
                                kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)

    return tf.keras.layers.ReLU()(x + h)

def seg_model(input_shape=(256, 256, 3),
                        n_blocks=9,
                        weight_decay=0.00001,
                        num_classes=11):

    regul = tf.keras.regularizers.l2
    h = inputs = tf.keras.Input(input_shape)

    Dx1, Dy1 = tf.image.image_gradients(inputs)
    M1 = tf.math.add(tf.math.abs(Dx1), tf.math.abs(Dy1))

    dim = 64
    h = tf.pad(h, [[0,0],[3,3],[3,3],[0,0]], mode="REFLECT")
    h = tf.keras.layers.Conv2D(filters=dim,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=regul(weight_decay))(h)
    Dx2, Dy2 = tf.image.image_gradients(h)        #  256 x 256 x 64
    M2 = tf.math.add(tf.math.abs(Dx2), tf.math.abs(Dy2))   #  256 x 256 x 64
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 256 x 256 x 64

    dim *= 2
    h = tf.keras.layers.Conv2D(filters=dim,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=regul(weight_decay))(h)
    Dx3, Dy3 = tf.image.image_gradients(h)        #  128 x 128 x 128
    M3 = tf.math.add(tf.math.abs(Dx3), tf.math.abs(Dy3))   #  128 x 128 x 128
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 128 x 128 x 128

    dim *= 2
    h = tf.keras.layers.Conv2D(filters=dim,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=regul(weight_decay))(h)
    Dx4, Dy4 = tf.image.image_gradients(h)        #  64 x 64 x 256
    M4 = tf.math.add(tf.math.abs(Dx4), tf.math.abs(Dy4))   #  64 x 64 x 256
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 64 x 64 x 256

    for _ in range(4):
        h = residual_block_with_conv(h)

    dim *= 2
    h = tf.keras.layers.Conv2D(filters=dim,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=regul(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)       # 32 x 32 x 512

    for _ in range(2):
        h = residual_block_with_conv(h)

    dim //= 2
    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False, kernel_regularizer=regul(weight_decay))(h)
    h = tf.math.multiply(M4, h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)       # 64 x 64 x 256

    dim //= 2
    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False, kernel_regularizer=regul(weight_decay))(h)
    h = tf.math.multiply(M3, h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)       # 128 x 128 x 128

    dim //= 2
    h = tf.keras.layers.Conv2DTranspose(dim, 3, strides=2, padding='same', use_bias=False, kernel_regularizer=regul(weight_decay))(h)
    h = tf.math.multiply(M2, h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.ReLU()(h)       # 256 x 256 x 64

    h = tf.pad(h, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    h = tf.keras.layers.Conv2D(num_classes, 7, padding='valid')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)