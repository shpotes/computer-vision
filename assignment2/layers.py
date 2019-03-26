import tensorflow as tf

class SubPooling(tf.keras.layers.Layer):
    def __init__(self, features, sz=[1, 2, 2, 1], st=[1, 2, 2, 1],
                 pad='VALID', **kwargs):
        super(SubPooling, self).__init__(**kwargs)
        self.sz=sz
        self.st=st
        self.pad=pad
        self.w = tf.Variable(tf.ones(features))
        self.b = tf.Variable(tf.zeros(features))
        
    def call(self, x):
        S = tf.nn.avg_pool2d(input=x, ksize=self.sz, 
                             strides=self.st, padding=self.pad)
        return tf.nn.sigmoid(self.w * S + self.b)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, bottleneck=False, **kwargs):
        super(ResBlock, self).__init_(**kwargs)
        self.bottleneck = bottleneck
        self.channels = channels
        
        st1 = (2, 2) if bottleneck else (1, 1)

        self.conv1 = tf.keras.layers.Conv2D(filters=self.channels,
                                            kernel_size=(3, 3),
                                            stride=st1,
                                            padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters=self.channels,
                                            kernel_size=(3, 3),
                                            padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if bottleneck:
            self.convT = tf.keras.layers.Conv2D(filters=self.channels,
                                                kernel_size=(1, 1),
                                                padding='same')
            self.bnT = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):

        input_transformed = input_tensor if not bottleneck \
            else tf.nn.relu(self.bnT(self.convT(input_tensor)))

        x = self.conv1(input_tensor)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x += input_transformed
        return  tf.nn.relu(x)

