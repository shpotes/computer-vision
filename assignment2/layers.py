class SubPooling(tf.keras.layers.Layer):
    def __init__(self, features, sz=[1, 2, 2, 1], st=[1, 2, 2, 1],
                 pad='VALID', **kwargs):
        super(SubPooling, self).__init__(**kwargs)
        self.sz=[1, 2, 2, 1]
        self.st=[1, 2, 2, 1]
        self.pad='VALID'
        self.w = tf.Variable(tf.ones(features))
        self.b = tf.Variable(tf.zeros(features))
        
    def call(self, x):
        S = tf.nn.avg_pool2d(input=x, ksize=self.sz,
                             strides=self.st, padding=self.pad)
        return tf.nn.sigmoid(self.w * S + self.b)
