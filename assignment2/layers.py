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
