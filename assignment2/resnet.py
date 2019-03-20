import tensorflow as tf
import numpy as np

def ResBlock(a0, bottleneck=False, plain=False):
    if not bottleneck:
        ch0 = int(a0.shape[-1])
        ch = ch0
        st1 = [1, 1, 1, 1]
        a0T = a0
    else:
        ch0 = int(a0.shape[-1])
        ch = 2 * int(a0.shape[-1])
        st1 = [1, 2, 2, 1]
        ws = tf.Variable(tf.truncated_normal([1, 1, ch0, ch], stddev=0.05))
        bs = tf.Variable(tf.zeros(ch))
        a0T = tf.nn.conv2d(input=a0, filter=ws, strides=[1, 2, 2, 1], padding='SAME') + bs
        a0T = tf.nn.relu(tf.layers.batch_normalization(a0T))
    if plain:
        a0T = 0

    w1 = tf.Variable(tf.truncated_normal([3, 3, ch0, ch], stddev=0.05))
    w2 = tf.Variable(tf.truncated_normal([3, 3, ch, ch], stddev=0.05))
    b1 = tf.Variable(tf.zeros(ch))
    b2 = tf.Variable(tf.zeros(ch))
    
    z1 = tf.nn.conv2d(input=a0, filter=w1, strides=st1, padding='SAME') + b1
    z1b = tf.layers.batch_normalization(z1)
    a1 = tf.nn.relu(z1b)
    
    z2 = tf.nn.conv2d(input=a1, filter=w2, strides=[1, 1, 1, 1], padding='SAME') + b2
    z2b = tf.layers.batch_normalization(z2)
    a2 = tf.nn.relu(z2b + a0T)

    return a2

def ResNet(x, num_classes=1000):
    w1 = tf.Variable(tf.truncated_normal([7, 7, 3, 64], stddev=0.05))
    w2 = tf.Variable(tf.truncated_normal([3 * 3 * 512, num_classes]))
    b1 = tf.Variable(tf.zeros(64))
    b2 = tf.Variable(tf.zeros(1000))
    
    z1 = tf.nn.conv2d(input=x, filter=w1, strides=[1, 2, 2, 1], padding='SAME') + b1
    a1 = tf.nn.relu(tf.layers.batch_normalization(z1))
    s2 = tf.nn.max_pool(a1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    r3 = ResBlock(ResBlock(ResBlock(s2)))
    r4 = ResBlock(ResBlock(ResBlock(ResBlock(r3, bottleneck=True))))
    r5 = ResBlock(ResBlock(ResBlock(ResBlock(ResBlock(r4, bottleneck=True)))))
    r6 = ResBlock(ResBlock(ResBlock(ResBlock(ResBlock(ResBlock(r5, bottleneck=True))))))
    
    s7 = tf.nn.avg_pool(r6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    f7 = tf.reshape(s7, [-1, 3 * 3 * 512])
    
    logits = tf.matmul(f7, w2) + b2

    return logits
