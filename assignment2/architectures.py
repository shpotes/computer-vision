import tensorflow as tf

def LeNet_5(x):
    w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.05))
    w1s = tf.Variable(tf.constant(1.0, shape=[6])) 
    w2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.05))
    w2s = tf.Variable(tf.constant(1.0, shape=[16]))
    w3 = tf.Variable(tf.truncated_normal([5, 5, 16, 120], stddev=0.05))
    w4 = tf.Variable(tf.truncated_normal([120, 84], stddev=0.05))
    w5 = tf.Variable(tf.truncated_normal([1, 10, 84], stddev=0.05))

    b1 = tf.Variable(tf.constant(0.0, shape=[6]))
    b1s = tf.Variable(tf.constant(0.0, shape=[6]))
    b2 = tf.Variable(tf.constant(0.0, shape=[16]))
    b2s = tf.Variable(tf.constant(0.0, shape=[16]))
    b3 = tf.Variable(tf.zeros(120))
    b4 = tf.Variable(tf.zeros(84))

    A = tf.constant(1.7159)
    
    C1 = tf.nn.conv2d(input=x, filter=w1,
                      strides=[1, 1, 1, 1], padding='VALID')
    C1 += b1
    
    S2 = tf.nn.avg_pool(value=C1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID')
    S2 = tf.multiply(w1s, S2) + b1s
    S2 = tf.nn.sigmoid(S2)

    # TODO: How to implement Partially connection (like lecun paper)?? 
    C3 = tf.nn.conv2d(input=S2, filter=w2,
                  strides=[1, 1, 1, 1], padding='VALID')
    C3 += b2
    
    S4 = tf.nn.avg_pool(value=C3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')
    S4 = tf.multiply(w2s, S4) + b2s
    S4 = tf.nn.sigmoid(S4)
    
    C5 = tf.nn.conv2d(input=S4, filter=w3,
                  strides=[1, 1, 1, 1], padding='VALID')
    
    C5 = tf.reshape(C5, [-1, 120])

    F6 = tf.matmul(C5, w4) + b4
    F6 = A * tf.nn.tanh(F6) # TODO: Review S
    
    F6 = tf.reshape(F6, [-1, 1, 84]) 
    scores = tf.reduce_sum(tf.square(F6 - w5), axis=2)
    
    return scores
