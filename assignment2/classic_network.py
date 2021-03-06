import tensorflow as tf

def LeNet_5(x):
    params = {}
    params['w1'] = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
    params['w1s'] = tf.Variable(tf.constant(1.0, shape=[6]))
    params['w2'] = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
    params['w2s'] = tf.Variable(tf.constant(1.0, shape=[16]))
    params['w3'] = tf.Variable(tf.truncated_normal([5, 5, 16, 120], stddev=0.1))
    params['w4'] = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
    params['w5'] = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))

    params['b1'] = tf.Variable(tf.constant(0.0, shape=[6]))
    params['b1s'] = tf.Variable(tf.constant(0.0, shape=[6]))
    params['b2'] = tf.Variable(tf.constant(0.0, shape=[16]))
    params['b2s'] = tf.Variable(tf.constant(0.0, shape=[16]))
    params['b3'] = tf.Variable(tf.zeros(120))
    params['b4'] = tf.Variable(tf.zeros(84))
    params['b5'] = tf.Variable(tf.zeros(10))
    
    A = tf.constant(1.7159)
    S = tf.constant(2/3)

    C1 = tf.nn.conv2d(input=x, filter=params['w1'],
                      strides=[1, 1, 1, 1], padding='VALID')
    C1 += params['b1']

    S2 = tf.nn.avg_pool(value=C1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID')
    S2 = tf.multiply(params['w1s'], S2) + params['b1s']
    S2 = tf.nn.sigmoid(S2)

    # TODO: How to implement Partially connection (like lecun paper)??
    C3 = tf.nn.conv2d(input=S2, filter=params['w2'],
                  strides=[1, 1, 1, 1], padding='VALID')
    C3 += params['b2']

    S4 = tf.nn.avg_pool(value=C3, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')
    S4 = tf.multiply(params['w2s'], S4) + params['b2s']
    S4 = tf.nn.sigmoid(S4)

    C5 = tf.nn.conv2d(input=S4, filter=params['w3'],
                  strides=[1, 1, 1, 1], padding='VALID')

    C5 = tf.reshape(C5, [-1, 120])

    F6 = tf.matmul(C5, params['w4']) + params['b4']
    F6 = A * tf.nn.tanh(S * F6) # TODO: Review S

    logits = tf.matmul(F6, params['w5']) + params['b5']
    return logits, params


def AlexNet(x):
    params = {}

    params['w1'] = tf.Variable(tf.truncated_normal([11, 11, 3, 96], stddev=0.05))
    params['w2'] = tf.Variable(tf.truncated_normal([5, 5, 96, 256], stddev=0.05))
    params['w3'] = tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=0.05))
    params['w4'] = tf.Variable(tf.truncated_normal([3, 3, 384, 384], stddev=0.05))
    params['w5'] = tf.Variable(tf.truncated_normal([3, 3, 384, 256], stddev=0.05))
    params['w6'] = tf.Variable(tf.truncated_normal([9216, 4096], stddev=0.05))
    params['w7'] = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.05))
    params['w8'] = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.05))

    params['b1'] = tf.Variable(tf.zeros(96))
    params['b2'] = tf.Variable(tf.zeros(256))
    params['b3'] = tf.Variable(tf.zeros(384))
    params['b4'] = tf.Variable(tf.zeros(384))
    params['b5'] = tf.Variable(tf.zeros(256))
    params['b6'] = tf.Variable(tf.zeros(4096))
    params['b7'] = tf.Variable(tf.zeros(4096))
    params['b8'] = tf.Variable(tf.zeros(1000))

    C1 = tf.nn.conv2d(input=x, filter=params['w1'],
                      strides=[1, 4, 4, 1], padding='VALID')
    C1 += params['b1']
    C1 = tf.nn.relu(C1)

    S2 = tf.nn.max_pool(C1, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    C3 = tf.nn.conv2d(input=S2, filter=params['w2'],
                      strides=[1,1,1,1], padding='SAME')
    C3 += params['b2']
    C3 = tf.nn.relu(C3)

    S4 = tf.nn.max_pool(C3, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

    C5 = tf.nn.conv2d(input=S4, filter=params['w3'],
                      strides=[1, 1, 1, 1], padding='SAME')
    C5 += params['b3']
    C5 = tf.nn.relu(C5)

    C6 = tf.nn.conv2d(input=C5, filter=params['w4'],
                      strides=[1, 1, 1, 1], padding='SAME')
    C6 += params['b4']
    C6 = tf.nn.relu(C6)

    C7 = tf.nn.conv2d(input=C5, filter=params['w5'],
                      strides=[1, 1, 1, 1], padding='SAME')
    C7 += params['b5']
    C7 = tf.nn.relu(C7)

    S8 = tf.nn.max_pool(C7, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='VALID')
    F8 = tf.reshape(S8, [-1, 9216])

    F9 = tf.matmul(F8, params['w6']) + params['b6']
    F9 = tf.nn.relu(F9)
    F9 = tf.nn.dropout(F9, keep_prob=0.5)

    F10 = tf.matmul(F9, params['w7']) + params['b7']
    F10 = tf.nn.relu(F10)
    F10 = tf.nn.dropout(F10, keep_prob=0.5)

    logits = tf.matmul(F10, params['w8']) + params['b8']

    return logits, params

def VGG16(x):
    def CONV(x, out_ch):
        inp_ch = int(x.shape[-1])
        w = tf.Variable(tf.truncated_normal([3, 3, inp_ch, out_ch], stddev=0.05))
        b = tf.Variable(tf.zeros(out_ch))

        C = tf.nn.conv2d(input=x, filter=w,
                         strides=[1, 1, 1, 1], padding='SAME')
        C += b
        C = tf.nn.relu(C)

        return C

    def POOL(x):
        S = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='VALID')
        return S

    w1 = tf.Variable(tf.truncated_normal([7 * 7 * 512, 4096], stddev=0.05))
    w2 = tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.05))
    w3 = tf.Variable(tf.truncated_normal([4096, 1000], stddev=0.05))

    b1 = tf.Variable(tf.zeros(4096))
    b2 = tf.Variable(tf.zeros(4096))
    b3 = tf.Variable(tf.zeros(1000))


    C1 = CONV(CONV(x, 64), 64) # Read Alex Normalization (LRN)
    S2 = POOL(C1)
    C3 = CONV(CONV(S2, 128), 128)
    S4 = POOL(C3)
    C5 = CONV(CONV(CONV(S4, 256), 256), 256)
    S6 = POOL(C5)
    C7 = CONV(CONV(CONV(S6, 512), 512), 512)
    S8 = POOL(C7)
    C9 = CONV(CONV(CONV(S8, 512), 512), 512)
    S10 = POOL(C9)

    F10 = tf.reshape(S10, [-1, 7 * 7 * 512])
    F11 = tf.nn.relu(tf.matmul(F10, w1) + b1)
    F11 = tf.nn.dropout(F11, keep_prob=0.5)
    F12 = tf.nn.relu(tf.matmul(F11, w2) + b2)
    F12 = tf.nn.dropout(F12, keep_prob=0.5)

    logits = tf.matmul(F12, w3) + b3
    
    return logits

