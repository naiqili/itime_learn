import tensorflow as tf
import numpy as np

sess = tf.InteractiveSession()

a_mat = np.asarray([[1,0,0], [0,1,0]])
b_mat = np.random.rand(5,3,3)

print 'a:'
print a_mat
print
print 'b:'
print b_mat
print

a = tf.constant(a_mat, dtype=tf.float32)
b = tf.constant(b_mat, dtype=tf.float32)

ab = tf.einsum('nu,zud->znd', a, b)

print 'a x b:'
print sess.run(ab)
print

c_mat = np.asarray([[1,0],
                    [0,1],
                    [0,0]])
c = tf.constant(c_mat, dtype=tf.float32)
print 'c:'
print c_mat
print

abc = tf.einsum('znu,uk->znk', ab, c)
print 'a x b x c:'
print sess.run(abc)
print

mp_abc = tf.reduce_max(abc, axis=2)
print 'maxpool(abc):'
print sess.run(mp_abc)
print

print 'transpose:'
print sess.run(tf.transpose(mp_abc))

sess.close()
