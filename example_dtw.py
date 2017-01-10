import tensorflow as tf
import numpy as np


nx = np.array([[2,2], [3,3], [4,4]],dtype=np.float32)
ny = np.array([[1,1], [2,2], [3,3], [4,4], [5,5]], dtype=np.float32)
x = tf.placeholder(tf.float32, (3,2))
y = tf.placeholder(tf.float32, (5,2))
sess=tf.InteractiveSession()

def squared_dist(tx, ty): 
    expanded_x = tf.expand_dims(tx, 1)
    expanded_y = tf.expand_dims(ty, 0)
    distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(expanded_x, expanded_y), 2))
    return distances

D = squared_dist(x, y)

#Ds = tf.unstack(D,axis=0)
#print Ds[0].eval(feed_dict={x:nx, y:ny})


def pseudo_code_warp(D):
    nQ, nD = D.shape
    M=np.zeros((nQ+1, nD+1),dtype= np.float32)
    M[:,0]=np.inf
    M[0,:]=0
    for i in range(nQ):
        for j in range(nD):
            M[i+1,j+1] = min(M[i,j], M[i,j+1], M[i+1,j]) + D[i,j]
    return M[1:, 1:]

def pseudo_code_warp2(D):
    nQ, nD = D.shape
    M=np.zeros((nQ+1, nD+1),dtype= np.float32)
    M[:,0]=np.inf
    M[0,:]=0
    for i in range(nQ):
        K[0]
        for j in range(nD):
            #M[i+1,j+1] = min(M[i,j], M[i,j+1], M[i+1,j]) + D[i,j]
            M[i+1,j+1] = min(M[i,j], M[i,j+1], M[i+1,j]) + D[i,j]
        K[i] = min(M[i,:nD], M[i,1:])
    return M[1:, 1:]

nD = np.array([[1,0,1,1,1,1],[1,1,0,1,1,1]],dtype=np.float32)
print pseudo_code_warp(nD) 


def warp_ele(m_i, vecs):
    m_t, m_tm1, v_i = vecs
    print 'm_t',m_t
    print 'm_tm1',m_tm1
    print 'v_i',v_i
    m_ip1 = tf.reduce_min(tf.pack([m_t, m_tm1, m_i]))+v_i
    return m_ip1

def warp_row(m_t, v_t):
    v_s = [tf.constant(np.inf)] + tf.unstack(m_t, axis=0)[:-1]
    m_tm1 = tf.pack(v_s)
    print 'm_tm1',m_tm1
    print 'm_t',m_t
    print 'v_s',v_s
    print 'v_t',v_t
    m_tp1 = tf.scan(warp_ele,(m_t,m_tm1,v_t), initializer=np.inf)
    return m_tp1



print D.eval(feed_dict={x:nx, y:ny})


#warped = tf.scan(warp_row, D, initializer=tf.zeros_initializer((5)))
warped = tf.scan(warp_row, nD, initializer=tf.zeros_initializer((6)))
print sess.run(warped, feed_dict={x:nx, y:ny})

########################################################
'''
import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def padded_euclidean(x, y, pad_tok = [0,0], dist=euclidean):
    if dist(x,pad_tok)==0 or dist(y,pad_tok)==0: return 0
    else: return dist(x, y)
a = [0,0]
b = [1,1]
c = [2,2]

print a,b,padded_euclidean(a,b)
print b,c, padded_euclidean(b,c)
x = np.array([[0,0],[0,0], [2,2], [3,3], [4,4], [0,0],[0,0]])
y = np.array([[2,2], [3,3], [4,4]])
distance, path = fastdtw(y, x, dist=padded_euclidean)
print(distance)
print path
'''
