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

nD = np.array([[1,0,1,1,1,1],[1,1,0,1,1,1]])
print pseudo_code_warp(nD) 


z = tf.zeros_initializer((1,5))

def warp_sca(s_im1, s_i):
    return s_im1+s_i

def warp_row(m_t, v_t):
    #return v_tm1+v_t
    vs = tf.unstack(v_t, axis=0)
    zz = 0.0#np.inf
    return m_t+tf.scan(warp_sca,v_t, initializer=zz)



print sess.run(z)
print z.eval()
print D.eval(feed_dict={x:nx, y:ny})


warped = tf.scan(warp_row, D, initializer=z)
#warped = tf.scan(warp_row, D)

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
