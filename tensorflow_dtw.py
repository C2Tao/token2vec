import tensorflow as tf
import numpy as np
sess=tf.InteractiveSession()

def squared_dist(tx, ty): 
    dm = len(tx.get_shape())
    expanded_x = tf.expand_dims(tx, dm-1)
    expanded_y = tf.expand_dims(ty, dm-2)
    distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(expanded_x, expanded_y), 2))
    return distances

def pseudo_code_warp(D):
    nQ, nD = D.shape
    M=np.zeros((nQ+1, nD+1),dtype= np.float32)
    M[:,0]=np.inf
    M[0,:]=0
    for i in range(nQ):
        for j in range(nD):
            M[i+1,j+1] = min(M[i,j], M[i,j+1], M[i+1,j]) + D[i,j]
    return M[1:, 1:]

#nD = np.array([[1,0,1,1,1,1],[1,1,0,1,1,1]],dtype=np.float32)
#print pseudo_code_warp(nD) 

def warp_ele(m_i, vecs):
    m_t, m_tm1, v_i = vecs
    #print 'm_tm1',m_tm1
    #print 'm_t',m_t
    #print 'v_i',v_i
    m_ip1 = tf.reduce_min(tf.pack([m_t, m_tm1, m_i]))+v_i
    return m_ip1

def warp_row(m_t, v_t):
    v_s = [tf.constant(np.inf)] + tf.unstack(m_t, axis=0)[:-1]
    m_tm1 = tf.pack(v_s)
    #print 'm_tm1',m_tm1
    #print 'm_t',m_t
    #print 'v_s',v_s
    #print 'v_t',v_t
    m_tp1 = tf.scan(warp_ele,(m_t, m_tm1, v_t), initializer=np.inf)
    return m_tp1

def warp_dtw(x, y):
    D = squared_dist(x, y)
    warped = tf.scan(warp_row, D, initializer=tf.zeros(D.get_shape()[1]))
    return tf.reduce_min(warped[-1])



if __name__=='__main__':
    nx = np.array([[2,2], [3,3], [4,4]], dtype=np.float32)
    ny = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]], dtype=np.float32)

    x = tf.placeholder(tf.float32, (3,2))
    y = tf.placeholder(tf.float32, (6,2))

    distance = warp_dtw(x,y)
    print distance.eval(feed_dict={x:nx, y:ny})

