import tensorflow as tf
import numpy as np
sess=tf.InteractiveSession()

def squared_dist(tx, ty): 
    dm = 3#len(tx.get_shape())
    expanded_x = tf.expand_dims(tx, dm-1)
    expanded_y = tf.expand_dims(ty, dm-2)
    distances = tf.sqrt(tf.reduce_sum(tf.squared_difference(expanded_x, expanded_y), dm))
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

def zero(val, dy_shape):
    
    x = tf.placeholder(tf.float32, shape=[None, 4])
    zeros_dims = tf.pack([tf.shape(x)[0], 7])
    tf.zeros([d1, d2])
    y = tf.fill(zeros_dims, 0.0)

def warp_ele(m_i, vecs):
    m_t, m_tm1, v_i = vecs
    print 'm_t',m_t
    print 'm_tm1',m_tm1
    print 'm_i',m_i
    print 'v_i',v_i
    m_ip1 = tf.reduce_min(tf.stack([m_t, m_tm1, m_i], axis=1), axis=1) 
    print 'm_ip1',m_ip1
    return m_ip1 + v_i

def warp_row(m_t, v_t):
    inf = tf.ones_like(m_t[0,:])*np.inf
    m_tm1 = tf.concat(0, [tf.expand_dims(inf, 0), m_t[:-1,:]])
    print 'inf', inf
    print 'm_tm1',m_tm1
    print 'm_t',m_t
    print 'v_t',v_t
    m_tp1 = tf.scan(warp_ele,(m_t, m_tm1, v_t), initializer=inf)
    #print 'm_tp1',m_tp1
    return m_tp1

def warp_dtw(x, y):
    '''
    print x,'x'
    print y,'y'
    D = squared_dist(x, y)
    print D
    warped = tf.scan(warp_row, D, initializer=tf.zeros(D.get_shape()[1]))
    print warped
    '''
    D = squared_dist(x, y)
    D = tf.transpose(D, perm=[1,2,0])
    #d0,d1,d2 = D.get_shape()
    dy_zero = tf.zeros_like(D[0,:,:])
    #print 'dy_zero',dy_zero

    warped = tf.scan(warp_row, D, initializer=dy_zero)
    #warped = tf.transpose(warped, perm = [2,0,1])
    #print 'warped',warped
    finals = warped[-1,:,:]
    #print 'finals',finals
    dist = tf.reduce_min(finals, axis = 0)
    return dist


'''
if __name__=='__main__':
    nx = np.array([[2,2], [3,3], [4,4]], dtype=np.float32)
    ny = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]], dtype=np.float32)
    nx = np.tile(nx,[100,1,1])
    ny = np.tile(ny,[100,1,1])
    x = tf.placeholder(tf.float32, (100,3,2))
    y = tf.placeholder(tf.float32, (100,6,2))
    dists = tf.map_fn(warp_dtw, (x, y))
    #distance = warp_dtw(x,y)
    print dists.eval(feed_dict={x:nx, y:ny})
'''
if __name__=='__main__':
    nx = np.array([[2,2], [3,3], [4,4]], dtype=np.float32)
    ny = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]], dtype=np.float32)
    nx = np.tile(nx,[100,1,1])
    ny = np.tile(ny,[100,1,1])
    x = tf.placeholder(tf.float32, None)
    y = tf.placeholder(tf.float32, None)
    #x = tf.placeholder(tf.float32, (100,3,2))
    #y = tf.placeholder(tf.float32, (100,6,2))
    
    dist = warp_dtw(x, y)
    print sess.run(dist, feed_dict = {x:nx, y:ny})
