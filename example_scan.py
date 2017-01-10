import tensorflow as tf
import numpy as np
############################################
# To achieve
# xs = some_func(ys, x0)
# need
# xs[i] = func(xs[i-1], ys[i])
ys = np.array([1, 1, 1, 1, 1, 1])
x0  = np.array(-5) 
def func(x_tm1, y_t):
    # x_t = func(x_t-1, y_t)
    x_t = x_tm1 + y_t
    return x_t
xs = tf.scan(func, ys, x0)
print sess.run(xs)
############################################
#[-4 -3 -2 -1  0  1]
sess=tf.InteractiveSession()
elems = np.array([1, 2, 3, 4, 5, 6]) 
sum0 = tf.scan(lambda a, x: a + x, elems) 
print sess.run(sum0)
#[ 1  3  6 10 15 21]

elems = np.array([0, 0, 0, 0, 0, 0])
initializer = np.array(0) 
sum_one = tf.scan(lambda a, x: x[0] - x[1] + a, (elems + 1, elems), initializer)
print sess.run(sum_one)
#[1 2 3 4 5 6]

elems = np.array([0, 0, 0, 0, 0, 0]) 
initializer = (np.array(0), np.array(1)) 
fibonaccis = tf.scan(lambda a, _: (a[1], a[0] + a[1]), elems, initializer)
print sess.run(fibonaccis)
#(array([1, 1, 2, 3, 5, 8]), array([ 1,  2,  3,  5,  8, 13]))


list_0 = np.array([0, 0, 0, 0, 0, 0])
list_1 = np.array([1, -1, 1, -1, 1, -1])
init= np.array(5) 
def func(init, current_element_of_lists):
    current_element_of_list_0, current_element_of_list_1 = current_element_of_lists
    next_init = current_element_of_list_0 + current_element_of_list_1 + init
    return next_init
print sess.run(tf.scan(func, (list_0, list_1), init))
#[6 5 6 5 6 5]





