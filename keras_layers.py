from keras.layers import merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Layer, BatchNormalization, Dropout, Flatten, Activation
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.regularizers import l1
from tensorflow_dtw_batch import warp_dtw
# theano weight indexing: (n_input), n_row, n_col, n_output
# theano tensor indexing: (n_batch), n_row, n_col, n_channel
# my order n_batch, n_time, n_feat_sub, n_feat 
n_post = 20
n_context = 3
n_seq = 1
#####################################################3
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))[:50,:,:,:]
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = np.reshape(x_train, (len(x_train)/n_seq, n_seq*28, 28, 1))
x_test = np.reshape(x_test, (len(x_test)/n_seq, n_seq*28, 28, 1))

#####################################################3
def dummy_objective(dummy_target, obj):
    return obj

def dummy_target(x):
    return np.zeros(len(x))

def obj_trans(inputs):
    from keras.objectives import categorical_crossentropy, binary_crossentropy
    y_answ, y_pred = inputs
    ep=1e-8
    trans = K.mean(categorical_crossentropy(y_answ, y_pred), axis=-1)
    tot_prob = K.mean(y_answ, axis=-2)
    tot_entro = K.mean(-tot_prob*K.log(tot_prob+ep), axis=-1)
    ind_entro = K.mean(K.mean(-y_answ*K.log(y_answ+ep), axis=-1), axis=-1)
    
    return 0.001*trans+1*(-tot_entro)+1*(ind_entro)

def obj_recon(inputs):
    y_answ, y_pred = inputs
    recon = K.mean(K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)
    return recon

def obj_entro(inputs):
    y_answ, y_pred = inputs
    prob = K.sum(y_pred, axis=-2)
    return K.sum(-prob*K.log(prob), axis=-1)

def obj_final(inputs):
    trans_err, recon_err = inputs
    return trans_err*1 + recon_err*1

def shape_final(input_shapes):
    assert len(input_shapes)==2
    return input_shapes[0][0], 1


class Example(Layer):
    def __init__(self, [INSERT ARGS HERE], **kwargs): 
        #define your own arguments, end with
        self.ARG1 = ARG1
        self.ARG2 = ARG2
        super(Context, self).__init__(**kwargs)

    def build(self, input_shape): 
        #define the layer parameters, end with
        super(Context, self).build(input_shape)

    def call(self, input_tensor_list, mask=None):
        #calculations happen here, input and output can be lists
        #used in the functional api like c,d = layer([a,b])
        x1, x2 = input_tensor_list
        return output_tensor_list

    def get_output_shape_for(self, input_shape): 
        #same as Lambda layer output size function
        #input and output includes batch size
        n_batch, dim[0], dim[1], dim[2] = input_shape
        return n_batch, dim[0], dim[1], dim[2]

    def get_config(self): 
        #add all parameters used in init to config
        config = {'ARG1': self.ARG1, 'ARG2':self.ARG2}
        base_config = super(Context, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Warp(Layer):
    def __init__(self, doc_len, qer_len, **kwargs): 
        #define your own arguments, end with
        self.ARG1 = ARG1
        self.ARG2 = ARG2
        super(Context, self).__init__(**kwargs)

    def build(self, input_shape): 
        #define the layer parameters, end with
        super(Context, self).build(input_shape)

    def call(self, input_tensor_list, mask=None):
        pD, pQ = input_tensor_list
        
        qer = tf.placeholder(tf.float32, (100,3,2))
        doc = tf.placeholder(tf.float32, (100,6,2))
    
        dist = warp_dtw(qer, doc)
        
        
        
        return output_tensor_list

    def get_output_shape_for(self, input_shape): 
        #same as Lambda layer output size function
        #input and output includes batch size
        n_batch, dim[0], dim[1], dim[2] = input_shape
        return n_batch, dim[0], dim[1], dim[2]

    def get_config(self): 
        #add all parameters used in init to config
        config = {'ARG1': self.ARG1, 'ARG2':self.ARG2}
        base_config = super(Context, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Context(Layer):
    def __init__(self, n_win, **kwargs):
        self.n_win = n_win
        super(Context, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Context, self).build(input_shape)

    def call(self, x, mask=None):
        return self._context(x, self.n_win)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]*self.n_win*2
    
    def get_config(self):
        config = {'n_win': self.n_win}
        base_config = super(Context, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def _pad_zero(self, x, n_left=0, n_right=0):
        return K.asymmetric_temporal_padding(x, left_pad = n_left, right_pad = n_right)

    def _context(self, x, n_win):
        w_list = list(reversed(range(2*n_win+1)))
        w_list.pop(n_win)
        x = K.concatenate([self._pad_zero(x, n_left = 2*n_win - w, n_right = w) for w in w_list], axis=-1)
        return Cropping1D(cropping=((n_win, n_win)))(x)


