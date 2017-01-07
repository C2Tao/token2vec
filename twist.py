from keras.layers import merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Layer, BatchNormalization, Dropout, Flatten, Activation
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.regularizers import l1
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

class BN(BatchNormalization):
    def get_output_shape_for(self, input_shape):
        return input_shape

class Context(Layer):
    '''
    basic requirement for a custom layer:
    define 5 functions:
    [__init__, build, call, get_output_shape_for, get_config]

    def __init__(self, [INSERT ARGS HERE], **kwargs): 
        #define your own parameters, end with
        super(Context, self).__init__(**kwargs)

    def build(self, input_shape): 
        #define the layer parameters, end with
        super(Context, self).build(input_shape)

    def call(self, x, mask=None):
        #calculations happen here

    def get_output_shape_for(self, input_shape): 
        #same as Lambda layer output size function

    def get_config(self): 
        #add all parameters used in init to config
    '''

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


#####################################################3
input_img = Input(shape=(28*n_seq, 28, 1), name='input')

x = input_img

# at this point the representation is (28, 28, 1)
x = Convolution2D(8, 3, 3, border_mode='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 1), border_mode='same')(x)
x = UpSampling2D((1, 2))(x)
x = Dropout(0.5)(x)
# at this point the representation is (14, 28*2, 8)
x = Convolution2D(8, 3, 3, border_mode='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 1), border_mode='same')(x)
x = UpSampling2D((1, 2))(x)
x = Dropout(0.5)(x)
# at this point the representation is (7, 28*4, 8)


x = Reshape((7*n_seq, 28*4*8))(x)
# at this point the representation is (7, 28*4*8)

#p = TimeDistributed(Dense(n_post, activation='softmax',W_regularizer=l1(0.01)), name = 'posterior')(x)
p = TimeDistributed(Dense(n_post, activation='softmax'), name = 'posterior')(x)

#p = Flatten()(p)
#p = BatchNormalization(mode=1)(p)
#p = Reshape((7*n_seq, n_post))(p)

# at this point the representation is (7, 100)

c = Context(n_context)(p)
c = TimeDistributed(Dense(n_post, activation = 'softmax'), name = 'predicted')(c)
err_trans = merge([p,c], mode=obj_trans, output_shape =  shape_final, name = 'err_trans')

x = TimeDistributed(Dense(28*4*8, activation='tanh'))(p)
# at this point the representation is (7, 28*4*8)
x = Reshape((7*n_seq, 28*4, 8))(x)


# at this point the representation is (7, 28*4, 8)
x = Convolution2D(8, 3, 3, border_mode='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((1, 2), border_mode='same')(x)
x = UpSampling2D((2, 1))(x)
x = Dropout(0.5)(x)
# at this point the representation is (14, 28*2, 8)
x = Convolution2D(8, 3, 3, border_mode='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((1, 2), border_mode='same')(x)
x = UpSampling2D((2, 1))(x)
x = Dropout(0.5)(x)
# at this point the representation is (28, 28, 8)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name = 'reconstructed')(x)
err_recon = merge([input_img, decoded], mode=obj_recon, output_shape = shape_final, name = 'err_recon')
err_final = merge([err_trans, err_recon], mode=obj_final, output_shape = shape_final, name = 'err_final')

tokenizer = Model(input_img, err_final)
tokenizer.compile(optimizer='adam', loss=dummy_objective)
#####################################################
#tokenizer = load_model('tokenizer.h5', custom_objects = {'Context':Context,'dummy_objective':dummy_objective})
####################################################
tokenizer.fit(x_train, dummy_target(x_train),
                nb_epoch=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, dummy_target(x_test)),
                callbacks=[])
tokenizer.save('tokenizer.h5')
#####################################################

input_img = tokenizer.get_layer("input").input
autoencoder = Model(input_img, tokenizer.get_layer("reconstructed").output)
posterior = Model(input_img, tokenizer.get_layer("posterior").output)
recon_err = Model(input_img, tokenizer.get_layer("err_recon").output)
trans_err = Model(input_img, tokenizer.get_layer("err_trans").output)


import matplotlib.pyplot as plt
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28*n_seq, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28*n_seq, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#####################################################3
plt.matshow(posterior.predict(x_test)[:1].reshape(-1,n_post*1))
plt.show()

rer = recon_err.predict(x_test)
ter = trans_err.predict(x_test)
print rer, ter

