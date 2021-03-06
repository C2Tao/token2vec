from keras.layers import merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Layer
from keras.models import Model
from keras.objectives import categorical_crossentropy, binary_crossentropy
from keras import backend as K
# theano weight indexing: (n_input), n_row, n_col, n_output
# theano tensor indexing: (n_batch), n_row, n_col, n_channel
# my order n_batch, n_time, n_feat_sub, n_feat 

def pad_zero(x, n_top=0, n_bot=0):
    return K.asymmetric_temporal_padding(x, left_pad = n_top, right_pad = n_bot)

def context(x, n_win):
    w_list = list(reversed(range(2*n_win+1)))
    w_list.pop(n_win)
    x = K.concatenate([pad_zero(x, n_top = 2*n_win - w, n_bot = w) for w in w_list], axis=-1)
    return Cropping1D(cropping=((n_win, n_win)))(x)

def dummy_objective(dummy_target, obj):
    return obj

def dummy_target(x):
    return np.zeros(len(x))


def obj_trans(inputs):
    from keras.objectives import categorical_crossentropy, binary_crossentropy
    y_answ, y_pred = inputs
    return K.mean(categorical_crossentropy(y_answ, y_pred), axis=-1)

def obj_recon(inputs):
    y_answ, y_pred = inputs
    return K.mean(K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)

def obj_final(inputs):
    obj1, obj2 = inputs
    return obj1+obj2

def shape_final(input_shapes):
    assert len(input_shapes)==2
    return input_shapes[0][0], 1

def shape_context(input_shapes, n_win):
    post_shape = input_shapes
    return post_shape[0], post_shape[1], post_shape[2]*n_win*2


class MyLayer(Layer):
    def __init__(self, n_win, **kwargs):
        self.n_win = n_win
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyLayer, self).build(input_shape)

    def call(self, x, mask=None):
        return context(x, self.n_win)

    def get_output_shape_for(self, input_shape):
        return shape_context(input_shape, self.n_win)
    
    def get_config(self):
        config = {'n_win': self.n_win}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


#####################################################
input_img = Input(shape=(28, 28, 1))
# at this point the representation is (28, 28, 1)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = UpSampling2D((1, 2))(x)
x = MaxPooling2D((2, 1), border_mode='same')(x)

# at this point the representation is (14, 28*2, 16)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((1, 2))(x)
x = MaxPooling2D((2, 1), border_mode='same')(x)

# at this point the representation is (7, 28*4, 8)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((1, 2))(x)
x = MaxPooling2D((2, 1), border_mode='same')(x)#encoded
encoded = x


# at this point the representation is (4, 28*8, 8)
x = Reshape((4, 28*8*8))(x)
# at this point the representation is (4, 28*8*8)
p = TimeDistributed(Dense(8, activation='softmax'), name = 'posterior')(x)
# at this point the representation is (4, 8)


#c = Lambda(context, shape_context, {'n_win':2})(p) 
c = MyLayer(2)(p)
#c = context(p, 2)
c = TimeDistributed(Dense(8, activation = 'softmax'), name = 'predicted')(c)


err_trans  = merge([p,c], mode=obj_trans, output_shape =  shape_final)

x = TimeDistributed(Dense(28*8*8, activation='tanh'))(x)
# at this point the representation is (4, 28*8*8)
x = Reshape((4, 28*8, 8))(x)


# at this point the representation is (4, 28*8, 8)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 1))(x)
x = MaxPooling2D((1, 2), border_mode='same')(x)

# at this point the representation is (8, 28*4, 8)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 1))(x)
x = MaxPooling2D((1, 2), border_mode='same')(x)

# at this point the representation is (16, 28*2, 8)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 1))(x)
x = MaxPooling2D((1, 2), border_mode='same')(x)

# at this point the representation is (32, 28, 16)
x = Convolution2D(16, 5, 1, activation='relu', border_mode = 'valid' )(x)
#context = Lambda(context)((x,2))

# at this point the representation is (28, 28, 16)
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name = 'reconstructed')(x)
print decoded
err_recon = merge([input_img, decoded], mode=obj_recon, output_shape = shape_final, name = 'recon')
err_final = merge([err_trans, err_recon], mode=obj_final, output_shape = shape_final, name = 'finalxxx')
print err_recon, err_trans
output = err_final

tokenizer = Model(input_img, output)
tokenizer.compile(optimizer='adadelta', loss=dummy_objective)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
#autoencoder.compile(optimizer='adadelta', loss=binary_crossentropy)


#####################################################3
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

nb_epoch = 1 #50

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))[:50,:,:,:]
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

tokenizer.fit(x_train, dummy_target(x_train),
                nb_epoch=nb_epoch,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, dummy_target(x_test)),
                callbacks=[])
#####################################################3
from keras.models import load_model
tokenizer.save('tokenizer.h5')
autoencoder.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
encoder.save('encoder.h5')  # creates a HDF5 file 'my_model.h5'
del tokenizer
del autoencoder  # deletes the existing model
del encoder  # deletes the existing model

# returns a compiled model
# identical to the previous one
tokenizer = load_model('tokenizer.h5', custom_objects = {'MyLayer':MyLayer,'dummy_objective':dummy_objective})
autoencoder = load_model('my_model.h5')
encoder = load_model('encoder.h5')
#####################################################3

import matplotlib.pyplot as plt
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n+1):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
#####################################################3

