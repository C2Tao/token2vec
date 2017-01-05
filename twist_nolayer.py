from keras.layers import Merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.objectives import categorical_crossentropy, binary_crossentropy
from keras import backend as K
# theano weight indexing: (n_input), n_row, n_col, n_output
# theano tensor indexing: (n_batch), n_row, n_col, n_channel
# my order n_batch, n_time, n_feat_sub, n_feat 
n_win = 2

def pad_zero(x, n_top=0, n_bot=0):
    return K.asymmetric_temporal_padding(x, left_pad = n_top, right_pad = n_bot)

def context(x, n_win):
    w_list = list(reversed(range(2*n_win+1)))
    w_list.pop(n_win)
    x = K.concatenate([pad_zero(x, n_top = 2*n_win - w, n_bot = w) for w in w_list], axis=-1)
    return Cropping1D(cropping=((n_win, n_win)))(x)

def dummy(holder):
    return holder



input_img = Input(shape=(28, 28, 1))
#####################################################
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


print x
# at this point the representation is (4, 28*8, 8)
x = Reshape((4, 28*8*8))(x)
# at this point the representation is (4, 28*8*8)
p = TimeDistributed(Dense(8, activation='softmax'), name = 'posterior')(x)
# at this point the representation is (4, 8)
c = context(p, 2)
c = TimeDistributed(Dense(8, activation = 'softmax'), name = 'predicted')(c)


def antirectifier(x):
    x -= K.mean(x, axis=1, keepdims=True)
    x = K.l2_normalize(x, axis=1)
    pos = K.relu(x)
    neg = K.relu(-x)
    return K.concatenate([pos, neg], axis=1)

def antirectifier_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2  # only valid for 2D tensors
    shape[-1] *= 2
    return tuple(shape)

#model.add(Lambda(antirectifier,
#         output_shape=antirectifier_output_shape))
#error1 = Lambda(categorical_crossentropy( ), output_shape= )

error1 = categorical_crossentropy(p, c)
error1 = K.sum(error1, axis=-1)

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

error2 = binary_crossentropy(input_img, decoded)
error2 = K.sum(error2, axis=-1)
error2 = K.sum(error2, axis=-1)
#print error1
#print error2
#errors = Merge([error1, error2], mode='concat', concat_axis=-1)
#error = Lambda(lambda x,y: x+y)(errors)
error = error1+error2
print error
autoencoder = Model(input_img, error)
encoder = Model(input_img, encoded)
autoencoder.compile(optimizer='adadelta', loss=dummy)


#####################################################3
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

nb_epoch = 1 #50

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

autoencoder.fit(x_train, x_train,
                nb_epoch=nb_epoch,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[])


'''
#####################################################3
from keras.models import load_model

autoencoder.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
encoder.save('encoder.h5')  # creates a HDF5 file 'my_model.h5'
del autoencoder  # deletes the existing model
del encoder  # deletes the existing model

# returns a compiled model
# identical to the previous one
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
'''
