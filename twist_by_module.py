from keras.layers import merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Layer, BatchNormalization, Dropout, Flatten, Activation
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.regularizers import l1
from keras_layers import Context, Twist
# theano weight indexing: (n_input), n_row, n_col, n_output
# theano tensor indexing: (n_batch), n_row, n_col, n_channel
# my order n_batch, n_time, n_feat_sub, n_feat 
n_post = 20
n_context = 3
n_seq = 1
################################################################################
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))[:50,:,:,:]
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = np.reshape(x_train, (len(x_train)/n_seq, n_seq*28, 28, 1))
x_test = np.reshape(x_test, (len(x_test)/n_seq, n_seq*28, 28, 1))

################################################################################
def dummy_objective(dummy_target, obj):
    return obj

def dummy_target(x):
    return np.zeros(len(x))

def dummy_shape(input_shapes):
    assert len(input_shapes)==2
    return input_shapes[0][0], 1
################################################################################
def obj_trans(inputs):
    from keras.objectives import categorical_crossentropy, binary_crossentropy
    y_answ, y_pred = inputs
    return K.mean(categorical_crossentropy(y_answ, y_pred), axis=-1)*0.001

def obj_sharp(inputs):
    y_dist, _ = inputs
    ep=1e-8
    tot_prob = K.mean(y_dist, axis=-2)
    tot_entro = K.mean(-tot_prob*K.log(tot_prob+ep), axis=-1)
    ind_entro = K.mean(K.mean(-y_dist*K.log(y_dist+ep), axis=-1), axis=-1)
    return ind_entro-tot_entro

def obj_recon(inputs):
    y_answ, y_pred = inputs
    return K.mean(K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)

def obj_final(inputs):
    errs = K.stack(inputs[:-1])
    weight = inputs[-1]
    print K.flatten(K.dot(K.expand_dims(weight, 0),errs))
    return K.flatten(K.dot(K.expand_dims(weight, 0),errs))
################################################################################

#def error_final(err_list, err_weights):
def error_final(err_list):
    #weights = K.stack(err_weights)
    #weighted_error = map(lambda e,w:e*w, err_list, err_weights)
    #err = merge(err_list+[weights], mode=obj_final, output_shape = dummy_shape, name = 'err_final')
    err = merge(err_list, mode='sum', name = 'err_final')
    return err

def error_recon(input_img, decoded):
    err = merge([input_img, decoded], mode=obj_recon, output_shape = dummy_shape, name = 'err_recon')
    return err

def error_sharp(p):
    err = merge([p, p], mode=obj_trans, output_shape =  dummy_shape, name = 'err_sharp')
    return err

def error_trans(p, e):
    e = Context(n_context)(e)
    c = TimeDistributed(Dense(n_post, activation = 'softmax'), name = 'predicted')(e)
    err = merge([p,c], mode=obj_trans, output_shape =  dummy_shape, name = 'err_trans')
    return err

def layer_twist(x, mode='encoder', (nX, nT, nF, nC)=(2,3,3,8),name=None):
    x = Convolution2D(nC, nT, nF, border_mode='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    if mode=='encoder':
        x = MaxPooling2D((nX, 1), border_mode='same')(x)
        x = UpSampling2D((1, nX))(x)
    elif mode=='decoder':
        x = MaxPooling2D((1, nX), border_mode='same')(x)
        x = UpSampling2D((nX, 1))(x)
    x = Dropout(0.5, name = name)(x)
    return x 

def layer_posterior(x, nP, nE):
    nB, nT, nF, nC = K.int_shape(x)
    x = Reshape((nT,nF*nC))(x)
    p = TimeDistributed(Dense(nP, activation='softmax'), name = 'posterior')(x)   
    e = TimeDistributed(Dense(nE, activation='relu'), name = 'embedding')(p)   
    return p, e

def layer_reverse(e, x):
    nB, nT, nF, nC = K.int_shape(x)
    x = TimeDistributed(Dense(nF*nC, activation='relu'))(e)
    x = Reshape((nT, nF, nC))(x)
    return x 
################################################################################
input_img = Input(shape=(28*n_seq, 28, 1), name='input')

x = input_img
x = layer_twist(x, 'encoder', (2,3,3,8))
x = layer_twist(x, 'encoder', (2,3,3,8))
p, e = layer_posterior(x, n_post, 20)

err_trans = error_trans(p, e)
err_sharp = error_sharp(p)

x = layer_reverse(e, x)
x = layer_twist(x, 'decoder', (2,3,3,8))
#x = layer_twist(x, 'decoder', (2,3,3,1), 'reconstructed')
x = layer_twist(x, 'decoder', (2,3,3,8))

#decoded = x
decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name = 'reconstructed')(x)
err_recon = error_recon(input_img, decoded)
err_final = error_final([err_trans, err_sharp, err_recon])

tokenizer = Model(input_img, err_final)
tokenizer.compile(optimizer='adam', loss=dummy_objective)

tokenizer.fit(x_train, dummy_target(x_train),
                nb_epoch=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, dummy_target(x_test)),
                callbacks=[])
tokenizer.save('tokenizer_module.h5')
#####################################################
#tokenizer = load_model('tokenizer_module.h5', custom_objects = {'Context':Context,'dummy_objective':dummy_objective,'Twist':Twist})
####################################################
import matplotlib.pyplot as plt

input_img = tokenizer.get_layer("input").input
autoencoder = Model(input_img, tokenizer.get_layer("reconstructed").output)
posterior = Model(input_img, tokenizer.get_layer("posterior").output)
recon_err = Model(input_img, tokenizer.get_layer("err_recon").output)
trans_err = Model(input_img, tokenizer.get_layer("err_trans").output)
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
plt.matshow(posterior.predict(x_test)[:10].reshape(-1,n_post*10))
plt.show()

rer = recon_err.predict(x_test)
ter = trans_err.predict(x_test)
print rer, ter
