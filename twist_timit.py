from keras.layers import merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Layer, BatchNormalization, Dropout, Flatten, Activation
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.regularizers import l1
from keras_layers import Context, Twist
import numpy as np
# theano weight indexing: (n_input), n_row, n_col, n_output
# theano tensor indexing: (n_batch), n_row, n_col, n_channel
# my order n_batch, n_time, n_feat_sub, n_feat 
n_post = 100
n_embed = n_post
n_context = 1
n_batch = 20
n_epoch = 100
n_size = 5
n_chan = 32
ew_trans, ew_sharp, ew_recon = 0.1, 1, 10
TRAIN=True
#TRAIN=False
ew = np.array([ew_trans, ew_sharp, ew_recon])
################################################################################
'''
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))[:50,:,:,:]
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_train = np.reshape(x_train, (len(x_train)/n_seq, n_seq*28, 28, 1))
x_test = np.reshape(x_test, (len(x_test)/n_seq, n_seq*28, 28, 1))
'''
import timit_parse
x_train_feat, x_train_mask = timit_parse.feat_load('feature/mini_document.pkl')
x_train = np.array(x_train_feat)[:,:,:, np.newaxis]  
mask_train = np.array(x_train_mask)
################################################################################
def dummy_objective(dummy_target, obj):
    return obj

def dummy_target(x):
    return np.zeros(len(x))

def dummy_shape(input_shapes):
    #print input_shapes
    #assert len(input_shapes)==2
    return input_shapes[0][0], 1
################################################################################
def obj_trans(inputs):
    from keras.objectives import categorical_crossentropy, binary_crossentropy
    y_answ, y_pred = inputs
    return K.mean(categorical_crossentropy(y_answ, y_pred), axis=-1)#*ew_trans

def obj_sharp(inputs):
    y_dist, _ = inputs
    ep=1e-8
    tot_prob = K.mean(y_dist, axis=-2)
    tot_entro = K.mean(-tot_prob*K.log(tot_prob+ep), axis=-1)
    ind_entro = K.mean(K.mean(-y_dist*K.log(y_dist+ep), axis=-1), axis=-1)
    return (ind_entro-tot_entro)#*ew_sharp

def obj_recon(inputs):
    y_answ, y_pred, mask = inputs
    #return K.mean(K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)#*ew_recon
    return K.mean(mask * K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)

def obj_final(inputs):
    errs = K.transpose(K.stack(inputs[:-1]))
    weight = inputs[-1]
    #print 'weight', weight
    #print 'errs', errs
    #print K.sum(weight*errs, axis = 1)
    return K.sum(weight*errs, axis = 1)
    #print K.flatten(K.dot(K.expand_dims(weight, 0),errs))
    #return K.flatten(K.dot(K.expand_dims(weight, 0),errs))
################################################################################

def error_final(err_list, err_weights):
    #def error_final(err_list):
    #weights = K.stack(err_weights)
    #weighted_error = map(lambda e,w:e*w, err_list, err_weights)
    err = merge(err_list+[err_weights], mode=obj_final, output_shape = dummy_shape, name = 'err_final')
    #err = merge(err_list, mode='sum', name = 'err_final')
    return err

def error_recon(input_feat, decoded, mask):
    err = merge([input_feat, decoded, mask], mode=obj_recon, output_shape = dummy_shape, name = 'err_recon')
    return err

def error_sharp(p):
    err = merge([p, p], mode=obj_trans, output_shape =  dummy_shape, name = 'err_sharp')
    return err

def error_trans(p, e):
    e = Context(n_context)(p)
    c = TimeDistributed(Dense(n_post, activation = 'softmax'), name = 'predicted')(e)
    err = merge([p,c], mode=obj_trans, output_shape =  dummy_shape, name = 'err_trans')
    #e = Context(n_context)(e)
    #c = TimeDistributed(Dense(n_post, activation = 'softmax'), name = 'predicted')(e)
    #err = merge([p,c], mode=obj_trans, output_shape =  dummy_shape, name = 'err_trans')
    return err

def layer_twist(x, mode='encoder', (cT, cF) = (2, 2), (nT, nF, nC)=(n_size,n_size,n_chan), w=None,name=None):
    if not w: 
        w = Convolution2D(nC, nT, nF, border_mode='same')
    x = w(x)
    x = BatchNormalization()(x)
    #x = Activation('linear')(x)
    x = Activation('tanh')(x)
    #x = Activation('relu')(x)
    if mode=='encoder':
        x = MaxPooling2D((cT, 1), border_mode='same')(x)
        x = UpSampling2D((1, cF))(x)
    elif mode=='decoder':
        x = MaxPooling2D((1, cF), border_mode='same')(x)
        x = UpSampling2D((cT, 1))(x)
    x = Dropout(0.5, name = name)(x)
    return x, w

def layer_posterior(x, nP, nE):
    nB, nT, nF, nC = K.int_shape(x)
    x = Reshape((nT,nF*nC))(x)
    p = TimeDistributed(Dense(nP, activation='softmax'), name = 'posterior')(x)   
    e = TimeDistributed(Dense(nE, activation='tanh'), name = 'embedding')(p)   
    return p, e

def layer_reverse(e, x):
    nB, nT, nF, nC = K.int_shape(x)
    x = TimeDistributed(Dense(nF*nC, activation='tanh'))(e)
    x = Reshape((nT, nF, nC))(x)
    return x 

def layer_rename(x, name = None):
    nB, nT, nF, nC = K.int_shape(x)
    x = Reshape((nT, nF, nC),name=name)(x)
    return x
################################################################################
input_feat = Input(shape=(512, 32, 1), name = 'input')
mask = Input(shape=(512, ), name = 'mask')
err_weight = Input(shape=(len(ew),),name='error_weight')
x = input_feat

x, _ = layer_twist(x, 'encoder', (4, 1))
x, _ = layer_twist(x, 'encoder', (2, 2))
x, _ = layer_twist(x, 'encoder', (2, 2))

p, e = layer_posterior(x, n_post, n_embed)
err_trans = error_trans(p, e)
err_sharp = error_sharp(p)
x = layer_reverse(e, x)

x, _ = layer_twist(x, 'decoder', (2, 2))
x, _ = layer_twist(x, 'decoder', (2, 2))
x, _ = layer_twist(x, 'decoder', (4, 1))

x = Convolution2D(1, n_size, n_size, activation='linear', border_mode='same')(x)

decoded = layer_rename(x, 'reconstructed')

err_recon = error_recon(input_feat, decoded, mask)
err_final = error_final([err_trans, err_sharp, err_recon], err_weight)

tokenizer = Model([input_feat, err_weight, mask], err_final)
tokenizer.compile(optimizer='adam', loss=dummy_objective)


def dummy_input(x, ew, mask):
    return [x, np.tile(ew, (len(x), 1)), mask]
    #return [x, ew]
################################################################################
if TRAIN:
    tokenizer.fit(dummy_input(x_train, ew, mask_train), dummy_target(x_train),
                    nb_epoch=n_epoch,
                    batch_size=n_batch,
                    shuffle=True,
                    validation_data=(dummy_input(x_train, ew, mask_train), dummy_target(x_train)),
                    callbacks=[])
    tokenizer.save('model/timit_mini.h5')
else:
    tokenizer = load_model('model/timit_mini.h5', custom_objects = {'Context':Context,'dummy_objective':dummy_objective,'Twist':Twist})
####################################################
import matplotlib.pyplot as plt

input_feat = tokenizer.get_layer("input").input
mask = tokenizer.get_layer("mask").input
autoencoder = Model(input_feat, tokenizer.get_layer("reconstructed").output)
posterior = Model(input_feat, tokenizer.get_layer("posterior").output)
recon_err = Model([input_feat, mask], tokenizer.get_layer("err_recon").output)
trans_err = Model(input_feat, tokenizer.get_layer("err_trans").output)
decoded_imgs = autoencoder.predict(x_train)

#plt.figure(figsize=(20, 4))

ax = plt.subplot(2, 1, 1)
plt.imshow(x_train[0].reshape(512, 32).T)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(2, 1, 2)
plt.imshow(decoded_imgs[0].reshape(512, 32).T)
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()
#####################################################3
plt.matshow(posterior.predict(x_train)[:1].reshape(-1,n_post).T)
plt.show()

rer = recon_err.predict([x_train,mask_train])
ter = trans_err.predict(x_train)
print rer, ter
