from keras.layers import merge, Reshape, TimeDistributed, Lambda, Cropping1D, Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Layer, BatchNormalization, Dropout, Flatten, Activation, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.regularizers import l1
from keras_layers import Context, Twist
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
#from tensorflow_dtw_batch import warp_dtw
import matplotlib.pyplot as plt
# theano weight indexing: (n_input), n_row, n_col, n_output
# theano tensor indexing: (n_batch), n_row, n_col, n_channel
# my order n_batch, n_time, n_feat_sub, n_feat 
n_tw = 2
#n_chan = 16
n_filtsize = 3
n_post = 128
n_embed = 64
n_context = 2
ACT = 'tanh'
MASK = True
n_batch = 8
drop = 0.5

n_epoch = 1
TRAIN = True
#TRAIN = False
#ew_trans, ew_sharp, ew_recon = 9, 1, 0
ew_dtw, ew_recon = 0, 0
patience = 5
sam_id = 0
model_name = 'dtw{}_filt{}_po{}_em{}_cxt{}_{}_mask{}_batch{}_drop{}_err{}_ep{}_p{}'.format(n_tw,n_filtsize, n_post, n_embed, n_context, ACT, int(MASK), n_batch, drop, ''.join(map(str, [ew_dtw, ew_recon])), n_epoch, patience)
ew = np.array([10**(-ew_dtw), 10**(-ew_recon)])
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
#x_train_feat, x_train_mask = timit_parse.feat_load('feature/mini_document.pkl')
x_train_feat, x_train_mask = timit_parse.feat_load('feature/train_query.pkl')
x_train = np.array(x_train_feat)[:,:,:, np.newaxis]  
mask_train = np.array(x_train_mask)#[:,:,np.newaxis]

x_test_feat, x_test_mask = timit_parse.feat_load('feature/test_query.pkl')
x_test = np.array(x_test_feat)[:,:,:, np.newaxis]  
mask_test = np.array(x_test_mask)#[:,:,np.newaxis]

if not MASK:
    mask_train = np.ones_like(mask_train, dtype = np.float32)
    mask_test = np.ones_like(mask_test, dtype = np.float32)
################################################################################
def dummy_objective(dummy_target, obj):
    return obj

def dummy_target(x):
    return np.zeros(len(x))

def dummy_shape(input_shapes):
    #print input_shapes
    #assert len(input_shapes)==2
    return input_shapes[0][0], 1

def dummy_input(x, ew, mask):
    return [x, x, np.tile(ew, (len(x), 1)), mask]
################################################################################
def obj_trans(inputs):
    from keras.objectives import categorical_crossentropy, binary_crossentropy
    y_answ, y_pred, mask = inputs
    return K.sum(categorical_crossentropy(y_answ, y_pred)*mask, axis=-1)/K.sum(mask, axis=-1)
    #return K.mean(categorical_crossentropy(y_answ, y_pred), axis=-1)*mask

def obj_sharp(inputs):
    y_dist, _, mask = inputs
    ep=1e-8
    print mask
    print y_dist
    y_dist = y_dist*mask

    tot_prob = K.sum(y_dist, axis=-2)/K.sum(mask, axis = -1)
    tot_entro = K.sum(-tot_prob*K.log(tot_prob+ep), axis=-1)
    ind_entro = K.sum(K.sum(-y_dist*K.log(y_dist+ep), axis=-1), axis=-1)/K.sum(mask, axis=-1)
    return (ind_entro-tot_entro)#*ew_sharp

def obj_recon(inputs):
    y_answ, y_pred, mask = inputs
    return K.sum(mask * K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)/K.sum(mask, axis=-1)
    #return K.mean(K.mean(K.mean(K.square(y_answ - y_pred), axis=-1), axis=-1),axis=-1)

def obj_final(inputs):
    errs = K.transpose(K.stack(inputs[:-1]))
    weight = inputs[-1]
    #print 'weight', weight
    #print 'errs', errs
    #print K.sum(weight*errs, axis = 1)
    return K.sum(weight*errs, axis = 1)
    #print K.flatten(K.dot(K.expand_dims(weight, 0),errs))
    #return K.flatten(K.dot(K.expand_dims(weight, 0),errs))

def obj_dtw(inputs):
    import tensorflow as tf
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
    qer, doc = inputs
    return warp_dtw(qer, doc) - warp_dtw(K.reverse(qer, axes = 1), doc)
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

def error_sharp(p, mask):
    err = merge([p, p, mask], mode=obj_trans, output_shape =  dummy_shape, name = 'err_sharp')
    return err

def error_trans(p, e, mask):
    e = Context(n_context)(e)
    c = TimeDistributed(Dense(n_post, activation = 'softmax'), name = 'predicted')(e)
    err = merge([p, c, mask], mode=obj_trans, output_shape =  dummy_shape, name = 'err_trans')
    #e = Context(n_context)(e)
    #c = TimeDistributed(Dense(n_post, activation = 'softmax'), name = 'predicted')(e)
    #err = merge([p,c], mode=obj_trans, output_shape =  dummy_shape, name = 'err_trans')
    return err

def error_dtw(qer, doc):
    err = merge([qer, doc], mode=obj_dtw, output_shape =  dummy_shape, name = 'err_dtw')
    return err

def layer_twist(x, mask, mode='encoder', (cT, cF) = (2, 2), nC = 1, act=ACT,(nT, nF)=(n_filtsize,n_filtsize),drop = drop, w=None):
    if not w: 
        w = Convolution2D(nC, nT, nF, border_mode='same')
    mask = Reshape((K.int_shape(mask)[1],1))(mask)

    x = w(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    if mode=='encoder':
        x = MaxPooling2D((cT, cF), border_mode='same')(x)
        #x = UpSampling2D((1, cF))(x)
        mask = MaxPooling1D(cT, border_mode='same')(mask)
    elif mode=='decoder':
        #x = MaxPooling2D((1, cF), border_mode='same')(x)
        x = UpSampling2D((cT, cF))(x)
        mask = UpSampling1D(cT)(mask)
    x = Dropout(drop)(x)
    
    mask = Reshape((K.int_shape(mask)[1],))(mask)
    return x, mask, w

def layer_posterior(x, nP, nE):
    nB, nT, nF, nC = K.int_shape(x)
    x = Reshape((nT,nF*nC))(x)
    p = TimeDistributed(Dense(nP, activation='softmax'))(x)   
    e = TimeDistributed(Dense(nE, activation='tanh'))(p)   
    return p, e

def layer_reverse(e, x):
    nB, nT, nF, nC = K.int_shape(x)
    x = TimeDistributed(Dense(nF*nC, activation='tanh'))(e)
    x = Reshape((nT, nF, nC))(x)
    return x 

def layer_rename(x, name = None):
    args = K.int_shape(x)
    x = Reshape(args[1:],name=name)(x)
    return x
################################################################################
input_mask = Input(shape=(32, ), name = 'mask')
input_feat = Input(shape=(32, 32, 1), name = 'input')
input_query = Input(shape=(32, 32, 1), name = 'query')
input_errweight = Input(shape=(len(ew),),name='error_weight')


input_holder = Input(shape=(32, 32, 1), name = 'input_holder')
x = input_holder
mask = input_mask
for i in range(n_tw):
    x, mask, _ = layer_twist(x, mask, 'encoder', (2, 2), 4**(i+1))
hidden, e = layer_posterior(x, n_post, n_embed)
#err_trans = error_trans(p, e, mask)
#err_sharp = error_sharp(p, mask)
x = layer_reverse(e, x)
for i in range(n_tw):
    x, mask, _ = layer_twist(x, mask, 'decoder', (2, 2), 4**(n_tw-i-1))
decoded = Convolution2D(1, n_filtsize, n_filtsize, activation='linear', border_mode='same',name='reconstructed')(x)

posterior = Model(input_holder, hidden, 'posterior')
autoencoder = Model(input_holder, decoded, 'autoencoder')

doc_post = layer_rename(posterior(input_feat),'doc_post')
qer_post = layer_rename(posterior(input_query),'qer_post')
err_dtw = error_dtw(doc_post, qer_post)

reconstructed = layer_rename(autoencoder(input_feat),'reconstructed')
err_recon = error_recon(input_feat, reconstructed, mask)
#err_final = error_final([err_trans, err_sharp, err_recon], input_errweight)
err_final = error_final([err_dtw, err_recon], input_errweight)


nndtw = Model([input_feat, input_query, input_errweight, input_mask], err_final)
nndtw.compile(optimizer='adam', loss=dummy_objective)

#tokenizer = Model([input_feat, input_errweight, input_mask], err_final)
#tokenizer.compile(optimizer='adam', loss=dummy_objective)


checkpoint = ModelCheckpoint(filepath='model/'+model_name+'.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')



################################################################################
if TRAIN:
    nndtw.fit(dummy_input(x_train, ew, mask_train), dummy_target(x_train),
                    nb_epoch=n_epoch,
                    batch_size=n_batch,
                    shuffle=True,
                    validation_data = (dummy_input(x_test, ew, mask_test), dummy_target(x_test)),
                    callbacks=[earlystop, checkpoint])
    #tokenizer.save(model_name)
else:
    nndtw = load_model('model/'+model_name+'.h5', custom_objects = {'Context':Context,'dummy_objective':dummy_objective})



################################################################################
x = x_test[sam_id:sam_id+1,:,:,:]
posterior = Model(nndtw.get_layer("input").input, nndtw.get_layer("doc_post").output)
#posterior = nndtw.posterior
autoencoder = Model(nndtw.get_layer("input").input, nndtw.get_layer("reconstructed").output)


f, ax = plt.subplots(3, 1)
ax[0].imshow(x.reshape(32, 32).T, aspect='auto')
ax[0].get_xaxis().set_visible(False)

ax[1].imshow(autoencoder.predict(x).reshape(32,32).T, aspect='auto')
#ax[1].get_xaxis().set_visible(False)

plt.gray()
ax[2].matshow(posterior.predict(x).reshape(-1,n_post).T, aspect='auto')
ax[2].get_xaxis().set_visible(False)

fig = plt.gcf()
fig.set_size_inches(12, 4)
#ax.get_yaxis().set_visible(False)

plt.savefig('figure/'+model_name+'_'+str(sam_id)+'.png')

################################################################################
################################################################################
'''
if TRAIN:
    tokenizer.fit(dummy_input(x_train, ew, mask_train), dummy_target(x_train),
                    nb_epoch=n_epoch,
                    batch_size=n_batch,
                    shuffle=True,
                    validation_data=(dummy_input(x_test, ew, mask_test), dummy_target(x_test)),
                    callbacks=[earlystop, checkpoint])
    #tokenizer.save(model_name)
else:
    tokenizer = load_model('model/'+model_name+'.h5', custom_objects = {'Context':Context,'dummy_objective':dummy_objective,'Twist':Twist})
####################################################

input_feat = tokenizer.get_layer("input").input
input_mask = tokenizer.get_layer("mask").input
autoencoder = Model(input_feat, tokenizer.get_layer("reconstructed").output)
posterior = Model(input_feat, tokenizer.get_layer("posterior").output)
recon_err = Model([input_feat, input_mask], tokenizer.get_layer("err_recon").output)
trans_err = Model([input_feat, input_mask], tokenizer.get_layer("err_trans").output)
x = x_test[sam_id:sam_id+1,:,:,:]

decoded_imgs = autoencoder.predict(x)


f, ax = plt.subplots(3, 1)
ax[0].imshow(x.reshape(512, 32).T, aspect='auto')
#ax[0].get_xaxis().set_visible(False)

ax[1].imshow(decoded_imgs.reshape(512,32).T, aspect='auto')
ax[1].get_xaxis().set_visible(False)

plt.gray()
ax[2].matshow(posterior.predict(x).reshape(-1,n_post).T, aspect='auto')
ax[2].get_xaxis().set_visible(False)

fig = plt.gcf()
fig.set_size_inches(12, 4)
#ax.get_yaxis().set_visible(False)

plt.savefig('figure/'+model_name+'_'+str(sam_id)+'.png')
#plt.show()
#####################################################3

#plt.matshow(posterior.predict(x).reshape(-1,n_post).T)

#plt.gray()
#plt.savefig('figure/'+model_name+'_'+str(sam_id)+'_post.png')
#plt.show()

#rer = recon_err.predict([x_test,mask_test])
#ter = trans_err.predict(x_test)
#print rer, ter
'''
