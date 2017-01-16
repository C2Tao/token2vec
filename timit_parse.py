import os
import numpy as np
import subprocess
root = '/home/c2tao/TIMIT/timit/'
wav_root = '/home/c2tao/corpus/timit_sentence_wav/'
word_root = '/home/c2tao/corpus/timit_word_wav/'

def get_timit_list():
    timit_list = []
    for type in sorted(['train','test']):
        for dir0 in sorted(os.listdir(root+type+'/')):
            for dir1 in sorted(os.listdir(root+type+'/'+dir0+'/')):
                for dir2 in sorted(os.listdir(root+type+'/'+dir0+'/'+dir1+'/')):
                    if 'phn' in dir2: 
                        name = type+'_'+dir0+'_'+dir1+'_'+dir2[:-4]
                        timit_list.append(name)
    return sorted(timit_list)

def timit_path(name):
    return os.path.join(root,*name.split('_'))

def timit_dur(name):
    '''
    try:
        return float(open(timit_path(name)+'.txt','r').readlines()[0].split()[1])
    except:
        print "something missing with in ", name, ".txt"
        return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
        # test_dr3_fpkt0_sil_1538.wrd
    '''
    if name=='test_dr3_fpkt0_si1538': 
        return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
    if name=='train_dr4_fpaf0_sx154':
        return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
    return float(open(timit_path(name)+'.phn','r').readlines()[-1].split()[1])
    #return float(open(timit_path(name)+'.txt','r').readlines()[0].split()[1])


def timit_parse(name, dot):
    #dot = word or phn
    tuples = []
    for line in open(timit_path(name)+'.'+dot,'r'):
        beg, end, tok = line.strip().split()
        tuples.append([float(beg), float(end), tok])
    beg_list, end_list, tag_list = zip(*tuples)
    tag_list = list(tag_list)
    int_list = list(end_list)
    dur = timit_dur(name)
    
    if beg_list[0]>0:
        int_list.insert(0, beg_list[0])
        tag_list.insert(0, 'sil')
    if end_list[-1]<dur:
        int_list.append(dur) 
        tag_list.append('sil') 
    while True:
        for i in range(1, len(int_list)):
            if int_list[i-1]>=int_list[i]:
                print "warning: the alignment of", name, dot," is abnormal"
                print tag_list[i-1], int_list[i-1]
                print tag_list[i], int_list[i]
                print int_list
                print tag_list
                int_list.pop(i-1)
                tag_list.pop(i-1)
                print int_list
                print tag_list
            break
        break
        
    int_list = np.array(int_list)/dur
    return int_list, tag_list 


def timit_mlf(name, ostream, dot, dur):
    ostream.write('"*/'+name+'.rec"\n')
    int_list, tag_list = timit_parse(name, dot)
    full = np.insert(int_list, 0, 0.0)
    for i in range(len(tag_list)):
        ostream.write('{} {} {} {}\n'.format(int(full[i]*dur)*100000, int(full[i+1]*dur)*100000, tag_list[i], 0.0))
    ostream.write('.\n')    

def timit_write(out_file, dot, opt):
    timit_list = get_timit_list()
    from zrst.util import MLF
    mlf = MLF('timit_dummy_train.mlf')
    mlf.merge(MLF('timit_dummy_test.mlf'))
    print mlf.wav_list
    dur_dict = dict(zip(mlf.wav_list, mlf.int_list))
    with open(out_file, 'w') as f:
        f.write('#!MLF!#\n')
        for t in timit_list:
            print t    
            if opt in t:
                timit_mlf(t, f, dot, dur_dict[t][-1])
##################################################################################
# wav splice functions
def timit_sox(name, folder, dot='wrd'):
    int_list, tag_list = timit_parse(name, dot)
    total_sec = timit_dur(name)/16000
    int_list = np.insert(int_list, 0, 0)*total_sec
    print name, total_sec, int_list, tag_list
    #sox in.mp3 out.mp3 trim offset_in_secs duration_in_secs
    for i, tag in enumerate(tag_list): 
        out_wav = os.path.join(folder, '_'.join([name, tag])+'.wav')
        in_wav  = os.path.join(wav_root, name+'.wav')
        sts = subprocess.Popen("sox {} {} trim {} {}".format(in_wav, out_wav, int_list[i], int_list[i+1]-int_list[i]), shell=True).wait()

def timit_splice_word(path):
    timit_list = get_timit_list()
    for wav in timit_list:
        timit_sox(wav, path) 

##################################################################################
# query selection functions
def timit_filter(filter_list = ['dr8','train','_sx'], folder=word_root):
    return set(filter(lambda y: all(map(lambda x: x in y, filter_list)), map(lambda w: w[:-4], os.listdir(folder))))

def last_word(query):
    return query.split('_')[-1]

def list_word(document):
    ___, words =  timit_parse(document, 'wrd')
    return words

def find_answ(query_list, document_list):
    answ = np.zeros([len(query_list), len(document_list)])
    query_words = map(last_word, query_list)
    document_words = map(list_word, document_list) 
    for i, q in enumerate(query_words):
        for j, d in enumerate(document_words):
            if q in d:
                answ[i,j]=1.0
    return answ

def query_filter_by_wordlen(query_list, n_min=4, n_max=1000):
    word_list = map(last_word, query_list)
    selected = filter(lambda i:len(word_list[i])>=n_min and len(word_list[i])<=n_max, range(len(query_list)))
    return map(lambda i: query_list[i], selected)

def query_filter_by_occurence(query_list, document_list, n_min = 20, n_max =1000):
    train_answ = find_answ(query_list, document_list)
    query_dist = np.sum(train_answ, axis=1)
    selected = filter(lambda i:query_dist[i]>=n_min and query_dist[i]<= n_max , range(len(query_list)))
    return map(lambda i: query_list[i], selected)

def query_filter(query_list, document_list, min1, min2):
    inx1 =query_filter_by_wordlen(query_list, min1)
    inx2 =query_filter_by_occurence(query_list, document_list, min2)
    selected = set(inx1).intersection(set(inx2))
    print len(query_list), len(inx1), len(inx2), len(selected)
    return selected

if __name__=='__main__':
    timit_list = get_timit_list()
    print timit_list[0]
    print timit_parse(timit_list[0], 'phn')
    print timit_dur(timit_list[0])/16000
    
    #timit_splice_word('/home/c2tao/timit_word_corpus')

    train_query = sorted(timit_filter(['dr8','train','_sx'], word_root))
    test_query = sorted(timit_filter(['dr8','test','_sx'], word_root))
    train_document = sorted(timit_filter(['train','_sx'], wav_root) - timit_filter(['train','_sx','dr8'], wav_root))
    test_document = sorted(timit_filter(['test','_sx'], wav_root) - timit_filter(['test','_sx','dr8'], wav_root))
     
    train_query = query_filter(train_query, train_document, 3, 20)
    test_query  = query_filter(test_query, test_document, 3, 20)
    print train_query
     
    #timit_write('timit_train_wrd.mlf', 'wrd', 'train')
    #timit_write('timit_test_wrd.mlf', 'wrd', 'test')
    #timit_write('timit_train_phn.mlf', 'phn', 'train')
    #timit_write('timit_test_phn.mlf', 'phn', 'test')
