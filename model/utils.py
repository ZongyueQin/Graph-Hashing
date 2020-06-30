from config import FLAGS
import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from MyGraph import MyGraph
import networkx as nx
from glob import glob
import xml

def sample_from_polynomial(cum_p, num):
    u = np.random.uniform(size=[num])
    cum_P = np.tile(cum_p, (num,1))
    U = (np.tile(u,(cum_p.shape[0],1))).T
    A = (U<cum_P)*(2-cum_P)
    ret = np.argmax(A,axis=1)
    return ret

def construct_input(data):
    features = tf.SparseTensor(data[0],data[1],data[2])
    laplacians = tf.SparseTensor(data[3],data[4],data[5])
    try:
        features = tf.sparse.reorder(features)
        laplacians = tf.sparse.reorder(laplacians)
    except AttributeError:
        features = tf.sparse_reorder(features)
        laplacians = tf.sparse_reorder(laplacians)
        
    return features, laplacians, data[6], data[7], data[8]

def construct_feed_dict_prefetch(data_fetcher, placeholders):

    feed_dict = dict()
#    nfn = data_fetcher.get_node_feature_dim()
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
  #  feed_dict.update({placeholders['num_features_nonzero']: [data_fetcher.batch_node_num]})
    
    return feed_dict

def construct_feed_dict_for_train(data_fetcher, placeholders):
    feed_dict = dict()
    if FLAGS.label_type == 'binary':
        features, laplacians, sizes, labels = \
        data_fetcher.sample_train_data(FLAGS.batchsize)
    else:
        features, laplacians, sizes, labels, generated_labels = \
        data_fetcher.sample_train_data(FLAGS.batchsize)
        
    nfn = data_fetcher.get_node_feature_dim()

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
 #   feed_dict.update({placeholders['num_features_nonzero']: [data_fetcher.batch_node_num]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    if FLAGS.label_type == 'ged':
        feed_dict.update({placeholders['generated_labels']: generated_labels})

    return feed_dict


def construct_feed_dict_for_encode(data_fetcher, placeholders, idx_list, tvt):
    feed_dict = dict()
    features, laplacians, sizes= \
        data_fetcher.get_data_without_label(idx_list, tvt)
    nfn = data_fetcher.get_node_feature_dim()
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
#    feed_dict.update({placeholders['num_features_nonzero']: [data_fetcher.batch_node_num]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    return feed_dict


def construct_feed_dict_for_query(data_fetcher, placeholders, idx_list, tvt):
    feed_dict = dict()
    features, laplacians, sizes= data_fetcher.get_data_without_label(idx_list, 
                                                                     tvt)
    nfn = data_fetcher.get_node_feature_dim()
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
   # feed_dict.update({placeholders['num_features_nonzero']: [data_fetcher.batch_node_num]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    return feed_dict



def node_match(n1, n2):
    return n1[FLAGS.node_feat_name] == n2[FLAGS.node_feat_name]

def sorted_nicely(l):
    def tryint(s):
        try:
            return int(s)
        except:
            return s

    import re
    def alphanum_key(s):
        return [tryint(c) for c in re.split('([0-9]+)', s)]

    return sorted(l, key=alphanum_key)

def get_similar_graphs_gid(inverted_index, code, thres):
    """ use bfs to find all similar codes """
    sets = []
    frontier = [(code, 0, -1)]
    count = 0
    while len(frontier) > 0:
        count = count + 1
        if count % 1000 == 0:
            print(count)
        cur_code, dist, last_flip_pos = frontier[0]
        frontier.pop(0)
        if cur_code in inverted_index.keys():
            sets = sets + inverted_index[cur_code]
        if dist < thres:
            for j in range(last_flip_pos+1, len(code)):
                temp_code = list(cur_code)
                temp_code[j] = bool(1-temp_code[j])
                frontier.append((tuple(temp_code), dist+1, j))
    print(count)
    return sets

def get_top_k_similar_graphs_gid(inverted_index, code, emb, top_k):
    """ use bfs to find all similar codes """
    sets = []
    frontier = [(code, 0, -1)]
    cur_ged = 0
    while len(frontier) > 0:
    #    print('t')
        cur_code, dist, last_flip_pos = frontier[0]
        frontier.pop(0)
        if dist > cur_ged and len(sets) > top_k:
            break
        if dist < cur_ged:
            raise RuntimeError('dist < cur_ged')
        cur_ged = dist
        
        if cur_code in inverted_index.keys():
            sets = sets + inverted_index[cur_code]
            
        for j in range(last_flip_pos+1, len(code)):
            temp_code = list(cur_code)
            temp_code[j] = bool(1-temp_code[j])
            frontier.append((tuple(temp_code), dist+1, j))
    #print('e')
    def func(x):
            dist_x = np.sum((np.array(x[1])-np.array(emb))**2)
            return dist_x*10000000 + x[0]
#            dist_y = sum((y-emb)**2)
#            if dist_x < dist_y:
#                return -1
#            if dist_x == dist_y:
#                return 0
#            else:
#                return 1
        
    if len(sets) > top_k:
        # sort sets according to embeddings' distance to emb
        sorted(sets, key = func)
        sets = sets[0:top_k]
    return sets

def tupleCode2IntegerCode(code):
    ret = 0
    curPow = 1
    for i in range(len(code)):
        if code[i] == True:
            ret = ret + curPow
        curPow = curPow * 2
        
    return int(ret)

def writeMapperDict(mapper_dict, fname):
    f = open(fname, 'w')
    for i in range(FLAGS.GED_threshold):
        f.write(str(max(mapper_dict[i]))+'\n')
    f.close()

def writeBitWeights(filename, bit_weights):
    f = open(filename, 'w')
    bit_weights = list(bit_weights)
    for w in bit_weights:
        f.write(str(w)+'\n')
    f.close()

def writeIndex(filename, index, embLen = FLAGS.hash_code_len):
    f = open(filename, 'w')
    f.write(str(len(index.keys()))+'\n')
    f.write(str(embLen)+'\n')
    for key in index.keys():
        code = tupleCode2IntegerCode(key)
        f.write(str(code)+'\n')
        f.write(str(len(index[key]))+'\n')
        for pair in index[key]:
            f.write(str(int(pair[0]))+'\n')
            string = ''
            for i in range(embLen):
                string = '{:s} {:f}'.format(string, pair[1][i])
            f.write(string+'\n')
            
    f.close()
    
""" encode training data with model that outputs both continuous embedding
    and discrete code """
def encodeTrainingData(sess, model, data_fetcher, placeholders,
                       use_emb=True, use_code=True, ecd_bs=-1):
    print('start encoding training data...')
    train_graph_num = data_fetcher.get_train_graphs_num()
    index = {}
    #encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
    if ecd_bs == -1:
        encode_batchsize=FLAGS.ecd_batchsize
    else:
        encode_batchsize = ecd_bs
    #all_codes = []
    #all_embs = []
    thres = np.zeros(FLAGS.hash_code_len)
    
    for i in range(0, train_graph_num, encode_batchsize):
        end = i + encode_batchsize
        if end > train_graph_num:
            end = train_graph_num
        idx_list = list(range(i,end))
        # padding to fit placeholders' shapes
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)
        
        feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                   placeholders, 
                                                   idx_list,
                                                   'train')
        feed_dict.update({placeholders['thres']: thres})
        
        if use_code and use_emb:
            codes, embs = sess.run([model.codes,
                                    model.ecd_embeddings], 
                                    feed_dict = feed_dict)
            codes = list(codes)
            codes = codes[0:end-i]
     #       all_codes = all_codes + codes
    
            embs = list(embs)
            embs = embs[0:end-i]
     #       all_embs = all_embs + embs
    
        elif use_code and use_emb == False:
            codes = sess.run(model.codes, feed_dict=feed_dict)
            codes = list(codes)
            codes = codes[0:end-i]
     #       all_codes = all_codes + codes
     #       all_embs = all_codes
        elif use_code == False and use_emb:
            embs = sess.run(model.ecd_embeddings, feed_dict=feed_dict)
            embs = list(embs)
            embs = embs[0:end-i]
     #       all_embs = all_embs + embs
     #       all_codes = all_embs
        else:
            raise RuntimeError('use_code and use_emb cannot both be False')
        for j, pair in enumerate(zip(codes, embs)):
            pos = i + j
            if pos % 1000000 == 0:
                print('Finish %d graphs'%pos)
            code = pair[0]
            emb = pair[1]
            tuple_code = tuple(code)
            gid = data_fetcher.get_train_graph_gid(pos)
            if use_emb and use_code:
                index.setdefault(tuple_code, [])
                index[tuple_code].append((gid, emb))
            if use_code and use_emb == False:
                index.setdefault(tuple_code, [])
                index[tuple_code].append((gid, None))            


    print('threshold is')
    print(thres)
    """
    id2emb = {}
    id2code = {}
    for i, pair in enumerate(zip(all_codes, all_embs)):
        code = pair[0]
        emb = pair[1]
        tuple_code = tuple(code)
        gid = data_fetcher.get_train_graph_gid(i)
        if use_emb and use_code:
            inverted_index.setdefault(tuple_code, [])
            inverted_index[tuple_code].append((gid, emb))
        if use_code and use_emb == False:
            inverted_index.setdefault(tuple_code, [])
            inverted_index[tuple_code].append((gid, None))            
        if use_emb:
            id2emb[gid] = emb
        if use_code:
            id2code[gid] = code
    """
    bit_weights = sess.run(model.bit_weights)
    if bit_weights is None:
        bit_weights = np.ones((FLAGS.hash_code_len))       
    print('finish encoding training data')
    return index, bit_weights

def readGroundTruth(f):
    ged_cnt = {}
    ground_truth = {}
    for line in f.readlines():
        g, q, d = line.split(' ')
        g = int(g)
        q = int(q)
        d = int(d)

        if q not in ground_truth.keys():
            ground_truth[q] = []
        ground_truth[q].append((g,d))

        ged_cnt.setdefault(d,0)
        ged_cnt[d] = ged_cnt[d] + 1
        
    return ground_truth, ged_cnt
    

            


def encodeData(sess, model, data_fetcher, graphs, placeholders,use_emb,use_code,
               inverted_index, id2emb, id2code):

    encode_batchsize=FLAGS.ecd_batchsize
    total_graph_num = len(graphs)
    thres = np.zeros(FLAGS.hash_code_len)
    all_codes = []
    all_embs = []
    

    for i in range(0, total_graph_num, encode_batchsize):

        end = i + encode_batchsize

        if end > total_graph_num:

            end = total_graph_num

        idx_list = list(range(i,end))

        # padding to fit placeholders' shapes

        while (len(idx_list) < encode_batchsize):

            idx_list.append(0)

        

        # Create wrapper graphs

        wrp_graphs = []

        for j in idx_list:

            mg = MyGraph(graphs[j], data_fetcher.max_label)

            wrp_graphs.append(mg)

        

        features = sp.vstack([g.sparse_node_inputs for g in wrp_graphs])

        features = data_fetcher._sparse_to_tuple(features)

        

        laplacians = sp.block_diag([g.laplacian for g in wrp_graphs])

        laplacians = data_fetcher._sparse_to_tuple(laplacians)

        

        sizes = [g.nxgraph.number_of_nodes() for g in wrp_graphs]

        data_fetcher.batch_node_num = sum(sizes)

                

        feed_dict = dict()

        #nfn = data_fetcher.get_node_feature_dim()

        feed_dict.update({placeholders['dropout']: 0})

        feed_dict.update({placeholders['features']: features})

        feed_dict.update({placeholders['support']: laplacians})

        #    feed_dict.update({placeholders['num_features_nonzero']: [data_fetcher.batch_node_num]})

        feed_dict.update({placeholders['graph_sizes']: sizes})

        feed_dict.update({placeholders['thres']: thres})

        

        if use_code and use_emb:

            codes, embs = sess.run([model.codes,

                                    model.ecd_embeddings], 

                                    feed_dict = feed_dict)

            codes = list(codes)

            codes = codes[0:end-i]

            all_codes = all_codes + codes

    

            embs = list(embs)

            embs = embs[0:end-i]

            all_embs = all_embs + embs

    

        elif use_code and use_emb == False:

            codes = sess.run(model.codes, feed_dict=feed_dict)

            codes = list(codes)

            codes = codes[0:end-i]

            all_codes = all_codes + codes

            all_embs = all_codes

        elif use_code == False and use_emb:

            embs = sess.run(model.ecd_embeddings, feed_dict=feed_dict)

            embs = list(embs)

            embs = embs[0:end-i]

            all_embs = all_embs + embs

            all_codes = all_embs

        else:

            raise RuntimeError('use_code and use_emb cannot both be False')


    for i, pair in enumerate(zip(all_codes, all_embs)):
        code = pair[0]

        emb = pair[1]

        tuple_code = tuple(code)

        gid = graphs[i].graph['gid']

        if use_emb and use_code:

            inverted_index.setdefault(tuple_code, [])

            inverted_index[tuple_code].append((gid, emb))

        if use_code and use_emb == False:

            inverted_index.setdefault(tuple_code, [])

            inverted_index[tuple_code].append((gid, None))            

        if use_emb:

            id2emb[gid] = emb

        if use_code:

            id2code[gid] = code    
 



def encodeDataInDir(sess, model, data_fetcher, graph_dir, placeholders, 

                    max_graph_in_mem=500000,

                    use_emb=True,

                    use_code=True):

    print('Start encoding...')

    graphs = []

    err_cnt = 0

    inverted_index = {}

    #encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize

    thres = np.zeros(FLAGS.hash_code_len)
    id2emb = {}
    id2code = {}


    

    for file in sorted_nicely(glob(graph_dir+'/*.gexf')):

        if len(graphs) >= max_graph_in_mem:
            print('Reach maximal graph in memory, start encoding')
            encodeData(sess, model, data_fetcher, graphs, placeholders,use_emb,
                       use_code, 
                       inverted_index, id2emb, id2code)
            print('finish this batch')
            graphs = []

        gid = int(os.path.basename(file).split('.')[0])

        try:

            g = nx.read_gexf(file)

        except xml.etree.ElementTree.ParseError:

            print('err in nx.read_fexf', end = ' ')

            print(file)

            err_cnt = err_cnt + 1

            continue

        g.graph['gid'] = gid

        graphs.append(g)

    print('encode last batch')
    encodeData(sess, model, data_fetcher, graphs, placeholders,use_emb,use_code,
               inverted_index, id2emb, id2code)     

    bit_weights = sess.run(model.bit_weights)
    if bit_weights is None:
        bit_weights = np.ones((FLAGS.hash_code_len))
    print('finish encoding training data') 
    return inverted_index, id2emb, id2code, bit_weights
