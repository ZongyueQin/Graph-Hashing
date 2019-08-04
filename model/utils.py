from config import FLAGS
import numpy as np
import tensorflow as tf

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

def get_similar_graphs_gid(inverted_index, code, thres=FLAGS.hamming_dist_thres):
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

def writeInvertedIndex(filename, index):
    f = open(filename, 'w')
    f.write(str(len(index.keys()))+'\n')
    f.write(str(FLAGS.hash_code_len)+'\n')
    for key in index.keys():
        code = tupleCode2IntegerCode(key)
        f.write(str(code)+'\n')
        f.write(str(len(index[key]))+'\n')
        for pair in index[key]:
            f.write(str(int(pair[0]))+'\n')
            string = ''
            for i in range(FLAGS.hash_code_len):
                string = '{:s} {:f}'.format(string, pair[1][i])
            f.write(string+'\n')
            
    f.close()
            
        
