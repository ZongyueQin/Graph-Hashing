from config import FLAGS
import numpy as np

def construct_feed_dict_prefetch(data, data_fetcher, placeholders):
    features = data[0]
    laplacians = data[1]
    sizes = data[2]
    labels = data[3]
    if FLAGS.label_type == 'ged':
        generated_labels = data[4]
    nfn = data_fetcher.get_node_feature_dim()
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacians})
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
    feed_dict.update({placeholders['graph_sizes']: sizes})
    if FLAGS.label_type == 'ged':
        feed_dict.update({placeholders['generated_labels']: generated_labels})

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
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
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
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
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
    feed_dict.update({placeholders['num_features_nonzero']: [nfn]})
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
    while len(frontier) > 0:
        cur_code, dist, last_flip_pos = frontier[0]
        frontier.pop(0)
        if cur_code in inverted_index.keys():
            sets = sets + inverted_index[cur_code]
        if dist < thres:
            for j in range(last_flip_pos+1, len(code)):
                temp_code = list(cur_code)
                temp_code[j] = bool(1-temp_code[j])
                frontier.append((tuple(temp_code), dist+1, j))
            
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
