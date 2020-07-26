import tensorflow as tf
import numpy as np
import os

import random

from utils import *
from graphHashFunctions import GraphHash_Emb_Code
from config import FLAGS
from DataFetcher import DataFetcher
from MyGraph import *

os.environ['CUDA_VISIBLE_DEVICES']='5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


""" Load datafetcher only to get node_feature_dim, probably should use more efficient way to do that in future """
data_fetcher = DataFetcher(dataset=FLAGS.dataset, exact_ged=True, wrp_train_graph = False)
node_feature_dim = data_fetcher.get_node_feature_dim()
# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, node_feature_dim)),
#    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'graph_sizes': tf.placeholder(tf.int32, shape=(1)),
#    'graph_sizes': tf.placeholder(tf.int32, shape=(FLAGS.batchsize*(1+FLAGS.k))),
#    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}

thres = np.zeros(FLAGS.hash_code_len)

# Create Model
#model = GraphHash_Emb_Code_Mapper(placeholders, 
model = GraphHash_Emb_Code(placeholders,
                           input_dim=node_feature_dim,
                           next_ele = None,
#                           mapper=None,
                           logging=True)

# Initialize session
sess = tf.Session(config=config)

# Init variables
saver = tf.train.Saver()

def loadModel(model_path):
    saver.restore(sess, model_path)
    print("Model restored from", model_path)
    return True

def getCodeAndEmbByQid(qid):
#    print(qid)
    
    query_graph = data_fetcher.gid2graph[qid] 
#    print(query_graph)
    features = query_graph.sparse_node_inputs
    features = data_fetcher._sparse_to_tuple(features)

    laplacian = query_graph.laplacian
    laplacian = data_fetcher._sparse_to_tuple(laplacian)
    
    size = [len(query_graph.ori_graph['nodes'])]

    code, emb = getCodeAndEmb(features, laplacian, size)
    code = tupleCode2IntegerCode(code[0])
#    print(code)
#    print('done')
#    print(tuple(emb[0]))
    emb = tuple(emb[0])
#    print(len(emb))

#    return code, emb
#    print(code)    
#    print(type(code))
#    print(emb)
#    emb = [3 for i in range(256)]
#    emb = tuple(emb)
    return (code, emb)

def getCodeAndEmbByString(string):
    #Parse String
    sequence = string.split("\n")
    graph = {}
    graph['gid'] = None
    nodeCnt = int(sequence[1])
    edgeCnt = int(sequence[2])
    graph['nodes']=[]
    graph['edges']=[]
    graph['adj_mat'] = np.zeros((nodeCnt, nodeCnt))
    for i in range(nodeCnt):
        graph['nodes'].append(int(sequence[i+3]))

    for i in range(edgeCnt):
        element = sequence[3+nodeCnt+i].split()
        f_node = int(element[0])
        t_node = int(element[1])
        graph['edges'].append((f_node, t_node))
        graph['adj_mat'][f_node, t_node] = 1
        graph['adj_mat'][t_node, f_node] = 1

    query_graph = MyGraph(graph, data_fetcher.max_label) 
     
#    print(query_graph)
    features = query_graph.sparse_node_inputs
    features = data_fetcher._sparse_to_tuple(features)

    laplacian = query_graph.laplacian
    laplacian = data_fetcher._sparse_to_tuple(laplacian)
    
    size = [len(query_graph.ori_graph['nodes'])]

    code, emb = getCodeAndEmb(features, laplacian, size)
    code = tupleCode2IntegerCode(code[0])
#    print(code)
#    print('done')
#    print(tuple(emb[0]))
    emb = tuple(emb[0])
#    print(len(emb))

#    return code, emb
#    print(code)    
#    print(type(code))
#    print(emb)
#    emb = [3 for i in range(256)]
#    emb = tuple(emb)
    return (code, emb)



def getCodeAndEmb(features, laplacian, size):
    feed_dict = dict()
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: laplacian})
   # feed_dict.update({placeholders['num_features_nonzero']: [data_fetcher.batch_node_num]})
    feed_dict.update({placeholders['graph_sizes']: size})
    feed_dict.update({placeholders['thres']:thres})

    code, emb = sess.run([model.codes, model.ecd_embeddings],
                         feed_dict = feed_dict)

    return code, emb
 
