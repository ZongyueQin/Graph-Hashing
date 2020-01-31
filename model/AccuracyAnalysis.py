import tensorflow as tf
import numpy as np
import os
import sys

import random

from utils import *
from graphHashFunctions import GraphHash_Emb_Code
from config import FLAGS
from DataFetcher import DataFetcher

os.environ['CUDA_VISIBLE_DEVICES']='5'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MinGED = 0
MaxGED = 11

if len(sys.argv) != 4:
    print('parameters are GT_fname, model_path, output_fname')
    os._exit(0)
GT_fname = str(sys.argv[1])
model_path = str(sys.argv[2])
output_fname = str(sys.argv[3])

print('read ground truth...')
# read Ground Truth
f = open(GT_fname, 'r')
ground_truth = {}
qids = set()
for line in f.readlines():
    g,q,d = line.split(' ')
    g = int(g)
    q = int(q)
    d = int(d)
    qids.add(q)
    ground_truth[(q,g)]=d
f.close()
print('done')



""" Load datafetcher and model"""
print('restoring model...')
data_fetcher = DataFetcher(dataset=FLAGS.dataset, exact_ged=True)
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
saver.restore(sess, model_path)
print("Model restored from", model_path)


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
#    code = tupleCode2IntegerCode(code[0])
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
 
# encoding training data
_, id2emb, id2code, bit_weights = encodeTrainingData(sess, model, data_fetcher,
                                        placeholders, True, True, 1)

f = open(output_fname, 'w')

# compute estimated ged
cnt = np.zeros((MaxGED-MinGED+1,12))
for qid in list(qids):
    ret = getCodeAndEmbByQid(qid)
    q_code = np.array(ret[0], dtype=np.int32)
    q_emb = np.array(ret[1])
    for g in data_fetcher.train_graphs:
        gid = g.ori_graph['gid']
        real_ged = ground_truth[(qid, gid)]
        if real_ged >= MinGED and real_ged <= MaxGED:
            g_emb = np.array(id2emb[gid])
            g_code = np.array(id2code[gid], dtype=np.int32)
            ged_by_code = np.sum((q_code-g_code)**2)
            ged_by_emb = np.sum((q_emb-g_emb)**2)
#            f.write(str(real_ged)+' '+str(ged_by_code)+' '+str(ged_by_emb)+'\n')
            if ged_by_code > 11:
                ged_by_code = 11
            cnt[real_ged-MinGED,ged_by_code]=cnt[real_ged-MinGED,ged_by_code]+1

for i in range(MinGED,MaxGED+1):
    for j in range(12):
        f.write('%d '%cnt[i-MinGED,j])
    f.write('\n')
f.close()



