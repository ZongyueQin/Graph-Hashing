""" Simply discretize continuous embedding """

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from random import randint
import numpy as np
import os
from scipy.stats import spearmanr, kendalltau

from utils import construct_feed_dict_for_train, construct_feed_dict_for_encode
from utils import construct_feed_dict_for_query
from utils import get_similar_graphs_gid, get_top_k_similar_graphs_gid
from graphHashFunctions import GraphHash_Rank
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
data_fetcher = DataFetcher(FLAGS.dataset)

# Some preprocessing

# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'graph_sizes': tf.placeholder(tf.int32, shape=((1+FLAGS.k)*FLAGS.batchsize)),
    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}

# Create model
model = GraphHash_Rank(placeholders, input_dim=data_fetcher.get_node_feature_dim(), 
                        logging=True)

# Initialize session
sess = tf.Session()
# Init variables
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
cost_val = []

print('start optimization...')
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict_for_train(data_fetcher, placeholders)

    # Training step
    outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

    # Validation

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), 
          "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

save_path = saver.save(sess, "SavedModel/model_rank.ckpt")
print("Model saved in path: {}".format(save_path))

print('start encoding training data...')
size = data_fetcher.get_train_graphs_num()
inverted_index = {}
encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
all_embs = []
for i in range(0,size, encode_batchsize):
    end = i + encode_batchsize
    if end > size:
        end = size
    idx_list = list(range(i,end))
    # To adjust to the size of placeholders, we add some graphs for padding
    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)
    feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                               placeholders, 
                                               idx_list,
                                               'train')
    embs = sess.run(model.outputs[0], 
                          feed_dict = feed_dict)
    embs = list(embs)
    embs = embs[0:end-i]
    all_embs = all_embs + embs
all_embs_np = np.array(all_embs)
thres = np.median(all_embs_np, axis=0)


for i, emb in enumerate(all_embs):
    code = (np.array(emb)>=thres).tolist()
    
    tuple_code = tuple(code)
    gid = data_fetcher.get_train_graph_gid(i)
    inverted_index.setdefault(tuple_code, [])
    inverted_index[tuple_code].append((gid, emb))

index_file = open('SavedModel/inverted_index_rank.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()
print('finish encoding, saved index to SavedModel/inverted_index_rank.pkl')

# Compute MSE of estimated GED for training data
MSE_train_con = 0
MSE_train_dis = 0
for i in range(100):
    idx1 = randint(0, size - 1)
    idx2 = randint(0, size - 1)
    while idx1 == idx2:
        idx2 = randint(0, size - 1)
    true_ged = data_fetcher.getLabelForPair(data_fetcher.train_graphs[idx1], 
                                            data_fetcher.train_graphs[idx2])
    idx_list = [idx1, idx2]
    # To adjust to the size of placeholders, we add some graphs for padding
    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)
    feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                               placeholders, 
                                               idx_list,
                                               'train')
    feed_dict.update({placeholders['thres']: thres})
    codes, embs = sess.run([model.codes, model.outputs[0]], 
                          feed_dict = feed_dict)
    emb1 = np.array(embs[0])
    emb2 = np.array(embs[1])
    est_ged = np.sum((emb1-emb2)**2)
    MSE_train_con = MSE_train_con + ((true_ged-est_ged)**2)/100
    
    code1 = np.array(codes[0], dtype=np.float32)
    code2 = np.array(codes[1], dtype=np.float32)
    est_ged = np.sum((code1-code2)**2)
    MSE_train_dis = MSE_train_dis + ((true_ged-est_ged)**2)/100
print('MSE for training (continuous) = {:f}'.format(MSE_train_con))
print('MSE for training (discrete) = {:f}'.format(MSE_train_dis))

print('Start testing')
total_query_num = data_fetcher.get_test_graphs_num()
# Read Ground Truth
ground_truth = {}
ground_truth_path = os.path.join('..','data',
                                 FLAGS.dataset,
                                 'test',
                                 FLAGS.ground_truth_file)
f = open(ground_truth_path, 'r')
for line in f.readlines():
    g, q, d = line.split(' ')
    g = int(g)
    q = int(q)
    d = int(d)
    if q not in ground_truth.keys():
        ground_truth[q] = []
    ground_truth[q].append((g,d))

PAtKs = []
SRCCs = []
KRCCs = []
precisions = [[] for i in range(10)]
recalls = [[] for i in range(10)]
f1_scores = [[] for i in range(10)]

for i in range(0, total_query_num, encode_batchsize):
    
    end = i + encode_batchsize
    if end > total_query_num:
        end = total_query_num
    idx_list = list(range(i,end))
    # To adjust to the size of placeholders, we add some graphs for padding
    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)
    feed_dict = construct_feed_dict_for_query(data_fetcher, 
                                              placeholders, 
                                              idx_list,
                                              'test')
    feed_dict.update({placeholders['thres']: thres})
    codes, embs = sess.run([model.codes, model.outputs[0]], feed_dict = feed_dict)
    for j, tup in enumerate(zip(codes, embs)):
        code = tup[0]
        emb = tup[1]
        # ignore all padding graphs
        if i + j >= size:
            break

        tuple_code = tuple(code)
        # top k query
        est_top_k = get_top_k_similar_graphs_gid(inverted_index, 
                                                    tuple_code,
                                                    emb,
                                                    FLAGS.top_k)
        q = data_fetcher.get_test_graph_gid(i + j)
        sorted(ground_truth[q], key=lambda x: x[1])
        true_top_k = ground_truth[q][0:FLAGS.top_k]
        
        PAtKs.append(len(set(est_top_k)&set(true_top_k)) / FLAGS.top_k)
        rho, _ = spearmanr(est_top_k, true_top_k)
        SRCCs.append(rho)
        tau, _ = kendalltau(est_top_k, true_top_k)
        KRCCs.append(tau)
        
        cur_pos = 0
        for t in range(1,9):
            similar_set = set(get_similar_graphs_gid(inverted_index,
                                                      tuple_code,
                                                      t))
            while ground_truth[q][cur_pos][1] <= t\
                  and cur_pos < len(ground_truth[q]):
                cur_pos = cur_pos + 1
            real_sim_set = set(ground_truth[q][0:cur_pos])
            tmp = similar_set & real_sim_set
            precision =  len(tmp)/len(similar_set)
            precisions[t-1].append(precision)
            recall = len(tmp)/len(real_sim_set)
            recalls[t-1].append(recall)
            if precision * recall == 0:
                if len(real_sim_set) == 0 and len(similar_set) == 0:
                    f1_score = 1
                else:
                    f1_score = 0
            else:
                f1_score = 2*precision*recall/(precision+recall)
            f1_scores[t-1].append(f1_score)
            
print('For Top-k query, k={:d}'.format(FLAGS.top_k))
print('average precision at k = {:f}'.format(sum(PAtKs)/len(PAtKs)))
print('average rho = {:f}'.format(sum(SRCCs)/len(SRCCs)))
print('average tau = {:f}'.format(sum(KRCCs)/len(KRCCs)))

print('For range query')
for t in range(1,9):
    print('threshold = {:d}'.format(t))    
    print('average precision = %f'%(sum(precisions[t-1])/len(precisions[t-1])))
    print('average recall = %f'%(sum(recalls[t-1])/len(recalls[t-1])))
    print('average f1-score = %f'%(sum(f1_scores[t-1])/len(f1_scores[t-1])))
                