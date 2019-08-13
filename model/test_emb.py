""" Simply discretize continuous embedding """

from __future__ import division
from __future__ import print_function
from queue import PriorityQueue as PQ

import time
import tensorflow as tf
from random import randint, sample
import numpy as np
import os
from scipy.stats import spearmanr, kendalltau
import subprocess
from tensorflow.python.tools import inspect_checkpoint as chkp
import sys


from utils import *
from graphHashFunctions import GraphHash_Rank
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle

os.environ['CUDA_VISIBLE_DEVICES']='3,4'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

test_top_k = True
test_range_query = True

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

train = False

#chkp.print_tensors_in_checkpoint_file("SavedModel/model_rank.ckpt", tensor_name='', all_tensors=True, all_tensor_names=True)



# Load data
data_fetcher = DataFetcher(FLAGS.dataset, True)
dataset = tf.data.Dataset.from_generator(data_fetcher.get_train_data, 
                                         (tf.int64, tf.float32, tf.int64,
                                          tf.int64, tf.float32, tf.int64,
                                          tf.int32, tf.float32, tf.float32), 
                                          (tf.TensorShape([None,2]), 
                                           tf.TensorShape([None]), 
                                           tf.TensorShape([2]),
                                           tf.TensorShape([None,2]), 
                                           tf.TensorShape([None]), 
                                           tf.TensorShape([2]),
                                           tf.TensorShape([(1+FLAGS.k)*FLAGS.batchsize]),
                                           tf.TensorShape([FLAGS.batchsize, FLAGS.batchsize]),
                                           tf.TensorShape([FLAGS.batchsize, FLAGS.k]),
                                           ))
dataset = dataset.prefetch(buffer_size=1)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
next_element = construct_input(one_element)


# Define placeholders
placeholders = {
    'support': tf.sparse_placeholder(tf.float32),
    'features': tf.sparse_placeholder(tf.float32, shape=(None, data_fetcher.get_node_feature_dim())),
    'labels': tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.batchsize)),
    'dropout': tf.placeholder_with_default(0., shape=()),
#    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'graph_sizes': tf.placeholder(tf.int32, shape=((1+FLAGS.k)*FLAGS.batchsize)),
    'generated_labels':tf.placeholder(tf.float32, shape=(FLAGS.batchsize, FLAGS.k)),
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.embedding_dim))
}



# Create model
model = GraphHash_Rank(placeholders, 
                       input_dim=data_fetcher.get_node_feature_dim(),
                       next_ele = next_element,
                       logging=True)
# Initialize session
sess = tf.Session(config=config)

# Init variables
saver = tf.train.Saver()
cost_val = []

if train:
    print('start optimization...')
    sess.run(tf.global_variables_initializer())
    train_start = time.time()
    for epoch in range(FLAGS.epochs):
    
        t = time.time()
    
        # Construct feed dictionary
        feed_dict = construct_feed_dict_prefetch(data_fetcher, placeholders)
        # Training step
        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)
    
        if (epoch+1) % 100 == 0:
            pred,lab = sess.run([model.pred, model.lab], feed_dict=feed_dict)
            print(pred)
            print(lab)
    
        # No Validation For Now

        # Print loss
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]), 
              "time=", "{:.5f}".format(time.time() - t))

#    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#        print("Early stopping...")
#        break

    print("Optimization Finished, tiem cost {:.5f} s"\
          .format(time.time()-train_start))

    save_path = saver.save(sess, "SavedModel/model_rank.ckpt")
    print("Model saved in path: {}".format(save_path))

else:
    saver.restore(sess, "../SavedModel/0811-emb/model_rank.ckpt")
    print("Model restored from ../SavedModel/0811-emb/model_rank.ckpt")



print('start encoding training data...')
train_graph_num = data_fetcher.get_train_graphs_num()
inverted_index = {}
encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
all_codes = []
all_embs = []

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
    codes, embs = sess.run([model.encode_outputs[0],
                            model.encode_outputs[0]], 
                            feed_dict = feed_dict)
    codes = list(codes)
    codes = codes[0:end-i]
    all_codes = all_codes + codes
    
    embs = list(embs)
    embs = embs[0:end-i]
    all_embs = all_embs + embs
    
all_codes_np = np.array(all_codes)
#thres = np.mean(all_codes_np, axis=0)
thres = np.zeros(FLAGS.embedding_dim)
id2emb = {}
for i, pair in enumerate(zip(all_codes, all_embs)):
    code = pair[0]
    emb = pair[1]
    code = (np.array(code) > thres).tolist()
    tuple_code = tuple(code)
    gid = data_fetcher.get_train_graph_gid(i)
    inverted_index.setdefault(tuple_code, [])
    inverted_index[tuple_code].append((gid, emb))
    id2emb[gid] = emb

index_file = open('SavedModel/id2emb.pkl', 'wb')
pickle.dump(id2emb, index_file)
index_file.close()

print('finish encoding, saved index to SavedModel/id2emb.pkl')
 

# Compute MSE of estimated GED for training data
print('Computing training MSE...')
MSE_train_con = 0
MSE_train_dis = 0
train_ged_cnt = {}
pred_cnt = {}
for i in range(100):
    
    #idx1 = data_fetcher.cur_train_sample_ptr
    #idx2 = idx1 + 1
    #if idx2 == train_graph_num:
    #    idx2 = 0
        
    #print(idx1)
    #print(idx2)
    #feed_dict = construct_feed_dict_prefetch(data_fetcher, placeholders)
    #lab, pred, embs, gids, var = sess.run([model.lab,model.pred,model.outputs[0],
    #                                       one_element[9],
    #                                      model.layers[0][0].vars['weights']], 
    #                            feed_dict=feed_dict)
    #print(var)
    #e=np.zeros((FLAGS.batchsize, FLAGS.batchsize))
    #for i in range(FLAGS.batchsize):
    #    for j in range(FLAGS.batchsize):
    #        emb1 = np.array(embs[i])
    #        emb2 = np.array(embs[j])
    #        e[i,j] = np.sum((emb1-emb2)**2)
    #print(e)
    #print(pred)
    #print(lab)
    #print(embs[0])
    #print(embs[1])
    #print(id2emb[gids[0]])
    #print(id2emb[gids[1]])
    
    idx1 = randint(0, train_graph_num - 1)
    idx2 = randint(0, train_graph_num - 1)
    while idx1 == idx2:
        idx2 = randint(0, train_graph_num - 1)
    
    true_ged = data_fetcher.getLabelForPair(data_fetcher.train_graphs[idx1], 
                                            data_fetcher.train_graphs[idx2])
    if true_ged == -1:
        true_ged = FLAGS.GED_threshold
    #print(true_ged)
    train_ged_cnt.setdefault(true_ged, 0)
    train_ged_cnt[true_ged] = train_ged_cnt[true_ged] + 1
#    idx_list = [idx1, idx2]


    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)

    feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                               placeholders, 
                                               idx_list,
                                               'train')
    feed_dict.update({placeholders['thres']: thres})
    codes, embs, inputs, act = sess.run([model.codes, model.encode_outputs[0], model.ecd_inputs, model.ecd_activations], 
                          feed_dict = feed_dict)
    emb1 = np.array(embs[0])
    emb2 = np.array(embs[1])
    """
    df = open('tmp1', 'wb')
    pickle.dump([inputs, act], df)
    """
    est_ged = np.sum((emb1-emb2)**2)

    if est_ged > FLAGS.GED_threshold:
        est_ged = FLAGS.GED_threshold
    pred_cnt.setdefault(int(est_ged),0)
    pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)]+1
    MSE_train_con = MSE_train_con + ((true_ged-est_ged)**2)/100
    
    code1 = np.array(codes[0], dtype=np.float32)
    code2 = np.array(codes[1], dtype=np.float32)
    est_ged = np.sum((code1-code2)**2)
    if est_ged > FLAGS.GED_threshold:
        est_ged = FLAGS.GED_threshold

    MSE_train_dis = MSE_train_dis + ((true_ged-est_ged)**2)/100
    


print(train_ged_cnt)
print(pred_cnt)
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
try:
    f = open(ground_truth_path, 'r')
    ged_cnt = {}
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
    
    has_GT = True

except IOError:
    print('Groundtruth file doesn\'t exist, ignore top-k and range query')
    has_GT = False



# Start testing
PAtKs = []
SRCCs = []
KRCCs = []
t_max = FLAGS.GED_threshold - 1
precisions = [[] for i in range(t_max)]
recalls = [[] for i in range(t_max)]
f1_scores = [[] for i in range(t_max)]
zero_cnt = [0 for i in range(t_max)]

precisions_nz = [[] for i in range(t_max)]
recalls_nz = [[] for i in range(t_max)]
f1_scores_nz = [[] for i in range(t_max)]

ret_size = [[] for i in range(t_max)]


top_k_query_time = []
range_query_time = [[] for i in range(t_max)]


MSE_test_con = 0
MSE_test_dis = 0
test_ged_cnt = {}
pred_cnt = {}

if not has_GT:
    for i in range(total_query_num):
        idx1 = i
        idx2 = randint(0, train_graph_num - 1)
        true_ged = data_fetcher.getLabelForPair(data_fetcher.test_graphs[idx1], 
                                                data_fetcher.train_graphs[idx2])
        if true_ged == -1:
            true_ged = FLAGS.GED_threshold

        test_ged_cnt.setdefault(true_ged, 0)
        test_ged_cnt[true_ged] = test_ged_cnt[true_ged] + 1

        # Compute code and embedding for query graph        
        idx_list = [idx1]    
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)
    
        feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                   placeholders, 
                                                   idx_list,
                                                   'test')
        feed_dict.update({placeholders['thres']: thres})
    
        codes, embs = sess.run([model.codes, model.encode_outputs[0]], 
                               feed_dict = feed_dict)
        emb1 = np.array(embs[0])
        code1 = codes[0]
        code1 = np.array(code1, dtype=np.float32)

        # Compute code and embedding for training graph
        idx_list = [idx2]   
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)
    
        feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                   placeholders, 
                                                   idx_list,
                                                   'train')
        feed_dict.update({placeholders['thres']: thres})

        codes, embs = sess.run([model.codes, model.encode_outputs[0]], 
                               feed_dict = feed_dict)
        emb2 = np.array(embs[0])
        code2 = codes[0]
        code2 = np.array(code2, dtype=np.float32)
    
        # Estimate GED
        est_ged = np.sum((emb1-emb2)**2) 
        if est_ged > FLAGS.GED_threshold:
            est_ged = FLAGS.GED_threshold
    
        pred_cnt.setdefault(int(est_ged),0)
        pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)]+1

        MSE_test_con = MSE_test_con + ((true_ged-est_ged)**2)/total_query_num

    
        est_ged = np.sum((code1-code2)**2)
        if est_ged > FLAGS.GED_threshold:
            est_ged = FLAGS.GED_threshold

        MSE_test_dis = MSE_test_dis + ((true_ged-est_ged)**2)/total_query_num
else:
    for i in range(0, total_query_num, encode_batchsize):   

        
        end = i + encode_batchsize
        if end > total_query_num:
            end = total_query_num

        idx_list = list(range(i,end))
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)
        
        

        feed_dict = construct_feed_dict_for_query(data_fetcher, 
                                                  placeholders, 
                                                  idx_list,
                                                  'test')
        feed_dict.update({placeholders['thres']: thres})
        
        codes, embs = sess.run([model.codes, model.encode_outputs[0]], 
                                   feed_dict = feed_dict)
        
    
        for j, tup in enumerate(zip(codes, embs)):
            code = tup[0]
            emb = tup[1]
            # ignore all padding graphs
            if i + j >= total_query_num:
                break

            tuple_code = tuple(code)
            q = data_fetcher.get_test_graph_gid(i + j)

            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
            
            # Second batch, the closest graphs
            for pair in ground_truth[q][0:encode_batchsize]:
                true_ged = pair[1]  
                gid = pair[0]
                emb_train = id2emb[gid]
                emb1 = np.array(emb)
                emb2 = np.array(emb_train)    
                est_ged = np.sum((emb1-emb2)**2)
                if est_ged > FLAGS.GED_threshold:
                    est_ged = FLAGS.GED_threshold


                MSE_test_con = MSE_test_con + ((true_ged-est_ged)**2)/(total_query_num*encode_batchsize)
                pred_cnt.setdefault(int(est_ged),0)
                pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)] + 1
    
            # compute code and embedding for training graphs
            # To adjust to the size of placeholders, we add some graphs for padding
            


            if test_top_k:
                # top k query
                query_time = time.time()
             

                est_top_k_wrp = PQ(maxsize=FLAGS.top_k)
                for gid in id2emb.keys():
                    emb1 = np.array(emb)
                    emb2 = np.array(id2emb[gid])
                    est_ged = np.sum((emb1-emb2)**2)
                    item = (-est_ged, gid)
                    if est_top_k_wrp.full():
                        maxItem = est_top_k_wrp.get()
                        if maxItem[0] > item[0]:
                            item = maxItem
                    est_top_k_wrp.put(item, block=False)

                est_top_k = [0 for idx in range(FLAGS.top_k)]
                for idx in range(FLAGS.top_k):
                    item = est_top_k_wrp.get()
                    est_top_k[FLAGS.top_k-1-idx] = item[1]
                
                top_k_query_time.append(time.time()-query_time)

                pos = FLAGS.top_k
                while ground_truth[q][pos][1] == ground_truth[q][pos-1][1]:
                    pos = pos + 1
                    if pos == len(ground_truth[q]):
                        break

                true_top_k_wrp = ground_truth[q][0:pos]
        
                true_top_k = [pair[0] for pair in true_top_k_wrp]
#               print(true_top_k)
#               print(est_top_k)

                PAtKs.append(len(set(est_top_k)&set(true_top_k)) / FLAGS.top_k)
   
                # compute real rank
                real_rank_all = {}
                last_d = -1
                last_rank = 1
                for pos, pair in enumerate(ground_truth[q]):
                    gid = pair[0]
                    d = pair[1]
                    if d == last_d:
                        real_rank_all[gid] = last_rank
                    else:
                        real_rank_all[gid] = pos + 1
                        last_rank = pos + 1
                        last_d = d

                real_rank = [real_rank_all[gid] for gid in est_top_k]
                tie_handler = {}
                for pos, rk in enumerate(real_rank):
                    if rk in tie_handler.keys():
                        real_rank[pos] = tie_handler[rk]
                        tie_handler[rk] = tie_handler[rk]+1
                    else:
                        tie_handler[rk] = rk + 1
                
                rho, _ = spearmanr(list(range(1,FLAGS.top_k+1)), real_rank)
                SRCCs.append(rho)
                tau, _ = kendalltau(list(range(1, FLAGS.top_k+1)), real_rank)
                KRCCs.append(tau)


        
            # range query
            if test_range_query:
                cur_pos = 0
                query_time = time.time()
                for t in range(1,t_max):
                    
                    query_time = time.time()
                    similar_set = set()
                    
                    for gid in id2emb.keys():
                        emb1 = np.array(emb)
                        emb2 = np.array(id2emb[gid])
                        est_ged = np.sum((emb1-emb2)**2)
                        if est_ged > FLAGS.GED_threshold:
                            est_ged = FLAGS.GED_threshold

                        if est_ged < t:
                            similar_set.add(gid)
                    
                    if i + j == 0:
                        print('t={:d}, cost {:f} s'.format(t, time.time()-query_time))

                    range_query_time[t-1].append(time.time()-query_time)
                    
                    ret_size[t-1].append(len(similar_set))

                    while cur_pos < len(ground_truth[q]) and\
                          ground_truth[q][cur_pos][1] <= t:
                        cur_pos = cur_pos + 1

                    real_sim_set_wrp = ground_truth[q][0:cur_pos]
                    real_sim_set = set([pair[0] for pair in real_sim_set_wrp])

                    tmp = similar_set & real_sim_set
                    if len(similar_set) == 0:
                        if len(real_sim_set) == 0:
                            precision = 1
                        else:
                            precision = 0
                    else:
                        precision =  len(tmp)/len(similar_set)

                    precisions[t-1].append(precision)
                    if len(real_sim_set) > 0:
                        precisions_nz[t-1].append(precision)

                    if len(real_sim_set) == 0:
                        zero_cnt[t-1] = zero_cnt[t-1] + 1
                        if len(similar_set) == 0:
                            recall = 1
                        else:
                            recall = 0
                    else:
                        recall = len(tmp)/len(real_sim_set)
                    recalls[t-1].append(recall)
                    if len(real_sim_set) > 0:
                        recalls_nz[t-1].append(recall)

                    if precision * recall == 0:
                        if len(real_sim_set) == 0 and\
                            len(similar_set) == 0:
                            f1_score = 1
                        else:
                            f1_score = 0
                    else:
                        f1_score = 2*precision*recall/(precision+recall)

                    f1_scores[t-1].append(f1_score)
                    if len(real_sim_set) > 0:
                        f1_scores_nz[t-1].append(f1_score)

print(test_ged_cnt)
print(pred_cnt)
print('MSE for test (continuous) = {:f}'.format(MSE_test_con))
print('MSE for test (discrete) = {:f}'.format(MSE_test_dis))

if test_top_k:
    print('For Top-k query, k={:d}'.format(FLAGS.top_k))
    print('average precision at k = {:f}'.format(sum(PAtKs)/len(PAtKs)))
    print('average rho = {:f}'.format(sum(SRCCs)/len(SRCCs)))
    print('average tau = {:f}'.format(sum(KRCCs)/len(KRCCs)))
    print('average search time = {:f}'.format(sum(top_k_query_time)/len(top_k_query_time)))

if test_range_query:
    print('For range query')
    for t in range(1,t_max):
        print('threshold = {:d}'.format(t), end=' ')
        print('empty cnt = {:d}'.format(zero_cnt[t-1]), end = ' ')
        print('average precision = %f'%(sum(precisions[t-1])/len(precisions[t-1])), end = ' ')
        print('average recall = %f'%(sum(recalls[t-1])/len(recalls[t-1])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores[t-1])/len(f1_scores[t-1])), end = ' ')
        print('average return size = %f'%(sum(ret_size[t-1])/len(ret_size[t-1])))
        print('average search time = {:f}'.format(sum(range_query_time[t-1])/len(range_query_time[t-1])))


    print('ignore empty answers')
    for t in range(1,t_max):
        print('threshold = {:d}'.format(t), end = ' ')
        print('average precision = %f'%(sum(precisions_nz[t-1])/len(precisions_nz[t-1])), end = ' ')
        print('average recall = %f'%(sum(recalls_nz[t-1])/len(recalls_nz[t-1])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores_nz[t-1])/len(f1_scores_nz[t-1])))

    print(ged_cnt)
    print('FLAGS.k={:d}'.format(FLAGS.k))


