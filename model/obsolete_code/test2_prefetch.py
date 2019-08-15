""" Simply discretize continuous embedding """

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from random import randint, sample
import numpy as np
import os
from scipy.stats import spearmanr, kendalltau
import subprocess

from utils import * 
from graphHashFunctions import GraphHash_Rank_Reg
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle

os.environ['CUDA_VISIBLE_DEVICES']='6'
test_top_k = False
test_range_query = True
train = True

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# Load data
data_fetcher = DataFetcher(FLAGS.dataset)
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
    'thres':tf.placeholder(tf.float32, shape=(FLAGS.hash_code_len))
}



# Create model
model = GraphHash_Rank_Reg(placeholders, 
                       input_dim=data_fetcher.get_node_feature_dim(),
                       next_ele = next_element,
                       logging=True)
# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
cost_val = []



if train == True:
    print('start optimization...')
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
    saver.restore(sess, "SavedModel/model_rank.ckpt")
    raise RuntimeError('restore have bug currently')


print('start encoding training data...')
train_graph_num = data_fetcher.get_train_graphs_num()
inverted_index = {}
encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
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
    embs = sess.run(model.encode_outputs[0], 
                    feed_dict = feed_dict)
    embs = list(embs)
    embs = embs[0:end-i]
    all_embs = all_embs + embs
    
all_embs_np = np.array(all_embs)
thres = np.median(all_embs_np, axis=0)
#thres = 0.5 * np.ones(FLAGS.hash_code_len)
id2emb = {}
for i, emb in enumerate(all_embs):
    code = (np.array(emb) > thres).tolist()
    tuple_code = tuple(code)
    gid = data_fetcher.get_train_graph_gid(i)
    inverted_index.setdefault(tuple_code, [])
    inverted_index[tuple_code].append((gid, emb))
    id2emb[gid] = emb

index_file = open('SavedModel/inverted_index_rank.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()

writeInvertedIndex('SavedModel/inverted_index.txt', inverted_index)
subprocess.check_output(['./processInvertedIndex', 
                        'SavedModel/inverted_index.txt',
                        'SavedModel/inverted_index.index', 
                        'SavedModel/inverted_index.value'])

print('finish encoding, saved index to SavedModel/inverted_index_rank.pkl')




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
    idx_list = [idx1, idx2]
    while (len(idx_list) < encode_batchsize):
        idx_list.append(0)
    feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                               placeholders, 
                                               idx_list,
                                               'train')
    feed_dict.update({placeholders['thres']: thres})
    codes, embs,var = sess.run([model.codes, model.encode_outputs[0],
                            model.layers[0][0].vars['weights']], 
                          feed_dict = feed_dict)
    #print('var')
    #print(var)
    emb1 = np.array(embs[0])
    emb2 = np.array(embs[1])
    #gid1 = data_fetcher.get_train_graph_gid(idx1)
    #gid2 = data_fetcher.get_train_graph_gid(idx2)
    #print(emb1)
    #print(id2emb[gid1])
    #print(emb2)
    #print(id2emb[gid2])
    est_ged = np.sum((emb1-emb2)**2)
    #print(est_ged)
    #break
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
            # MSE
            idx_list = sample(range(train_graph_num), 
                              encode_batchsize)
            
            # get ground truth
            # Use two batch when testing MSE, first one is randomly selected,
            # the other one is the closest ones
            true_ged = []
            for idx in idx_list:
                g = data_fetcher.get_train_graph_gid(idx)
                for val in ground_truth[q]:
                    if val[0] == g:
                        true_ged.append(val[1])
                        break
               
            # compute code and embedding for training graphs
            # To adjust to the size of placeholders, we add some graphs for padding
            feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                       placeholders, 
                                                       idx_list,
                                                       'train')
            feed_dict.update({placeholders['thres']: thres})
        
            codes, embs = sess.run([model.codes, model.encode_outputs[0]], 
                                   feed_dict = feed_dict)
            
            l = 0
            for code_train, emb_train in zip(codes, embs):
                test_ged_cnt.setdefault(true_ged[l], 0)
                test_ged_cnt[true_ged[l]] = test_ged_cnt[true_ged[l]] + 1

                emb1 = np.array(emb)
                emb2 = np.array(emb_train)    
                est_ged = np.sum((emb1-emb2)**2)
                if est_ged > FLAGS.GED_threshold:
                    est_ged = FLAGS.GED_threshold


                MSE_test_con = MSE_test_con + ((true_ged[l]-est_ged)**2)/(total_query_num*encode_batchsize*2)
                pred_cnt.setdefault(int(est_ged),0)
                pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)] + 1
    
                code1 = np.array(code, dtype=np.float32)
                code2 = np.array(code_train, dtype=np.float32)
                est_ged = np.sum((code1-code2)**2)
                if est_ged > FLAGS.GED_threshold:
                    est_ged = FLAGS.GED_threshold

                MSE_test_dis = MSE_test_dis + ((true_ged[l]-est_ged)**2)/(total_query_num*encode_batchsize*2)
                l = l + 1

            # Second batch, the closest graphs
            true_ged = []
            idx_list = []
            for pair in ground_truth[q][0:encode_batchsize]:
                idx_list.append(data_fetcher.get_pos_by_gid(pair[0], 'train'))
                true_ged.append(pair[1])  
               
            # compute code and embedding for training graphs
            # To adjust to the size of placeholders, we add some graphs for padding
            feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                       placeholders, 
                                                       idx_list,
                                                       'train')
            feed_dict.update({placeholders['thres']: thres})
        
            codes, embs = sess.run([model.codes, model.encode_outputs[0]], 
                                   feed_dict = feed_dict)
            
            l = 0
            for code_train, emb_train in zip(codes, embs):
                test_ged_cnt.setdefault(true_ged[l], 0)
                test_ged_cnt[true_ged[l]] = test_ged_cnt[true_ged[l]] + 1

                emb1 = np.array(emb)
                emb2 = np.array(emb_train)    
                est_ged = np.sum((emb1-emb2)**2)
                if est_ged > FLAGS.GED_threshold:
                    est_ged = FLAGS.GED_threshold


                MSE_test_con = MSE_test_con + ((true_ged[l]-est_ged)**2)/(total_query_num*encode_batchsize*2)
                pred_cnt.setdefault(int(est_ged),0)
                pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)] + 1
    
                code1 = np.array(code, dtype=np.float32)
                code2 = np.array(code_train, dtype=np.float32)
                est_ged = np.sum((code1-code2)**2)
                if est_ged > FLAGS.GED_threshold:
                    est_ged = FLAGS.GED_threshold

                MSE_test_dis = MSE_test_dis + ((true_ged[l]-est_ged)**2)/(total_query_num*encode_batchsize*2)
                l = l + 1






            if test_top_k:
                # top k query
                est_top_k_wrp = get_top_k_similar_graphs_gid(inverted_index, 
                                                             tuple_code,
                                                             emb,
                                                             FLAGS.top_k)
            

                pos = FLAGS.top_k
                while ground_truth[q][pos][1] == ground_truth[q][pos-1][1]:
                    pos = pos + 1
                    if pos == len(ground_truth[q]):
                        break

                true_top_k_wrp = ground_truth[q][0:pos]
        
                est_top_k = [pair[0] for pair in est_top_k_wrp]
                true_top_k = [pair[0] for pair in true_top_k_wrp]
#               print(true_top_k)
#               print(est_top_k)

                PAtKs.append(len(set(est_top_k)&set(true_top_k)) / FLAGS.top_k)
                true_top_k = true_top_k[0:FLAGS.top_k]
                rho, _ = spearmanr(est_top_k, true_top_k)
                SRCCs.append(rho)
                tau, _ = kendalltau(est_top_k, true_top_k)
                KRCCs.append(tau)


        
            # range query
            if test_range_query:
                cur_pos = 0
                for t in range(1,t_max):
                #    similar_set_wrp = get_similar_graphs_gid(inverted_index,
                #                                         tuple_code,
                #                                         t)
                #    similar_set = set([pair[0] for pair in similar_set_wrp])
                    query_time = time.time()
                    if FLAGS.fine_grained:
                        ret = subprocess.check_output(['./query',
                                                   str(tupleCode2IntegerCode(tuple_code)),
                                                   str(t),
                                                   'SavedModel/inverted_index.index',
                                                   'SavedModel/inverted_index.value',
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.hash_code_len),
                                                   '1'] + [str(dim) for dim in emb])
                    else:
                        ret = subprocess.check_output(['./query',
                                                   str(tupleCode2IntegerCode(tuple_code)),
                                                   str(t),
                                                   'SavedModel/inverted_index.index',
                                                   'SavedModel/inverted_index.value',
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.hash_code_len),
                                                   '-1'])
                    if i + j == 0:
                        print('t={:d}, cost {:f} s'.format(t, time.time()-query_time))

                    similar_set = set([int(gid) for gid in ret.split()])

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

if test_range_query:
    print('For range query')
    for t in range(1,t_max):
        print('threshold = {:d}'.format(t), end=' ')    
        print('empty cnt = {:d}'.format(zero_cnt[t-1]), end = ' ')
        print('average precision = %f'%(sum(precisions[t-1])/len(precisions[t-1])), end = ' ')
        print('average recall = %f'%(sum(recalls[t-1])/len(recalls[t-1])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores[t-1])/len(f1_scores[t-1])))

    print('ignore empty answers') 
    for t in range(1,t_max):
        print('threshold = {:d}'.format(t), end = ' ')    
        print('average precision = %f'%(sum(precisions_nz[t-1])/len(precisions_nz[t-1])), end = ' ')
        print('average recall = %f'%(sum(recalls_nz[t-1])/len(recalls_nz[t-1])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores_nz[t-1])/len(f1_scores_nz[t-1])))
                
    print(ged_cnt)
    print('FLAGS.k={:d}'.format(FLAGS.k))
