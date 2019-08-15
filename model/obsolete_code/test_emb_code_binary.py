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
from tensorflow.python.tools import inspect_checkpoint as chkp
import sys


from utils import *
from graphHashFunctions import GraphHash_Emb_Code_Binary
import numpy as np
from config import FLAGS
from DataFetcher import DataFetcher
import pickle

train = False

os.environ['CUDA_VISIBLE_DEVICES']='4,6'
test_classification = True

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

#chkp.print_tensors_in_checkpoint_file("SavedModel/model_rank.ckpt", tensor_name='', all_tensors=True, all_tensor_names=True)



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
model = GraphHash_Emb_Code_Binary(placeholders, 
                       input_dim=data_fetcher.get_node_feature_dim(),
                       next_ele = next_element,
                       logging=True)
# Initialize session
sess = tf.Session()

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
    
        if (epoch+1) % 50 == 0:
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

    save_path = saver.save(sess, "SavedModelBinary/model_rank.ckpt")
    print("Model saved in path: {}".format(save_path))

else:
    saver.restore(sess, "SavedModelBinary/model_rank.ckpt")
    print("Model restored")



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
                            model.ecd_embeddings[0]], 
                            feed_dict = feed_dict)
    codes = list(codes)
    codes = codes[0:end-i]
    all_codes = all_codes + codes
    
    embs = list(embs)
    embs = embs[0:end-i]
    all_embs = all_embs + embs
    
all_codes_np = np.array(all_codes)
thres = np.mean(all_codes_np, axis=0)
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

index_file = open('SavedModelBinary/inverted_index_rank.pkl', 'wb')
pickle.dump(inverted_index, index_file)
index_file.close()
writeInvertedIndex('SavedModelBinary/inverted_index.txt', inverted_index, FLAGS.embedding_dim)
subprocess.check_output(['./processInvertedIndex', 
                        'SavedModelBinary/inverted_index.txt',
                        'SavedModelBinary/inverted_index.index', 
                        'SavedModelBinary/inverted_index.value'])

print('finish encoding, saved index to SavedModelBinary/inverted_index_rank.pkl')
 

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
    embs, inputs, act = sess.run([model.ecd_embeddings[0], model.ecd_inputs, model.ecd_activations], 
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
    
print(train_ged_cnt)
print(pred_cnt)
print('MSE for training (continuous) = {:f}'.format(MSE_train_con))




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
    
        embs = sess.run(model.ecd_embeddings[0], 
                               feed_dict = feed_dict)
        emb1 = np.array(embs[0])

        # Compute code and embedding for training graph
        idx_list = [idx2]   
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)
    
        feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                   placeholders, 
                                                   idx_list,
                                                   'train')
        feed_dict.update({placeholders['thres']: thres})

        embs = sess.run(model.ecd_embeddings[0], 
                               feed_dict = feed_dict)
        emb2 = np.array(embs[0])
    
        # Estimate GED
        est_ged = np.sum((emb1-emb2)**2) 
        if est_ged > FLAGS.GED_threshold:
            est_ged = FLAGS.GED_threshold
    
        pred_cnt.setdefault(int(est_ged),0)
        pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)]+1

        MSE_test_con = MSE_test_con + ((true_ged-est_ged)**2)/total_query_num

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
        
        codes, embs = sess.run([model.codes, model.ecd_embeddings[0]], 
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
        
            codes, embs = sess.run([model.codes, model.ecd_embeddings[0]], 
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
        
            codes, embs = sess.run([model.codes, model.ecd_embeddings[0]], 
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
    
                l = l + 1




            # classification query
            if test_classification:
                cur_pos = 0
                while ground_truth[q][cur_pos][1] < FLAGS.GED_threshold:
                    cur_pos = cur_pos + 1
                real_sim_set_wrp = ground_truth[q][0:cur_pos]
                real_sim_set = set([pair[0] for pair in real_sim_set_wrp])

                for t in range(1,t_max):
                    
                    query_time = time.time()
                    if FLAGS.fine_grained:
                        
                        ret = subprocess.check_output(['./query',
                                                   str(tupleCode2IntegerCode(tuple_code)),
                                                   str(t),
                                                   'SavedModelBinary/inverted_index.index',
                                                   'SavedModelBinary/inverted_index.value',
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.embedding_dim),
                                                   '1',
                                                   str(FLAGS.GED_threshold)] + [str(dim) for dim in emb])
                        
 
                    else:
                        ret = subprocess.check_output(['./query',
                                                   str(tupleCode2IntegerCode(tuple_code)),
                                                   str(t),
                                                   'SavedModelBinary/inverted_index.index',
                                                   'SavedModelBinary/inverted_index.value',
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.embedding_dim),
                                                   '-1'])
                    if i + j == 0:
                        print('t={:d}, cost {:f} s'.format(t, time.time()-query_time))

                    similar_set = set([int(gid) for gid in ret.split()])

                    ret_size[t-1].append(len(similar_set))

                    while cur_pos < len(ground_truth[q]) and\
                          ground_truth[q][cur_pos][1] <= t:
                        cur_pos = cur_pos + 1


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

if test_classification:
    print('For classification query')
    for t in range(1,t_max):
        print('threshold = {:d}'.format(t), end=' ')
        print('empty cnt = {:d}'.format(zero_cnt[t-1]), end = ' ')
        print('average precision = %f'%(sum(precisions[t-1])/len(precisions[t-1])), end = ' ')
        print('average recall = %f'%(sum(recalls[t-1])/len(recalls[t-1])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores[t-1])/len(f1_scores[t-1])))
        print('average return size = %f'%(sum(ret_size[t-1])/len(ret_size[t-1])))

    print('ignore empty answers')
    for t in range(1,t_max):
        print('threshold = {:d}'.format(t), end = ' ')
        print('average precision = %f'%(sum(precisions_nz[t-1])/len(precisions_nz[t-1])), end = ' ')
        print('average recall = %f'%(sum(recalls_nz[t-1])/len(recalls_nz[t-1])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores_nz[t-1])/len(f1_scores_nz[t-1])))

    print(ged_cnt)
    print('FLAGS.k={:d}'.format(FLAGS.k))


