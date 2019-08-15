from random import randint
from config import FLAGS
from utils import *
import numpy as np
from scipy.stats import spearmanr, kendalltau
import subprocess
import time
import os

def computeTrainingMSE(sess, model, thres, data_fetcher, placeholders,
                       pair_num=100, use_code=True, use_emb=True):
    # Compute MSE of estimated GED for training data
    # Randomly sample 100 pairs and compute MSE
    train_graph_num = data_fetcher.get_train_graphs_num()
    encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
    print('Computing training MSE...')
    MSE_train_con = 0
    MSE_train_dis = 0
    train_ged_cnt = {}
    pred_cnt = {}
    
    for i in range(pair_num):
        idx1 = randint(0, train_graph_num - 1)
        idx2 = randint(0, train_graph_num - 1)
        while idx1 == idx2:
            idx2 = randint(0, train_graph_num - 1)
    
        true_ged = data_fetcher.getLabelForPair(data_fetcher.train_graphs[idx1], 
                                                data_fetcher.train_graphs[idx2])
        if FLAGS.clip == True and true_ged == -1:
            true_ged = FLAGS.GED_threshold

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
        if use_code == True and use_emb == True:
            codes, embs = sess.run([model.codes, model.ecd_embeddings],
                                   feed_dict = feed_dict)
        elif use_code == True and use_emb == False:
            codes = sess.run(model.codes,
                             feed_dict = feed_dict)
        elif use_code == False and use_emb == True:
            embs = sess.run(model.ecd_embeddings, 
                            feed_dict=feed_dict)
        else:
            raise RuntimeError('use_code and use_emb cannnot both be false')

        if use_emb:
            emb1 = np.array(embs[0])
            emb2 = np.array(embs[1])    
            est_ged = model.getGEDByEmb(emb1, emb2)
            if est_ged > FLAGS.GED_threshold:
                est_ged = FLAGS.GED_threshold
    
            pred_cnt.setdefault(int(est_ged),0)
            pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)]+1
            MSE_train_con = MSE_train_con + ((true_ged-est_ged)**2)/pair_num

        if use_code:    
            code1 = np.array(codes[0], dtype=np.float32)
            code2 = np.array(codes[1], dtype=np.float32)
            est_ged = model.getGEDByCode(code1, code2)
            if est_ged > FLAGS.GED_threshold:
                est_ged = FLAGS.GED_threshold

            MSE_train_dis = MSE_train_dis + ((true_ged-est_ged)**2)/pair_num
    


    print(train_ged_cnt)
    if use_emb:
        print(pred_cnt)
        print('MSE for training (continuous) = {:f}'.format(MSE_train_con))
    if use_code:
        print('MSE for training (discrete) = {:f}'.format(MSE_train_dis))



def computeTestMSEWithoutGroundTruth(model, sess, thres, data_fetcher, placeholders,
                         use_code=True, use_emb=True):
    total_query_num = data_fetcher.get_test_graphs_num()
    train_graph_num = data_fetcher.get_train_graphs_num()
    test_ged_cnt = {}
    pred_cnt = {}
    encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
    MSE_test_con = 0
    MSE_test_dis = 0
    
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
        
        if use_code == True and use_emb == True:
            codes, embs = sess.run([model.codes, model.ecd_embeddings],
                                   feed_dict = feed_dict)            
            code1 = codes[0]
            code1 = np.array(code1, dtype=np.float32)
            emb1 = np.array(embs[0])
        elif use_code == True and use_emb == False:
            codes = sess.run(model.codes,
                             feed_dict = feed_dict)    
            code1 = codes[0]
            code1 = np.array(code1, dtype=np.float32)
        elif use_code == False and use_emb == True:
            embs = sess.run(model.ecd_embeddings, 
                            feed_dict=feed_dict)
            emb1 = np.array(embs[0])
        else:
            raise RuntimeError('use_code and use_emb cannnot both be false')
        
        # Compute code and embedding for training graph
        idx_list = [idx2]   
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)
    
        feed_dict = construct_feed_dict_for_encode(data_fetcher, 
                                                   placeholders, 
                                                   idx_list,
                                                   'train')
        feed_dict.update({placeholders['thres']: thres})

        if use_code == True and use_emb == True:
            codes, embs = sess.run([model.codes, model.ecd_embeddings],
                                   feed_dict = feed_dict)            
            code2 = codes[0]
            code2 = np.array(code1, dtype=np.float32)
            emb2 = np.array(embs[0])
        elif use_code == True and use_emb == False:
            codes = sess.run(model.codes,
                             feed_dict = feed_dict)    
            code2 = codes[0]
            code2 = np.array(code1, dtype=np.float32)
        elif use_code == False and use_emb == True:
            embs = sess.run(model.ecd_embeddings, 
                            feed_dict=feed_dict)
            emb2 = np.array(embs[0])
        else:
            raise RuntimeError('use_code and use_emb cannnot both be false')
    
        # Estimate GED
        if use_emb:
            est_ged = model.getGEDByEmb(emb1, emb2) 
            if est_ged > FLAGS.GED_threshold:
                est_ged = FLAGS.GED_threshold
    
            pred_cnt.setdefault(int(est_ged),0)
            pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)]+1

            MSE_test_con = MSE_test_con + ((true_ged-est_ged)**2)/total_query_num

        if use_code:    
            est_ged = model.getGEDByCode(code1, code2)
            if est_ged > FLAGS.GED_threshold:
                est_ged = FLAGS.GED_threshold

            MSE_test_dis = MSE_test_dis + ((true_ged-est_ged)**2)/total_query_num
            
    print(test_ged_cnt)
    if use_emb:
        print(pred_cnt)
        print('MSE for test (continuous) = {:f}'.format(MSE_test_con))
    if use_code:
        print('MSE for test (discrete) = {:f}'.format(MSE_test_dis))
 
       
def computeTestMSEWithGroundTruth(sess, model, thres, placeholders, 
                                  data_fetcher, ground_truth,
                                  id2emb, id2code,
                                  use_code=True, use_emb=True):
    total_query_num = data_fetcher.get_test_graphs_num()
    encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
    test_ged_cnt = {}
    pred_cnt = {}
    MSE_test_con = 0
    MSE_test_dis = 0
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
        
        if use_code == True and use_emb == True:
            codes, embs = sess.run([model.codes, model.ecd_embeddings],
                                   feed_dict = feed_dict)            
        elif use_code == True and use_emb == False:
            codes = sess.run(model.codes,
                             feed_dict = feed_dict)
            embs = codes
        elif use_code == False and use_emb == True:
            embs = sess.run(model.ecd_embeddings, 
                            feed_dict=feed_dict)
            codes = embs
        else:
            raise RuntimeError('use_code and use_emb cannnot both be false')
    

        for j, tup in enumerate(zip(codes, embs)):
            code = tup[0]
            emb = tup[1]
            # ignore all padding graphs
            if i + j >= total_query_num:
                break
            
            q = data_fetcher.get_test_graph_gid(i + j)

            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
            
            # Second batch, the closest graphs
            for pair in ground_truth[q][0:encode_batchsize]:
                true_ged = pair[1]  
                gid = pair[0]
                
                if use_emb:
                    emb_train = id2emb[gid]
                    emb1 = np.array(emb)
                    emb2 = np.array(emb_train)    
                    est_ged = model.getGEDByEmb(emb1, emb2)
                    if est_ged > FLAGS.GED_threshold:
                        est_ged = FLAGS.GED_threshold

                    MSE_test_con = MSE_test_con + ((true_ged-est_ged)**2)/(total_query_num*encode_batchsize)
                    pred_cnt.setdefault(int(est_ged),0)
                    pred_cnt[int(est_ged)] = pred_cnt[int(est_ged)] + 1
    
                if use_code:
                    code_train = id2code[gid]
                    code1 = np.array(code, dtype=np.float32)
                    code2 = np.array(code_train, dtype=np.float32)
                    est_ged = model.getGEDByCode(code1, code2)
                    
                    if est_ged > FLAGS.GED_threshold:
                        est_ged = FLAGS.GED_threshold

                    MSE_test_dis = MSE_test_dis + ((true_ged-est_ged)**2)/(total_query_num*encode_batchsize)
            
    print(test_ged_cnt)
    if use_emb:
        print(pred_cnt)
        print('MSE for test (continuous) = {:f}'.format(MSE_test_con))
    if use_code:
        print('MSE for test (discrete) = {:f}'.format(MSE_test_dis))
        

def topKQuery(sess, model, data_fetcher, ground_truth,
              inverted_index,
              placeholders,
              index_index_fname='SavedModel/inverted_index.index',
              index_value_fname='SavedModel/inverted_index.value',
              #input_batchsize = 1,
              use_code=True, use_emb=True):
    thres = np.zeros(FLAGS.hash_code_len)

    total_query_num = data_fetcher.get_test_graphs_num()
    train_graph_num = data_fetcher.get_train_graphs_num()
    PAtKs = []
    SRCCs = []
    KRCCs = []
    search_time = []
    encode_time = []
    encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize

    for i in range(0, total_query_num, encode_batchsize):   

        end = i + encode_batchsize
        if end > total_query_num:
            end = total_query_num

        
        for j in range(i, end):

            if i + j >= total_query_num:
                break

            q = data_fetcher.get_test_graph_gid(i + j)
            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
 

#        idx_list = list(range(i,end))
#        while (len(idx_list) < encode_batchsize):
#            idx_list.append(0)
        idx_list = list(range(i,end))           
        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)

        feed_dict = construct_feed_dict_for_query(data_fetcher, 
                                                  placeholders, 
                                                  idx_list,
                                                  'test')
        feed_dict.update({placeholders['thres']: thres})
        start_time = time.time()
        if use_code == True and use_emb == True:
            codes, embs = sess.run([model.codes, model.ecd_embeddings],
                                   feed_dict = feed_dict)            
        #elif use_code == True and use_emb == False:
            #codes = sess.run(model.codes,
         #                    feed_dict = feed_dict)
         #   embs = codes
        elif use_code == False and use_emb == True:
            embs = sess.run(model.ecd_embeddings, 
                            feed_dict=feed_dict)
            codes = embs
        else:
            raise RuntimeError('topKQuery: For now use_emb cannnot be false')
        encode_time.append(time.time()-start_time)
    
        for j, tup in enumerate(zip(codes, embs)):
            if i + j >= total_query_num:
                break

            code = tup[0]
            emb = tup[1]
            # ignore all padding graphs
#            if i + j >= total_query_num:
#                break

            tuple_code = tuple(code)
            
            start_time = time.time()
            ret = subprocess.check_output(['./topKQuery',
                                           str(tupleCode2IntegerCode(tuple_code)),
                                           str(FLAGS.hash_code_len),
                                           index_index_fname,
                                           index_value_fname,
                                           str(len(inverted_index.keys())),
                                           str(train_graph_num),
                                           str(FLAGS.hash_code_len),
                                           str(FLAGS.embedding_dim),
                                           str(FLAGS.top_k)] + [str(dim) for dim in emb])
             
            search_time.append(time.time()-start_time)
            if i + j == 0:
                print('top {:d}, cost {:f} s'.format(FLAGS.top_k, time.time()-start_time))
                # print(ret)
            est_top_k = [int(gid) for gid in ret.split()]

            pos = FLAGS.top_k
            while ground_truth[q][pos][1] == ground_truth[q][pos-1][1]:
                pos = pos + 1
                if pos == len(ground_truth[q]):
                    break

            true_top_k_wrp = ground_truth[q][0:pos]
        
            true_top_k = [pair[0] for pair in true_top_k_wrp]

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

    print('For Top-k query, k={:d}'.format(FLAGS.top_k))
    print('average precision at k = {:f}'.format(sum(PAtKs)/len(PAtKs)))
    print('average rho = {:f}'.format(sum(SRCCs)/len(SRCCs)))
    print('average tau = {:f}'.format(sum(KRCCs)/len(KRCCs)))
    print('average search time = {:f}'.format(sum(search_time)/len(search_time)))
    print('average encode time = {:f}'.format(sum(encode_time)/len(encode_time)))
    
    
def rangeQueryVerification(q_idx, candidate_set, data_fetcher, upbound):
    query_graph = data_fetcher.test_graphs[q_idx]
    q_fname = data_fetcher.writeGraph2TempFile(query_graph)
    ret_set = set()
    for gid in candidate_set:
        g2 = data_fetcher.getGraphByGid(gid)
        g2_fname = data_fetcher.writeGraph2TempFile(g2)

        ged = subprocess.check_output(['./ged', q_fname, '1', g2_fname, '1', 
                                       str(upbound),
                                       str(FLAGS.beam_width)])
        # remove temporary files
        os.remove(q_fname)
        os.remove(g2_fname)
        os.remove(q_fname+'_ordered')
        os.remove(g2_fname+'_ordered')
        
        if int(ged) != -1:
            ret_set.add(gid)
            
    return ret_set
    
    
def rangeQuery(sess, model, data_fetcher, ground_truth,
              placeholders,
              inverted_index,
              t_min = 1, t_max=FLAGS.GED_threshold-2,
              index_index_fname='SavedModel/inverted_index.index',
              index_value_fname='SavedModel/inverted_index.value',
              #input_batchsize = 1,
              use_code=True, use_emb=True):
    thres = np.zeros(FLAGS.hash_code_len)

    total_query_num = data_fetcher.get_test_graphs_num()
    train_graph_num = data_fetcher.get_train_graphs_num()
    encode_batchsize=(1+FLAGS.k) * FLAGS.batchsize    

    precisions = [[] for i in range(t_min, t_max+1)]
    recalls = [[] for i in range(t_min, t_max+1)]
    f1_scores = [[] for i in range(t_min, t_max+1)]
    zero_cnt = [0 for i in range(t_min, t_max+1)]

    precisions_nz = [[] for i in range(t_min, t_max+1)]
    recalls_nz = [[] for i in range(t_min, t_max+1)]
    f1_scores_nz = [[] for i in range(t_min, t_max+1)]

    ret_size = [[] for i in range(t_min, t_max+1)]

    
    search_time = [[] for i in range(t_min, t_max+1)]
    encode_time = []
    verify_time = [[] for i in range(t_min, t_max+1)]

    for i in range(0, total_query_num, encode_batchsize):   

        end = i + encode_batchsize
        if end > total_query_num:
            end = total_query_num


        for j in range(i, end):
            if i + j >= total_query_num:
                break
            q = data_fetcher.get_test_graph_gid(i + j)
            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
 

#        idx_list = list(range(i,end))
#        while (len(idx_list) < encode_batchsize):
#            idx_list.append(0)
        idx_list = list(range(i,end))           

        while (len(idx_list) < encode_batchsize):
            idx_list.append(0)

        feed_dict = construct_feed_dict_for_query(data_fetcher, 
                                                  placeholders, 
                                                  idx_list,
                                                  'test')
        feed_dict.update({placeholders['thres']: thres})
        start_time = time.time()
        if use_code == True and use_emb == True:
            codes, embs = sess.run([model.codes, model.ecd_embeddings],
                                   feed_dict = feed_dict)            
        elif use_code == True and use_emb == False:
            codes = sess.run(model.codes,
                             feed_dict = feed_dict)
            embs = codes
        elif use_code == False and use_emb == True:
            embs = sess.run(model.ecd_embeddings, 
                            feed_dict=feed_dict)
            codes = embs
        else:
            raise RuntimeError('rangeQuery: use_code and use_emb cannnot both be false')
        
        encode_time.append(time.time()-start_time)
    
        for j, tup in enumerate(zip(codes, embs)):
            if i + j >= total_query_num:
                break
            code = tup[0]
            emb = tup[1]
            # ignore all padding graphs
#            if i + j >= total_query_num:
#                break

            tuple_code = tuple(code)
            
            cur_pos = 0
            for t in range(t_min,t_max+1):
                    
                start_time = time.time()
                if use_emb:
                        ret = subprocess.check_output(['./query',
                                                   str(tupleCode2IntegerCode(tuple_code)),
                                                   str(t + 1),
                                                   index_index_fname,
                                                   index_value_fname,
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.embedding_dim),
                                                   '1',
                                                   str(t)] + [str(dim) for dim in emb])
                else:
                    ret = subprocess.check_output(['./query',
                                                   str(tupleCode2IntegerCode(tuple_code)),
                                                   str(t),
                                                   'SavedModel/inverted_index.index',
                                                   'SavedModel/inverted_index.value',
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.embedding_dim),
                                                   '-1'])
                search_time[t-t_min].append(time.time()-start_time)
                if i + j == 0:
                    print('t={:d}, cost {:f} s'.format(t, time.time()-start_time))
                    
                candidate_set = set([int(gid) for gid in ret.split()])

                start_time = time.time()
                similar_set = rangeQueryVerification(i+j, candidate_set, 
                                                     data_fetcher,
                                                     upbound=t)                
                verify_time[t-t_min].append(time.time()-start_time)
#                similar_set = set([int(gid) for gid in ret.split()])

                ret_size[t-t_min].append(len(similar_set))

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

                precisions[t-t_min].append(precision)
                if len(real_sim_set) > 0:
                    precisions_nz[t-t_min].append(precision)

                if len(real_sim_set) == 0:
                    zero_cnt[t-t_min] = zero_cnt[t-1] + 1
                    if len(similar_set) == 0:
                        recall = 1
                    else:
                        recall = 0
                else:
                    recall = len(tmp)/len(real_sim_set)
                recalls[t-t_min].append(recall)
                if len(real_sim_set) > 0:
                    recalls_nz[t-t_min].append(recall)

                if precision * recall == 0:
                    if len(real_sim_set) == 0 and len(similar_set) == 0:
                        f1_score = 1
                    else:
                        f1_score = 0
                else:
                    f1_score = 2*precision*recall/(precision+recall)

                f1_scores[t-t_min].append(f1_score)
                if len(real_sim_set) > 0:
                    f1_scores_nz[t-t_min].append(f1_score)

    print('For range query')
    print('average encode time = {:f}'.format(sum(encode_time)/len(encode_time)))
    for t in range(t_min, t_max+1):
        print('threshold = {:d}'.format(t), end=' ')
        print('empty cnt = {:d}'.format(zero_cnt[t-t_min]), end = ' ')
        print('average precision = %f'%(sum(precisions[t-t_min])/len(precisions[t-t_min])), end = ' ')
        print('average recall = %f'%(sum(recalls[t-t_min])/len(recalls[t-t_min])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores[t-t_min])/len(f1_scores[t-t_min])), end = ' ')
        print('average return size = %f'%(sum(ret_size[t-t_min])/len(ret_size[t-t_min])), end = ' ')
        print('average search time = {:f}'.format(sum(search_time[t-t_min])/len(search_time[t-t_min])), end= ' ')
        print('average search time = {:f}'.format(sum(verify_time[t-t_min])/len(verify_time[t-t_min])))

    print('ignore empty answers')
    for t in range(t_min,t_max+1):
        print('threshold = {:d}'.format(t), end = ' ')
        if len(precisions_nz[t-t_min]) > 0:
            ave_pre_nz = sum(precisions_nz[t-t_min])/len(precisions_nz[t-t_min])
        else:
            ave_pre_nz = float('nan')
        print('average precision = %f'%(ave_pre_nz), end = ' ')
 
        if len(recalls_nz[t-t_min]) > 0:
            ave_rc_nz = sum(recalls_nz[t-t_min])/len(recalls_nz[t-t_min])
        else:
            ave_rc_nz = float('nan') 
        print('average recall = %f'%(ave_rc_nz), end = ' ')

        if len(f1_scores_nz[t-t_min]) > 0:
            ave_f1_nz = sum(f1_scores_nz[t-t_min])/len(f1_scores_nz[t-t_min])
        else:
            ave_f1_nz = float('nan')
        print('average f1-score = %f'%(ave_f1_nz))


