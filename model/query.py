from random import randint
import socket
from config import FLAGS
from CSM_config import CSM_FLAGS
from utils import *
import numpy as np
from scipy.stats import spearmanr, kendalltau
import subprocess
import time
import os
from nx_to_gxl import nx_to_gxl

def computeCSMTestMSEWithGroundTruth(sess, saver, csm,
                                     csm_data_fetcher, 
                                     ground_truth,
                                     per_test_cnt = 10):
    raise NotImplementedError
    """
    total_query_num = csm_data_fetcher.get_test_graphs_num()
    MSE_test = 0
    
    for i in range(0, total_query_num):
        g1 = csm_data_fetcher.test_graphs[i]
        g_list1 = [g1]

        q = csm_data_fetcher.get_test_graph_gid(i)

        ground_truth[q] = sorted(ground_truth[q],
                                key=lambda x: x[1]*10000000 + x[0])

            # Second batch, the closest graphs
        g_list2 = []
        GT = []
        thres_scores = []
        for pair in ground_truth[q][0:per_test_cnt]:
            true_ged = pair[1]
            gid = pair[0]
            g2 = csm_data_fetcher.getGraphByGid(gid)
            g_list2.append(g2)
            normalzied_ged = true_ged*2/(len(g1.nxgraph.nodes())+len(g2.nxgraph.nodes()))
            thres = CSM_FLAGS.csm_GED_threshold*2/(len(g1.nxgraph.nodes())+len(g2.nxgraph.nodes()))
            thres = np.exp(-thres)
            score = np.exp(-normalzied_ged)
            GT.append(score)
            thres_scores.append(thres)
            
        pred = csm.predict(sess, saver, g_list1, g_list2)
        GT = np.array(GT)
        pred = np.squeeze(pred)
        thres_scores = np.array(thres_scores)
        
        GT[GT == thres_scores] = 0
            
        MSE_test = MSE_test + (np.mean((GT-pred)**2))/total_query_num

    print('MSE for test = {:f}'.format(MSE_test))
    """

def computeTrainingMSE(sess, model, thres, data_fetcher, placeholders,
                       pair_num=100, use_code=True, use_emb=True,
                       encode_batchsize=(1+FLAGS.k)*FLAGS.batchsize):
    # Compute MSE of estimated GED for training data
    # Randomly sample 100 pairs and compute MSE
    if encode_batchsize < 2:
        raise RuntimeError('encode_batchsize need to > 2 to compute training MSE')

    train_graph_num = data_fetcher.get_train_graphs_num()
#    encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
    
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
                         use_code=True, use_emb=True, 
                         encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize):


    total_query_num = data_fetcher.get_test_graphs_num()
    train_graph_num = data_fetcher.get_train_graphs_num()
    test_ged_cnt = {}
    pred_cnt = {}
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
                                  use_code=True, use_emb=True,
                                  encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize):
    total_query_num = data_fetcher.get_test_graphs_num()
    
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
            
            # compute the closest graphs
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
              index_index_fname='SavedModel/inverted_index_'+FLAGS.dataset+'.index',
              index_value_fname='SavedModel/inverted_index_'+FLAGS.dataset+'.value',
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
#    encode_batchsize = (1+FLAGS.k) * FLAGS.batchsize
    encode_batchsize=1

    for i in range(0, total_query_num, encode_batchsize):   

        end = i + encode_batchsize
        if end > total_query_num:
            end = total_query_num

        """    
        for j in range(i, end):

            if i + j >= total_query_num:
                break

            q = data_fetcher.get_test_graph_gid(i + j)
            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
         """

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
    
        # start query
        for j, tup in enumerate(zip(codes, embs)):
            if i + j >= total_query_num:
                break

            q = data_fetcher.get_test_graph_gid(i+j)
            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])

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
            #ret = ret.split() 
            search_time.append(time.time()-start_time)
            #search_time.append(float(ret[-1]))
            #ret = ret[0:-1]
            if i + j == 0:
                print('top {:d}, cost {:f} s'.format(FLAGS.top_k, time.time()-start_time))
                # print(ret)
            try:
                est_top_k = [int(gid) for gid in ret.split()]
            except ValueError:
                print(ret)
                raise RuntimeError('subprocess ret'+str(ret))

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
            try:
                real_rank = [real_rank_all[gid] for gid in est_top_k]
            except KeyError:
                print(ret)
                print(est_top_k)
                raise RuntimeError('')

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
    
    
def traditionalApproxVerification(q_idx, candidate_set, data_fetcher, upbound):
    candidate_set = list(candidate_set)    
    n = len(candidate_set)
    if n == 0:
        return set()
    
    query_xml = os.path.join('tmpfile',
                             str(time.time())+'.xml')
    qf = open(query_xml, 'w')
    qf.write('<?xml version="1.0"?>\n<GraphCollection>\n<graphs>\n')        
    query_graph = data_fetcher.test_graphs[q_idx]
    qfname = os.path.join('tmpfile',
                          str(time.time())+\
                          str(query_graph.nxgraph.graph['gid'])+'.gxl')
    nx_to_gxl(query_graph.nxgraph, query_graph.nxgraph.graph['gid'],
              qfname)
    qf.write('<print file="{}"/>\n'.format(qfname))
    qf.write('</graphs>\n</GraphCollection>')
    qf.close()
        
    collection_file = os.path.join('tmpfile',
                                   str(time.time())+'.xml')
    f = open(collection_file, 'w')
    f.write('<?xml version="1.0"?>\n<GraphCollection>\n<graphs>\n')        
    fnames = []
    for gid in candidate_set:
        g = data_fetcher.getGraphByGid(gid)
        fname = os.path.join('tmpfile',
                             str(time.time())+\
                             str(g.nxgraph.graph['gid'])+'.gxl')

        nx_to_gxl(g.nxgraph, g.nxgraph.graph['gid'], fname)
        f.write('<print file="{}"/>\n'.format(fname))
        fnames.append(fname)     
            
    f.write('</graphs>\n</GraphCollection>')
    f.close()
        
    ged_1 = subprocess.check_output(['java', '-cp', 
                                     'graph-matching-toolkit/src',
                                     'algorithms.GraphMatching',
                                     data_fetcher.data_dir+'/prop/VJ.prop',
                                     collection_file, 
                                     query_xml])
    
    ged_2 = subprocess.check_output(['java', '-cp', 
                                     'graph-matching-toolkit/src',
                                     'algorithms.GraphMatching',
                                     data_fetcher.data_dir+'/prop/beam.prop',
                                     collection_file, 
                                     query_xml])
    
    ged_3 = subprocess.check_output(['java', '-cp', 
                                     'graph-matching-toolkit/src',
                                     'algorithms.GraphMatching',
                                     data_fetcher.data_dir+'/prop/hungarian.prop',
                                     collection_file, 
                                     query_xml])

    ged_1_list = ged_1.split()
    ged_2_list = ged_2.split()
    ged_3_list = ged_3.split()
    ret_set = set()
    for i in range(n):
        ged = min([float(ged_1_list[i]), 
                   float(ged_2_list[i]), 
                   float(ged_3_list[i])])
        if ged <= upbound:
            ret_set.add(candidate_set[i])
    
    os.remove(collection_file)
    for fname in fnames:
        os.remove(fname)
    os.remove(query_xml)
    os.remove(qfname)
    return ret_set

def BssGedServerVerification(q_idx, candidate_set, data_fetcher, upbound, port=12345):
    raise NotImplementedError
    """
    if len(candidate_set) == 0:
        return set(), 0

    candidate_set = list(candidate_set)
    start_time = time.time()
    addr = ("127.0.0.1", port)
    s = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

    query_graph = data_fetcher.test_graphs[q_idx]
    #print(query_graph.nxgraph.graph['gid'])
    print('python: %d %d'%(len(query_graph.nxgraph.nodes()), len(query_graph.nxgraph.edges())))
    q_gid = str(query_graph.nxgraph.graph['gid'])
    #q_fname = data_fetcher.writeGraph2TempFile(query_graph)
#    g_fname = data_fetcher.writeGraphList2TempFile(candidate_set)
    ret_set = set()

#    geds = subprocess.check_output(['./ged', q_fname, '1', g_fname, 
#                                   str(len(candidate_set)), 
#                                   str(upbound),
#                                   str(FLAGS.beam_width)])

    EOS = False
    s.sendto(bytes(q_gid, 'utf-8'), addr)
    s.sendto(bytes(str(upbound),'utf-8'), addr)
    for i in range(0, len(candidate_set), 1000):
        start = i
        end = min([i+1000, len(candidate_set)])
        gid_send = ''

        for gid in candidate_set[start:end]:
            gid_send=gid_send+str(gid)+' '
        if end - start < 1000:
            gid_send = gid_send+'-1'
            EOS = True

        gid_send = bytes(gid_send, 'utf-8')
        s.sendto(gid_send, addr)
    if not EOS:
        s.sendto(bytes('-1', 'utf-8'), addr)

    geds = s.recvfrom(10000)
    verify_time = time.time() - start_time
    #verify_time = verify_time * 1000
    geds = geds[0]
    geds = geds.split()

    
#    verify_time = float(geds[-1])
#    geds = geds[0:-1]
    #print(geds)
    assert(len(geds) == len(candidate_set))
    for i, ged in enumerate(geds):

        if int(ged) != -1:
            ret_set.add(candidate_set[i])
        #print(len(ret_set))
            
    #os.remove(q_fname)
    #os.remove(g_fname)
    return ret_set, verify_time
    """

def BssGedVerification(q_idx, candidate_set, data_fetcher, upbound):
    if len(candidate_set) == 0:
        return set(), 0
    candidate_set = list(candidate_set)
    query_graph = data_fetcher.test_graphs[q_idx]
    q_fname = data_fetcher.writeGraph2TempFile(query_graph)
    g_fname = data_fetcher.writeGraphList2TempFile(candidate_set)
    ret_set = set()

    #start_time = time.time()
    geds = subprocess.check_output(['./ged', q_fname, '1', g_fname, 
                                   str(len(candidate_set)), 
                                   str(upbound),
                                   str(FLAGS.beam_width)])
    #verify_time = time.time() - start_time
    geds = geds.split()
    verify_time = float(geds[-1])
    
    geds = geds[0:-1]
    #print(geds)
    for i, ged in enumerate(geds):

        if int(ged) != -1:
            ret_set.add(candidate_set[i])
        #print(len(ret_set))
            
    os.remove(q_fname)
    os.remove(g_fname)
    return ret_set, verify_time
    
def CSMVerification(q_idx, candidate_set, csm_data_fetcher, upbound, csm, sess, csm_saver):
    if len(candidate_set) == 0:
        return set(), 0
    candidate_set = list(candidate_set)
    g_graphs = [csm_data_fetcher.getGraphByGid(gid) for gid in candidate_set]
    query_graph = csm_data_fetcher.test_graphs[q_idx]

    start_time = time.time()
    scores = csm.predict(sess, csm_saver, [query_graph], g_graphs)
    scores = np.resize(scores, (scores.size))

    q_size = len(query_graph.nxgraph.nodes())
    g_sizes = [len(g.nxgraph.nodes()) for g in g_graphs]
    thres_scores = np.exp(-(np.array([(upbound+0.5)*2/(q_size+g_size) for g_size in g_sizes], dtype=np.float32)))

    ret_set = set()
    for i in range(len(candidate_set)):
        if thres_scores[i] <= scores[i]:
            ret_set.add(candidate_set[i])

    verify_time = time.time() - start_time
    return ret_set, verify_time

def rangeQueryVerification(q_idx, candidate_set, data_fetcher, upbound, 
                           csm=None, csm_data_fetcher=None, sess = None, csm_saver=None):
    """ 
    if upbound > 3:
        return traditionalApproxVerification(q_idx, candidate_set, data_fetcher,
                                             upbound)
    
    query_graph = data_fetcher.test_graphs[q_idx]
    if len(query_graph.nxgraph.nodes()) > 20:
        return traditionalApproxVerification(q_idx, candidate_set,
                                             data_fetcher,
                                             upbound)
    
    candidate_set_small = set()
    candidate_set_big = set()
    for gid in candidate_set:
        g = data_fetcher.getGraphByGid(gid)
        if len(g.nxgraph.nodes()) > 20:
            candidate_set_big.add(gid)
        else:
            candidate_set_small.add(gid)
            
    return BssGedVerification(q_idx, candidate_set_small, data_fetcher, upbound) |\
    traditionalApproxVerification(q_idx, candidate_set_big, data_fetcher, upbound)
    """
    #return traditionalApproxVerification(q_idx, candidate_set, data_fetcher, upbound)
    if csm is None:
      return BssGedVerification(q_idx, candidate_set, data_fetcher, upbound)
    else:
      return CSMVerification(q_idx, candidate_set, csm_data_fetcher, upbound, csm, sess, csm_saver)

def rangeQuery(sess, model, data_fetcher, ground_truth,
              placeholders,
              inverted_index,
              csm = None,
              csm_data_fetcher = None,
              csm_saver = None,
              t_min = 1, t_max=FLAGS.GED_threshold-2,
              index_index_fname='SavedModel/inverted_index_'+FLAGS.dataset+'.index',
              index_value_fname='SavedModel/inverted_index_'+FLAGS.dataset+'.value',
              #input_batchsize = 1,
              use_code=True, use_emb=True):
#    f = open('QINFO.txt', 'w')
    thres = np.zeros(FLAGS.hash_code_len)

    total_query_num = data_fetcher.get_test_graphs_num()
    train_graph_num = data_fetcher.get_train_graphs_num()
    #encode_batchsize=(1+FLAGS.k) * FLAGS.batchsize    
    encode_batchsize = 1

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

    precisions_cdd = [[] for i in range(t_min, t_max+1)]
    recalls_cdd = [[] for i in range(t_min, t_max+1)]
    f1_scores_cdd = [[] for i in range(t_min, t_max+1)]

    precisions_nz_cdd = [[] for i in range(t_min, t_max+1)]
    recalls_nz_cdd = [[] for i in range(t_min, t_max+1)]
    f1_scores_nz_cdd = [[] for i in range(t_min, t_max+1)]

    ret_size_cdd = [[] for i in range(t_min, t_max+1)]

    
    search_time = [[] for i in range(t_min, t_max+1)]
    encode_time = []
    verify_time = [[] for i in range(t_min, t_max+1)]


    for i in range(0, total_query_num, encode_batchsize):   

        end = i + encode_batchsize
        if end > total_query_num:
            end = total_query_num


 

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

            q = data_fetcher.get_test_graph_gid(i+j)
            ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
#            if q != 27764:
#                continue
            tuple_code = tuple(code)
            
            cur_pos = 0
            #line = str(q) + ' ' + str(tupleCode2IntegerCode(tuple_code))+'\n'
            #f.write(line)
#            line = ''
#            for dim in emb:
#                line = line + ('%.6f'%dim) + ' '
#            line = line + '\n'
#            f.write(line)
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
                                                   str(t+1),
                                                   index_index_fname,
                                                   index_value_fname,
                                                   str(len(inverted_index.keys())),
                                                   str(train_graph_num),
                                                   str(FLAGS.hash_code_len),
                                                   str(FLAGS.embedding_dim),
                                                   '-1'])
                ret = ret.split()
#                search_time[t-t_min].append(time.time()-start_time)
                search_time[t-t_min].append(float(ret[-1]))
                ret = ret[0:-1]
                if i + j == 0:
                    print('t={:d}, cost {:f} s'.format(t, time.time()-start_time))
                    
#                candidate_set = set([int(gid) for gid in ret.split()])
                candidate_set = set([int(gid) for gid in ret])

#                for tok in candidate_set:
#                    line = line + str(tok) + ' '
#                f.write(line)
#                f.write('\n')
 

                while cur_pos < len(ground_truth[q]) and\
                        ground_truth[q][cur_pos][1] <= t:
                    cur_pos = cur_pos + 1

                real_sim_set_wrp = ground_truth[q][0:cur_pos]
                real_sim_set = set([pair[0] for pair in real_sim_set_wrp])

                ret_size_cdd[t-t_min].append(len(candidate_set))

                tmp = candidate_set & real_sim_set
                if len(candidate_set) == 0:
                    if len(real_sim_set) == 0:
                        precision = 1
                    else:
                        precision = 0
                else:
                    precision =  len(tmp)/len(candidate_set)

                precisions_cdd[t-t_min].append(precision)
                if len(real_sim_set) > 0:
                    precisions_nz_cdd[t-t_min].append(precision)
                 

                if len(real_sim_set) == 0:
                    if len(candidate_set) == 0:
                        recall = 1
                    else:
                        recall = 0
                else:
                    recall = len(tmp)/len(real_sim_set)
                recalls_cdd[t-t_min].append(recall)
                if len(real_sim_set) > 0:
                    recalls_nz_cdd[t-t_min].append(recall)

                if precision * recall == 0:
                    if len(real_sim_set) == 0 and len(candidate_set) == 0:
                        f1_score = 1
                    else:
                        f1_score = 0
                else:
                    f1_score = 2*precision*recall/(precision+recall)

                f1_scores_cdd[t-t_min].append(f1_score)
                if len(real_sim_set) > 0:
                    f1_scores_nz_cdd[t-t_min].append(f1_score)
                   



                start_time = time.time()
                similar_set, verify_t = rangeQueryVerification(i+j, candidate_set, 
                                                     data_fetcher,
                                                     upbound=t,
                                                     csm=csm,
                                                     csm_data_fetcher=csm_data_fetcher,
                                                     sess = sess,
                                                     csm_saver=csm_saver)                
#                verify_time[t-t_min].append(verify_t)
                verify_time[t-t_min].append(verify_t)
#                similar_set = set([int(gid) for gid in ret.split()])
                if i + j == 0:
                    print('verification time: {:f} s'.format(verify_t))


                ret_size[t-t_min].append(len(similar_set))


                tmp = similar_set & real_sim_set
                if len(similar_set) == 0:
                    if len(real_sim_set) == 0:
                        precision = 1
                    else:
                        precision = 0
                else:
                    precision =  len(tmp)/len(similar_set)
                
                if precision != 1:
                    if len(similar_set) > 0:
                        print(q)
                        print('t=%d'%t)
                        print(similar_set)
                        print(real_sim_set_wrp)
                        print(ground_truth[q][0:5])
                        raise RuntimeError('bug')
                
                precisions[t-t_min].append(precision)
                if len(real_sim_set) > 0:
                    precisions_nz[t-t_min].append(precision)

                if len(real_sim_set) == 0:
                    zero_cnt[t-t_min] = zero_cnt[t-t_min] + 1
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
    te = sum(encode_time)/len(encode_time)
    for t in range(t_min, t_max+1):
        print('threshold = {:d}'.format(t), end=' ')
        print('empty cnt = {:d}'.format(zero_cnt[t-t_min]), end = ' ')

        print('average cdd precision = %f'%(sum(precisions_cdd[t-t_min])/len(precisions_cdd[t-t_min])), end = ' ')
        print('average cdd recall = %f'%(sum(recalls_cdd[t-t_min])/len(recalls_cdd[t-t_min])), end = ' ')
        print('average cdd f1-score = %f'%(sum(f1_scores_cdd[t-t_min])/len(f1_scores_cdd[t-t_min])), end = ' ')
        print('average cdd return size = %f'%(sum(ret_size_cdd[t-t_min])/len(ret_size_cdd[t-t_min])))
 
        print('average precision = %f'%(sum(precisions[t-t_min])/len(precisions[t-t_min])), end = ' ')
        print('average recall = %f'%(sum(recalls[t-t_min])/len(recalls[t-t_min])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores[t-t_min])/len(f1_scores[t-t_min])), end = ' ')
#        print('average return size = %f'%(sum(ret_size[t-t_min])/len(ret_size[t-t_min])), end = ' ')
        print('average search time = {:f}'.format(sum(search_time[t-t_min])/len(search_time[t-t_min])), end= ' ')
        print('average verify time = {:f}'.format(sum(verify_time[t-t_min])/len(verify_time[t-t_min])), end = ' ')
        ts = sum(search_time[t-t_min])/len(search_time[t-t_min])
        tv = sum(verify_time[t-t_min])/len(verify_time[t-t_min])
        print('average total time = {:f}'.format(te+ts+tv))

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


        if len(precisions_nz_cdd[t-t_min]) > 0:
            ave_pre_nz = sum(precisions_nz_cdd[t-t_min])/len(precisions_nz_cdd[t-t_min])
        else:
            ave_pre_nz = float('nan')
        print('average cdd precision = %f'%(ave_pre_nz), end = ' ')
 
        if len(recalls_nz_cdd[t-t_min]) > 0:
            ave_rc_nz = sum(recalls_nz_cdd[t-t_min])/len(recalls_nz_cdd[t-t_min])
        else:
            ave_rc_nz = float('nan') 
        print('average cdd recall = %f'%(ave_rc_nz), end = ' ')

        if len(f1_scores_nz_cdd[t-t_min]) > 0:
            ave_f1_nz = sum(f1_scores_nz_cdd[t-t_min])/len(f1_scores_nz_cdd[t-t_min])
        else:
            ave_f1_nz = float('nan')
        print('average cdd f1-score = %f'%(ave_f1_nz))

def rangeQueryCSM(sess, 
                  csm,
                  csm_data_fetcher,
                  csm_saver,
                  ground_truth,
                  t_min = 1, t_max=FLAGS.GED_threshold-2):


    total_query_num = csm_data_fetcher.get_test_graphs_num()
    train_graph_num = csm_data_fetcher.get_train_graphs_num()

    precisions = [[] for i in range(t_min, t_max+1)]
    recalls = [[] for i in range(t_min, t_max+1)]
    f1_scores = [[] for i in range(t_min, t_max+1)]

    ret_size = [[] for i in range(t_min, t_max+1)]

    precisions_nz = [[] for i in range(t_min, t_max+1)]
    recalls_nz = [[] for i in range(t_min, t_max+1)]
    f1_scores_nz = [[] for i in range(t_min, t_max+1)]

    verify_time = [[] for i in range(t_min, t_max+1)]

    for i in range(0, total_query_num):   

        q = csm_data_fetcher.get_test_graph_gid(i)
        ground_truth[q] = sorted(ground_truth[q], 
                                     key=lambda x: x[1]*10000000 + x[0])
#            if q != 27764:
#                continue
            
        cur_pos = 0
        for t in range(t_min,t_max+1):
                    
            while cur_pos < len(ground_truth[q]) and\
                  ground_truth[q][cur_pos][1] <= t:
                cur_pos = cur_pos + 1

            real_sim_set_wrp = ground_truth[q][0:cur_pos]
            real_sim_set = set([pair[0] for pair in real_sim_set_wrp])



            start_time = time.time()

            similar_set = CSMVerification(i, list(csm_data_fetcher.gid2graph.keys()), 
                                          csm_data_fetcher, 
                                          t, csm, sess, csm_saver)
            verify_time[t-t_min].append(time.time()-start_time)
#                similar_set = set([int(gid) for gid in ret.split()])
            if i == 0:
                print('verification time: {:f} s'.format(time.time()-start_time))

            ret_size[t-t_min].append(len(similar_set))


            tmp = similar_set & real_sim_set
            if len(similar_set) == 0:
                if len(real_sim_set) == 0:
                    precision = 1
                else:
                    precision = 0
            else:
                precision =  len(tmp)/len(similar_set)
            """
                if precision != 1:
                    if len(similar_set) > 0:
                        print(q)
                        print('t=%d'%t)
                        print(similar_set)
                        print(real_sim_set_wrp)
                        print(ground_truth[q][0:5])
                        raise RuntimeError('bug')
            """
            precisions[t-t_min].append(precision)
            if len(real_sim_set) > 0:
                precisions_nz[t-t_min].append(precision)

            if len(real_sim_set) == 0:
                #zero_cnt[t-t_min] = zero_cnt[t-t_min] + 1
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
    for t in range(t_min, t_max+1):
        print('threshold = {:d}'.format(t), end=' ')

        print('average precision = %f'%(sum(precisions[t-t_min])/len(precisions[t-t_min])), end = ' ')
        print('average recall = %f'%(sum(recalls[t-t_min])/len(recalls[t-t_min])), end = ' ')
        print('average f1-score = %f'%(sum(f1_scores[t-t_min])/len(f1_scores[t-t_min])), end = ' ')
        print('average return size = %f'%(sum(ret_size[t-t_min])/len(ret_size[t-t_min])), end = ' ')
        print('average verify time = {:f}'.format(sum(verify_time[t-t_min])/len(verify_time[t-t_min])))

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

