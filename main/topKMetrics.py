# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 21:20:49 2019

@author: dell
"""
import sys
import os
import scipy.stats as ss

sys.path.insert(0, '../model')
from DataFetcher import DataFetcher

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('parameters are GT_fname, output_fname, K')
        os._exit(0)

    GT_fname = str(sys.argv[1])
    output_fname = str(sys.argv[2])
#    GED_thres = int(sys.argv[3])
    K = int(sys.argv[3])
    ground_truth = {}

#    data_fetcher = DataFetcher(sys.argv[5], True)

    f = open(GT_fname, 'r')
    ged_cnt = {}
    for line in f.readlines():
        g, q, d = line.split(' ')
        g = int(g)
        q = int(q)
        d = int(d)

        if q not in ground_truth.keys():
            ground_truth[q] = []
#            ground_truth[q] = set()
#        if d < GED_thres:
#        ground_truth[q].append((g,d))
        ground_truth[q].append((g,d))
        ged_cnt.setdefault(d,0)
        ged_cnt[d] = ged_cnt[d] + 1
    
    f.close()

    f = open(output_fname, 'r')
    res = {}
    for line in f.readlines():
        ids = line.split()
        q = int(ids[0])
        if q not in res.keys():
            res[q] = []
        for gid in ids[1:]:
            res[q].append(int(gid))

    print('finish reading files')

    PAtKs = []
    SRCCs = []
    KRCCs = []

    for q in ground_truth.keys():
#        break
        
        #true_top_k = set()
        ground_truth[q] = sorted(ground_truth[q], 
                                 key=lambda x: x[1]*10000000 + x[0])
        #print(q)
        #print(ground_truth[q][0:4])
#        pos = 0
#        while ground_truth[q][pos][1] <= GED_thres:
            
#            real_sim_set.add(ground_truth[q][pos][0])
#            pos = pos + 1
        
        est_top_k = set(res[q])

        pos = K
        while ground_truth[q][pos][1] == ground_truth[q][pos-1][1]:
            pos = pos + 1
            if pos == len(ground_truth[q]):
                break

        true_top_k_wrp = ground_truth[q][0:pos]
        
        true_top_k = [pair[0] for pair in true_top_k_wrp]

        PAtKs.append(len(set(est_top_k)&set(true_top_k)) / K)
   
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
              
        rho, _ = ss.spearmanr(list(range(1,K+1)), real_rank)
        SRCCs.append(rho)
        tau, _ = ss.kendalltau(list(range(1, K+1)), real_rank)
        KRCCs.append(tau)
        

    print('For Top-k query, k={:d}'.format(K))
    print('average precision at k = {:f}'.format(sum(PAtKs)/len(PAtKs)))
    print('average rho = {:f}'.format(sum(SRCCs)/len(SRCCs)))
    print('average tau = {:f}'.format(sum(KRCCs)/len(KRCCs)))

