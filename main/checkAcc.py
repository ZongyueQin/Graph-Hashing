# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 21:20:49 2019

@author: dell
"""
import sys
import os

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print('parameters are GT_fname, output_fname, GED_thres')
        os._exit(0)
    GT_fname = str(sys.argv[1])
 
    output_fname = str(sys.argv[2])

    GED_thres = int(sys.argv[3])

    f = open(GT_fname, 'r')
    ged_cnt = {}
    ground_truth = {}
    for line in f.readlines():
        g, q, d = line.split(' ')
        g = int(g)
        q = int(q)
        d = int(d)

        if q not in ground_truth.keys():
#            ground_truth[q] = []
            ground_truth[q] = set()
        if d <= GED_thres:
#        ground_truth[q].append((g,d))
            ground_truth[q].add(g)
        ged_cnt.setdefault(d,0)
        ged_cnt[d] = ged_cnt[d] + 1
    print(ged_cnt)
    
    f.close()

    f = open(output_fname, 'r')
    res = {}
    for line in f.readlines():
        ids = line.split()
        q = int(ids[0])
        if q not in res.keys():
            res[q] = set()
#        print('q = %d'%q)
        for gid in ids[1:]:
#            print(gid, end = ' ')
            res[q].add(int(gid))
#        print('')
#        print(q)
#        print(res[q])
#        break

        
            
#        q, g, p = line.split(' ')
#        g = int(g)
#        q = int(q)
#        p = float(p)
#
#        if q not in res.keys():
#            res[q] = []
#        if p > p_thres:
#            res[q].append(g)

    print('finish reading files')
    precisions = []
    recalls = []
    f1_scores = []
    zero_cnt = 0

    precisions_nz = []
    recalls_nz = []
    f1_scores_nz = []

    ret_size = []
    intersect_size = []

    for q in ground_truth.keys():
#        break
        
        real_sim_set = set()
#        ground_truth[q] = sorted(ground_truth[q], 
#                                 key=lambda x: x[1]*10000000 + x[0])
        #print(q)
        #print(ground_truth[q][0:4])
#        pos = 0
#        while ground_truth[q][pos][1] <= GED_thres:
            
#            real_sim_set.add(ground_truth[q][pos][0])
#            pos = pos + 1
        
        similar_set = res[q]
        real_sim_set = ground_truth[q]
        ret_size.append(len(similar_set))
        tmp = similar_set & real_sim_set
        intersect_size.append(len(tmp))
        if len(similar_set) == 0:
            if len(real_sim_set) == 0:
                precision = 1
            else:
                precision = 0
        else:
            precision =  len(tmp)/len(similar_set)

        precisions.append(precision)
        if len(real_sim_set) > 0:
            precisions_nz.append(precision)

        if len(real_sim_set) == 0:
            zero_cnt = zero_cnt + 1
            if len(similar_set) == 0:
                recall = 1
            else:
#                print('error', q)
                recall = 0
        else:
            recall = len(tmp)/len(real_sim_set)
        recalls.append(recall)
              
        if len(real_sim_set) > 0:
            recalls_nz.append(recall)

        if precision * recall == 0:
            if len(real_sim_set) == 0 and\
               len(similar_set) == 0:
                f1_score = 1
            else:
                f1_score = 0
        else:
            f1_score = 2*precision*recall/(precision+recall)

        f1_scores.append(f1_score)
        if len(real_sim_set) > 0:
            f1_scores_nz.append(f1_score)
#        else:
#            assert(f1_score == 1 and precision == 1 and recall == 1)


    print('average precision = %f'%(sum(precisions)/len(precisions)))
    print('average recall = %f'%(sum(recalls)/len(recalls)))
    print('average f1-score = %f'%(sum(f1_scores)/len(f1_scores)))
    print('average returned size = %f'%(sum(ret_size)/len(ret_size)))
    print('average tmp size = %f'%(sum(intersect_size)/len(intersect_size)))
    print('zero_cnt = %d'%zero_cnt)

    if zero_cnt < len(ground_truth.keys()):
        print('ignore empty answers')
        print('average precision = %f'%(sum(precisions_nz)/len(precisions_nz)))
        print('average recall = %f'%(sum(recalls_nz)/len(recalls_nz)))
        print('average f1-score = %f'%(sum(f1_scores_nz)/len(f1_scores_nz)))

#print(ged_cnt)
 
